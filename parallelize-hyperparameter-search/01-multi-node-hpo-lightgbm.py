# Databricks notebook source
# MAGIC %md
# MAGIC # Parallelized Bayesian Hyperparameter Tuning for LightGBM with Ray 
# MAGIC
# MAGIC Boosting algorithms, like LightGBM, offers a simple, yet powerful, model to solve many regression and classification problems. However, they are prone to overfitting and require hyperparameter tuning with validation datasets to ensure they can be generalized to the real-world problems they are meant to solve. When it comes to hyperparameter tuning, traditional grid-search is inefficient (i.e. unnecessarily time-consuming). It offers little benefit over more efficient methods like Bayesian search, especially when the search space is large. To double-click on this, Bayesian search balances exploration and exploitation. It explores the search space and uses this as a prior to determine which area to search more in-depth for later trials. 
# MAGIC
# MAGIC This notebook outlines two powerful additions to LightGBM to improve (i.e. make more efficient) hyperparameter search. They are:
# MAGIC 1. Ray for parallelized search across a multi-node cluster
# MAGIC 2. Optuna for Bayesian search
# MAGIC
# MAGIC The below are the cluster resources used to train a 50M x 100 column dataset. In short:
# MAGIC - 8 worker nodes each with 48 CPU and 192 GB RAM
# MAGIC - no autoscaling
# MAGIC
# MAGIC Our Ray cluster will not use the driver node, hence why we provision a smaller driver. We will use only the workers in the Ray cluster to run 8 HPO trials in parallel. 
# MAGIC
# MAGIC ```json
# MAGIC {
# MAGIC   "cluster_name": "Multi Node MLR",
# MAGIC   "spark_version": "16.4.x-cpu-ml-scala2.13",
# MAGIC   "aws_attributes": {
# MAGIC       "first_on_demand": 1,
# MAGIC       "availability": "SPOT_WITH_FALLBACK",
# MAGIC       "zone_id": "auto",
# MAGIC       "spot_bid_price_percent": 100
# MAGIC   },
# MAGIC   "node_type_id": "m5d.12xlarge",
# MAGIC   "driver_node_type_id": "m5d.4xlarge",
# MAGIC   "spark_env_vars": {
# MAGIC       "DATABRICKS_SERVER_HOSTNAME": "e2-demo-field-eng.cloud.databricks.com",
# MAGIC       "DATABRICKS_HTTP_PATH": "/sql/1.0/warehouses/856a528773be741d",
# MAGIC       "DATABRICKS_TOKEN": "{{secrets/notebooks/ray-gtm-examples-sql-warehouse-token}}"
# MAGIC   },
# MAGIC   "enable_elastic_disk": true,
# MAGIC   "single_user_name": "jon.cheung@databricks.com",
# MAGIC   "enable_local_disk_encryption": false,
# MAGIC   "data_security_mode": "SINGLE_USER",
# MAGIC   "runtime_engine": "STANDARD",
# MAGIC   "assigned_principal": "user:jon.cheung@databricks.com",
# MAGIC   "num_workers": 8,
# MAGIC }
# MAGIC ```

# COMMAND ----------

# MAGIC %pip install --quiet lightgbm bayesian-optimization==1.5.1 ray[all]=2.49.1 optuna mlflow
# MAGIC dbutils.library.restartPython()
# MAGIC

# COMMAND ----------

num_training_rows = 50_000_000
num_training_columns = 100
num_labels = 5
catalog = "main"
schema = "ray_gtm_examples"

table = f"synthetic_data_{num_training_rows}_rows_{num_training_columns}_columns_{num_labels}_labels"
label="target"
mlflow_experiment_name = f'/Users/jon.cheung@databricks.com/ray-lgbm'

# Set the number of trials to run
num_samples = 16

# If running in a multi-node cluster, this is where you
# should configure the run's persistent storage that is accessible
# across all worker nodes.
ray_xgboost_path = '/dbfs/Users/jon.cheung@databricks.com/ray_xgboost/' 
# This is for stashing the cluster logs
ray_logs_path = "/dbfs/Users/jon.cheung@databricks.com/ray_collected_logs/"

# COMMAND ----------

# MAGIC %md 
# MAGIC ## OPTIONAL Baseline: one LightGBM model
# MAGIC This short code-snippet below builds one LightGBM model using a specific set of hyperparameters. It'll provide a starting point for us before we parallelize. Ensure you understand what's going on here before we move onto the next steps. 

# COMMAND ----------

# from sklearn.datasets import make_regression
# from sklearn.model_selection import train_test_split
# import lightgbm as lgb
# import numpy as np
# from sklearn.metrics import mean_squared_error

# # Create a pseudo-dataset to test
# data, labels = make_regression(n_samples=10_000_000, 
#                                    n_features=100, 
#                                    n_informative=10, 
#                                    n_targets=1)
# # Perform train test split
# train_x, test_x, train_y, test_y = train_test_split(data, labels, test_size=0.25)

# # Build input matrices for LightGBM
# train_set = lgb.Dataset(train_x, label=train_y)
# test_set = lgb.Dataset(test_x, label=test_y)

# # LightGBM hyperparameter configs
# config = {'objective':'regression',
#           'metric': 'rmse',
#           'num_leaves':31,
#           'learning_rate':0.05,
#           'n_estimators':1000,
#           'num_threads': 16, 
#           'random_state':42}
# # Train the classifier
# results = {}
# gbm = lgb.train(config,
#                 train_set,
#                 valid_sets=[train_set, test_set],
#                 valid_names=["train", "validation"],
#                 callbacks = [lgb.record_evaluation(results)]
#                 )

# # Plot tarin and validation metric across time 
# lgb.plot_metric(results)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## 2. Parallelize hyperparameter tuning for LightGBM
# MAGIC To parallelize hyperparameter tuning we will perform two steps:
# MAGIC - 2a. instantiate a Ray cluster - a Ray cluster is composed of multi-nodes for computing. Since this is Ray on Spark, we can assign `min/max worker_nodes` equal to (or less than) the number of worker nodes in the Spark cluster and `num_cpus_per_node` to the number of CPUs allocated per worker in the Spark cluster. 
# MAGIC - 2b. Use Ray Tune to define and search the hyperparameter space. 
# MAGIC
# MAGIC ### 2a. Instantiate a Ray cluster
# MAGIC
# MAGIC The recommended configuration for a Ray cluster is as follows:
# MAGIC - set the `num_cpus_per_node` to the CPU count per worker node (with this configuration, each Apache Spark worker node launches one Ray worker node that will fully utilize the resources of each Apache Spark worker node.)
# MAGIC - set `min_worker_nodes` to the number of Ray worker nodes you want to launch on each node.
# MAGIC - set `max_worker_nodes` to the total amount of worker nodes (this and `min_worker_nodes` together enable autoscaling)

# COMMAND ----------

import ray
from ray.util.spark import setup_ray_cluster, shutdown_ray_cluster
import os

restart = True
if restart is True:
  try:
    shutdown_ray_cluster()
  except:
    pass
  try:
    ray.shutdown()
  except:
    pass

# Set the parameters here so mlflow works properly at all the Ray nodes
os.environ['DATABRICKS_HOST'] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
os.environ['DATABRICKS_TOKEN'] = dbutils.secrets.get(scope = "development", key = "jon_cheung_PAT")

# The below configuration mirrors my Spark worker cluster set up. Change this to match your cluster configuration. 
setup_ray_cluster(
  min_worker_nodes=8,
  max_worker_nodes=8,
  num_cpus_worker_node=48,
  num_gpus_worker_node=0,
  collect_log_to_path="/dbfs/Users/jon.cheung@databricks.com/ray_collected_logs"
)

# COMMAND ----------

import ray
import os

try: 
  ## Option 1 (PREFERRED): Build a Ray Dataset using a Databricks SQL Warehouse
  # Insert your SQL warehouse ID here. I've queried my 100M row dataset using a Small t-shirt sized cluster.

  # Ensure you've set the DATABRICKS_TOKEN so you can query using the warehouse compute
  ds = ray.data.read_databricks_tables(
    warehouse_id='2a72600bb68f00ee',
    catalog=catalog,
    schema=schema,
    query=f'SELECT * FROM {table}',
  )

  print('read directly from UC')
except: 
  ## Option 2: Fallback to building a Ray Dataset using a Parquet files
  # If you have too many Ray nodes, you may not be able to create a Ray dataset using the warehouse method above because of rate limits. One back up solution is to create parquet files from the delta table and build a ray dataset from that. This is not the recommended route because, in essence, you are duplicating data.
  parquet_path = f'/Volumes/{catalog}/{schema}/synthetic_data/{table}'
  ds = ray.data.read_parquet(parquet_path)
  print('read directly from parquet')

train_dataset, val_dataset = ds.train_test_split(test_size=0.25)



# COMMAND ----------

# MAGIC %md
# MAGIC ### 2b. Use Ray Tune to parallelize hyperparameter search
# MAGIC
# MAGIC ![](images/xgboost_ray_tune.jpg)

# COMMAND ----------

import os
import numpy as np
import lightgbm as lgb
import mlflow
from ray import train, tune
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.tune.search.optuna import OptunaSearch

# Define resources per HPO trial and calculate max concurrent HPO trials
num_cpu_per_trial = 48
resources = ray.cluster_resources()
total_cluster_cpus = resources.get("CPU") 
max_concurrent_trials = max(1, int(total_cluster_cpus // num_cpu_per_trial))


# Define a training function to parallelize
def train_global_lgbm(config: dict,
                            train_data: ray.data.Dataset,
                            val_data: ray.data.Dataset,
                            ):
    """
    This objective function trains a LGBM model given a set of sampled hyperparameters. There is no returned value but a metric that is sent back to the may orchestrating tune function to update the progress of the HPO run.
    config: dict, defining the sampled hyperparameters to train the model on.
    """
    # Convert Ray data to pandas DataFrame
    # - materialize dumps the dataset from disk/cloud to cluster object store.
    # - to_pandas pulls from object store to local node
    train_df = train_data.materialize().to_pandas()
    val_df = val_data.materialize().to_pandas()

    train_X = train_df.drop(label, axis=1)
    train_y = train_df[label]
    val_X = val_df.drop(label, axis=1)
    val_y = val_df[label]

    # Build input matrices for LightGBM
    train_set = lgb.Dataset(train_X, label=train_y)
    test_set = lgb.Dataset(val_X, label=val_y)

    # Train the classifier
    results = {}
    gbm = lgb.train(config,
                    train_set,
                    valid_sets=[test_set],
                    valid_names=["validation"],
                    callbacks = [lgb.record_evaluation(results)]
                    )
    # get RMSE of validation set for last iteration
    eval_metric = results['validation']['multi_logloss'][-1]

    # Return evaluation results back to driver node
    tune.report({"multi_logloss": eval_metric, "done": True})

# By default, Ray Tune uses 1 CPU/trial. LightGBM leverages hyper-threading so we will utilize all CPUs in a node per instance. Since I've set up my nodes to have 48 CPUs each, I'll set the "cpu" parameter to 48. Feel free to tune this down if you're seeing that you're not utilizing all the CPUs in the cluster. 
trainable_with_resources = tune.with_resources(train_global_lgbm, 
                                               {"CPU": num_cpu_per_trial})

# Define the hyperparameter search space.
param_space = {
    "objective": "multiclass",
    "num_class": num_labels,
    "metric": "multi_logloss",
    "num_threads": num_cpu_per_trial,    
    "n_estimators": tune.randint(100, 1000),
    "num_leaves": tune.randint(10, 100),
    "early_stopping_round": tune.randint(3, 20),
}

# Set up search algorithm. Here we use Optuna and set the sampler to a Bayesian one (i.e. TPES)
optuna = OptunaSearch(metric=param_space['metric'], 
                      mode="min")

with mlflow.start_run(run_name ='parallelized_64_cores'):
    tuner = tune.Tuner(
        ray.tune.with_parameters(
            trainable_with_resources,
            train_data=train_dataset,
            val_data=val_dataset),
        run_config=tune.RunConfig(name='mlflow',
                          callbacks=[MLflowLoggerCallback(
                              experiment_name=mlflow_experiment_name,
                              save_artifact=True,
                              log_params_on_trial_end=True)]
                          ),
        tune_config=tune.TuneConfig(num_samples=num_samples,
                                    max_concurrent_trials=max_concurrent_trials,
                                    search_alg=optuna),
        param_space=param_space
        )
    results = tuner.fit()

results.get_best_result(metric=param_space['metric'], 
                        mode="min").config
