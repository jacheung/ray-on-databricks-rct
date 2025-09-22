# Databricks notebook source
# MAGIC %md
# MAGIC # Parallelized Bayesian Hyperparameter Tuning for XGBoost with Ray 
# MAGIC
# MAGIC Boosting algorithms, like XGboost, offers a simple, yet powerful, model to solve many regression and classification problems. However, they are prone to overfitting and require hyperparameter tuning with validation datasets to ensure they can be generalized to the real-world problems they are meant to solve. When it comes to hyperparameter tuning, traditional grid-search is inefficient (i.e. unnecessarily time-consuming). It offers little benefit over more efficient methods like Bayesian search, especially when the search space is large. To double-click on this, Bayesian search balances exploration and exploitation. It explores the search space and uses this as a prior to determine which area to search more in-depth for later trials. 
# MAGIC
# MAGIC This notebook outlines two powerful additions to XGBoost to improve (i.e. make more efficient) hyperparameter search. They are:
# MAGIC 1. Ray for parallelized search
# MAGIC 2. Optuna for Bayesian search

# COMMAND ----------

# MAGIC %md
# MAGIC Specs for the compute I used
# MAGIC ```json
# MAGIC {
# MAGIC     "autoscale": {
# MAGIC         "min_workers": 2,
# MAGIC         "max_workers": 8
# MAGIC     },
# MAGIC     "cluster_name": "Multi Node MLR",
# MAGIC     "spark_version": "15.4.x-cpu-ml-scala2.12",
# MAGIC     "spark_conf": {
# MAGIC         "spark.databricks.pyspark.dataFrameChunk.enabled": "true"
# MAGIC     },
# MAGIC     "aws_attributes": {
# MAGIC         "first_on_demand": 1,
# MAGIC         "availability": "SPOT_WITH_FALLBACK",
# MAGIC         "zone_id": "auto",
# MAGIC         "spot_bid_price_percent": 100,
# MAGIC         "ebs_volume_type": "GENERAL_PURPOSE_SSD",
# MAGIC         "ebs_volume_count": 3,
# MAGIC         "ebs_volume_size": 100
# MAGIC     },
# MAGIC     "node_type_id": "m5d.4xlarge",
# MAGIC     "driver_node_type_id": "m4.4xlarge",
# MAGIC     "ssh_public_keys": [],
# MAGIC     "custom_tags": {},
# MAGIC     "autotermination_minutes": 45,
# MAGIC     "enable_elastic_disk": false,
# MAGIC     "init_scripts": [],
# MAGIC     "single_user_name": "jon.cheung@databricks.com",
# MAGIC     "enable_local_disk_encryption": false,
# MAGIC     "data_security_mode": "SINGLE_USER",
# MAGIC     "runtime_engine": "STANDARD",
# MAGIC     "effective_spark_version": "15.4.x-cpu-ml-scala2.12",
# MAGIC     "assigned_principal": "user:jon.cheung@databricks.com",
# MAGIC     "cluster_id": "1023-181805-kbecrjfb"
# MAGIC }
# MAGIC ```

# COMMAND ----------

!pip install --quiet xgboost bayesian-optimization==1.5.1 ray[all]=2.49.1 optuna mlflow
dbutils.library.restartPython()


# COMMAND ----------

# MAGIC %md 
# MAGIC ## Parallelize hyperparameter tuning for XGboost
# MAGIC To parallelize hyperparameter tuning we will perform two steps:
# MAGIC - 1a. instantiate a Ray cluster - a Ray cluster is composed of multi-nodes for computing. Since this is Ray on Spark, we can assign `min/max worker_nodes` equal to (or less than) the number of worker nodes in the Spark cluster and `num_cpus_per_node` to the number of CPUs allocated per worker in the Spark cluster. 
# MAGIC - 1b. Use Ray Tune to define and search the hyperparameter space. 

# COMMAND ----------

num_training_rows = 20_000_000
num_training_columns = 100
num_labels = 5

catalog = "main"
schema = "ray_gtm_examples"
table = f"synthetic_data_{num_training_rows}_rows_{num_training_columns}_columns_{num_labels}_labels"
label="target"
# Set mlflow experiment name
mlflow_experiment_name = '/Users/jon.cheung@databricks.com/ray-xgb-nike-fresh'

# Set the number of HPO trials to run
num_samples = 48

# If running in a multi-node cluster, this is where you
# should configure the run's persistent storage that is accessible
# across all worker nodes.
ray_xgboost_path = '/dbfs/Users/jon.cheung@databricks.com/ray_xgboost/' 
# This is for stashing the cluster logs
ray_logs_path = "/dbfs/Users/jon.cheung@databricks.com/ray_collected_logs/"

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1a. Instantiate a Ray cluster
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
# MAGIC ### 1b. Use Ray Tune to parallelize hyperparameter search

# COMMAND ----------

num_cpu_per_trial=48

# Define the hyperparameter search space.
# XGB sample hyperparameter configs
param_space = {
    "objective": "multi:softmax", #TODO change to reg:squarederror for regression
    'eval_metric': 'mlogloss', #TODO change to rmse/mae/rmsle for regression
    "num_class": num_labels, #TODO remove for regression
    "learning_rate": tune.uniform(0.01, 0.3),
    "num_estimators": tune.randint(100, 1000),
    "early_stopping_rounds": tune.randint(3, 20),
    "n_jobs": num_cpu_per_trial, 
    "random_state": 42
}

# COMMAND ----------

import os
import numpy as np
import mlflow
from mlflow.utils.databricks_utils import get_databricks_env_vars
import xgboost
from ray import train, tune
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.tune.search.optuna import OptunaSearch
from ray.train.xgboost import XGBoostTrainer, RayTrainReportCallback


resources = ray.cluster_resources()
total_cluster_cpus = resources.get("CPU") 
max_concurrent_trials = max(1, int(total_cluster_cpus // num_cpu_per_trial))


# Set mlflow experiment name
experiment_name = '/Users/jon.cheung@databricks.com/ray-xgb-nike-fresh'
mlflow.set_experiment(experiment_name)

# Define a training function to parallelize
def train_global_xgb(config: dict,
                    train_data: ray.data.Dataset,
                    val_data: ray.data.Dataset,
                            ):
    """
    This objective function trains an XGBoost model given a set of sampled hyperparameters. There is no returned value but a metric that is sent back to the driver node to update the progress of the HPO run.

    config: dict, defining the sampled hyperparameters to train the model on.
    **The below three parameters are used for nesting each HPO run as a child run**
    experiment_name: str, the name of the mlflow experiment to log to. This is inherited from the driver node that initiates the mlflow parent run.
    parent_run_id: str, the ID of the parent run. This is inherited from the driver node that initiates the mlflow parent run.
    mlflow_credentials: dict, the credentials for logging to mlflow. This is inherited from the driver node. 
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

    dtrain = xgboost.DMatrix(train_X, label=train_y)
    deval = xgboost.DMatrix(val_X, label=val_y)

    evals_results = {}
    bst = xgboost.train(
        config,
        dtrain=dtrain,
        evals=[(deval, "validation")],
        num_boost_round=config['num_estimators'],
        evals_result=evals_results,
        early_stopping_rounds=config['early_stopping_rounds']
    )
    eval_metric = evals_results['validation'][config['eval_metric']]

    tune.report({config['eval_metric']: eval_metric, "done": True})


# By default, Ray Tune uses 1 CPU/trial. XGB leverages hyper-threading so we will utilize all CPUs in a node per instance. Since I've set up my nodes to have 64 CPUs each, I'll set the "cpu" parameter to 64. Feel free to tune this down if you're seeing that you're not utilizing all the CPUs in the cluster. 
trainable_with_resources = tune.with_resources(train_global_xgb, 
                                               {"cpu": num_cpu_per_trial})

# Set up search algorithm. Here we use Optuna and use the default the Bayesian sampler (i.e. TPES)
optuna = OptunaSearch(metric=param_space['eval_metric'], 
                      mode="min")

with mlflow.start_run(run_name ='20M_dataset_48_cores') as parent_run:
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

results.get_best_result(metric=param_space['eval_metric'], 
                        mode="min").config
