# Databricks notebook source
# MAGIC %md
# MAGIC # Parallelized Bayesian Hyperparameter Tuning for XGBoost with Ray 
# MAGIC
# MAGIC Boosting algorithms, like XGboost, offers a simple, yet powerful, model to solve many regression and classification problems. However, they are prone to overfitting and require hyperparameter tuning with validation datasets to ensure they can be generalized to the real-world problems they are meant to solve. When it comes to hyperparameter tuning, traditional grid-search is inefficient (i.e. unnecessarily time-consuming). It offers little benefit over more efficient methods like Bayesian search, especially when the search space is large. To double-click on this, Bayesian search balances exploration and exploitation. It explores the search space and uses this as a prior to determine which area to search more in-depth for later trials. 
# MAGIC
# MAGIC This notebook outlines three powerful additions to XGBoost to improve (i.e. make more efficient and speed up) hyperparameter search. They are:
# MAGIC 1. `Ray Tune` for parallelized search across a cluster
# MAGIC 2. `Ray Data` for further parallelizing grouped data HPO search
# MAGIC 3. `Optuna` for Bayesian search

# COMMAND ----------

# MAGIC %md
# MAGIC Specs for the compute I used
# MAGIC ```json
# MAGIC {
# MAGIC     "num_workers": 8,
# MAGIC     "cluster_name": "Multi Node MLR",
# MAGIC     "spark_version": "16.4.x-cpu-ml-scala2.13",
# MAGIC     "spark_conf": {
# MAGIC         "spark.databricks.pyspark.dataFrameChunk.enabled": "true"
# MAGIC     },
# MAGIC     "node_type_id": "m5d.12xlarge",
# MAGIC     "driver_node_type_id": "m5d.4xlarge",
# MAGIC     "spark_env_vars": {
# MAGIC         "DATABRICKS_SERVER_HOSTNAME": "e2-demo-field-eng.cloud.databricks.com",
# MAGIC         "DATABRICKS_HTTP_PATH": "/sql/1.0/warehouses/856a528773be741d",
# MAGIC         "DATABRICKS_TOKEN": "{{secrets/notebooks/ray-gtm-examples-sql-warehouse-token}}"
# MAGIC     },
# MAGIC     "autotermination_minutes": 45,
# MAGIC     "enable_elastic_disk": true,
# MAGIC     "init_scripts": [],
# MAGIC     "single_user_name": "jon.cheung@databricks.com",
# MAGIC     "enable_local_disk_encryption": false,
# MAGIC     "data_security_mode": "SINGLE_USER",
# MAGIC     "runtime_engine": "STANDARD",
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
groups=4

catalog = "main"
schema = "ray_gtm_examples"
table = f"synthetic_data_{num_training_rows}_rows_{num_training_columns}_columns_{num_labels}_labels_{groups}_groups"
label="target"
# Set mlflow experiment name
mlflow_experiment_name = '/Users/jon.cheung@databricks.com/ray-xgb-groups'

# Set the number of HPO trials to run per group
num_samples = 16

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
os.environ['DATABRICKS_TOKEN'] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

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


# train_dataset, val_dataset = ds.train_test_split(test_size=0.25)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1b. Use Ray Tune and Ray Data to parallelize hyperparameter search for groups

# COMMAND ----------

import os
import numpy as np
import mlflow
import xgboost
from ray import train, tune
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.tune.search.optuna import OptunaSearch
from ray.train.xgboost import XGBoostTrainer, RayTrainReportCallback
from sklearn.model_selection import train_test_split
import pandas as pd
from datetime import datetime
from typing import List
import itertools


mlflow.set_experiment(mlflow_experiment_name)

# Define a training function to parallelize
def trainable_xgb(config: dict, 
                  train_X: pd.DataFrame,
                  train_y: pd.DataFrame,
                  val_X: pd.DataFrame,
                  val_y: pd.DataFrame):
    """
    This objective function trains an XGBoost model given a set of sampled hyperparameters.

    config: dict, defining the sampled hyperparameters to train the model on.
    **The below three parameters are used for nesting each HPO run as a child run**
    experiment_name: str, the name of the mlflow experiment to log to. This is inherited from the driver node that initiates the mlflow parent run.
    parent_run_id: str, the ID of the parent run. This is inherited from the driver node that initiates the mlflow parent run.
    mlflow_credentials: dict, the credentials for logging to mlflow. This is inherited from the driver node. 
    """
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
    eval_metric = evals_results['validation'][config['eval_metric']][-1]

    tune.report({config['eval_metric']: eval_metric, "done": True})
    
    return bst, evals_results

def global_tuner(df: pd.DataFrame,
                 columns_to_group: List[str],
                 cpu_resources_per_trial: int,
                 max_concurrent_trials: int,
                 param_space: dict):
    """
    Runs distributed hyperparameter optimization for XGBoost models using Ray Tune and MLflow logging, grouped by specified columns.

    Args:
        df (pd.DataFrame): Input DataFrame containing features, label, and group columns.
        columns_to_group (List[str]): List of column names to group by for separate model training.
        cpu_resources_per_trial (int): Number of CPUs allocated per trial.
        max_concurrent_trials (int): Maximum number of concurrent trials per group.
        param_space (dict): Hyperparameter search space for Ray Tune.

    Returns:
        pd.DataFrame: DataFrame with group identifiers, experiment IDs, and local model artifact paths.
    """
    mlflow.set_experiment(mlflow_experiment_name)
    
    # Today's date
    today = datetime.now()
    formatted_date = today.strftime("%y%m%d")
    
    # Check for single unique values and create unique group name for run
    non_unique_cols = [col for col in columns_to_group if df[col].nunique() != 1]
    if non_unique_cols:
        print(f"❌ Error: The following columns do not have a single unique value: {non_unique_cols}")
        return None
    unique_values = [str(df[col].iloc[0]) for col in columns_to_group]
    group_name = '_'.join(unique_values)
    
    # Train test split
    X = df.drop([label] + columns_to_group, axis=1)
    y = df[label]
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    # Define trainable with resources for each trial
    trainable_with_resources = tune.with_resources(trainable_xgb, {"cpu": cpu_resources_per_trial})

    with mlflow.start_run(run_name=f"group_{group_name}_{formatted_date}") as parent_run:
        # Set up search algorithm. Here we use Optuna and use the default the Bayesian sampler (i.e. TPES)
        optuna = OptunaSearch(metric=param_space['eval_metric'], 
                            mode="min",
                            study_name=str(group_name))
        # Kick off HPO run across the cluster with Ray Tune
        tuner = tune.Tuner(
            ray.tune.with_parameters(
                trainable_with_resources,
                train_X=X_train,
                train_y=y_train,
                val_X=X_test, 
                val_y=y_test),
            run_config=tune.RunConfig(name='mlflow',
                                callbacks=[MLflowLoggerCallback(
                                    experiment_name=mlflow_experiment_name,
                                    save_artifact=True,
                                    log_params_on_trial_end=True,
                                    tags={"mlflow.parentRunId": parent_run.info.run_id})]
                                ),
            tune_config=tune.TuneConfig(num_samples=num_samples,
                                        max_concurrent_trials=max_concurrent_trials,
                                        search_alg=optuna,
                                        reuse_actors = True # Highly recommended for short training jobs (NOT RECOMMENDED FOR GPU AND LONG
                                        ),
            param_space=param_space
            )
        
        results = tuner.fit()
        
        best_config = results.get_best_result(metric=param_space['eval_metric'], 
                            mode="min").config
        
        best_model, evaluation_results = trainable_xgb(config=best_config,
                                                       train_X=X_train,
                                                       train_y=y_train,
                                                       val_X=X_test, 
                                                       val_y=y_test)
        
        with mlflow.start_run(run_name=f"BEST_group_{group_name}_{formatted_date}",
                               nested=True) as best_run:
            mlflow.xgboost.log_model(best_model, 
                                    name="best_model")
            local_run_model_path = os.path.join(best_run.info.run_id, "best_model", "model.xgb")
            mlflow.log_metrics({best_config['eval_metric']: evaluation_results['validation'][best_config['eval_metric']][-1]})
            mlflow.log_params(best_config)
    
    best_runs = pd.DataFrame.from_dict({'group_id': [group_name],
                                        'experiment_id': mlflow.get_experiment_by_name(mlflow_experiment_name).experiment_id,
                                        'local_run_model_path': [local_run_model_path]})
    return best_runs


# COMMAND ----------

# MAGIC %md
# MAGIC Based on my cluster of 384 CPUs in the Ray cluster, I can run 96 trials in parallel (i.e. 24 trials per group). The below should run in 6 mins. 

# COMMAND ----------

#TODO define parameters for grouping and runs.
group_column=['group_id']
group_concurrency = 4
num_cpu_per_trial = 4

# Get total number of CPUs and calculate the max number of concurrent trials based on the number of concurrent groups.
resources = ray.cluster_resources()
total_cluster_cpus = resources.get("CPU") 
max_concurrent_trials_per_group = max(1, int(total_cluster_cpus // num_cpu_per_trial // group_concurrency))


# Define the hyperparameter search space.
# XGB sample hyperparameter configs
param_space = {
    "objective": "multi:softmax", # change to reg:squarederror for regression
    'eval_metric': 'mlogloss', # change to rmse/mae/rmsle for regression
    "num_class": num_labels, # remove for regression
    "learning_rate": tune.uniform(0.01, 0.3),
    "num_estimators": tune.randint(100, 1000),
    "early_stopping_rounds": tune.randint(3, 20),
    "n_jobs": num_cpu_per_trial, 
    "random_state": 42
}


results = ds.groupby(group_column).map_groups(global_tuner, 
                                    concurrency=group_concurrency,
                                    fn_kwargs={'columns_to_group': group_column,
                                               'cpu_resources_per_trial': num_cpu_per_trial,
                                               'max_concurrent_trials': max_concurrent_trials_per_group, 
                                               'param_space': param_space})

ls = results.to_pandas()
ls.to_json('artifact_map.json', orient='records', lines=True)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## OPTIONAL: Create and log a router model
# MAGIC
# MAGIC When dealing with many models for all the groups, it sometimes make more sense to route all of them behind a single model/endpoint. It becomes easier to manage and maintain. 
# MAGIC
# MAGIC This model router pyfunc packages all the model artifacts (i.e. model file) and retrieves it at inference time.
# MAGIC

# COMMAND ----------

import mlflow.pyfunc
import xgboost
import pandas as pd
import os
import pickle

class ModelRouter(mlflow.pyfunc.PythonModel):
    def __init__(self, group_columns: List[str]):
        self.group_columns = group_columns
    
    def load_model(self, context, group_identifier):
        # Retrieve the local (i.e. within the serving endpoint) path for the model.
        path = context.artifacts[group_identifier]
        
        # For XGBoost classifier/regressor but feel free to swap to a different model. 
        bst =  xgboost.Booster()
        bst.load_model(path)
        
        return bst

    def predict(self, context, model_input):
        # Retrieve the unique model IDs and then loop through all of them for inference. 
        # Consider using an async call to leverage all the CPU threads    
        unique_combinations = model_input[self.group_columns].drop_duplicates()
        
        preds = []
        for _, combo in unique_combinations.iterrows():
            mask = pd.Series([True] * len(model_input))
            for col in group_column:
                mask &= (model_input[col] == combo[col])
            group_df = model_input[mask]
            
            # stitch together unique combos 
            combo_str = '_'.join(str(combo[col]) for col in group_column)
            model = self.load_model(context, combo_str)

            # Remove group_id column for prediction
            X = group_df.drop(columns=['group_id'])
            dmatrix = xgboost.DMatrix(X)
            pred = model.predict(dmatrix)
            # Store predictions with index
            preds.append(pd.DataFrame({'index': group_df.index, 'prediction': pred}))
        # Concatenate all predictions and sort by index
        result = pd.concat(preds).sort_values('index')
        return result['prediction'].values


# Create a local map for the models
local_map = {}
for idx, row in ls.iterrows():
    path = row['local_run_model_path']
    local_dir = mlflow.artifacts.download_artifacts(f'runs:/{path}')
    local_map[row['group_id']]=local_dir

# Save the ModelRouter as a pyfunc model
mlflow.pyfunc.save_model(path='model_router',
    python_model=ModelRouter(group_column),
    artifacts=local_map
)

# COMMAND ----------

shutdown_ray_cluster()

df = spark.table(
    f"{catalog}.{schema}.{table}"
).sample(
    withReplacement=False,
    fraction=0.1
).limit(100000)

pdf = df.toPandas().drop(label, axis=1)

model = mlflow.pyfunc.load_model('model_router')

results = model.predict(pdf)

# COMMAND ----------

results
