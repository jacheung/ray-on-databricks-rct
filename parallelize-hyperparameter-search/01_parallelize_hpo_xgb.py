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

!pip install --quiet scikit-learn bayesian-optimization==1.5.1 ray[default]==2.42.0 ray[tune]==2.42.0 optuna mlflow==2.20.1
dbutils.library.restartPython()


# COMMAND ----------

# MAGIC %md 
# MAGIC ## Baseline: one XGB model
# MAGIC This short code-snippet below builds one XGBoost model using a specific set of hyperparameters. It'll provide a starting point for us before we parallelize. Ensure you understand what's going on here before we move onto the next steps. 

# COMMAND ----------

# from sklearn.datasets import make_classification
# from sklearn.model_selection import train_test_split
# from xgboost import XGBClassifier
# import numpy as np
# from sklearn.metrics import mean_squared_error

# # Create a pseudo-dataset to test
# data, labels = make_classification(n_samples=1_000_000, 
#                                    n_features=100, 
#                                    n_informative=10, 
#                                    n_classes=3)
# # Perform train test split
# train_x, test_x, train_y, test_y = train_test_split(data, labels, test_size=0.25)

# # XGB sample hyperparameter configs
# config = {'objective':'multi:softmax',
#           'eval_metric': 'mlogloss', 
#           'learning_rate':0.05,
#           'n_estimators':1000,
#           'early_stopping_round': 20, 
#           'n_jobs': 16, 
#           'random_state':42}

# # Train the classifier
# bst = XGBClassifier(**config)
# bst.fit(train_x, train_y, eval_set=[(test_x, test_y)])

# # Retrieve the evaluation metric values from the training process
# results = bst.evals_result()
# final_eval_metric = results['validation_0'][config['eval_metric']][-1]
# final_eval_metric

# COMMAND ----------

# MAGIC %md 
# MAGIC ## 2. Parallelize hyperparameter tuning for XGboost
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

# The below configuration mirrors my Spark worker cluster set up. Change this to match your cluster configuration. 
setup_ray_cluster(
  min_worker_nodes=2,
  max_worker_nodes=8,
  num_cpus_per_node=16,
  num_gpus_worker_node=0,
  collect_log_to_path="/dbfs/Users/jon.cheung@databricks.com/ray_collected_logs"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2b. Use Ray Tune to parallelize hyperparameter search
# MAGIC
# MAGIC ![](images/xgboost_ray_tune.jpg)

# COMMAND ----------

import os
import numpy as np
import mlflow
from mlflow.utils.databricks_utils import get_databricks_env_vars
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from xgboost import XGBClassifier
from ray import train, tune
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.tune.search.optuna import OptunaSearch

# Set the number of trials to run
num_samples = 48

# Set mlflow experiment name
experiment_name = '/Users/jon.cheung@databricks.com/ray-xgb-nike-fresh'
mlflow.set_experiment(experiment_name)
mlflow_db_creds = get_databricks_env_vars("databricks")

# Define a training function to parallelize
def train_classifier(config: dict,
                    experiment_name: str,
                    parent_run_id: str,
                    mlflow_credentials: dict,
                    ):
    """
    This objective function trains an XGBoost model given a set of sampled hyperparameters. There is no returned value but a metric that is sent back to the driver node to update the progress of the HPO run.

    config: dict, defining the sampled hyperparameters to train the model on.
    **The below three parameters are used for nesting each HPO run as a child run**
    experiment_name: str, the name of the mlflow experiment to log to. This is inherited from the driver node that initiates the mlflow parent run.
    parent_run_id: str, the ID of the parent run. This is inherited from the driver node that initiates the mlflow parent run.
    mlflow_credentials: dict, the credentials for logging to mlflow. This is inherited from the driver node. 
    """
    # Set mlflow credentials and active MLflow experiment within each Ray task
    os.environ.update(mlflow_credentials)
    mlflow.set_experiment(experiment_name)

    # Write code to import your dataset here
    data, labels = make_classification(n_samples=1_000_000, 
                                     n_features=100, 
                                     n_informative=10, 
                                     n_classes=3)
    # Split into train and test set
    train_x, test_x, train_y, test_y = train_test_split(data, labels, test_size=0.25)
    with mlflow.start_run(run_name='xgb_model_hpo_250307', 
                          nested=True,
                          parent_run_id=parent_run_id):

        # Train the classifier
        bst = XGBClassifier(**config)
        bst.fit(train_x, train_y, eval_set=[(test_x, test_y)], verbose=False)

        # Retrieve the evaluation metric values from the training process
        eval_results = bst.evals_result()
        final_eval_metric = eval_results['validation_0'][config['eval_metric']][-1]
        
        # write mlflow metrics
        mlflow.log_params(config)
        mlflow.log_metrics({f'validation_{config["eval_metric"]}': final_eval_metric})

    # Return evaluation results back to driver node
    train.report({config['eval_metric']: final_eval_metric, "done": True})

# By default, Ray Tune uses 1 CPU/trial. LightGBM leverages hyper-threading so we will utilize all CPUs in a node per instance. Since I've set up my nodes to have 64 CPUs each, I'll set the "cpu" parameter to 64. Feel free to tune this down if you're seeing that you're not utilizing all the CPUs in the cluster. 
trainable_with_resources = tune.with_resources(train_classifier, 
                                               {"cpu": 16})

# Define the hyperparameter search space.
# XGB sample hyperparameter configs
param_space = {
    "objective": "multi:softmax",
    'eval_metric': 'mlogloss', 
    "learning_rate": tune.uniform(0.01, 0.3),
    "n_estimators": tune.randint(100, 1000),
    "early_stopping_rounds": tune.randint(3, 20),
    "n_jobs": 16, 
    "random_state": 42
}

# Set up search algorithm. Here we use Optuna and use the default the Bayesian sampler (i.e. TPES)
optuna = OptunaSearch(metric="mlogloss", 
                      mode="min")

with mlflow.start_run(run_name ='parallelized_16_cores') as parent_run:
    tuner = tune.Tuner(
        ray.tune.with_parameters(
            trainable_with_resources,
            experiment_name=experiment_name,
            parent_run_id = parent_run.info.run_id,
            mlflow_credentials=mlflow_db_creds),
        tune_config=tune.TuneConfig(num_samples=num_samples,
                                    search_alg=optuna),
        param_space=param_space
        )
    results = tuner.fit()

results.get_best_result(metric="mlogloss", 
                        mode="min").config
