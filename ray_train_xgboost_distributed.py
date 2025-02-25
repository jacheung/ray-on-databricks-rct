# Databricks notebook source
# MAGIC %md
# MAGIC # Distributed XGBoost Training + Inference with Ray
# MAGIC
# MAGIC XGBoost is one of the most common and powerful boosting library out there; it can be used for both regression and classification problems. Amidst its powerful feature set, there are drawbacks with training time and the requirement to perform extensive hyperparameter search to reduce overfitting. To meet this compute demand, XGBoost natively leverages multi-threading, allowing it to use all the CPU cores on a single-machine. However, what if multi-threading on a single-node is not fast enough?
# MAGIC
# MAGIC Ray offers a distributed version of XGBoost to perform distributed data parallelism. With drop-in replacements of `xgboost` native classes, XGboost Ray allows you to leverage multi-node clusters to distribute your training. 
# MAGIC
# MAGIC [xgboost_example](https://docs.ray.io/en/latest/train/examples/xgboost/xgboost_example.html)

# COMMAND ----------

# MAGIC %pip install -qU databricks-feature-engineering skforecast scikit-lego ray[default] ray[data,train]
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

catalog = "jon_cheung" # Change This
schema = "ray_gtm_examples"
table = "m4_daily"
label="value"

# COMMAND ----------

# MAGIC %md 
# MAGIC ## 1. Create and prepare a dataset
# MAGIC We'll be using the M4 dataset which has 100k different time-series available. We'll be building one global time-series model. This means we'll be using one model to forecast and predict. For the data preparation, we will:
# MAGIC 1. use a radial basis function to encode the day of year 
# MAGIC 2. one-hot encode the unique identifier

# COMMAND ----------

from skforecast.datasets import fetch_dataset

if not spark.catalog.tableExists(f"{catalog}.{schema}.{table}"): 
  df = fetch_dataset(name="m4_daily") 
  df.reset_index(drop=False, inplace=True)
  # Write m4 dataset to the catalog
  spark.createDataFrame(df).write.mode('overwrite').saveAsTable(f"{catalog}.{schema}.{table}")
  print(f"... OK!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Start a Ray Cluster and train a distributed XGBoost model

# COMMAND ----------

import ray
from ray.util.spark import setup_ray_cluster, shutdown_ray_cluster


workers = 4
cpus_per_worker = 16

# The recommended configuration for a Ray cluster is as follows:
# - set the num_cpus_per_node to the CPU count per worker node (with this configuration, each Apache Spark worker node launches one Ray worker node that will fully utilize the resources of each Apache Spark worker node.)
# - set min_worker_nodes to the number of Spark workers. SUBTRACT ONE if you're running Spark tasks as we need to reserve a Spark worker for Spark jobs
# - set max_worker_nodes to the total amount of worker nodes (this and `min_worker_nodes` together enable autoscaling)
setup_ray_cluster(
  min_worker_nodes=workers,
  max_worker_nodes=workers,
  num_cpus_per_node=cpus_per_worker,
  num_gpus_worker_node=0,
  collect_log_to_path="/dbfs/Users/jon.cheung@databricks.com/ray_collected_logs",
  RAY_memory_monitor_refresh_ms=0
)


# COMMAND ----------

from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.ml.functions import vector_to_array
from pyspark.sql.functions import col
from ray.data import from_spark

# 1. Create Spark OHE pipeline on distributed dataset
indexer = StringIndexer(inputCol="series_id", outputCol="series_id_idx")
encoder = OneHotEncoder(
    inputCols=["series_id_idx"],
    outputCols=["series_id_encoded"],
    dropLast=False
)
spark_pipeline = Pipeline(stages=[indexer, encoder])


def ohe_to_columns(fitted_pipeline, pipeline_output_dataframe):
    # 1. Get category mapping from trained model
    string_indexer_model = fitted_pipeline.stages[0]
    categories = string_indexer_model.labels  # Preserves frequency-based order

    # 2. Convert SparseVector to expanded columns
    pipeline_output_dataframe = pipeline_output_dataframe.withColumn("ohe_array", vector_to_array("series_id_encoded"))

    # 3. Create individual binary columns with proper naming
    binary_cols = [
        col("ohe_array")[i].alias(f"series_{category}") 
        for i, category in enumerate(categories)
    ]

    # 4. Create final dataframe with original columns + new binary features
    final_sdf = pipeline_output_dataframe.select(
        [col for col in pipeline_output_dataframe.columns if col not in {"series_id_encoded", "series_id_idx", "ohe_array", "series_id"}] +
        binary_cols
    )

    return final_sdf

# COMMAND ----------

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklego.preprocessing import RepeatingBasisFunction
import numpy as np
import mlflow

# Define our custom day of year transformer for time encoding using Radial Basis Functions
class TimeDayOfYearEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, n_periods=12, input_range=(1, 365)):
        self.n_periods = n_periods
        self.input_range = input_range
        self.rbf = RepeatingBasisFunction(n_periods=n_periods, input_range=input_range)
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_datetime = pd.to_datetime(X)
        year = X_datetime.dt.year.values.reshape(-1, 1)
        doy = X_datetime.dt.dayofyear.values.reshape(-1, 1)
        
        # Apply RepeatingBasisFunction to day of year feature
        doy_encoded = self.rbf.fit_transform(doy)
        
        return np.hstack([year, doy_encoded])

    def get_feature_names_out(self, input_features=None):
        n_rbf_features = self.n_periods
        return ['year'] + [f'doy_rbf_{i}' for i in range(n_rbf_features)]


# Create a sklearn transform pipeline to operate on Pandas Dataframes
def m4_preprocessing_pipeline():
  # Define the categorical columns for one-hot encoding
  time_cols = "timestamp"
  mlflow.autolog(disable=True)
        
  time_pipeline = Pipeline(steps=[("time_decomposition", TimeDayOfYearEncoder())])

  # Create the ColumnTransformer
  preprocessor = ColumnTransformer(
      transformers=[
          ("time", time_pipeline, time_cols),  # Apply time_pipeline to the timestamp column
      ],
      remainder="passthrough",  # Pass through any remaining columns
      sparse_threshold=0  # Ensure dense output
  )

  return preprocessor

# Create a batch transform function for Ray Datasets
def preprocess_batch(batch):
  preprocessor = m4_preprocessing_pipeline()
  df = pd.DataFrame(preprocessor.fit_transform(batch),
                    columns=preprocessor.get_feature_names_out())
  df = df.infer_objects()
  return df

# COMMAND ----------

from pyspark.sql.functions import col
from ray.train.xgboost import XGBoostTrainer
from ray.train import ScalingConfig, RunConfig
from ray.air.integrations.mlflow import MLflowLoggerCallback, setup_mlflow

experiment_name = '/Users/jon.cheung@databricks.com/xgboost-distributed'
mlflow.set_experiment(experiment_name)
distributed_object_store = "/dbfs/Users/jon.cheung@databricks.com/ray_xgb_checkpoints"

# Distributed XGBoost Training
with mlflow.start_run(run_name='xgboost-distributed'):
    # 1. Load Data
    n_series = 500
    ids = [f"D{i}" for i in range(1, n_series+1)]
    spark_df = spark.read.table(f"{catalog}.{schema}.{table}").filter(col("series_id").isin(ids)).select("timestamp", "series_id", "value")

    train_dataset_sdf, validation_dataset_sdf = spark_df.randomSplit(weights=[0.7,0.3], seed=100)

    # 2. Run Spark preprocessing pipeline
    pipeline_model = spark_pipeline.fit(train_dataset_sdf)
    train_dataset_sdf = pipeline_model.transform(train_dataset_sdf)
    validation_dataset_sdf = pipeline_model.transform(validation_dataset_sdf)

    train_dataset_sdf = ohe_to_columns(pipeline_model, train_dataset_sdf)
    validation_dataset_sdf = ohe_to_columns(pipeline_model, validation_dataset_sdf)
    
    # 3. Run Ray preprocessing pipeline
    train_dataset_ray = from_spark(train_dataset_sdf)  
    valid_dataset_ray = from_spark(validation_dataset_sdf)

    train_dataset = train_dataset_ray.map_batches(
                    preprocess_batch,
                    batch_format="pandas"
                )
    valid_dataset = valid_dataset_ray.map_batches(
                        preprocess_batch,
                        batch_format="pandas"
                    )

    # Sample params config ideally you'd load your own from HPO tuning
    params = {"objective": "reg:squarederror",
            "n_estimators": 100,
            "max_depth": 8, 
            "learning_rate": .3,
            "early_stopping_rounds": 10, 
            "num_boosting_rounds": 500,
            "max_bin": 256}

    # 4. Train Distributed XGBoost
    trainer = XGBoostTrainer(
            label_column="remainder__value",
            params=params,
            num_boost_round = params["num_boosting_rounds"],
            run_config=RunConfig(storage_path=distributed_object_store),
            scaling_config=ScalingConfig(num_workers=workers,
                                        resources_per_worker={'CPU': cpus_per_worker-1},
                                        use_gpu=False),
            datasets={"train": train_dataset, "valid": valid_dataset},
    )

    results = trainer.fit()
    # Save spark preprocessor
    mlflow.spark.log_model(pipeline_model, "ohe_pipeline")
    # Save final model checkpoint
    mlflow.log_artifact(results.checkpoint.to_directory(), "model_checkpoint")
    mlflow.log_metrics(results.metrics_dataframe[['train-rmse', 'valid-rmse']].to_dict())
    mlflow.log_params(params)

