# Databricks notebook source
# MAGIC %pip install dbldatagen
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

catalog_name = 'main'
schema_name = 'ray_gtm_examples'

# COMMAND ----------

dbutils.widgets.text("num_training_rows", "100", "num_training_rows")
num_training_rows = int(dbutils.widgets.get("num_training_rows"))
print(f"Generating {num_training_rows} synthetic rows")

dbutils.widgets.text("num_training_columns", "1000", "num_training_columns")
num_training_columns = int(dbutils.widgets.get("num_training_columns"))
print(f"Generating {num_training_columns} synthetic columns")

dbutils.widgets.text("num_labels", "2", "num_labels")
num_labels = int(dbutils.widgets.get("num_labels"))
print(f"Generating {num_labels} synthetic labels")

concurrency = sc.defaultParallelism
print(f"Setting Spark.XGBoost num_workers to {concurrency} = num cores on workers in cluster")

# COMMAND ----------

import dbldatagen as dg
from pyspark.sql.types import FloatType, IntegerType, StringType

testDataSpec = (
    dg.DataGenerator(spark, name="synthetic_data", rows=num_training_rows, partitions=concurrency)
    .withIdOutput()
    .withColumn(
        "r",
        FloatType(),
        expr="rand()",
        numColumns=num_training_columns,
    )
    .withColumn(
      "target",
      IntegerType(),
      expr=f"floor(rand()*{num_labels})",
      numColumns=1
      )
)

df = testDataSpec.build()
df = df.repartition(50)

# COMMAND ----------


import os
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import catalog
from databricks.sdk.errors import ResourceAlreadyExists

w = WorkspaceClient()

# Create a CreateSchema object with the desired schema name and catalog
try:   
    created_schema = w.schemas.create(name=schema_name,
                                      catalog_name=catalog_name)
    print(f"Schema '{created_schema.name}' created successfully")
except:
    # Handle the case where the schema already exists
    print(f"Schema '{schema_name}' already exists in catalog '{catalog_name}'. Skipping catalog creation.")

# Create a volume for the parquet files if that doesn't exist
parquet_write_path = f'/Volumes/{catalog_name}/{schema_name}/synthetic_data'
if not os.path.exists(parquet_write_path):
    created_volume = w.volumes.create(catalog_name=catalog_name,
                                        schema_name=schema_name,
                                        name='synthetic_data',
                                        volume_type=catalog.VolumeType.MANAGED
                                        )
    print(f"Volume 'synthetic_data' at {parquet_write_path} created successfully")
else:
    print(f"Volume {parquet_write_path} already exists. Skipping volumes creation.")

# write table
df.write.format("delta").mode("overwrite").saveAsTable(f"{catalog_name}.{schema_name}.synthetic_data_{num_training_rows}_rows_{num_training_columns}_columns_{num_labels}_labels")
# write parquet
df.write.mode("overwrite").format("parquet").save(f"{parquet_write_path}/synthetic_data_{num_training_rows}_rows_{num_training_columns}_columns_{num_labels}_labels")
