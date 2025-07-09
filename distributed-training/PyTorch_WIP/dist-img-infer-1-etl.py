# Databricks notebook source
dbutils.widgets.text("path", "/tmp/flowers/delta", "Output Delta table path")
dbutils.widgets.text("ckpt_path", "/tmp/flowers/ckpt", "checkpointLocation path")

# COMMAND ----------

# MAGIC %md ## 1. ETL images into a Delta table
# MAGIC
# MAGIC ---
# MAGIC * Use [flowers dataset](https://www.tensorflow.org/datasets/catalog/tf_flowers) hosted under `/databricks-datasets`.
# MAGIC * Use the Auto Loader with binary file data source to load images in a Delta table.
# MAGIC * Extract image metadata and store them together with image data.
# MAGIC * Use Delta Lake to simplify data management.

# COMMAND ----------

import io
import numpy as np
import pandas as pd
from pyspark.sql.functions import col, pandas_udf, regexp_extract
from PIL import Image

# COMMAND ----------

# MAGIC %md ### The flowers dataset
# MAGIC
# MAGIC This example uses the [flowers dataset](https://www.tensorflow.org/datasets/catalog/tf_flowers) from the TensorFlow team.
# MAGIC It contains flower photos stored under five sub-directories, one per class, and is available in Databricks Datasets for easy access.

# COMMAND ----------

# MAGIC %fs ls /databricks-datasets/flower_photos

# COMMAND ----------

# MAGIC %md ### Use the Auto Loader with binary file data source to load images in a Delta table
# MAGIC
# MAGIC Databricks Runtime supports the binary file data source, which reads binary files and converts each file into a single record that contains the raw content and metadata of the file.
# MAGIC
# MAGIC Auto Loader (`cloudFiles` data source) incrementally and efficiently processes existing and new data files as they arrive.
# MAGIC
# MAGIC Auto Loader supports two modes for detecting new files. This notebook demonstrates the default directory listing mode. The file notification mode might provide better performance. For more information, see the Auto Loader documentation ([AWS](https://docs.databricks.com/spark/latest/structured-streaming/auto-loader.html)|[Azure](https://docs.microsoft.com/en-us/azure/databricks/spark/latest/structured-streaming/auto-loader)|[GCP](https://docs.gcp.databricks.com/spark/latest/structured-streaming/auto-loader.html)).

# COMMAND ----------

images = spark.readStream.format("cloudFiles") \
  .option("cloudFiles.format", "binaryFile") \
  .option("recursiveFileLookup", "true") \
  .option("pathGlobFilter", "*.jpg") \
  .load("/databricks-datasets/flower_photos") \
  .repartition(4)

# COMMAND ----------

# MAGIC %md ###Expand the DataFrame with extra metadata columns.
# MAGIC
# MAGIC Extract some frequently used metadata from `images` DataFrame:
# MAGIC * extract labels from file paths,
# MAGIC * extract image sizes.

# COMMAND ----------

def extract_label(path_col):
  """Extract label from file path using built-in SQL functions."""
  return regexp_extract(path_col, "flower_photos/([^/]+)", 1)

# COMMAND ----------

def extract_size(content):
  """Extract image size from its raw content."""
  image = Image.open(io.BytesIO(content))
  return image.size

@pandas_udf("width: int, height: int")
def extract_size_udf(content_series):
  sizes = content_series.apply(extract_size)
  return pd.DataFrame(list(sizes))

# COMMAND ----------

df = images.select(
  col("path"),
  extract_size_udf(col("content")).alias("size"),
  extract_label(col("path")).alias("label"),
  col("content"))

# COMMAND ----------

# MAGIC %md ### Save the DataFrame in Delta format.
# MAGIC The following code uses "trigger once" mode to run the streaming job. At the first run, it ingests all the existing image files into the Delta table and exits. If you have a static image folder, this is all that is required.
# MAGIC
# MAGIC If you have continuously arriving new images:
# MAGIC * If the latency requirement is loose, you can schedule a job that runs every day (or other appropriate intervals), and subsequent runs will ingest new images into the Delta table.
# MAGIC * If the latency requirement is strict, you can run this notebook as a "real-time" streaming job by removing `.trigger(once=True)`, and new images will be ingested into the Delta table instantly.
# MAGIC
# MAGIC For more information about Auto Loader and how to optimize its configuration, see the Auto Loader documentation ([AWS](https://docs.databricks.com/spark/latest/structured-streaming/auto-loader.html)|[Azure](https://docs.microsoft.com/en-us/azure/databricks/spark/latest/structured-streaming/auto-loader)|[GCP](https://docs.gcp.databricks.com/spark/latest/structured-streaming/auto-loader.html)) and this [blog post](https://databricks.com/blog/2020/02/24/introducing-databricks-ingest-easy-data-ingestion-into-delta-lake.html).

# COMMAND ----------

# Image data is already compressed, so you can turn off Parquet compression.
# spark.conf.set("spark.sql.parquet.compression.codec", "uncompressed")
# Replace the paths by your preferred paths
# path = dbutils.widgets.get("path")
path = 'main.ray_gtm_examples.flowers_dataset'
ckpt_path = dbutils.widgets.get("ckpt_path")
df.writeStream.format("delta") \
  .option("checkpointLocation", ckpt_path) \
  .trigger(once=True) \
  .start(path)

# COMMAND ----------

# MAGIC %md
# MAGIC On Databricks Runtime 9.0 and above, images in `binaryFile` format that are loaded or saved as Delta tables using Auto Loader have annotations attached so that the image thumbnails are shown when displayed. The command below shows an example. For more information, see the binary file documentation ([AWS](https://docs.databricks.com/data/data-sources/binary-file.html#images)|[Azure](https://docs.microsoft.com/en-us/azure/databricks/data/data-sources/binary-file#images)|[GCP](https://docs.gcp.databricks.com/data/data-sources/binary-file.html#images)).

# COMMAND ----------

# Load the saved Delta table and preview
df_saved = spark.read.format("delta").load(path)
display(df_saved.limit(5))

# COMMAND ----------

# MAGIC %md ###Make SQL queries (optional).

# COMMAND ----------

df_saved.createOrReplaceTempView("tmp_flowers")

# COMMAND ----------

# MAGIC %sql SELECT COUNT(*) FROM tmp_flowers WHERE label = 'daisy'

# COMMAND ----------

# MAGIC %sql SELECT label, COUNT(*) AS cnt FROM tmp_flowers
# MAGIC   WHERE size.width >= 400 AND size.height >= 400
# MAGIC   GROUP BY label ORDER BY cnt
