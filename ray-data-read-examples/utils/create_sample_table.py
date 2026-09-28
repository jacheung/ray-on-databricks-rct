# Databricks notebook source
# DBTITLE 1,Introduction
# MAGIC %md
# MAGIC # create_sample_table
# MAGIC
# MAGIC Shared utility that ensures a lightweight sample Delta table exists in Unity Catalog.
# MAGIC
# MAGIC Designed to be called from any `ray-data-read-examples` notebook via
# MAGIC `dbutils.notebook.run("../utils/create_sample_table", 300, {"catalog": ..., "schema": ..., "table_name": ...})`.
# MAGIC
# MAGIC **Behaviour:**
# MAGIC * If `catalog.schema.table_name` already exists → skip creation and print the path.
# MAGIC * Otherwise → generate 100 000 rows with 10 numeric feature columns and a label column using pure Spark (no extra libraries).
# MAGIC * Returns the fully qualified table name so callers can capture it.
# MAGIC
# MAGIC **Parameters** (all optional — sensible defaults provided):
# MAGIC | Parameter | Default | Description |
# MAGIC |---|---|---|
# MAGIC | `catalog` | `main` | Unity Catalog catalog |
# MAGIC | `schema` | `ray_gtm_examples` | Unity Catalog schema |
# MAGIC | `table_name` | `ray_data_source` | Target table name |

# COMMAND ----------

# DBTITLE 1,Parameters
# Accept parameters from callers; fall back to sensible defaults.
dbutils.widgets.text("catalog", "main", "Unity Catalog catalog")
dbutils.widgets.text("schema", "ray_gtm_examples", "Unity Catalog schema")
dbutils.widgets.text("table_name", "ray_data_source", "Target table name")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
table_name = dbutils.widgets.get("table_name")

full_table_name = f"{catalog}.{schema}.{table_name}"
print(f"Target table: {full_table_name}")

# COMMAND ----------

# DBTITLE 1,Idempotent creation
from pyspark.sql import functions as F

NUM_ROWS = 100_000
NUM_FEATURES = 10
NUM_LABELS = 5

if spark.catalog.tableExists(full_table_name):
    print(f"Table {full_table_name} already exists — skipping creation.")
else:
    print(f"Creating {full_table_name} with {NUM_ROWS:,} rows, {NUM_FEATURES} feature columns, and {NUM_LABELS} labels ...")

    # Build the DataFrame from spark.range + rand() — no extra libraries needed.
    df = spark.range(NUM_ROWS).withColumn("id", F.col("id").cast("long"))

    for i in range(NUM_FEATURES):
        df = df.withColumn(f"feature_{i}", F.rand(seed=42 + i))

    df = df.withColumn("label", (F.rand(seed=0) * NUM_LABELS).cast("int"))

    # Ensure the target schema exists before writing.
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")

    df.write.format("delta").mode("overwrite").saveAsTable(full_table_name)
    print(f"Created {full_table_name} — {spark.table(full_table_name).count():,} rows.")

# COMMAND ----------

# DBTITLE 1,Return table name to caller
# When invoked via dbutils.notebook.run(), this value is returned to the caller.
dbutils.notebook.exit(full_table_name)
