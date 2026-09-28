# Databricks notebook source
# /// script
# [tool.databricks.environment]
# base_environment = "databricks_ai_v5"
# environment_version = "5"
# ///
# DBTITLE 1,Introduction
# MAGIC %md
# MAGIC ## Reading Unity Catalog Tables with `ray.data.read_delta` on Serverless GPU (AI Runtime)
# MAGIC
# MAGIC This notebook demonstrates how to load Unity Catalog tables into Ray Datasets on **Databricks Serverless GPU** (AI Runtime) using `ray.data.read_delta` with the `DatabricksUnityCatalog` helper. This is the serverless GPU counterpart to the [classic-compute/read_delta](../classic-compute/read_delta) notebook.
# MAGIC
# MAGIC ### How it works
# MAGIC 1. A fresh copy of the source Delta table is created with deletion vectors disabled (delta-rs cannot read tables with deletion vectors or column mapping in the protocol).
# MAGIC 2. `ray_init()` from `serverless_gpu` starts a single-node Ray instance on the serverless GPU node.
# MAGIC 3. `DATABRICKS_HOST` and `DATABRICKS_TOKEN` are set as environment variables.
# MAGIC 4. A `DatabricksUnityCatalog` object is created with the workspace URL, token, and cloud region.
# MAGIC 5. `ray.data.read_delta(table_path, catalog=catalog)` resolves the table's storage location and credentials internally.
# MAGIC 6. Ray reads the underlying Delta Parquet files directly from cloud storage — no SQL Warehouse or Spark cluster in the data path.
# MAGIC
# MAGIC ### Key difference from Classic Compute
# MAGIC On serverless GPU, there is no Spark cluster to host Ray workers. Instead of `setup_ray_cluster()`, we use:
# MAGIC ```python
# MAGIC from serverless_gpu import ray_init
# MAGIC ray_init()
# MAGIC ```
# MAGIC This starts a single-node Ray instance directly on the serverless GPU node.
# MAGIC
# MAGIC ### Lineage
# MAGIC Because `read_delta` reads directly from cloud storage, **Unity Catalog does not automatically capture lineage** for these reads.
# MAGIC
# MAGIC ### Requirements
# MAGIC * Serverless GPU compute (AI Runtime).
# MAGIC * `SELECT` on the target table, `USE CATALOG`, and `USE SCHEMA`.
# MAGIC * `deltalake` and `ray[all]==2.58.0` Python packages.
# MAGIC * The target table must have **deletion vectors disabled** and **no column mapping** in the protocol.
# MAGIC
# MAGIC ### Notebook overview
# MAGIC | Cell | Purpose |
# MAGIC |---|---|
# MAGIC | **2 – Install packages** | Pins `deltalake` and `ray[all]`. |
# MAGIC | **3 – Configuration** | Calls `utils/create_sample_table` to ensure the shared source table exists, then sets the notebook-owned target table. |
# MAGIC | **4 – Create clean table** | Creates a fresh CTAS copy with deletion vectors disabled. |
# MAGIC | **5 – Ray init** | Initialises single-node Ray via `ray_init()` and sets credentials as env vars. |
# MAGIC | **6 – Read table** | Creates a `DatabricksUnityCatalog` and reads the table via `read_delta`. |

# COMMAND ----------

# DBTITLE 1,Install packages
# MAGIC %pip install ray[all]==2.58.0 deltalake
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Configuration: Unity Catalog table coordinates
uc_catalog = "main"
uc_schema  = "ray_gtm_examples"
uc_table   = "ray_data_source"
forked_uc_table = "ray_data_source_read_delta"

source_table_path = dbutils.notebook.run(
    "../utils/create_sample_table", 300,
    {"catalog": uc_catalog, "schema": uc_schema, "table_name": uc_table}
)

uc_table_path = f"{uc_catalog}.{uc_schema}.{forked_uc_table}"
print(f"Source table: {source_table_path}")
print(f"Target table: {uc_table_path}")

# COMMAND ----------

# DBTITLE 1,Create a notebook-owned table for read_delta
# Create a notebook-owned copy with deletion vectors disabled.
# ray.data.read_delta (delta-rs) cannot read tables with DVs enabled today,
# so we fork the source table with a clean protocol for this example.

spark.sql(f"""
  CREATE OR REPLACE TABLE {uc_table_path}
  TBLPROPERTIES ('delta.enableDeletionVectors' = false)
  AS SELECT * FROM {source_table_path}
""")
print(f"Created {uc_table_path} from {source_table_path}")

# COMMAND ----------

# DBTITLE 1,Ray init and set credentials
import os
from serverless_gpu import ray_init

# Start single-node Ray on the serverless GPU node
ray_init()

# Set env vars so the DatabricksUnityCatalog can authenticate
os.environ['DATABRICKS_HOST'] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
os.environ['DATABRICKS_TOKEN'] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

print(f"DATABRICKS_HOST: {os.environ['DATABRICKS_HOST']}")
print("DATABRICKS_TOKEN: set")

# COMMAND ----------

# DBTITLE 1,Read Unity Catalog table with read_delta
from ray.data import DatabricksUnityCatalog
import ray

# Initialize the catalog with workspace credentials
catalog = DatabricksUnityCatalog(
    url=os.environ['DATABRICKS_HOST'],
    token=os.environ['DATABRICKS_TOKEN'],
    region="us-west-2",  # Required for AWS; not needed for Azure/GCP
)

# Read the Unity Catalog table via read_delta
ds = ray.data.read_delta(uc_table_path, catalog=catalog)

print(ds.schema())
print(f"Count: {ds.count()}")
