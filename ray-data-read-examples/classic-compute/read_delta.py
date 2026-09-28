# Databricks notebook source
# DBTITLE 1,Introduction
# MAGIC %md
# MAGIC ## Reading Unity Catalog Tables with `ray.data.read_delta` on Classic Compute
# MAGIC
# MAGIC This notebook is centered on the `ray.data.read_delta` API, so the notebook name stays aligned to the Ray entrypoint even though the workflow also includes a small `write_delta` append example at the end. On **Classic Compute** (Ray on Spark), this path reads a Unity Catalog table directly from cloud storage by resolving access through `DatabricksUnityCatalog`.
# MAGIC
# MAGIC ### How it works
# MAGIC 1. The driver sets `DATABRICKS_HOST` and `DATABRICKS_TOKEN` as environment variables **before** calling `setup_ray_cluster()` so Ray workers inherit them.
# MAGIC 2. A `DatabricksUnityCatalog` object is created with the workspace URL, token, and cloud region.
# MAGIC 3. `ray.data.read_delta("catalog.schema.table", catalog=catalog)` resolves the table's storage location and credentials internally via the catalog object.
# MAGIC 4. Ray workers read the underlying Delta files directly from cloud storage — no SQL Warehouse is in the data path.
# MAGIC 5. The final cell appends a small sample back to the same notebook-owned table with `write_delta` so the read and write path can be exercised together.
# MAGIC
# MAGIC ### Migration from `read_unity_catalog`
# MAGIC `ray.data.read_unity_catalog` is being deprecated. `ray.data.read_delta` with a `DatabricksUnityCatalog` is the recommended replacement. The key improvement is that credential resolution is handled by the `DatabricksUnityCatalog` object — you no longer need to manually vend storage tokens via the Databricks SDK.
# MAGIC
# MAGIC See the [Ray docs for read_delta](https://docs.ray.io/en/latest/data/api/doc/ray.data.read_delta.html) for the full API reference.
# MAGIC
# MAGIC ### Lineage
# MAGIC Because `read_delta` reads directly from cloud storage, **Unity Catalog does not automatically capture lineage** for these reads. See the [read_from_unity_catalog](./read_from_unity_catalog) reference notebook in this folder for an optional lineage registration pattern using the External Metadata APIs.
# MAGIC
# MAGIC ### Requirements
# MAGIC * Databricks Runtime **17.3 LTS** or later.
# MAGIC * `SELECT` on the target table, `USE CATALOG`, and `USE SCHEMA`.
# MAGIC * `deltalake>=1.5.0` and `ray[all]==2.58.0` Python packages.
# MAGIC
# MAGIC > **Note on `deltalake` 1.5.0+:** The latest delta-rs release adds native support for [deletion vectors](https://docs.delta.io/latest/delta-deletion-vectors.html), making the end-to-end OSS read path fully seamless.
# MAGIC
# MAGIC ### Notebook overview
# MAGIC | Cell | Purpose |
# MAGIC |---|---|
# MAGIC | **2 – Install packages** | Pins `deltalake` and `ray[all]`. |
# MAGIC | **3 – Configuration** | Calls `utils/create_sample_table` to ensure the shared source table exists, then sets the notebook-owned target table. |
# MAGIC | **4 – Prepare demo table** | Creates a separate Delta table for this notebook so the append example does not modify the shared source table. |
# MAGIC | **5 – Ray cluster setup** | Sets credentials as env vars, then initialises a Ray-on-Spark cluster using all 4 worker nodes. |
# MAGIC | **6 – Read table** | Creates a `DatabricksUnityCatalog` and reads the table via `read_delta`. |
# MAGIC | **7 – Append rows** | Appends a small sample back to the same table with `write_delta` and verifies the row count change. |

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

# DBTITLE 1,Ray cluster setup
import ray
from ray.util.spark import setup_ray_cluster, shutdown_ray_cluster
import os

# Cleanly restart Ray if it was already running
try:
    shutdown_ray_cluster()
except:
    pass
try:
    ray.shutdown()
except:
    pass

# Dynamically grab the current user for log paths
user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
ray_logs_path = f"/dbfs/Users/{user}/ray_collected_logs/"

# Set env vars so Ray workers can resolve Unity Catalog access through Databricks-managed credential vending.
os.environ['DATABRICKS_HOST'] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
os.environ['DATABRICKS_TOKEN'] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

setup_ray_cluster(
    min_worker_nodes=4,
    max_worker_nodes=4,
    num_cpus_worker_node=4,
    num_gpus_worker_node=0,
    collect_log_to_path=ray_logs_path,
)

# COMMAND ----------

# DBTITLE 1,Read Unity Catalog table with read_delta
from ray.data import DatabricksUnityCatalog

# Initialize the catalog with workspace credentials
catalog = DatabricksUnityCatalog(
    url=os.environ['DATABRICKS_HOST'],
    token=os.environ['DATABRICKS_TOKEN'],
    region="us-west-2",  # Required for AWS; not needed for Azure/GCP
)

# Primary demo: read the Unity Catalog table via ray.data.read_delta.
ds = ray.data.read_delta(uc_table_path, catalog=catalog)

print(ds.schema())
print(f"Count: {ds.count()}")

# COMMAND ----------

# DBTITLE 1,Append rows with write_delta
# Companion write example: append a small sample back to the same Delta table.
pre_count = ds.count()
print(f"Row count before append: {pre_count}")

# Take a small sample to append back.
sample_ds = ds.limit(100)
print(f"Rows to append: {sample_ds.count()}")

# Append to the existing table.
print(f"\nAppending to: {uc_table_path}")
sample_ds.write_delta(uc_table_path, catalog=catalog, mode="append")

# Re-read the table and verify that the row count increased.
verify_ds = ray.data.read_delta(uc_table_path, catalog=catalog)
post_count = verify_ds.count()
print(f"\nRow count after append: {post_count}")
print(f"Rows added: {post_count - pre_count}")
