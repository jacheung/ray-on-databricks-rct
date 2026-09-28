# Databricks notebook source
# DBTITLE 1,Introduction
# MAGIC %md
# MAGIC ## Reading Unity Catalog Tables with `ray.data.read_iceberg` on Classic Compute
# MAGIC
# MAGIC This notebook is centered on the `ray.data.read_iceberg` API, so the notebook name stays aligned to the Ray entrypoint even though the workflow also includes a small `write_iceberg` append example at the end. On **Classic Compute** (Ray on Spark), this path reads a Unity Catalog table through its **Iceberg-compatible surface** (UniForm / External Iceberg Reads) using `DatabricksUnityCatalog`.
# MAGIC
# MAGIC ### How it works
# MAGIC 1. The target Delta table is configured with **UniForm (Iceberg compatibility v2)**, which automatically maintains Iceberg metadata alongside Delta.
# MAGIC 2. The driver sets `DATABRICKS_HOST` and `DATABRICKS_TOKEN` as environment variables **before** calling `setup_ray_cluster()` so Ray workers inherit them.
# MAGIC 3. A `DatabricksUnityCatalog` object is created with the workspace URL, token, and cloud region.
# MAGIC 4. `ray.data.read_iceberg(table_identifier, catalog=catalog)` resolves the table's Iceberg REST endpoint and vends short-lived cloud credentials internally.
# MAGIC 5. Ray workers read the underlying files directly from cloud storage — no SQL Warehouse is in the data path.
# MAGIC 6. The final cell appends a small sample back to the same notebook-owned table with `write_iceberg` so the read and write path can be exercised together.
# MAGIC
# MAGIC ### When to use this path
# MAGIC `read_iceberg` is the right companion to the Delta example when you want the notebook to stay aligned to the Ray API name but demonstrate access through the Iceberg-compatible table surface.
# MAGIC
# MAGIC ### Requirements
# MAGIC * Databricks Runtime **14.3 LTS** or later (for UniForm v2 support).
# MAGIC * `SELECT` on the target table, `USE CATALOG`, `USE SCHEMA`, and `EXTERNAL USE SCHEMA`.
# MAGIC * The workspace must have **external data access** enabled.
# MAGIC * `ray[all]==2.58.0` and `pyiceberg` Python packages.
# MAGIC * The target table must have **UniForm (Iceberg)** enabled.
# MAGIC
# MAGIC ### Notebook overview
# MAGIC | Cell | Purpose |
# MAGIC |---|---|
# MAGIC | **2 – Install packages** | Pins `pyiceberg` and `ray[all]`. |
# MAGIC | **3 – Configuration** | Calls `utils/create_sample_table` to ensure the shared source table exists, then sets the notebook-owned target table. |
# MAGIC | **4 – Prepare demo table** | Creates a separate Delta table for this notebook and enables UniForm Iceberg on it. |
# MAGIC | **5 – Ray cluster setup** | Sets credentials as env vars, then initialises a Ray-on-Spark cluster using all 4 worker nodes. |
# MAGIC | **6 – Read table** | Creates a `DatabricksUnityCatalog` and reads the table via `read_iceberg`. |
# MAGIC | **7 – Append rows** | Appends a small sample back to the same table with `write_iceberg` and verifies the row count change. |

# COMMAND ----------

# DBTITLE 1,Install packages
# MAGIC %pip install ray[all]==2.58.0 pyiceberg
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Configuration: Unity Catalog table coordinates
uc_catalog = "main"
uc_schema  = "ray_gtm_examples"
uc_table   = "ray_data_source"
forked_uc_table = "ray_data_source_read_iceberg"

source_table_path = dbutils.notebook.run(
    "../utils/create_sample_table", 300,
    {"catalog": uc_catalog, "schema": uc_schema, "table_name": uc_table}
)

uc_table_path = f"{uc_catalog}.{uc_schema}.{forked_uc_table}"
print(f"Source table: {source_table_path}")
print(f"Target table: {uc_table_path}")

# COMMAND ----------

# DBTITLE 1,Create a notebook-owned table with UniForm Iceberg enabled
# Create a notebook-owned copy so the read/write demo does not modify the shared source table.
# Then enable UniForm Iceberg on that copy so the example stays isolated and repeatable.

# Step 1: Create a clean demo table from the shared source data.
spark.sql(f"""
  CREATE OR REPLACE TABLE {uc_table_path}
  TBLPROPERTIES ('delta.enableDeletionVectors' = false)
  AS SELECT * FROM {source_table_path}
""")
print(f"Created {uc_table_path} from {source_table_path}")

# Step 2: Enable column mapping and UniForm Iceberg on the demo table.
spark.sql(f"""
  ALTER TABLE {uc_table_path}
  SET TBLPROPERTIES (
    'delta.columnMapping.mode' = 'name',
    'delta.enableIcebergCompatV2' = 'true',
    'delta.universalFormat.enabledFormats' = 'iceberg'
  )
""")
print(f"UniForm Iceberg enabled on {uc_table_path}")

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

# DBTITLE 1,Read Unity Catalog table with read_iceberg
from ray.data.catalog import DatabricksUnityCatalog

# Initialize the catalog with workspace credentials
catalog = DatabricksUnityCatalog(
    url=os.environ["DATABRICKS_HOST"],
    token=os.environ["DATABRICKS_TOKEN"],
    region="us-west-2",  # Required for AWS; not needed for Azure/GCP
)

# Primary demo: read the Unity Catalog table via ray.data.read_iceberg.
ds = ray.data.read_iceberg(
    table_identifier=uc_table_path,
    catalog=catalog,
)

print(ds.schema())
print(f"Count: {ds.count()}")

# COMMAND ----------

# DBTITLE 1,Append rows with write_iceberg
# Companion write example: append a small sample back to the same Iceberg-compatible table.
pre_count = ds.count()
print(f"Row count before append: {pre_count}")

# Take a small sample to append back.
sample_ds = ds.limit(100)
print(f"Rows to append: {sample_ds.count()}")

# Append to the existing table.
print(f"\nAppending to: {uc_table_path}")
sample_ds.write_iceberg(table_identifier=uc_table_path, catalog=catalog, mode="append")

# Re-read the table and verify that the row count increased.
verify_ds = ray.data.read_iceberg(table_identifier=uc_table_path, catalog=catalog)
post_count = verify_ds.count()
print(f"\nRow count after append: {post_count}")
print(f"Rows added: {post_count - pre_count}")
