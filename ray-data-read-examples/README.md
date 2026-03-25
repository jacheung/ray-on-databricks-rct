# Reading Databricks Tables into Ray Data

Ray workloads on Databricks — distributed training, hyperparameter tuning, batch inference — need data. That data lives in Databricks as tables managed by Unity Catalog. The challenge is bridging the two: getting governed, production data out of your tables and into Ray's distributed dataset format without bottlenecks, permission gaps, or loss of lineage.

Ray Data provides two native connectors for reading Databricks tables. Both read from Unity Catalog — they differ in **how** the data travels from storage to your Ray workers. This folder provides a notebook for each.

## Two data paths

### [Via SQL Warehouse](./read_from_databricks_tables) (`ray.data.read_databricks_tables`)

Submits a SQL query to a **Databricks SQL Warehouse** through the Statement Execution API. The warehouse executes the query, and Ray workers fetch result chunks over HTTP.

- Supports arbitrary SQL via the `query=` parameter — filters, joins, aggregations, `LIMIT`.
- Unity Catalog captures lineage automatically — every read appears in the table's Lineage tab.
- Requires a running SQL Warehouse.
- Performance is bounded by warehouse size and Statement Execution API result limits.

### [Via direct cloud storage](./read_from_unity_catalog) (`ray.data.read_unity_catalog`)

Uses **Unity Catalog credential vending** to issue short-lived cloud storage tokens. Ray workers read the underlying Delta Parquet files directly from S3/ADLS/GCS — no warehouse, no intermediary.

- Performance scales with the Ray cluster — reads go straight to object storage.
- Requires the `EXTERNAL USE SCHEMA` privilege (only the catalog owner can grant it).
- Reads the full table; does not support custom SQL.
- Lineage is **not** captured automatically. The notebook includes an optional `log_lineage=True` flag that registers reads via the External Metadata API.

## Choosing between them

| | Via SQL Warehouse | Via direct cloud storage |
|---|---|---|
| **API** | `ray.data.read_databricks_tables` | `ray.data.read_unity_catalog` |
| **Data path** | Statement Execution API → HTTP chunks | Credential vending → S3/ADLS/GCS |
| **Requires** | SQL Warehouse (running) | `EXTERNAL USE SCHEMA` grant |
| **Custom SQL** | Yes (`query=` parameter) | No (full table only) |
| **Lineage** | Automatic | Manual (External Metadata API) |
| **Performance ceiling** | Warehouse size + API limits | Ray cluster size |
| **Best for** | Filtered/transformed reads, smaller result sets, lineage-sensitive workflows | Full table reads, large-scale training, maximum throughput |

## Prerequisites

Both notebooks expect:

- **Databricks Classic Compute** with Ray installed (`ray[all]==2.54.0`).
- A Ray-on-Spark cluster initialized via `ray.util.spark.setup_ray_cluster`.
- `DATABRICKS_HOST` and `DATABRICKS_TOKEN` environment variables set before cluster init so Ray workers inherit credentials.

See each notebook's introduction cell for method-specific requirements.
