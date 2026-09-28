# Reading Databricks Tables into Ray Data

Ray workloads on Databricks — distributed training, hyperparameter tuning, batch inference — need data. That data lives in Databricks as tables managed by Unity Catalog. The challenge is bridging the two: getting governed, production data out of your tables and into Ray's distributed dataset format without bottlenecks, permission gaps, or loss of lineage.

Ray Data provides three native connectors for reading Databricks tables. All read from Unity Catalog — they differ in **how** the data travels from storage to your Ray workers. Each notebook is named after its primary Ray API entry point. This folder provides notebooks for each approach across two compute environments.

## Shared source table

All notebooks call `utils/create_sample_table` to ensure a lightweight sample table (`ray_data_source`) exists in Unity Catalog. The utility is idempotent — if the table already exists, it skips creation.

## Two compute environments

Ray on Databricks runs in two modes, each with its own cluster lifecycle:

### Classic Compute (`classic-compute/`)

Uses **Ray on Spark** — Ray workers are launched on Spark executor nodes via `setup_ray_cluster()`. Best when you already have a Classic Compute cluster or need tight Spark interop.

### AI Runtime / Serverless GPU (`ai-runtime/`)

Uses **Databricks Serverless GPU** — no Spark cluster to manage. Single-node workloads use `ray.init()` directly; multi-node workloads use the `@ray_launch` decorator from `serverless_gpu.ray` to provision GPU nodes on demand.

## Three data paths

### Via SQL Warehouse (`ray.data.read_databricks_tables`)

Submits a SQL query to a **Databricks SQL Warehouse** through the Statement Execution API. The warehouse executes the query and Ray workers fetch result chunks over HTTP.

- Supports arbitrary SQL via the `query=` parameter (filters, joins, aggregations).
- Unity Catalog captures lineage automatically — every read appears in the table's Lineage tab.
- Requires a running SQL Warehouse.
- **Read-only** on Ray-on-Spark clusters — `write_databricks_table` stages data via Spark + FUSE volumes, which conflicts with Ray-on-Spark.

### Via Delta / delta-rs (`ray.data.read_delta`)

Uses `DatabricksUnityCatalog` to resolve table location and vend short-lived cloud storage credentials. Ray workers read the underlying Delta files directly from S3/ADLS/GCS — no warehouse, no Spark.

- Performance scales with the Ray cluster — reads go straight to object storage.
- Also supports `write_delta` for appending or overwriting data.
- Requires `deltalake` (delta-rs) Python package.
- Target table must have **deletion vectors disabled** (delta-rs limitation).
- Lineage is **not** captured automatically.

### Via Iceberg / UniForm (`ray.data.read_iceberg`)

Uses `DatabricksUnityCatalog` to resolve the table's Iceberg REST endpoint and vend credentials. Ray workers read via the Iceberg-compatible surface (UniForm / External Iceberg Reads).

- Avoids the delta-rs deletion vector limitation by reading through the Iceberg metadata layer.
- Also supports `write_iceberg` for appending or overwriting data.
- Requires `pyiceberg` Python package and **UniForm (Iceberg)** enabled on the target table.
- Lineage **is captured** through the credential-vending flow.

## Choosing between them

| | SQL Warehouse | Delta (delta-rs) | Iceberg (UniForm) |
|---|---|---|---|
| **API** | `read_databricks_tables` | `read_delta` | `read_iceberg` |
| **Data path** | Statement Execution API → HTTP | Credential vending → S3/ADLS/GCS | Iceberg REST → S3/ADLS/GCS |
| **Requires** | SQL Warehouse (running) | `deltalake` package, DVs disabled | `pyiceberg` package, UniForm enabled |
| **Custom SQL** | Yes (`query=` parameter) | No (full table) | No (full table) |
| **Write support** | No (on Ray-on-Spark) | Yes (`write_delta`) | Yes (`write_iceberg`) |
| **Lineage** | Automatic | Not captured | Captured |
| **Best for** | Filtered reads, lineage-sensitive workflows | Full table reads, maximum throughput | Full table reads when DVs are present |

## Folder contents

```
ray-data-read-examples/
├── README.md
├── utils/
│   └── create_sample_table              # Shared utility: ensures ray_data_source table exists
├── classic-compute/
│   ├── read_databricks_tables           # SQL Warehouse path (Ray on Spark)
│   ├── read_delta                       # Delta (delta-rs) path (Ray on Spark)
│   └── read_iceberg                     # Iceberg (UniForm) path (Ray on Spark)
├── ai-runtime/
│   ├── read_from_databricks_tables_serverless_gpu   # SQL Warehouse path (Serverless GPU)
│   └── read_delta_serverless_gpu                    # Delta (delta-rs) path (Serverless GPU)
└── deprecated/
    └── ...
```

## Prerequisites

**Classic Compute:**
- Databricks Runtime **17.3 LTS** or later.
- `ray[all]==2.58.0` installed via `%pip`.

**AI Runtime (Serverless GPU):**
- Databricks Serverless GPU compute, environment version **client.4.10** or later.
- `ray[all]==2.58.0` installed via `%pip`.

See each notebook's introduction cell for method-specific requirements.
