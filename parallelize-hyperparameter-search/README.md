# Parallelized Hyperparameter Tuning with Ray

Boosting algorithms like XGBoost and LightGBM are powerful but prone to overfitting. Hyperparameter tuning with proper validation is essential — yet traditional grid search is inefficient, especially in large search spaces. Bayesian search (Optuna) balances exploration and exploitation, intelligently narrowing in on promising regions of the search space.

This folder combines **Ray Tune** for parallelized trial execution with **Optuna** for Bayesian search, running on Databricks. Each notebook demonstrates a different HPO pattern — varying by model, cluster topology, or data partitioning strategy.

## Getting started

All `01-*` notebooks depend on the `00-create-dataset` notebook to generate their training data. **Run it first.**

### [`00-create-dataset`](./00-create-dataset)

Generates synthetic classification datasets using `dbldatagen` and saves them as both Delta tables and Parquet files in Unity Catalog. Parameterized via notebook widgets:

| Parameter | Description |
|---|---|
| `num_training_rows` | Number of rows to generate |
| `num_training_columns` | Number of feature columns |
| `num_labels` | Number of target classes |
| `groups` | Number of group partitions (for the map-groups notebook) |

Outputs are written to `main.ray_gtm_examples` (Delta table) and `/Volumes/main/ray_gtm_examples/synthetic_data/` (Parquet).

## HPO notebooks

| Notebook | Model | Cluster topology | What it demonstrates |
|---|---|---|---|
| [`01-single-node-multi-gpu-hpo-xgb`](./01-single-node-multi-gpu-hpo-xgb) | XGBoost | Single node, multi-GPU | GPU vs CPU training benchmark. Each Optuna trial runs on one GPU. Useful for determining whether GPU acceleration is worth it for your data size. |
| [`01-multi-node-hpo-xgb`](./01-multi-node-hpo-xgb) | XGBoost | Multi-node CPU (8 × 48 CPU) | Standard multi-node HPO. One trial per worker node, all CPUs allocated per trial. Ray on Spark distributes trials across the cluster. |
| [`01-multi-node-hpo-lightgbm`](./01-multi-node-hpo-lightgbm) | LightGBM | Multi-node CPU (8 × 48 CPU) | Same parallelization pattern as XGBoost but for LightGBM. Demonstrates that the Ray Tune + Optuna pattern is model-agnostic. |
| [`01-multi-node-hpo-xgb-map-groups`](./01-multi-node-hpo-xgb-map-groups) | XGBoost | Multi-node CPU (8 × 48 CPU) | Grouped HPO — runs independent Optuna studies per group (e.g., region, segment) concurrently. Includes a model router PyFunc for deploying group-specific models behind a single endpoint. |

## Common pattern across notebooks

Every `01-*` notebook follows the same structure:

1. **Install packages** — `ray`, `optuna`, `mlflow`, and the model library (`xgboost` or `lightgbm`).
2. **Configure** — set catalog/schema/table coordinates, MLflow experiment name, and trial count.
3. **Initialize Ray** — `setup_ray_cluster` (multi-node) or `ray.init` (single-node GPU).
4. **Load data** — read from Unity Catalog via SQL Warehouse or from Parquet files in Volumes.
5. **Define trainable** — a function that trains one model with a sampled hyperparameter config and reports metrics back to Ray Tune.
6. **Run Tune** — `tune.Tuner` with `OptunaSearch`, resource allocation per trial, and MLflow logging of all trials as nested runs.

## Cluster guidance

| Notebook | Recommended compute |
|---|---|
| `01-single-node-multi-gpu-hpo-xgb` | Single node with multiple GPUs (e.g., `g5.12xlarge` with 4 × A10G) |
| `01-multi-node-*` | Multi-node CPU cluster (e.g., 8 × `m5d.12xlarge` with 48 CPUs each) |

For multi-node notebooks, set `num_cpus_worker_node` to match your Spark worker CPU count and `min/max_worker_nodes` to your cluster size. See each notebook's cluster spec cell for the exact configuration used.
