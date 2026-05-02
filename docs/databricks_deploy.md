# Deploying FrothIQ to Azure Databricks

This guide explains how to take the FrothIQ codebase from local development to a production Databricks deployment. The principle is **local ≡ cloud**: the same Python code runs unchanged.

## 1. Prerequisites

- Azure Databricks workspace (your organization's or trial / community).
- Databricks CLI installed (`pip install databricks-cli`) and configured (`databricks configure --token`).
- Unity Catalog enabled on the workspace (recommended for governance).
- A storage account (Azure Blob / ADLS Gen2) accessible from Databricks.

## 2. Upload the dataset

```bash
# Once: upload the Kaggle CSV to a Databricks volume.
databricks fs cp \
  data/raw/flotation/MiningProcess_Flotation_Plant_Database.csv \
  dbfs:/Volumes/main/frothiq/raw/MiningProcess_Flotation_Plant_Database.csv
```

## 3. Bronze loader as a Databricks notebook

Convert `notebooks/00_eda.ipynb` to Databricks `.py` notebook format (with `# COMMAND ----------` separators) using `jupyter nbconvert --to script` and minor adjustments. The `load_flotation()` function works unchanged — just point it to the DBFS path.

## 4. Promote model to Unity Catalog Model Registry

```python
import mlflow

mlflow.set_registry_uri("databricks-uc")

# Inside the training notebook:
mlflow.lightgbm.log_model(
    model,
    artifact_path="model",
    registered_model_name="main.frothiq.iron_concentrate_predictor",
    signature=signature,
    input_example=X_train.head(2),
)
```

Then in the Databricks UI:
- Navigate to Catalog → main → frothiq → Models.
- Find `iron_concentrate_predictor`.
- Add alias `@staging` to the latest version.

## 5. Schedule the training job

Create a `databricks.yml` (Databricks Asset Bundles) at the repo root:

```yaml
bundle:
  name: frothiq

resources:
  jobs:
    train_baseline:
      name: FrothIQ — Train LightGBM Baseline
      tasks:
        - task_key: train
          notebook_task:
            notebook_path: ./notebooks/02_baseline_lightgbm.ipynb
          job_cluster_key: ml_cluster
      job_clusters:
        - job_cluster_key: ml_cluster
          new_cluster:
            spark_version: 14.3.x-scala2.12
            node_type_id: Standard_DS3_v2
            num_workers: 2
      schedule:
        quartz_cron_expression: "0 0 6 * * ?"  # daily at 6am
        timezone_id: America/Santiago

targets:
  dev:
    workspace:
      host: https://<your-workspace>.azuredatabricks.net
```

Deploy with: `databricks bundle deploy -t dev`.

## 6. Serve predictions

Two options:

### Option A — Databricks Model Serving (managed)

In the UI: Catalog → models → `iron_concentrate_predictor` → "Serve this model". Databricks creates a REST endpoint:

```
https://<workspace>.azuredatabricks.net/serving-endpoints/iron_concentrate_predictor/invocations
```

### Option B — FastAPI + Docker (cross-cloud portability)

Use the FastAPI app in `src/frothiq/serving/`, point it to `models:/main.frothiq.iron_concentrate_predictor@production` from the Databricks Model Registry. Deploy as a container in AKS / Container Apps / your provider of choice.

## 7. Monitoring with Lakehouse Monitoring

Enable [Lakehouse Monitoring](https://docs.databricks.com/lakehouse-monitoring/index.html) on the predictions table:

```python
from databricks.lakehouse_monitoring import create_monitor

create_monitor(
    table_name="main.frothiq.predictions",
    profile_type="InferenceLog",
    inference_log={
        "timestamp_col": "prediction_time",
        "model_id_col": "model_version",
        "prediction_col": "predicted_pct_iron",
        "label_col": "actual_pct_iron",  # joined from lab measurements
    },
)
```

This generates dashboards for data drift, model performance over time, and SLA breaches.

## 8. Cost guidance

For a workspace serving a flotation plant in real-time:

- **Compute**: 2-node `Standard_DS3_v2` cluster ~$2-3 USD / hour during training (a few hours/day).
- **Storage**: trivial (~$0.02 / GB / month).
- **Model serving**: ~$0.07 / 1000 predictions on managed endpoint.

Total for a single plant: typically $200-500 USD / month at moderate utilization. Cheaper than a single human analyst.

## 9. Hand-off checklist

Before declaring the deployment "production-ready":

- [ ] Unity Catalog ACLs configured (read-only for operators, full for ML team).
- [ ] CI/CD pipeline runs tests on every PR.
- [ ] Model Registry has `@staging` and `@production` aliases.
- [ ] Lakehouse Monitoring dashboards configured for both targets.
- [ ] Alerts configured for prediction latency > SLA, drift > threshold.
- [ ] Runbook documented in `docs/runbook.md`.
- [ ] Operator training session scheduled.
