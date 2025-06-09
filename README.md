# Azure Customer Churn Prediction

End‑to‑end customer churn prediction pipeline built on **Azure ML**, **Databricks (Spark)** and **LightGBM**.

## Architecture
```
+---------------------+       +-------------------+       +---------------------+
| Azure Data Factory  |  -->  |   Azure Databricks|  -->  |  Azure Feature Store|
+---------------------+       +-------------------+       +---------------------+
         |                                                           |
         v                                                           v
+---------------------+       +-------------------+       +---------------------+
|   ADLS Gen‑2 (Raw)  |       |  LightGBM Model   |  -->  | Azure ML Endpoint   |
+---------------------+       +-------------------+       +---------------------+
```

## Quick Start

```bash
git clone https://github.com/<your‑org>/azure-customer-churn-prediction.git
cd azure-customer-churn-prediction
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Run training locally
python src/training/train_lightgbm.py --config configs/local_train.yml
```

## Repository Layout

| Path | Purpose |
|------|---------|
| `src/` | All Python source code grouped by stage (ingestion, features, training, deployment, monitoring) |
| `configs/` | YAML/JSON configs for pipelines, HyperDrive, Spark, etc. |
| `notebooks/` | Exploratory analysis & Databricks notebooks (exported as `.ipynb`) |
| `ci_cd/` | GitHub Actions / Azure DevOps pipelines |
| `docs/` | Diagrams and additional documentation |

## License
[MIT](LICENSE)
