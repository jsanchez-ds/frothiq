🌐 **English** · [Español](README.es.md)

# ⚗️ FrothIQ — Mineral Process Quality Forecasting on Databricks

> **Production-grade ML platform for mineral flotation process quality prediction. Built local-first, deployable to Azure Databricks. Includes statistical process control (SPC) charts and what-if simulation for operators.**

End-to-end ML platform that ingests sensor data from a mineral flotation plant (real industrial data from a Brazilian iron-ore concentration plant), processes it through a **Medallion architecture** (Bronze → Silver → Gold) on Delta Lake, trains predictive models for **% Iron** and **% Silica** in concentrate output, and serves them through a Streamlit dashboard with SPC charts and a what-if simulator. The whole pipeline is reproducible local ≡ cloud — same code runs on a laptop or on a Databricks cluster.

> ⚠️ **Status — work in progress (started 2026-05-02).** Scaffolding + data ingestion + LightGBM baseline are live; LSTM, SPC, what-if dashboard and Databricks deployment docs are added iteratively. See the [Roadmap](#-roadmap).

---

## 🎯 What this project proves

| Capability | Evidence |
|---|---|
| **Mining domain** | Real industrial data from a flotation plant (737K rows, 24 sensors) |
| **Big data on Spark** | PySpark pipeline runs local and on Databricks unchanged |
| **Medallion architecture** | Bronze (raw) → Silver (clean) → Gold (features) on Delta Lake |
| **Production ML** | LightGBM + LSTM (PyTorch) tracked in MLflow with model signatures |
| **Statistical Process Control** | Shewhart charts, CUSUM, Western Electric rules for operator alerts |
| **What-if simulation** | Operator dashboard: "what if I change pH from X to Y?" |
| **MLOps rigor** | Tests + CI, Docker, drift monitoring, model registry with aliases |
| **Local-first → Cloud** | Reproducible local equiv to Databricks; deployment guide included |

---

## 🏗️ Architecture

```
┌──────────────────────┐    ┌────────────────────┐    ┌─────────────────────┐
│  Kaggle CSV          │───▶│  Bronze Layer      │───▶│  Silver Layer       │
│  (737K × 24)         │    │  (Delta — append)  │    │  (clean, typed)     │
│  Real flotation data │    └────────────────────┘    └──────────┬──────────┘
└──────────────────────┘                                          │
                                                                  ▼
       ┌─────────────────────────────┐                ┌────────────────────┐
       │  Streamlit dashboard        │◀───────────────│  Gold Layer        │
       │  • SPC charts               │                │  (features+target) │
       │  • What-if simulator        │                └──────────┬─────────┘
       │  • Live predictions         │                           │
       └─────────────┬───────────────┘                           ▼
                     │                                ┌────────────────────┐
                     │ HTTP                           │  MLflow Tracking   │
                     ▼                                │  + Model Registry  │
       ┌─────────────────────────────┐                │  (alias @prod)     │
       │  FastAPI service            │◀───────────────└──────────┬─────────┘
       │  /predict_quality           │                           │
       │  /sim_whatif                │                           ▼
       └─────────────────────────────┘                ┌────────────────────┐
                                                      │  Evidently AI      │
                                                      │  (drift monitoring)│
                                                      └────────────────────┘
```

---

## 📊 Dataset

**[Quality Prediction in a Mining Process](https://www.kaggle.com/datasets/edumagalhaes/quality-prediction-in-a-mining-process)** by Edson Antonio Magalhaes (Kaggle, ~700K downloads).

Real industrial data from a Brazilian iron-ore concentration plant. The dataset:

- **737,453 rows** sampled at 20-second intervals over ~6 months
- **24 columns**: `% Iron Feed`, `% Silica Feed`, `Starch Flow`, `Amina Flow`, `Ore Pulp Flow`, `Ore Pulp pH`, `Ore Pulp Density`, 7 `Flotation Column Air Flow` columns, 7 `Flotation Column Level` columns, target `% Iron Concentrate` and `% Silica Concentrate`
- **Targets** (lab measurements, hourly): `% Iron Concentrate` and `% Silica Concentrate`
- **Goal**: predict outputs from upstream sensor measurements so operators can adjust parameters before product goes off-spec

---

## 📂 Project structure

```
.
├── src/frothiq/
│   ├── data/             # Kaggle ingestion, Bronze loaders
│   ├── features/         # Rolling stats, lag aggregations, frequency-domain
│   ├── models/
│   │   ├── baseline/     # LightGBM, XGBoost, scikit-learn baselines
│   │   ├── deep/         # LSTM (PyTorch) sequence models
│   │   ├── spc/          # Shewhart, CUSUM, Western Electric rules
│   │   └── whatif/       # What-if simulation (counterfactual predictions)
│   ├── serving/          # FastAPI app, Streamlit dashboard
│   └── utils/            # MLflow helpers, logging, configs
├── notebooks/            # 00_eda → 01_features → 02_baseline → 03_lstm → 04_spc → 05_whatif
├── data/                 # raw / interim / processed (gitignored)
├── configs/              # YAML configs per dataset / model
├── scripts/              # download_data.sh, train_*.py, deploy_databricks.sh
├── tests/                # pytest suite
├── .github/workflows/    # CI: ruff + pytest
└── docs/                 # ADRs, dataset cards, Databricks deployment guide
```

---

## 🚀 Quickstart

```bash
# 1. Clone
git clone https://github.com/jsanchez-ds/frothiq.git
cd frothiq

# 2. Install (Python 3.11+ recommended)
python -m venv .venv
.venv\Scripts\activate          # Windows (use 'source .venv/bin/activate' on Mac/Linux)
pip install -e ".[dev]"

# 3. Get Kaggle API token (one-time)
#    https://www.kaggle.com/settings → API → Create New Token
#    Save kaggle.json to ~/.kaggle/kaggle.json (or %USERPROFILE%\.kaggle on Windows)

# 4. Download the dataset (~125 MB)
bash scripts/download_data.sh

# 5. Open the EDA notebook
jupyter lab notebooks/00_eda.ipynb
```

---

## 🧪 Modeling approach

| Model | Library | Target | Notes |
|---|---|---|---|
| LightGBM | `lightgbm` | `% Iron Concentrate`, `% Silica Concentrate` | Tabular baseline with rolling features |
| Quantile LightGBM | `lightgbm` | P10 / P50 / P90 of targets | Confidence intervals for SPC alerts |
| LSTM (PyTorch) | `torch` | Same | Sequence-aware; benchmark vs tabular |
| Western Electric SPC | `pyspc` (custom) | Sensor channels | Detection rules for operator alerts |

Validation: **temporal split** (train: first 70%, val: next 15%, test: last 15%) — no random shuffling. The hour-level lab measurements are joined with 20-second sensor data via temporal forward-fill.

---

## 🗺️ Roadmap

- [x] Repo scaffolding + bilingual READMEs + CI skeleton
- [x] Data download script (Kaggle API)
- [x] Bronze + Silver + Gold pipeline (PySpark, Delta Lake)
- [x] Notebook 00 — EDA on flotation sensors and targets
- [x] Notebook 01 — Feature engineering (rolling, lag, calendar)
- [x] Notebook 02 — LightGBM baseline + MLflow tracking
- [ ] Notebook 03 — LSTM (PyTorch) sequence model
- [ ] Notebook 04 — Statistical Process Control (Shewhart, CUSUM, WE rules)
- [ ] Notebook 05 — What-if simulator on Streamlit
- [ ] FastAPI serving + drift monitoring
- [ ] Databricks deployment guide (`docs/databricks_deploy.md`)

---

## 📜 License

MIT — see [LICENSE](LICENSE).

---

## 👤 Author

**Jonathan Sánchez Pesantes** — Industrial Engineer · Data Scientist
🔗 [linkedin.com/in/jonasanchez](https://www.linkedin.com/in/jonasanchez) · [github.com/jsanchez-ds](https://github.com/jsanchez-ds)
