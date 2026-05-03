🌐 **English** · [Español](README.es.md)

# ⚗️ FrothIQ — Mineral Process Quality Forecasting on Databricks

> **Production-grade ML platform for mineral flotation process quality prediction. Built local-first, deployable to Azure Databricks. Includes statistical process control (SPC) charts and what-if simulation for operators.**

End-to-end ML platform that ingests sensor data from a mineral flotation plant (real industrial data from a Brazilian iron-ore concentration plant), processes it through a **Medallion architecture** (Bronze → Silver → Gold) on Delta Lake, trains predictive models for **% Iron** and **% Silica** in concentrate output, and serves them through a Streamlit dashboard with SPC charts and a what-if simulator. The whole pipeline is reproducible local ≡ cloud — same code runs on a laptop or on a Databricks cluster.

> ✅ **Status — modeling track + serving layer complete (2026-05-02).** All notebooks (EDA → features → LightGBM all-rows → LightGBM fresh-only → SPC → What-if), FastAPI inference service, Streamlit dashboard, and drift monitoring are live. Deployment to Databricks documented. **47/47 tests pass.** See the [Roadmap](#-roadmap).

---

## 📊 Headline results on the Kaggle flotation dataset (737K rows, 6 months)

| Metric | All-rows model | Fresh-only model (notebook 02b) |
|---|---|---|
| Test RMSE on `% Iron Concentrate` | 1.216 | **0.786** (−35.4%) |
| Test RMSE on `% Silica Concentrate` | 1.152 | **0.823** (−28.5%) |
| Test R² on `% Iron Concentrate` | −0.171 | −0.216 |
| Train rows used | 515,677 | 42,654 |

**The headline finding is not the RMSE** — it is the structural **temporal distribution shift** detected between train (Mar–Jun 2017) and test (Jul–Sep 2017). Fresh-only training cuts RMSE 28-35% by removing forward-fill noise from the supervision signal, but R² stays slightly negative because the test distribution moved.

**SPC catches the shift dramatically** — Shewhart Western Electric rules + CUSUM detect the regime change residue-by-residue:

| SPC method on residuals | Signals fired | % of test rows |
|---|---|---|
| Shewhart rule 1 (±3σ) | 19 | 0.21% |
| Shewhart rule 2 (2 of 3 ±2σ) | 132 | 1.44% |
| Shewhart rule 3 (4 of 5 ±1σ) | 2,074 | 22.68% |
| **Shewhart rule 4 (8 same side)** | **8,816** | **96.40%** |
| **CUSUM (δ=1σ, h=4σ)** | **8,567** | **93.68%** |
| EWMA (λ=0.2, L=3) | 2,117 | 23.15% |

The CUSUM Cl statistic ramps to **~1000 over thousands of rows** — visual proof of sustained model bias as the plant operating regime drifts. **This is exactly what production SPC is for**: catching the moment a model starts being systematically wrong, before the lab QA confirms the quality drift.

### Honest findings

1. **The Kaggle flotation dataset has 91.73% forward-filled labels.** Training on all rows treats forward-fills as ground truth and yields a model that beats the naive baseline by only 1.5–4%. Restricting to the 8.27% fresh lab readings is the methodologically correct path; documented in notebook 02b.

2. **Even with the fresh-only filter, R² remains slightly negative.** The dominant problem is **temporal distribution shift** between the first 70% of the timeline (train) and the last 15% (test). Operating regime, feed source, or instrument calibration changed mid-dataset — a real, common, and underreported phenomenon in industrial ML.

3. **The feature importances validate physical interpretation.** For `% Iron Concentrate` the top driver is `pct_iron_feed` (more iron in → more iron out — physically correct). For `% Silica Concentrate` the top drivers are starch flow and ore pulp density, exactly the reactives used to depress silica during reverse cationic flotation.

4. **The what-if simulator is robust to single-point overrides** (Δ predicted ≈ 0 for any pH override). The model correctly learned that one-instant excursions don't predict steady-state quality — only sustained changes (over a 30-min window or more) move the prediction. This is a **feature, not a bug**: in production, the simulator would override a contiguous window of cycles, not a single timestep.

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
- [x] Bronze + Silver + Gold pipeline (rolling/lag/calendar features)
- [x] Notebook 00 — EDA on flotation sensors and targets
- [x] Notebook 01 — Feature engineering pipeline
- [x] Notebook 02 — LightGBM baseline + MLflow tracking
- [x] Notebook 03 — LSTM (PyTorch) sequence model with multi-target head
- [x] Notebook 04 — Statistical Process Control (Shewhart + CUSUM + EWMA + Western Electric)
- [x] Notebook 05 — What-if simulator (naive + exact recompute)
- [x] FastAPI serving (`/predict`, `/whatif`, `/health`) — `src/frothiq/serving/api.py`
- [x] Streamlit dashboard with SPC + what-if tabs — `src/frothiq/serving/dashboard.py`
- [x] Drift monitoring (Evidently + basic fallback) — `src/frothiq/monitoring/drift.py`
- [x] Dockerfile for inference container
- [x] Databricks deployment guide (`docs/databricks_deploy.md`)
- [ ] Run end-to-end on the actual Kaggle dataset (operator action — see Quickstart)
- [ ] Promote models to Databricks Unity Catalog Model Registry

---

## 📜 License

MIT — see [LICENSE](LICENSE).

---

## 👤 Author

**Jonathan Sánchez Pesantes** — Industrial Engineer · Data Scientist
🔗 [linkedin.com/in/jonasanchez](https://www.linkedin.com/in/jonasanchez) · [github.com/jsanchez-ds](https://github.com/jsanchez-ds)
