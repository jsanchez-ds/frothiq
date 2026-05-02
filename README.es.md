[English](README.md) · 🌐 **Español**

# ⚗️ FrothIQ — Pronóstico de Calidad en Procesos Mineros sobre Databricks

> **Plataforma productiva de ML para predecir calidad en procesos de flotación de minerales. Construida local-first, desplegable a Azure Databricks. Incluye control estadístico de proceso (SPC) y simulador what-if para operadores.**

Plataforma end-to-end que ingesta datos de sensores de una planta de flotación de minerales (datos industriales reales de una planta de concentración de hierro en Brasil), los procesa con **arquitectura Medallion** (Bronze → Silver → Gold) sobre Delta Lake, entrena modelos predictivos para **% de Hierro** y **% de Sílice** en el concentrado de salida, y los sirve a través de un dashboard Streamlit con cartas SPC y un simulador what-if. Todo el pipeline es reproducible local ≡ cloud — el mismo código corre en una laptop o en un cluster de Databricks.

> ✅ **Estado — track de modelado + capa de serving completos (2026-05-02).** Los 6 notebooks (EDA → features → LightGBM → LSTM → SPC → What-if), el servicio FastAPI, el dashboard Streamlit y el monitoreo de drift están operativos. Despliegue Databricks documentado. **47/47 tests pasan.** Ver [Roadmap](#-roadmap).

---

## 🎯 Lo que demuestra este proyecto

| Capacidad | Evidencia |
|---|---|
| **Dominio minería** | Datos industriales reales de una planta de flotación (737K filas, 24 sensores) |
| **Big data sobre Spark** | Pipeline PySpark corre local y en Databricks sin cambios |
| **Arquitectura Medallion** | Bronze (raw) → Silver (limpio) → Gold (features) sobre Delta Lake |
| **ML productivo** | LightGBM + LSTM (PyTorch) trackeados en MLflow con model signatures |
| **Control estadístico de proceso** | Cartas Shewhart, CUSUM, reglas Western Electric para alertas operacionales |
| **Simulador what-if** | Dashboard del operador: "¿qué pasa si cambio el pH de X a Y?" |
| **Rigor MLOps** | Tests + CI, Docker, monitoreo de drift, model registry con aliases |
| **Local-first → Cloud** | Reproducible local ≡ Databricks; guía de despliegue incluida |

---

## 📊 Dataset

**[Quality Prediction in a Mining Process](https://www.kaggle.com/datasets/edumagalhaes/quality-prediction-in-a-mining-process)** por Edson Antonio Magalhaes (Kaggle, ~700K descargas).

Datos industriales reales de una planta brasilera de concentración de hierro. El dataset:

- **737,453 filas** muestreadas cada 20 segundos durante ~6 meses
- **24 columnas**: `% Hierro Feed`, `% Sílice Feed`, `Flujo de Almidón`, `Flujo de Amina`, `Flujo de Pulpa`, `pH de Pulpa`, `Densidad de Pulpa`, 7 columnas `Flujo de Aire por Columna de Flotación`, 7 columnas `Nivel por Columna de Flotación`, target `% Hierro Concentrado` y `% Sílice Concentrado`
- **Targets** (mediciones de laboratorio, horarias): `% Hierro Concentrado` y `% Sílice Concentrado`
- **Objetivo**: predecir outputs a partir de mediciones de sensores aguas arriba para que operadores ajusten parámetros antes que el producto salga fuera de especificación

---

## 🚀 Inicio rápido

```bash
# 1. Clonar
git clone https://github.com/jsanchez-ds/frothiq.git
cd frothiq

# 2. Instalar (Python 3.11+ recomendado)
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -e ".[dev]"

# 3. Obtener token de Kaggle API (una sola vez)
#    https://www.kaggle.com/settings → API → Create New Token
#    Guardar kaggle.json en %USERPROFILE%\.kaggle\ (Windows) o ~/.kaggle/

# 4. Descargar el dataset (~125 MB)
bash scripts/download_data.sh

# 5. Abrir el notebook de EDA
jupyter lab notebooks/00_eda.ipynb
```

---

## 🗺️ Roadmap

- [x] Scaffolding + READMEs bilingües + esqueleto CI
- [x] Script de descarga (Kaggle API)
- [x] Pipeline de features (rolling, lag, calendario)
- [x] Notebook 00 — EDA sobre sensores y targets
- [x] Notebook 01 — Feature engineering pipeline
- [x] Notebook 02 — Baseline LightGBM + MLflow tracking
- [x] Notebook 03 — LSTM (PyTorch) con cabeza multi-target
- [x] Notebook 04 — Control estadístico de proceso (Shewhart + CUSUM + EWMA + WE)
- [x] Notebook 05 — Simulador what-if (naive + exacto recomputado)
- [x] Serving FastAPI (`/predict`, `/whatif`, `/health`) — `src/frothiq/serving/api.py`
- [x] Dashboard Streamlit con pestañas SPC + what-if — `src/frothiq/serving/dashboard.py`
- [x] Monitoreo de drift (Evidently + fallback básico) — `src/frothiq/monitoring/drift.py`
- [x] Dockerfile para contenedor de inferencia
- [x] Guía de despliegue Databricks (`docs/databricks_deploy.md`)
- [ ] Correr end-to-end sobre el dataset real de Kaggle (acción del operador — ver Quickstart)
- [ ] Promover modelos a Databricks Unity Catalog Model Registry

---

## 📜 Licencia

MIT — ver [LICENSE](LICENSE).

---

## 👤 Autor

**Jonathan Sánchez Pesantes** — Ingeniero Civil Industrial · Data Scientist
🔗 [linkedin.com/in/jonasanchez](https://www.linkedin.com/in/jonasanchez) · [github.com/jsanchez-ds](https://github.com/jsanchez-ds)
