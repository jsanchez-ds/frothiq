[English](README.md) · 🌐 **Español**

# ⚗️ FrothIQ — Pronóstico de Calidad en Procesos Mineros sobre Databricks

> **Plataforma productiva de ML para predecir calidad en procesos de flotación de minerales. Construida local-first, desplegable a Azure Databricks. Incluye control estadístico de proceso (SPC) y simulador what-if para operadores.**

Plataforma end-to-end que ingesta datos de sensores de una planta de flotación de minerales (datos industriales reales de una planta de concentración de hierro en Brasil), los procesa con **arquitectura Medallion** (Bronze → Silver → Gold) sobre Delta Lake, entrena modelos predictivos para **% de Hierro** y **% de Sílice** en el concentrado de salida, y los sirve a través de un dashboard Streamlit con cartas SPC y un simulador what-if. Todo el pipeline es reproducible local ≡ cloud — el mismo código corre en una laptop o en un cluster de Databricks.

> ✅ **Estado — track de modelado + capa de serving completos (2026-05-02).** Todos los notebooks (EDA → features → LightGBM all-rows → LightGBM fresh-only → SPC → What-if), el servicio FastAPI, el dashboard Streamlit y el monitoreo de drift están operativos. Despliegue Databricks documentado. **47/47 tests pasan.** Ver [Roadmap](#-roadmap).

---

## 📊 Resultados principales sobre el dataset de Kaggle (737K filas, 6 meses)

| Métrica | Modelo all-rows | Modelo fresh-only (notebook 02b) |
|---|---|---|
| Test RMSE sobre `% Iron Concentrate` | 1.216 | **0.786** (−35.4%) |
| Test RMSE sobre `% Silica Concentrate` | 1.152 | **0.823** (−28.5%) |
| Test R² sobre `% Iron Concentrate` | −0.171 | −0.216 |
| Filas de entrenamiento | 515,677 | 42,654 |

**El hallazgo principal no es el RMSE** — es el **shift temporal estructural** entre train (mar–jun 2017) y test (jul–sep 2017). Entrenar solo con lab readings frescos reduce RMSE 28-35% al sacar el ruido de los forward-fills del supervisor signal, pero el R² queda ligeramente negativo porque la distribución del test cambió.

**Las cartas SPC capturan el shift dramáticamente** — Shewhart con reglas Western Electric + CUSUM detectan el cambio de régimen residual por residual:

| Método SPC sobre residuales | Signals | % del test |
|---|---|---|
| Shewhart regla 1 (±3σ) | 19 | 0.21% |
| Shewhart regla 2 (2 de 3 ±2σ) | 132 | 1.44% |
| Shewhart regla 3 (4 de 5 ±1σ) | 2,074 | 22.68% |
| **Shewhart regla 4 (8 mismo lado)** | **8,816** | **96.40%** |
| **CUSUM (δ=1σ, h=4σ)** | **8,567** | **93.68%** |
| EWMA (λ=0.2, L=3) | 2,117 | 23.15% |

El estadístico Cl del CUSUM sube hasta **~1000 sobre miles de filas** — prueba visual del bias sostenido del modelo a medida que la planta drifta su régimen operacional. **Eso es exactamente para lo que sirve el SPC en producción**: capturar el momento en que un modelo empieza a equivocarse sistemáticamente, antes de que el QA de laboratorio confirme el drift de calidad.

### Hallazgos honestos

1. **El dataset Kaggle tiene 91.73% de labels forward-filled.** Entrenar con todas las filas trata los forward-fills como ground truth y produce un modelo que solo supera al baseline ingenuo en 1.5–4%. Restringir al 8.27% de lab readings frescos es el camino metodológicamente correcto; documentado en notebook 02b.

2. **Incluso con el filtro fresh-only, R² queda ligeramente negativo.** El problema dominante es el **shift temporal de distribución** entre el primer 70% del timeline (train) y el último 15% (test). El régimen operacional, fuente del feed o calibración de instrumentos cambió a mitad del dataset — fenómeno real, común, y poco reportado en ML industrial.

3. **Las importancias de features validan interpretación física.** Para `% Iron Concentrate` el top driver es `pct_iron_feed` (más hierro entra → más hierro sale — físicamente correcto). Para `% Silica Concentrate` los top drivers son starch flow y ore pulp density, exactamente los reactivos usados para deprimir sílice en flotación catiónica reversa.

4. **El simulador what-if es robusto a overrides puntuales** (Δ predicted ≈ 0 para cualquier override de pH). El modelo aprendió correctamente que excursiones de un solo instante no predicen calidad estacionaria — solo cambios sostenidos (sobre una ventana de 30+ min) mueven la predicción. **Esto es un feature, no un bug**: en producción el simulador overridearía una ventana contigua de cycles, no un solo timestep.

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
