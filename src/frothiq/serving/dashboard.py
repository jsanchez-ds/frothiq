"""Streamlit dashboard for FrothIQ.

Three tabs:
  1. Live Predictions — point predictions on an uploaded feature CSV.
  2. SPC Charts — Shewhart, CUSUM, EWMA over a sensor or target.
  3. What-if Simulator — operator slider to ask "what if I change pH?".

Run with:
    streamlit run src/frothiq/serving/dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Allow running directly without installing.
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from frothiq.models.spc.cusum import CusumParams, cusum_chart  # noqa: E402
from frothiq.models.spc.ewma import EwmaParams, ewma_chart  # noqa: E402
from frothiq.models.spc.shewhart import (  # noqa: E402
    fit_control_limits,
    western_electric_violations,
)

st.set_page_config(page_title="FrothIQ", page_icon="⚗️", layout="wide")
st.title("⚗️ FrothIQ — Mineral Process Quality")
st.caption("Predicción de calidad de concentrado · Cartas de control · Simulador what-if")


tab_pred, tab_spc, tab_whatif = st.tabs(["📈 Predictions", "🚦 SPC Charts", "🎛️ What-if"])


# ----- Tab 1: Predictions -----------------------------------------------------

with tab_pred:
    st.subheader("Predicciones de calidad")
    st.markdown(
        "Sube un CSV con las features para obtener predicciones de "
        "`% Iron Concentrate` y `% Silica Concentrate`. "
        "Los modelos se cargan desde MLflow Registry; si no están disponibles, "
        "se usa un fallback local en `artifacts/`."
    )

    uploaded = st.file_uploader("CSV con features", type=["csv"], key="pred_uploader")
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.write(f"**{len(df):,} filas × {df.shape[1]} columnas**")
        st.dataframe(df.head(), use_container_width=True)

        st.info(
            "Para predecir, levanta el servicio FastAPI y conecta este dashboard a "
            "`http://localhost:8000/predict`. En modo demo, este tab solo muestra el preview."
        )


# ----- Tab 2: SPC Charts ------------------------------------------------------

with tab_spc:
    st.subheader("Cartas de control estadístico de proceso")
    st.markdown(
        "Tres tipos de cartas: **Shewhart** (puntos individuales ±3σ), "
        "**CUSUM** (acumula desviaciones, detecta shifts pequeños persistentes), "
        "**EWMA** (suaviza con ponderación exponencial)."
    )

    col_left, col_right = st.columns(2)
    with col_left:
        spc_uploaded = st.file_uploader(
            "CSV con timestamp + columna a monitorear", type=["csv"], key="spc_uploader"
        )
    with col_right:
        chart_type = st.selectbox("Tipo de carta", ["Shewhart", "CUSUM", "EWMA"], key="spc_type")

    if spc_uploaded is not None:
        df = pd.read_csv(spc_uploaded, parse_dates=[0] if True else None)
        all_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not all_cols:
            st.error("No hay columnas numéricas en el CSV.")
        else:
            col_to_chart = st.selectbox("Columna a monitorear", all_cols, key="spc_col")
            baseline_pct = st.slider(
                "% inicial usado como baseline para fit", 5, 50, 20, key="spc_baseline_pct",
            )
            n_baseline = max(30, int(len(df) * baseline_pct / 100))
            baseline_values = df[col_to_chart].iloc[:n_baseline].dropna().values

            if len(baseline_values) < 30:
                st.warning("Baseline muy corto. Necesitas al menos 30 puntos.")
            else:
                limits = fit_control_limits(baseline_values)
                values = df[col_to_chart].values
                ts = df.iloc[:, 0] if pd.api.types.is_datetime64_any_dtype(df.iloc[:, 0]) else df.index

                fig = go.Figure()

                if chart_type == "Shewhart":
                    violations = western_electric_violations(values, limits)
                    any_violation = (
                        violations["rule_1"] | violations["rule_2"]
                        | violations["rule_3"] | violations["rule_4"]
                    )
                    fig.add_trace(go.Scatter(x=ts, y=values, mode="lines", name=col_to_chart, line={"color": "#1f77b4"}))
                    fig.add_hline(y=limits.center, line={"color": "green", "dash": "dot"}, annotation_text="μ")
                    fig.add_hline(y=limits.ucl_3, line={"color": "red", "dash": "dash"}, annotation_text="UCL ±3σ")
                    fig.add_hline(y=limits.lcl_3, line={"color": "red", "dash": "dash"})
                    if any_violation.any():
                        fig.add_trace(go.Scatter(
                            x=np.array(ts)[any_violation], y=values[any_violation],
                            mode="markers", name="Violación WE",
                            marker={"color": "red", "size": 10, "symbol": "x"},
                        ))
                    st.metric("Violaciones Western Electric (total)", int(any_violation.sum()))
                elif chart_type == "CUSUM":
                    delta = st.slider("δ (tamaño de shift a detectar, en σ)", 0.5, 3.0, 1.0, step=0.5, key="cusum_delta")
                    h = st.slider("h (decision interval, en σ)", 3.0, 6.0, 4.0, step=0.5, key="cusum_h")
                    params = CusumParams(target=limits.center, sigma=limits.sigma, delta_sigma=delta, h_sigma=h)
                    chart = cusum_chart(values, params)
                    fig.add_trace(go.Scatter(x=ts, y=chart["cu"], name="Cu (upward)", line={"color": "#1f77b4"}))
                    fig.add_trace(go.Scatter(x=ts, y=chart["cl"], name="Cl (downward)", line={"color": "#ff7f0e"}))
                    fig.add_hline(y=params.h, line={"color": "red", "dash": "dash"}, annotation_text="h")
                    st.metric("Señales CUSUM (total)", int(chart["signal"].sum()))
                else:  # EWMA
                    lam = st.slider("λ (smoothing parameter)", 0.05, 0.5, 0.2, step=0.05, key="ewma_lambda")
                    L = st.slider("L (control limit width, σ)", 2.0, 4.0, 3.0, step=0.5, key="ewma_L")
                    params = EwmaParams(target=limits.center, sigma=limits.sigma, lambda_=lam, L=L)
                    chart = ewma_chart(values, params)
                    fig.add_trace(go.Scatter(x=ts, y=values, mode="lines", name="raw", line={"color": "lightgray"}))
                    fig.add_trace(go.Scatter(x=ts, y=chart["z"], name="EWMA", line={"color": "#1f77b4"}))
                    fig.add_trace(go.Scatter(x=ts, y=chart["ucl"], name="UCL", line={"color": "red", "dash": "dash"}))
                    fig.add_trace(go.Scatter(x=ts, y=chart["lcl"], name="LCL", line={"color": "red", "dash": "dash"}))
                    st.metric("Señales EWMA (total)", int(chart["signal"].sum()))

                fig.update_layout(
                    height=500, hovermode="x unified",
                    title=f"{chart_type} chart — {col_to_chart}",
                    xaxis_title="time", yaxis_title=col_to_chart,
                )
                st.plotly_chart(fig, use_container_width=True)


# ----- Tab 3: What-if ---------------------------------------------------------

with tab_whatif:
    st.subheader("Simulador what-if")
    st.markdown(
        "Pregunta: *¿qué pasa con la calidad si cambio el pH de 9.4 a 9.8?* "
        "El modelo evalúa la predicción actual y la contrafactual con el override aplicado."
    )

    st.info(
        "💡 **Modo demo**: para usar este simulador con un modelo entrenado, levanta "
        "FastAPI (`uvicorn frothiq.serving.api:app`) y este tab consultará "
        "`POST /whatif` con tus inputs.\n\n"
        "Mientras tanto, puedes explorar la interfaz:"
    )

    col_a, col_b = st.columns(2)
    with col_a:
        target = st.selectbox(
            "Target a evaluar",
            ["pct_iron_concentrate", "pct_silica_concentrate"],
            key="whatif_target",
        )
        current_ph = st.number_input("pH actual", value=9.4, step=0.1, key="whatif_curr_ph")
        new_ph = st.slider("Nuevo pH (override)", 8.0, 11.0, 9.8, 0.1, key="whatif_new_ph")
    with col_b:
        current_density = st.number_input("Densidad actual", value=1.7, step=0.05, key="whatif_curr_d")
        new_density = st.slider("Nueva densidad (override)", 1.5, 2.0, 1.7, 0.05, key="whatif_new_d")

    overrides = {}
    if new_ph != current_ph:
        overrides["ore_pulp_ph"] = new_ph
    if new_density != current_density:
        overrides["ore_pulp_density"] = new_density

    if st.button("Evaluar what-if", key="whatif_btn"):
        if not overrides:
            st.warning("No hay overrides — ajusta al menos un slider.")
        else:
            st.success("Overrides preparados.")
            st.json(overrides)
            st.caption(
                "Esta acción enviaría un `POST /whatif` al servicio FastAPI con los overrides + features actuales. "
                "Conecta el endpoint para ver los resultados reales."
            )
