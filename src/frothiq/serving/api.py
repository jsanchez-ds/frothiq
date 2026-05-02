"""FastAPI service for FrothIQ.

Exposes:
  - GET  /health         → service + model status
  - POST /predict        → batch prediction for one or both targets
  - POST /whatif         → naive counterfactual prediction

The service loads each target's model from MLflow Model Registry by alias
(``models:/frothiq-{target}@production``). If the registry is unavailable,
it falls back to local pickle files in ``artifacts/`` for development.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException

from frothiq.models.whatif.simulator import simulate_whatif_naive

from .schemas import (
    HealthResponse,
    PredictRequest,
    PredictResponse,
    WhatIfRequest,
    WhatIfResponse,
)

logger = logging.getLogger("frothiq.api")
logging.basicConfig(level=logging.INFO)


DEFAULT_TARGETS = ("pct_iron_concentrate", "pct_silica_concentrate")
MODEL_ALIAS = os.environ.get("FROTHIQ_MODEL_ALIAS", "production")


def _model_uri(target: str) -> str:
    """Return the MLflow model URI for a given target."""
    return f"models:/frothiq-{target}@{MODEL_ALIAS}"


class ModelStore:
    """In-process cache of loaded models per target."""

    def __init__(self) -> None:
        self._models: dict[str, Any] = {}
        self._versions: dict[str, str] = {}

    def load(self, target: str) -> Any:
        if target in self._models:
            return self._models[target]

        # Try MLflow Model Registry first.
        try:
            uri = _model_uri(target)
            model = mlflow.pyfunc.load_model(uri)
            self._models[target] = model
            self._versions[target] = uri
            logger.info(f"Loaded model from registry: {uri}")
            return model
        except Exception as exc:
            logger.warning(f"Could not load from registry ({exc}); trying local fallback.")

        # Local fallback: artifacts/{target}.pkl
        local_path = Path("artifacts") / f"{target}.pkl"
        if local_path.exists():
            import joblib

            model = joblib.load(local_path)
            self._models[target] = model
            self._versions[target] = f"local:{local_path}"
            logger.info(f"Loaded model from local: {local_path}")
            return model

        raise RuntimeError(
            f"No model available for target '{target}'. "
            f"Train one and either register it in MLflow as 'frothiq-{target}@{MODEL_ALIAS}' "
            f"or save it locally at {local_path}."
        )

    def versions(self) -> dict[str, str]:
        return dict(self._versions)

    def loaded_targets(self) -> list[str]:
        return list(self._models)


store = ModelStore()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Try loading both default targets at startup so the first request isn't cold."""
    for target in DEFAULT_TARGETS:
        try:
            store.load(target)
        except Exception as exc:
            logger.warning(f"Startup warmup failed for {target}: {exc}")
    yield


app = FastAPI(
    title="FrothIQ Inference API",
    version="0.1.0",
    description="Mineral process quality predictions and what-if simulator.",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        models_loaded=store.loaded_targets(),
        info={"model_alias": MODEL_ALIAS},
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    targets = (req.target,) if req.target else DEFAULT_TARGETS
    feature_rows = [r.features for r in req.rows]
    if not feature_rows:
        raise HTTPException(status_code=400, detail="At least one feature row is required.")

    df = pd.DataFrame(feature_rows)

    predictions: list[dict[str, float]] = [{} for _ in feature_rows]
    for target in targets:
        try:
            model = store.load(target)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        try:
            preds = model.predict(df)
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"Prediction failed for {target}: {exc}"
            ) from exc
        for i, p in enumerate(preds):
            predictions[i][target] = float(p)

    return PredictResponse(predictions=predictions, model_versions=store.versions())


@app.post("/whatif", response_model=WhatIfResponse)
def whatif(req: WhatIfRequest) -> WhatIfResponse:
    try:
        model = store.load(req.target)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    feature_row = pd.Series(req.current_features)
    feature_cols = list(req.current_features.keys())

    result = simulate_whatif_naive(model, feature_row, feature_cols, req.overrides)
    return WhatIfResponse(
        baseline=float(result.baseline_pred[0]),
        counterfactual=float(result.counterfactual_pred[0]),
        delta=float(result.delta[0]),
        overrides=req.overrides,
        target=req.target,
    )
