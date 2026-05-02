"""Pydantic schemas for the FrothIQ FastAPI service."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class FeatureRow(BaseModel):
    """A single row of features (one observation) — keys are feature names,
    values are numeric. The schema is intentionally open because the feature
    set evolves with retraining; the model's signature enforces the contract
    at inference time.
    """

    features: dict[str, float] = Field(
        ..., description="Mapping from feature name to numeric value."
    )


class PredictRequest(BaseModel):
    """Request body for ``POST /predict``."""

    rows: list[FeatureRow] = Field(
        ..., description="One or more feature rows to score."
    )
    target: str | None = Field(
        default=None,
        description=(
            "Optional: which target to predict ('pct_iron_concentrate' or "
            "'pct_silica_concentrate'). If omitted, both are returned."
        ),
    )


class PredictResponse(BaseModel):
    """Response body from ``POST /predict``."""

    predictions: list[dict[str, float]] = Field(
        ..., description="Per-row predictions: {target_name: value}."
    )
    model_versions: dict[str, str] = Field(
        ..., description="Model versions / aliases used per target."
    )


class WhatIfRequest(BaseModel):
    """Request body for ``POST /whatif``."""

    current_features: dict[str, float] = Field(
        ..., description="Current feature row (latest observation)."
    )
    overrides: dict[str, float] = Field(
        ..., description="Sensor overrides — {sensor_name: hypothetical_value}."
    )
    target: str = Field(
        ..., description="Which target to evaluate the what-if for."
    )


class WhatIfResponse(BaseModel):
    baseline: float
    counterfactual: float
    delta: float
    overrides: dict[str, float]
    target: str


class HealthResponse(BaseModel):
    status: str
    models_loaded: list[str]
    info: dict[str, Any] = {}
