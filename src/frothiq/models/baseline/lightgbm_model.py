"""LightGBM baseline for predicting iron and silica concentrate quality.

Trains one LightGBM regressor per target (multi-output via independent models)
with full MLflow tracking: params, val metrics, model with signature.
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass
class TargetResult:
    target: str
    model: lgb.LGBMRegressor
    val_metrics: dict[str, float]
    test_metrics: dict[str, float] | None = None


LGBM_DEFAULTS: dict = {
    "objective": "regression",
    "metric": "rmse",
    "n_estimators": 1500,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 20,
    "subsample": 0.85,
    "subsample_freq": 1,
    "colsample_bytree": 0.85,
    "reg_lambda": 1.0,
    "n_jobs": -1,
    "verbosity": -1,
    "random_state": 0,
}


def _eval(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        f"{prefix}_mae": float(mean_absolute_error(y_true, y_pred)),
        f"{prefix}_r2": float(r2_score(y_true, y_pred)),
    }


def train_one_target(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    target_name: str,
    params: dict | None = None,
    X_test: pd.DataFrame | None = None,
    y_test: np.ndarray | None = None,
    run_name: str | None = None,
) -> TargetResult:
    """Train a LightGBM regressor for a single target column with MLflow."""
    cfg = {**LGBM_DEFAULTS, **(params or {})}
    run_name = run_name or f"lgbm-{target_name}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(cfg)
        mlflow.log_param("target", target_name)
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_train_rows", len(X_train))
        mlflow.log_param("n_val_rows", len(X_val))

        model = lgb.LGBMRegressor(**cfg)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )

        y_val_pred = model.predict(X_val)
        val_metrics = _eval(y_val, y_val_pred, "val")
        mlflow.log_metrics(val_metrics)

        test_metrics = None
        if X_test is not None and y_test is not None:
            y_test_pred = model.predict(X_test)
            test_metrics = _eval(y_test, y_test_pred, "test")
            mlflow.log_metrics(test_metrics)

        signature = infer_signature(X_train, model.predict(X_train.head(5)))
        mlflow.lightgbm.log_model(
            model,
            name="model",
            signature=signature,
            input_example=X_train.head(2),
        )
        return TargetResult(
            target=target_name,
            model=model,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
        )
