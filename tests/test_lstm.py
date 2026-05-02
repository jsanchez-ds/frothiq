"""Tests for the LSTM module — exercises shapes, scaler invariants, and a smoke training run."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from frothiq.models.deep.lstm import (
    LSTMRegressor,
    SensorScaler,
    make_windows,
    train_lstm,
)


def _toy_flotation(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2017-01-01", periods=n, freq="20s"),
            "ore_pulp_ph": rng.normal(9.5, 0.2, size=n),
            "ore_pulp_density": rng.normal(1.7, 0.05, size=n),
            "amina_flow": rng.normal(500, 50, size=n),
            "pct_iron_concentrate": rng.normal(64, 1, size=n),
            "pct_silica_concentrate": rng.normal(2, 0.5, size=n),
        }
    )


def test_sensor_scaler_zero_centers_train_data():
    df = _toy_flotation()
    cols = ["ore_pulp_ph", "ore_pulp_density", "amina_flow"]
    scaler = SensorScaler.fit(df, cols)
    z = scaler.transform(df)
    assert np.allclose(z.mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(z.std(axis=0), 1.0, atol=1e-6)


def test_sensor_scaler_handles_constant_columns():
    df = pd.DataFrame({"a": [1.0, 1.0, 1.0], "b": [0.0, 1.0, 2.0]})
    scaler = SensorScaler.fit(df, ["a", "b"])
    z = scaler.transform(df)
    # Constant column collapses to zero (mean centered, std forced to 1.0).
    assert np.allclose(z[:, 0], 0.0)


def test_make_windows_shape():
    df = _toy_flotation(n=100)
    cols = ["ore_pulp_ph", "ore_pulp_density"]
    target_cols = ["pct_iron_concentrate", "pct_silica_concentrate"]
    scaler = SensorScaler.fit(df, cols)
    X, y = make_windows(df, cols, window=20, target_cols=target_cols, scaler=scaler)
    assert X.shape == (81, 20, 2)  # 100 - 20 + 1 = 81 windows
    assert y.shape == (81, 2)


def test_make_windows_raises_on_too_short():
    df = _toy_flotation(n=10)
    with pytest.raises(ValueError, match="at least"):
        make_windows(
            df,
            sensor_cols=["ore_pulp_ph"],
            window=50,
            target_cols=["pct_iron_concentrate"],
        )


def test_lstm_forward_output_shape():
    model = LSTMRegressor(n_features=3, n_targets=2, hidden_size=8, num_layers=1, dropout=0.0)
    x = torch.randn(7, 12, 3)
    out = model(x)
    assert out.shape == (7, 2)


def test_train_lstm_smoke(tmp_path):
    """Tiny end-to-end run on synthetic data — verifies the loop & MLflow logging."""
    import mlflow

    mlflow.set_tracking_uri(f"file:///{tmp_path.as_posix()}/mlruns")
    df = _toy_flotation(n=400)
    train_df = df.iloc[:300]
    val_df = df.iloc[300:]
    cols = ["ore_pulp_ph", "ore_pulp_density", "amina_flow"]
    targets = ["pct_iron_concentrate", "pct_silica_concentrate"]

    result = train_lstm(
        train_df,
        val_df,
        sensor_cols=cols,
        target_cols=targets,
        window=20,
        hidden_size=8,
        num_layers=1,
        max_epochs=2,
        patience=2,
        batch_size=32,
        device="cpu",
        run_name="smoke",
    )
    assert isinstance(result.val_metrics["val_rmse"], float)
    assert np.isfinite(result.val_metrics["val_rmse"])
    assert result.best_epoch >= 1
