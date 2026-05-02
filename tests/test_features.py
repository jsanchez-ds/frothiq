"""Tests for feature engineering pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from frothiq.features.pipeline import FeatureConfig, build_features, list_feature_cols
from frothiq.features.rolling import (
    add_calendar_features,
    add_lag_features,
    add_rolling_features,
)


def _toy_flotation(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(0)
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


def test_add_rolling_features_shape():
    df = _toy_flotation(n=50)
    out = add_rolling_features(
        df, sensor_cols=["ore_pulp_ph", "amina_flow"], windows=(5, 15)
    )
    # 4 stats × 2 sensors × 2 windows = 16 new columns.
    new_cols = set(out.columns) - set(df.columns)
    assert len(new_cols) == 16
    assert {"ore_pulp_ph_mean_5", "amina_flow_std_15"} <= new_cols


def test_add_rolling_features_no_nan_in_first_row():
    df = _toy_flotation(n=10)
    out = add_rolling_features(df, sensor_cols=["ore_pulp_ph"], windows=(5,))
    # min_periods=1 in our implementation — first row is the value itself, not NaN.
    assert pd.notna(out.iloc[0]["ore_pulp_ph_mean_5"])


def test_add_lag_features_shifts_correctly():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2017-01-01", periods=10, freq="20s"),
            "x": np.arange(10, dtype=float),
        }
    )
    out = add_lag_features(df, sensor_cols=["x"], lags=(2,))
    # Lag-2 of [0,1,2,...,9] is [NaN, NaN, 0, 1, 2, ..., 7].
    assert pd.isna(out["x_lag_2"].iloc[1])
    assert out["x_lag_2"].iloc[2] == 0.0
    assert out["x_lag_2"].iloc[5] == 3.0


def test_add_calendar_features():
    df = pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp("2017-03-13 10:00:00"),  # Monday 10am
                pd.Timestamp("2017-03-18 23:00:00"),  # Saturday 11pm
            ]
        }
    )
    out = add_calendar_features(df)
    assert out.iloc[0]["hour_of_day"] == 10
    assert out.iloc[0]["day_of_week"] == 0  # Monday
    assert out.iloc[0]["is_weekend"] == 0
    assert out.iloc[1]["is_weekend"] == 1


def test_build_features_combines_all():
    df = _toy_flotation(n=50)
    cfg = FeatureConfig(rolling_windows=(5,), lag_steps=(3,))
    out = build_features(
        df,
        sensor_cols=["ore_pulp_ph", "amina_flow"],
        feed_cols=[],
        config=cfg,
    )
    expected_new = {
        "ore_pulp_ph_mean_5", "ore_pulp_ph_std_5",
        "ore_pulp_ph_min_5", "ore_pulp_ph_max_5",
        "ore_pulp_ph_lag_3", "amina_flow_lag_3",
        "hour_of_day", "day_of_week", "is_weekend",
    }
    assert expected_new <= set(out.columns)


def test_list_feature_cols_excludes_targets_and_timestamp():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2017-01-01", periods=10, freq="20s"),
            "ph": np.arange(10, dtype=float),
            "pct_iron_concentrate": np.arange(10, dtype=float) + 60,
        }
    )
    cols = list_feature_cols(df, target_cols=["pct_iron_concentrate"])
    assert "ph" in cols
    assert "pct_iron_concentrate" not in cols
    assert "timestamp" not in cols
