"""Rolling-window feature engineering for flotation sensor data.

Mineral flotation is a slow-dynamics process — changes in pH, density, or air
flow translate to concentrate quality with delays of minutes to tens of minutes.
Rolling statistics over those time horizons are typically the single most
predictive feature family.

This module computes rolling mean / std / min / max over multiple windows for
each sensor column. Windows are expressed in *number of rows* (the dataset is
sampled at 20-second intervals, so 30 rows = 10 minutes, 180 rows = 1 hour).
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def add_rolling_features(
    df: pd.DataFrame,
    sensor_cols: Iterable[str],
    windows: Iterable[int] = (30, 180, 540),  # 10 min, 1 h, 3 h at 20-s sampling
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Add rolling mean / std / min / max features per sensor and window.

    Parameters
    ----------
    df : DataFrame
        Long-format sensor table sorted by timestamp.
    sensor_cols : iterable of str
        Sensor columns to compute features for.
    windows : iterable of int
        Window sizes (in rows). Default: 30, 180, 540 = 10 min, 1 h, 3 h.

    Builds every new column once into a single ``pd.DataFrame`` and concatenates
    in a single shot — this avoids the ``PerformanceWarning: DataFrame is highly
    fragmented`` that fires when each new column is assigned individually.
    """
    sensor_cols = list(sensor_cols)
    base = df.sort_values(timestamp_col).reset_index(drop=True)

    new_frames: list[pd.DataFrame] = []
    for w in windows:
        roll = base[sensor_cols].rolling(window=w, min_periods=1)
        mean = roll.mean()
        std = roll.std().fillna(0.0)
        rmin = roll.min()
        rmax = roll.max()

        mean.columns = [f"{c}_mean_{w}" for c in sensor_cols]
        std.columns = [f"{c}_std_{w}" for c in sensor_cols]
        rmin.columns = [f"{c}_min_{w}" for c in sensor_cols]
        rmax.columns = [f"{c}_max_{w}" for c in sensor_cols]

        new_frames.extend([mean, std, rmin, rmax])

    return pd.concat([base, *new_frames], axis=1)


def add_lag_features(
    df: pd.DataFrame,
    sensor_cols: Iterable[str],
    lags: Iterable[int] = (30, 180, 540),
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Add lag features (sensor reading N rows ago) per sensor and lag.

    For mineral flotation, lags reflect process delay — feeding the model the
    pH 10 minutes ago captures the propagation through the column.
    """
    sensor_cols = list(sensor_cols)
    base = df.sort_values(timestamp_col).reset_index(drop=True)

    new_frames: list[pd.DataFrame] = []
    for lag in lags:
        shifted = base[sensor_cols].shift(lag)
        shifted.columns = [f"{c}_lag_{lag}" for c in sensor_cols]
        new_frames.append(shifted)

    return pd.concat([base, *new_frames], axis=1)


def add_calendar_features(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Add hour-of-day and day-of-week features. Useful when the plant has
    operational shifts that affect feed quality."""
    out = df.copy()
    ts = pd.to_datetime(out[timestamp_col])
    out["hour_of_day"] = ts.dt.hour.astype("int8")
    out["day_of_week"] = ts.dt.dayofweek.astype("int8")
    out["is_weekend"] = (out["day_of_week"] >= 5).astype("int8")
    return out
