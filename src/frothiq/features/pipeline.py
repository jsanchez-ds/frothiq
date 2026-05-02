"""End-to-end feature engineering pipeline for FrothIQ.

Combines:
  1. Rolling statistics (mean / std / min / max) over multiple windows.
  2. Lag features.
  3. Calendar features (hour, dow, weekend).

Returns a "gold" DataFrame ready to feed into a tabular model.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .rolling import add_calendar_features, add_lag_features, add_rolling_features


@dataclass
class FeatureConfig:
    """Knobs for the feature pipeline."""

    rolling_windows: tuple[int, ...] = (30, 180, 540)  # 10 min, 1 h, 3 h
    lag_steps: tuple[int, ...] = (30, 180, 540)
    add_calendar: bool = True
    include_feed: bool = True
    extra_drop: tuple[str, ...] = field(default_factory=tuple)


def build_features(
    df: pd.DataFrame,
    sensor_cols: Iterable[str],
    feed_cols: Iterable[str] | None = None,
    config: FeatureConfig | None = None,
) -> pd.DataFrame:
    """Run the full feature engineering pipeline.

    Parameters
    ----------
    df : DataFrame
        Tidy DataFrame from `frothiq.data.loader.load_flotation`.
    sensor_cols : iterable of str
        Sensor columns to engineer features for.
    feed_cols : iterable of str, optional
        Feed quality columns (% iron / silica feed). Included as raw
        features if `config.include_feed` is True.
    config : FeatureConfig, optional
    """
    cfg = config or FeatureConfig()
    sensor_cols = list(sensor_cols)
    feed_cols = list(feed_cols) if feed_cols else []

    # Drop any columns the user explicitly asked to remove.
    extra_drop = list(cfg.extra_drop)
    sensor_cols = [c for c in sensor_cols if c not in extra_drop]

    out = add_rolling_features(df, sensor_cols, windows=cfg.rolling_windows)
    out = add_lag_features(out, sensor_cols, lags=cfg.lag_steps)

    if cfg.add_calendar:
        out = add_calendar_features(out)

    if not cfg.include_feed:
        for col in feed_cols:
            if col in out.columns:
                out = out.drop(columns=col)

    return out


def save_gold(df: pd.DataFrame, path: str | Path) -> Path:
    """Persist the gold features as Parquet (snappy compression)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, compression="snappy", index=False)
    return p


def load_gold(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def list_feature_cols(
    gold: pd.DataFrame,
    target_cols: Iterable[str],
    timestamp_col: str = "timestamp",
) -> list[str]:
    """Return the columns of `gold` that should be used as model features
    (excludes timestamp, targets, and any non-numeric columns).
    """
    target_cols = set(target_cols)
    drop = target_cols | {timestamp_col}
    return [
        c for c in gold.columns
        if c not in drop and gold[c].dtype.kind in "fi"
    ]
