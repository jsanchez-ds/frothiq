"""What-if simulator for flotation operators.

Lets an operator ask: "what happens to predicted % iron concentrate if I change
pH from 9.4 to 9.8?". Two implementations:

* :func:`apply_overrides_naive` — fast, approximate. Substitutes the new value
  for the latest reading and nudges rolling means proportionally by 1/window.
  Suitable for real-time UI sliders where speed matters.

* :func:`apply_overrides_exact` — slower, accurate. Replays the recent window
  of raw sensor data with the override applied and re-runs the rolling
  pipeline so derived features (mean / std / min / max / lag) are exact.
  Suitable for committed decisions where precision matters.

Both return a feature row that the trained model can score directly. The
:func:`simulate_whatif_*` helpers compute baseline + counterfactual + delta in one call.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from frothiq.features.rolling import add_lag_features, add_rolling_features


@dataclass
class WhatIfResult:
    baseline_pred: np.ndarray
    counterfactual_pred: np.ndarray
    delta: np.ndarray
    overrides: dict[str, float]


# ----- Naive override (fast) --------------------------------------------------


_ROLLING_SUFFIXES = ("_mean_", "_std_", "_min_", "_max_")


def apply_overrides_naive(
    feature_row: pd.Series,
    overrides: dict[str, float],
) -> pd.Series:
    """Apply ``overrides`` to a feature row and propagate naively to derived stats.

    For each overridden sensor ``X``:
      - Replaces ``X`` with the new value.
      - For every derived rolling feature ``X_{stat}_{w}``: nudges by weight 1/w
        toward the new value (only for ``mean``); ``min``/``max`` are updated only
        if the new value is more extreme; ``std`` is left unchanged.
    """
    out = feature_row.copy()
    for col, value in overrides.items():
        if col not in out.index:
            continue
        out[col] = value
        for k in out.index:
            if not k.startswith(f"{col}_"):
                continue
            for suffix in _ROLLING_SUFFIXES:
                if suffix in k:
                    parts = k.split(suffix)
                    if len(parts) == 2 and parts[0] == col:
                        try:
                            window = int(parts[1])
                        except ValueError:
                            continue
                        weight = 1.0 / window
                        if "_mean_" in k:
                            out[k] = (1 - weight) * out[k] + weight * value
                        elif "_min_" in k:
                            out[k] = min(out[k], value)
                        elif "_max_" in k:
                            out[k] = max(out[k], value)
                        # std left unchanged in the naive path.
    return out


# ----- Exact override (recompute) ---------------------------------------------


def apply_overrides_exact(
    recent_window: pd.DataFrame,
    sensor_cols: Iterable[str],
    feature_cols: Iterable[str],
    overrides: dict[str, float],
    rolling_windows: Iterable[int] = (30, 180, 540),
    lag_steps: Iterable[int] = (30, 180, 540),
    timestamp_col: str = "timestamp",
) -> pd.Series:
    """Recompute features exactly after applying overrides to the most recent row.

    Parameters
    ----------
    recent_window : DataFrame
        At least ``max(rolling_windows)`` rows ending at the cycle the operator is
        evaluating, sorted by timestamp ascending.
    sensor_cols : iterable of str
        Sensor columns whose rolling/lag features will be re-derived.
    feature_cols : iterable of str
        Final feature column names expected by the model.
    overrides : dict
        ``{sensor_name: new_value}`` applied to the **last row** of recent_window.
    """
    sensor_cols = list(sensor_cols)
    feature_cols = list(feature_cols)
    rolling_windows = tuple(rolling_windows)
    lag_steps = tuple(lag_steps)

    modified = recent_window.copy()
    last_idx = modified.index[-1]
    for col, value in overrides.items():
        if col in modified.columns:
            modified.loc[last_idx, col] = value

    rebuilt = add_rolling_features(
        modified, sensor_cols, windows=rolling_windows, timestamp_col=timestamp_col
    )
    rebuilt = add_lag_features(
        rebuilt, sensor_cols, lags=lag_steps, timestamp_col=timestamp_col
    )

    last = rebuilt.iloc[-1]
    return last.reindex(feature_cols)


# ----- Simulator entrypoints --------------------------------------------------


def simulate_whatif_naive(
    model,
    feature_row: pd.Series,
    feature_cols: list[str],
    overrides: dict[str, float],
) -> WhatIfResult:
    """Naive what-if: O(1) — fast, approximate."""
    X_base = feature_row[feature_cols].to_frame().T
    baseline_pred = np.asarray(model.predict(X_base)).reshape(-1)

    modified = apply_overrides_naive(feature_row, overrides)
    X_cf = modified[feature_cols].to_frame().T
    counterfactual_pred = np.asarray(model.predict(X_cf)).reshape(-1)

    return WhatIfResult(
        baseline_pred=baseline_pred,
        counterfactual_pred=counterfactual_pred,
        delta=counterfactual_pred - baseline_pred,
        overrides=overrides,
    )


def simulate_whatif_exact(
    model,
    recent_window: pd.DataFrame,
    sensor_cols: list[str],
    feature_cols: list[str],
    overrides: dict[str, float],
    rolling_windows: Iterable[int] = (30, 180, 540),
    lag_steps: Iterable[int] = (30, 180, 540),
    timestamp_col: str = "timestamp",
) -> WhatIfResult:
    """Exact what-if: recompute features over the recent window. Slower, accurate."""
    base_row = apply_overrides_exact(
        recent_window, sensor_cols, feature_cols, overrides={},
        rolling_windows=rolling_windows, lag_steps=lag_steps, timestamp_col=timestamp_col,
    )
    X_base = base_row.to_frame().T
    baseline_pred = np.asarray(model.predict(X_base)).reshape(-1)

    cf_row = apply_overrides_exact(
        recent_window, sensor_cols, feature_cols, overrides=overrides,
        rolling_windows=rolling_windows, lag_steps=lag_steps, timestamp_col=timestamp_col,
    )
    X_cf = cf_row.to_frame().T
    counterfactual_pred = np.asarray(model.predict(X_cf)).reshape(-1)

    return WhatIfResult(
        baseline_pred=baseline_pred,
        counterfactual_pred=counterfactual_pred,
        delta=counterfactual_pred - baseline_pred,
        overrides=overrides,
    )


# ----- Backward-compat aliases (used by older tests) --------------------------

apply_overrides = apply_overrides_naive
simulate_whatif = simulate_whatif_naive
