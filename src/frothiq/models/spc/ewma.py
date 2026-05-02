"""EWMA (Exponentially Weighted Moving Average) control chart.

EWMA is the third pillar of classical SPC alongside Shewhart and CUSUM. It
weights recent observations more than old ones via a smoothing parameter λ:

    z_t = λ * x_t + (1 - λ) * z_{t-1},    z_0 = target

Control limits widen with t until they stabilize at:

    UCL = target + L * σ * sqrt(λ / (2 - λ))
    LCL = target - L * σ * sqrt(λ / (2 - λ))

Detects medium-sized shifts (between Shewhart's domain and CUSUM's). Common
parameter choice: λ ∈ [0.05, 0.25], L = 3 (standard 3-sigma equivalent).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class EwmaParams:
    """Parameters for an EWMA chart."""

    target: float
    sigma: float
    lambda_: float = 0.2  # smoothing parameter
    L: float = 3.0  # control limit width in σ units


def ewma_chart(values: np.ndarray, params: EwmaParams) -> dict[str, np.ndarray]:
    """Compute the EWMA series and time-varying control limits.

    Returns
    -------
    dict with keys:
        z : (n,) EWMA series
        ucl : (n,) upper control limit (time-varying)
        lcl : (n,) lower control limit (time-varying)
        signal : (n,) boolean — True when z > ucl or z < lcl
    """
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    lam = params.lambda_

    z = np.empty(n)
    z_prev = params.target
    for i in range(n):
        z[i] = lam * arr[i] + (1 - lam) * z_prev
        z_prev = z[i]

    # Time-varying control-limit factor; converges to sqrt(λ / (2 - λ)).
    t = np.arange(1, n + 1)
    factor = np.sqrt((lam / (2 - lam)) * (1 - (1 - lam) ** (2 * t)))
    half_width = params.L * params.sigma * factor
    ucl = params.target + half_width
    lcl = params.target - half_width

    signal = (z > ucl) | (z < lcl)
    return {"z": z, "ucl": ucl, "lcl": lcl, "signal": signal}


def annotate_ewma(df: pd.DataFrame, column: str, params: EwmaParams) -> pd.DataFrame:
    """Append EWMA columns to ``df``: ``{column}_z``, ``_ucl``, ``_lcl``, ``_signal``."""
    out = df.copy()
    chart = ewma_chart(out[column].to_numpy(dtype=float), params)
    for k, v in chart.items():
        out[f"{column}_{k}"] = v
    return out
