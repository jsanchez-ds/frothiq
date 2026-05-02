"""CUSUM (Cumulative Sum) control chart for detecting small mean shifts.

CUSUM accumulates deviations from a target value and signals when the cumulative
sum exceeds a threshold. It detects **small persistent shifts** that Shewhart
charts (sensitive to single-point excursions) tend to miss.

The two-sided CUSUM tracks deviations above (Cu) and below (Cl) the target:

    Cu_t = max(0, Cu_{t-1} + (x_t - target) - k)
    Cl_t = max(0, Cl_{t-1} + (target - x_t) - k)

A signal is raised when ``Cu_t > h`` or ``Cl_t > h`` (h is the decision interval).

Standard parameter choice (Page 1954, Hawkins & Olwell 1998):
    k = δ * σ / 2   (slack = half the shift size we want to detect, in σ units)
    h = 4 σ to 5 σ  (decision interval; 5σ ≈ ARL_0 of 465 in-control runs)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CusumParams:
    """Parameters for a two-sided CUSUM chart."""

    target: float
    sigma: float
    delta_sigma: float = 1.0  # detect 1-σ shifts by default
    h_sigma: float = 4.0  # decision interval = 4σ

    @property
    def k(self) -> float:
        """Slack parameter (a.k.a. reference value)."""
        return 0.5 * self.delta_sigma * self.sigma

    @property
    def h(self) -> float:
        """Decision interval."""
        return self.h_sigma * self.sigma


def cusum_chart(values: np.ndarray, params: CusumParams) -> dict[str, np.ndarray]:
    """Compute Cu (upward) and Cl (downward) CUSUM statistics and signal flags.

    Returns
    -------
    dict with keys:
        cu : (n,) cumulative sum above target
        cl : (n,) cumulative sum below target
        signal : (n,) boolean — True when either Cu > h or Cl > h
        signal_up : (n,) boolean — True when Cu > h (mean shifted up)
        signal_down : (n,) boolean — True when Cl > h (mean shifted down)
    """
    arr = np.asarray(values, dtype=float)
    n = len(arr)

    cu = np.zeros(n)
    cl = np.zeros(n)
    for i in range(n):
        prev_cu = cu[i - 1] if i > 0 else 0.0
        prev_cl = cl[i - 1] if i > 0 else 0.0
        cu[i] = max(0.0, prev_cu + (arr[i] - params.target) - params.k)
        cl[i] = max(0.0, prev_cl + (params.target - arr[i]) - params.k)

    signal_up = cu > params.h
    signal_down = cl > params.h
    return {
        "cu": cu,
        "cl": cl,
        "signal": signal_up | signal_down,
        "signal_up": signal_up,
        "signal_down": signal_down,
    }


def annotate_cusum(df: pd.DataFrame, column: str, params: CusumParams) -> pd.DataFrame:
    """Append CUSUM columns to ``df``: ``{column}_cu``, ``_cl``, ``_signal``, ``_signal_up``,
    ``_signal_down``.
    """
    out = df.copy()
    chart = cusum_chart(out[column].to_numpy(dtype=float), params)
    for k, v in chart.items():
        out[f"{column}_{k}"] = v
    return out
