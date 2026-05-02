"""Shewhart control chart and Western Electric rules.

A Shewhart chart plots a process variable over time with a centerline (mean)
and control limits at ±3σ. Western Electric rules detect "out of control"
patterns even when no single point breaches ±3σ — for example, 8 consecutive
points on the same side of the mean indicates a sustained shift.

This module provides minimal implementations suitable for a flotation plant.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ControlLimits:
    """Centerline + ±1σ, ±2σ, ±3σ control limits, fitted from a baseline period."""

    center: float
    sigma: float

    @property
    def lcl_3(self) -> float:
        return self.center - 3 * self.sigma

    @property
    def ucl_3(self) -> float:
        return self.center + 3 * self.sigma

    @property
    def lcl_2(self) -> float:
        return self.center - 2 * self.sigma

    @property
    def ucl_2(self) -> float:
        return self.center + 2 * self.sigma

    @property
    def lcl_1(self) -> float:
        return self.center - self.sigma

    @property
    def ucl_1(self) -> float:
        return self.center + self.sigma


def fit_control_limits(values: np.ndarray) -> ControlLimits:
    """Fit Shewhart control limits from a baseline period (must be in-control)."""
    arr = np.asarray(values, dtype=float)
    return ControlLimits(center=float(arr.mean()), sigma=float(arr.std(ddof=1)))


def western_electric_violations(
    values: np.ndarray,
    limits: ControlLimits,
) -> dict[str, np.ndarray]:
    """Detect Western Electric rule violations.

    Returns a dict with one boolean array per rule, marking the index where the
    violation is concluded (i.e. last point of the offending pattern).

    Rules implemented (the canonical four):
      1. One point beyond ±3σ.
      2. Two of three consecutive points beyond ±2σ on the same side.
      3. Four of five consecutive points beyond ±1σ on the same side.
      4. Eight consecutive points on the same side of the centerline.
    """
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    above = arr > limits.center
    below = arr < limits.center

    out: dict[str, np.ndarray] = {}

    # Rule 1: any point beyond ±3σ.
    out["rule_1"] = (arr > limits.ucl_3) | (arr < limits.lcl_3)

    # Rule 2: 2 of 3 consecutive points beyond ±2σ same side.
    rule2 = np.zeros(n, dtype=bool)
    for i in range(2, n):
        window_above_2sig = (arr[i - 2 : i + 1] > limits.ucl_2).sum() >= 2
        window_below_2sig = (arr[i - 2 : i + 1] < limits.lcl_2).sum() >= 2
        rule2[i] = window_above_2sig or window_below_2sig
    out["rule_2"] = rule2

    # Rule 3: 4 of 5 consecutive points beyond ±1σ same side.
    rule3 = np.zeros(n, dtype=bool)
    for i in range(4, n):
        window_above = (arr[i - 4 : i + 1] > limits.ucl_1).sum() >= 4
        window_below = (arr[i - 4 : i + 1] < limits.lcl_1).sum() >= 4
        rule3[i] = window_above or window_below
    out["rule_3"] = rule3

    # Rule 4: 8 consecutive points on same side of centerline.
    rule4 = np.zeros(n, dtype=bool)
    for i in range(7, n):
        if above[i - 7 : i + 1].all() or below[i - 7 : i + 1].all():
            rule4[i] = True
    out["rule_4"] = rule4

    return out


def annotate_violations(
    df: pd.DataFrame,
    column: str,
    limits: ControlLimits,
) -> pd.DataFrame:
    """Append per-row boolean columns for each Western Electric rule violation."""
    out = df.copy()
    violations = western_electric_violations(out[column].values, limits)
    for rule, flags in violations.items():
        out[f"{column}_{rule}"] = flags
    out[f"{column}_any_violation"] = (
        out[f"{column}_rule_1"]
        | out[f"{column}_rule_2"]
        | out[f"{column}_rule_3"]
        | out[f"{column}_rule_4"]
    )
    return out
