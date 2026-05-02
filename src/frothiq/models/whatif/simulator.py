"""What-if simulator for flotation operators.

Lets an operator ask: "what happens to predicted % iron concentrate if I change
pH from 9.4 to 9.8?". Implementation:

1. Take the current sensor state (latest cycle of features).
2. Override one or more sensor values with the operator's hypothetical change.
3. Re-compute the dependent rolling features by replaying the change forward.
4. Run the model on the modified state.
5. Return the delta vs the current prediction.

This is a counterfactual prediction — useful for control room decisions before
committing a parameter change in the plant.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class WhatIfResult:
    baseline_pred: float
    counterfactual_pred: float
    delta: float
    overrides: dict[str, float]


def apply_overrides(
    feature_row: pd.Series,
    overrides: dict[str, float],
) -> pd.Series:
    """Apply a dict of {feature_name: new_value} to a feature row.

    Also updates derived rolling features (mean / std) of the overridden columns
    in a *naive* way: substitutes the new value for the latest reading and
    nudges the rolling means proportionally. This is an approximation; the exact
    way is to re-run the rolling pipeline on the recent window with the
    override applied.
    """
    out = feature_row.copy()
    for col, value in overrides.items():
        if col in out.index:
            out[col] = value
            # Naive propagation to derived rolling features.
            for derived_suffix in ("_mean_30", "_mean_180", "_mean_540"):
                derived = f"{col}{derived_suffix}"
                if derived in out.index:
                    # Move the mean a bit toward the new value (1 / window_size weight).
                    window_size = int(derived_suffix.split("_")[-1])
                    weight = 1.0 / window_size
                    out[derived] = (1 - weight) * out[derived] + weight * value
    return out


def simulate_whatif(
    model,
    feature_row: pd.Series,
    feature_cols: list[str],
    overrides: dict[str, float],
) -> WhatIfResult:
    """Predict baseline + counterfactual and return the delta."""
    X_base = feature_row[feature_cols].to_frame().T
    baseline_pred = float(model.predict(X_base)[0])

    modified = apply_overrides(feature_row, overrides)
    X_cf = modified[feature_cols].to_frame().T
    counterfactual_pred = float(model.predict(X_cf)[0])

    return WhatIfResult(
        baseline_pred=baseline_pred,
        counterfactual_pred=counterfactual_pred,
        delta=counterfactual_pred - baseline_pred,
        overrides=overrides,
    )
