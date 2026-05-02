"""Tests for what-if simulator."""

from __future__ import annotations

import pandas as pd
import pytest

from frothiq.models.whatif.simulator import (
    apply_overrides,
    simulate_whatif,
)


class _DummyModel:
    """Linear model: y = 2 * ph + 0.1 * density. For easy verification."""

    def predict(self, X):
        return (2.0 * X["ore_pulp_ph"] + 0.1 * X["ore_pulp_density"]).to_numpy()


def test_apply_overrides_basic():
    row = pd.Series({"ore_pulp_ph": 9.4, "ore_pulp_density": 1.7})
    out = apply_overrides(row, overrides={"ore_pulp_ph": 9.8})
    assert out["ore_pulp_ph"] == 9.8
    assert out["ore_pulp_density"] == 1.7  # not changed


def test_apply_overrides_propagates_to_rolling_mean():
    row = pd.Series({"ore_pulp_ph": 9.4, "ore_pulp_ph_mean_30": 9.4})
    out = apply_overrides(row, overrides={"ore_pulp_ph": 11.0})
    # Naive propagation: mean shifts toward new value with weight 1/30.
    expected_mean = (29 / 30) * 9.4 + (1 / 30) * 11.0
    assert out["ore_pulp_ph_mean_30"] == pytest.approx(expected_mean, rel=1e-6)


def test_simulate_whatif_returns_delta():
    row = pd.Series({"ore_pulp_ph": 9.4, "ore_pulp_density": 1.7})
    feature_cols = ["ore_pulp_ph", "ore_pulp_density"]
    model = _DummyModel()
    result = simulate_whatif(
        model, row, feature_cols, overrides={"ore_pulp_ph": 9.8}
    )
    # baseline = 2*9.4 + 0.1*1.7 = 18.97
    # counterfactual = 2*9.8 + 0.1*1.7 = 19.77
    # delta = 0.8
    assert result.baseline_pred == pytest.approx(18.97, rel=1e-3)
    assert result.counterfactual_pred == pytest.approx(19.77, rel=1e-3)
    assert result.delta == pytest.approx(0.8, rel=1e-3)
    assert result.overrides == {"ore_pulp_ph": 9.8}
