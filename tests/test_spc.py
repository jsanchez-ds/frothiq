"""Tests for Statistical Process Control module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from frothiq.models.spc.shewhart import (
    ControlLimits,
    annotate_violations,
    fit_control_limits,
    western_electric_violations,
)


def test_fit_control_limits_basic():
    rng = np.random.default_rng(0)
    values = rng.normal(loc=64.0, scale=1.0, size=1000)
    limits = fit_control_limits(values)
    assert limits.center == pytest.approx(64.0, abs=0.1)
    assert limits.sigma == pytest.approx(1.0, abs=0.1)


def test_control_limits_properties():
    limits = ControlLimits(center=10.0, sigma=2.0)
    assert limits.lcl_3 == 4.0
    assert limits.ucl_3 == 16.0
    assert limits.lcl_2 == 6.0
    assert limits.ucl_2 == 14.0
    assert limits.lcl_1 == 8.0
    assert limits.ucl_1 == 12.0


def test_rule_1_detects_3sigma_violation():
    limits = ControlLimits(center=0.0, sigma=1.0)
    values = np.array([0.1, 0.2, -0.1, 5.0, 0.3])  # 5.0 > +3σ
    out = western_electric_violations(values, limits)
    assert out["rule_1"][3]
    assert not out["rule_1"][0]


def test_rule_2_detects_two_of_three_2sigma():
    limits = ControlLimits(center=0.0, sigma=1.0)
    # Two of three points beyond +2σ on the same side.
    values = np.array([0.1, 0.2, 2.5, 0.3, 2.6])
    out = western_electric_violations(values, limits)
    # At index 4: window is [0.3, 2.6] looking back 2 — actually indices 2,3,4 = [2.5, 0.3, 2.6] = 2 above.
    assert out["rule_2"][4]


def test_rule_4_detects_eight_consecutive_same_side():
    limits = ControlLimits(center=0.0, sigma=1.0)
    # Eight consecutive points above the centerline (positive).
    values = np.array([0.5] * 8 + [-0.1])
    out = western_electric_violations(values, limits)
    assert out["rule_4"][7]
    assert not out["rule_4"][8]


def test_annotate_violations_adds_columns():
    limits = ControlLimits(center=0.0, sigma=1.0)
    df = pd.DataFrame({"q": [0.0, 0.1, 5.0, 0.2, -3.5]})
    out = annotate_violations(df, "q", limits)
    expected_cols = {"q_rule_1", "q_rule_2", "q_rule_3", "q_rule_4", "q_any_violation"}
    assert expected_cols <= set(out.columns)
    assert out["q_rule_1"].iloc[2]  # 5.0 > +3σ
    assert out["q_rule_1"].iloc[4]  # -3.5 < -3σ
    assert out["q_any_violation"].iloc[2]
