"""Tests for CUSUM and EWMA control charts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from frothiq.models.spc.cusum import CusumParams, annotate_cusum, cusum_chart
from frothiq.models.spc.ewma import EwmaParams, annotate_ewma, ewma_chart

# ----- CUSUM ------------------------------------------------------------------


def test_cusum_in_control_no_signal():
    rng = np.random.default_rng(0)
    values = rng.normal(loc=0.0, scale=1.0, size=500)
    params = CusumParams(target=0.0, sigma=1.0, delta_sigma=1.0, h_sigma=4.0)
    chart = cusum_chart(values, params)
    # In-control with h=4σ: signal share should be very low (~ 0%).
    assert chart["signal"].mean() < 0.05


def test_cusum_detects_upward_shift():
    # First 100 in-control, next 100 shifted up by 2σ.
    in_ctrl = np.zeros(100)
    shifted = np.full(100, 2.0)
    values = np.concatenate([in_ctrl, shifted])
    params = CusumParams(target=0.0, sigma=1.0, delta_sigma=1.0, h_sigma=4.0)
    chart = cusum_chart(values, params)
    # Signal should fire somewhere after the shift starts.
    assert chart["signal_up"][100:].any()
    assert not chart["signal_down"].any()


def test_cusum_detects_downward_shift():
    values = np.concatenate([np.zeros(100), np.full(100, -2.0)])
    params = CusumParams(target=0.0, sigma=1.0, delta_sigma=1.0, h_sigma=4.0)
    chart = cusum_chart(values, params)
    assert chart["signal_down"][100:].any()
    assert not chart["signal_up"].any()


def test_cusum_params_helpers():
    p = CusumParams(target=10.0, sigma=2.0, delta_sigma=1.0, h_sigma=4.0)
    assert p.k == pytest.approx(1.0)  # 0.5 * 1.0 * 2.0
    assert p.h == pytest.approx(8.0)  # 4.0 * 2.0


def test_annotate_cusum_appends_columns():
    df = pd.DataFrame({"q": np.zeros(20)})
    params = CusumParams(target=0.0, sigma=1.0)
    out = annotate_cusum(df, "q", params)
    expected = {"q_cu", "q_cl", "q_signal", "q_signal_up", "q_signal_down"}
    assert expected <= set(out.columns)


# ----- EWMA -------------------------------------------------------------------


def test_ewma_in_control_no_signal():
    rng = np.random.default_rng(0)
    values = rng.normal(loc=0.0, scale=1.0, size=500)
    params = EwmaParams(target=0.0, sigma=1.0, lambda_=0.2, L=3.0)
    chart = ewma_chart(values, params)
    assert chart["signal"].mean() < 0.05


def test_ewma_detects_shift():
    values = np.concatenate([np.zeros(100), np.full(100, 1.5)])
    params = EwmaParams(target=0.0, sigma=1.0, lambda_=0.2, L=3.0)
    chart = ewma_chart(values, params)
    # Signal should fire somewhere in the post-shift segment.
    assert chart["signal"][100:].any()


def test_ewma_z_starts_near_target():
    values = np.array([1.0, 1.0, 1.0])
    params = EwmaParams(target=0.0, sigma=1.0, lambda_=0.2, L=3.0)
    chart = ewma_chart(values, params)
    # z_0 = 0.2 * 1 + 0.8 * 0 = 0.2.
    assert chart["z"][0] == pytest.approx(0.2, rel=1e-6)


def test_annotate_ewma_appends_columns():
    df = pd.DataFrame({"q": np.zeros(20)})
    params = EwmaParams(target=0.0, sigma=1.0)
    out = annotate_ewma(df, "q", params)
    expected = {"q_z", "q_ucl", "q_lcl", "q_signal"}
    assert expected <= set(out.columns)
