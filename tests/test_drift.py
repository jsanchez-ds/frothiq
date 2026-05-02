"""Tests for drift monitoring."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from frothiq.monitoring.drift import (
    _basic_drift_report,
    _population_stability_index,
    compute_drift_report,
    save_drift_html,
)


def test_psi_zero_when_distributions_identical():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, size=2000)
    psi = _population_stability_index(a, a)
    assert abs(psi) < 0.05  # identical distributions ⇒ ~0


def test_psi_nonzero_when_distributions_differ():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, size=2000)
    b = rng.normal(2, 1, size=2000)  # mean shifted
    psi = _population_stability_index(a, b)
    assert psi > 0.5  # very different ⇒ large PSI


def test_basic_drift_report_no_drift():
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({"x": rng.normal(0, 1, 500), "y": rng.normal(0, 1, 500)})
    cur = pd.DataFrame({"x": rng.normal(0, 1, 500), "y": rng.normal(0, 1, 500)})
    report = _basic_drift_report(ref, cur, ["x", "y"])
    # Same distributions: no column should drift.
    assert report["n_drifted_columns"] == 0


def test_basic_drift_report_detects_shift():
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({"x": rng.normal(0, 1, 500), "y": rng.normal(0, 1, 500)})
    cur = pd.DataFrame({"x": rng.normal(3, 1, 500), "y": rng.normal(0, 1, 500)})  # x shifted
    report = _basic_drift_report(ref, cur, ["x", "y"])
    assert report["n_drifted_columns"] >= 1


def test_compute_drift_report_returns_dict():
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({"a": rng.normal(0, 1, 500)})
    cur = pd.DataFrame({"a": rng.normal(0, 1, 500)})
    report = compute_drift_report(ref, cur, feature_cols=["a"])
    assert "engine" in report
    # Either 'evidently' or 'frothiq.basic' depending on whether evidently is installed.
    assert report["engine"] in ("evidently", "frothiq.basic")


def test_save_drift_html_basic_engine(tmp_path: Path):
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({"x": rng.normal(0, 1, 500)})
    cur = pd.DataFrame({"x": rng.normal(2, 1, 500)})
    report = _basic_drift_report(ref, cur, ["x"])
    out = save_drift_html(report, tmp_path / "report.html")
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert "FrothIQ" in content
