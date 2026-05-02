"""Tests for the flotation data loader.

Doesn't require the actual Kaggle dataset — exercises path resolution and
splitting logic against synthetic fixtures so CI passes without 125 MB downloads.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from frothiq.data.loader import (
    COLUMN_RENAMES,
    SENSOR_COLS,
    TARGET_COLS,
    detect_constant_lab_measurements,
    load_flotation,
    temporal_split,
)


def _write_synthetic_csv(path: Path, n: int = 100) -> None:
    """Write a tiny synthetic Kaggle-shaped CSV."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        ts = pd.Timestamp("2017-03-10") + pd.Timedelta(seconds=20 * i)
        row = {"date": ts.strftime("%Y-%m-%d %H:%M:%S")}
        for orig_col, _new_col in COLUMN_RENAMES.items():
            if orig_col == "date":
                continue
            row[orig_col] = float(rng.normal(50, 5))
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, decimal=",")


def test_load_flotation_basic(tmp_path):
    p = tmp_path / "synthetic.csv"
    _write_synthetic_csv(p, n=50)
    result = load_flotation(path=p)
    assert len(result.df) == 50
    assert "timestamp" in result.df.columns
    assert all(c in result.df.columns for c in SENSOR_COLS)
    assert all(c in result.df.columns for c in TARGET_COLS)


def test_load_flotation_with_nrows(tmp_path):
    p = tmp_path / "synthetic.csv"
    _write_synthetic_csv(p, n=100)
    result = load_flotation(path=p, nrows=20)
    assert len(result.df) == 20


def test_load_flotation_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_flotation(path="/nonexistent/path.csv")


def test_temporal_split_no_overlap():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2017-01-01", periods=100, freq="20s"),
            "value": np.arange(100),
        }
    )
    train, val, test = temporal_split(df, train_frac=0.7, val_frac=0.15)
    assert len(train) + len(val) + len(test) == 100
    # Train should come strictly before val, val before test.
    assert train["timestamp"].max() < val["timestamp"].min()
    assert val["timestamp"].max() < test["timestamp"].min()


def test_temporal_split_invalid_fractions():
    df = pd.DataFrame(
        {"timestamp": pd.date_range("2017-01-01", periods=10, freq="20s")}
    )
    with pytest.raises(ValueError):
        temporal_split(df, train_frac=0.6, val_frac=0.5)


def test_detect_constant_lab_measurements():
    # Lab measurement repeated for 10 cycles, then changes, then repeated 5.
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2017-01-01", periods=20, freq="20s"),
            "pct_iron_concentrate": [60.0] * 10 + [62.0] * 5 + [63.0] * 5,
            "pct_silica_concentrate": [10.0] * 20,
        }
    )
    fresh = detect_constant_lab_measurements(df)
    # First row is fresh by convention. Then changes at indices 10 and 15.
    assert fresh.iloc[0]
    assert fresh.iloc[10]
    assert fresh.iloc[15]
    # Indices 1-9 are forward-fills (not fresh).
    assert not fresh.iloc[5]
