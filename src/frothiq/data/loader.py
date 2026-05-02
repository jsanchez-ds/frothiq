"""Loader for the Kaggle 'Quality Prediction in a Mining Process' dataset.

The raw CSV uses Brazilian-style decimals (commas). It contains 24 columns sampled
at varying frequencies — sensor readings every 20 seconds and lab measurements
(% Iron Concentrate, % Silica Concentrate) every hour. This module:

1. Reads the CSV with the right decimal handling.
2. Standardises column names (snake_case ASCII).
3. Returns a clean pandas DataFrame ready for downstream processing.

The hourly lab measurements are forward-filled at the 20-second resolution to align
sensor readings with their corresponding output quality measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# Canonical column names (snake_case ASCII), aligned with the original Kaggle
# columns (some have unicode percent signs and non-ASCII characters).
COLUMN_RENAMES: dict[str, str] = {
    "date": "timestamp",
    "% Iron Feed": "pct_iron_feed",
    "% Silica Feed": "pct_silica_feed",
    "Starch Flow": "starch_flow",
    "Amina Flow": "amina_flow",
    "Ore Pulp Flow": "ore_pulp_flow",
    "Ore Pulp pH": "ore_pulp_ph",
    "Ore Pulp Density": "ore_pulp_density",
    "Flotation Column 01 Air Flow": "flot_col_01_air_flow",
    "Flotation Column 02 Air Flow": "flot_col_02_air_flow",
    "Flotation Column 03 Air Flow": "flot_col_03_air_flow",
    "Flotation Column 04 Air Flow": "flot_col_04_air_flow",
    "Flotation Column 05 Air Flow": "flot_col_05_air_flow",
    "Flotation Column 06 Air Flow": "flot_col_06_air_flow",
    "Flotation Column 07 Air Flow": "flot_col_07_air_flow",
    "Flotation Column 01 Level": "flot_col_01_level",
    "Flotation Column 02 Level": "flot_col_02_level",
    "Flotation Column 03 Level": "flot_col_03_level",
    "Flotation Column 04 Level": "flot_col_04_level",
    "Flotation Column 05 Level": "flot_col_05_level",
    "Flotation Column 06 Level": "flot_col_06_level",
    "Flotation Column 07 Level": "flot_col_07_level",
    "% Iron Concentrate": "pct_iron_concentrate",
    "% Silica Concentrate": "pct_silica_concentrate",
}


SENSOR_COLS: list[str] = [
    "starch_flow",
    "amina_flow",
    "ore_pulp_flow",
    "ore_pulp_ph",
    "ore_pulp_density",
] + [f"flot_col_{i:02d}_air_flow" for i in range(1, 8)] + [
    f"flot_col_{i:02d}_level" for i in range(1, 8)
]

FEED_COLS: list[str] = ["pct_iron_feed", "pct_silica_feed"]
TARGET_COLS: list[str] = ["pct_iron_concentrate", "pct_silica_concentrate"]


@dataclass
class FlotationData:
    """Container for the loaded flotation dataset."""

    df: pd.DataFrame
    sensor_cols: list[str]
    feed_cols: list[str]
    target_cols: list[str]


def _resolve_root(root: str | Path) -> Path:
    """Walk up from cwd looking for the data folder. Lets notebooks under
    ``notebooks/`` and scripts under ``scripts/`` use the same default path.
    """
    p = Path(root)
    if p.is_absolute() and p.exists():
        return p
    cwd = Path.cwd().resolve()
    for parent in [cwd, *cwd.parents]:
        candidate = parent / p
        if candidate.exists():
            return candidate
    return p


def load_flotation(
    path: str | Path = "data/raw/flotation/MiningProcess_Flotation_Plant_Database.csv",
    nrows: int | None = None,
) -> FlotationData:
    """Load the Kaggle flotation dataset and return a tidy DataFrame.

    Parameters
    ----------
    path : str or Path
        Location of the Kaggle CSV.
    nrows : int, optional
        Limit number of rows (useful for tests / quick experiments).
    """
    p = _resolve_root(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Dataset not found at {p}. Run `bash scripts/download_data.sh` first."
        )

    # Brazilian-style numbers use comma as decimal separator.
    df = pd.read_csv(p, decimal=",", nrows=nrows)
    df = df.rename(columns=COLUMN_RENAMES)

    # Parse timestamp and sort.
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Cast all sensor / feed / target columns to float64 (defensive).
    numeric_cols = SENSOR_COLS + FEED_COLS + TARGET_COLS
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return FlotationData(
        df=df,
        sensor_cols=SENSOR_COLS,
        feed_cols=FEED_COLS,
        target_cols=TARGET_COLS,
    )


def temporal_split(
    df: pd.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    timestamp_col: str = "timestamp",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split chronologically into (train, val, test) — no shuffling.

    Returns three DataFrames with the same columns as the input. Cutoffs are
    computed by row count after sorting by timestamp ascending.
    """
    if not 0 < train_frac < 1 or not 0 < val_frac < 1:
        raise ValueError("Fractions must be in (0, 1).")
    if train_frac + val_frac >= 1:
        raise ValueError("train_frac + val_frac must be < 1 (test gets the rest).")

    df_sorted = df.sort_values(timestamp_col).reset_index(drop=True)
    n = len(df_sorted)
    train_end = int(round(n * train_frac))
    val_end = int(round(n * (train_frac + val_frac)))

    train = df_sorted.iloc[:train_end].copy()
    val = df_sorted.iloc[train_end:val_end].copy()
    test = df_sorted.iloc[val_end:].copy()
    return train, val, test


def detect_constant_lab_measurements(
    df: pd.DataFrame,
    target_cols: list[str] = TARGET_COLS,
    timestamp_col: str = "timestamp",
) -> pd.Series:
    """Lab measurements (target) are sampled hourly but forward-filled to 20-second
    resolution. This helper returns a boolean series flagging the rows where the
    lab measurement was just forward-filled (still equals the previous value).

    Useful for sample weighting or for restricting training to "fresh" lab readings.
    """
    df_sorted = df.sort_values(timestamp_col).reset_index(drop=True)
    is_fresh = pd.Series(False, index=df_sorted.index)
    for col in target_cols:
        if col in df_sorted.columns:
            is_fresh = is_fresh | (df_sorted[col].diff().abs() > 1e-9)
    is_fresh.iloc[0] = True  # first row counts as fresh.
    return is_fresh
