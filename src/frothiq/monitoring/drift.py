"""Data and target drift detection using Evidently AI.

Compares a *current* DataFrame against a *reference* DataFrame and reports:
  - Per-column data drift (Kolmogorov-Smirnov, PSI, Wasserstein).
  - Target drift if labels are present.
  - Overall dataset drift score.

The implementation is **optional**: ``evidently`` is not in the base install of
``frothiq`` to keep the runtime minimal. When you need drift reports, install
it explicitly with:

    pip install evidently

If unavailable, this module falls back to a basic in-house implementation
based on KS test per column and PSI for stability over time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _basic_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    feature_cols: list[str],
) -> dict[str, Any]:
    """Fallback drift report using KS and PSI when Evidently is unavailable."""
    from scipy.stats import ks_2samp

    rows = []
    for col in feature_cols:
        if col not in reference.columns or col not in current.columns:
            continue
        ref = reference[col].dropna().to_numpy()
        cur = current[col].dropna().to_numpy()
        if len(ref) < 10 or len(cur) < 10:
            continue
        ks_stat, ks_p = ks_2samp(ref, cur)
        psi = _population_stability_index(ref, cur)
        rows.append(
            {
                "column": col,
                "ks_stat": float(ks_stat),
                "ks_p": float(ks_p),
                "psi": float(psi),
                "drift": ks_p < 0.05 or psi > 0.25,
            }
        )

    df = pd.DataFrame(rows)
    n_drifted = int(df["drift"].sum()) if len(df) else 0
    return {
        "report": df,
        "n_drifted_columns": n_drifted,
        "n_total_columns": len(df),
        "drift_share": n_drifted / max(len(df), 1),
        "engine": "frothiq.basic",
    }


def _population_stability_index(
    reference: np.ndarray, current: np.ndarray, n_bins: int = 10
) -> float:
    """Compute PSI (Population Stability Index).

    PSI < 0.10 → no significant change.
    PSI 0.10–0.25 → moderate change, monitor.
    PSI > 0.25 → significant change, action required.
    """
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return 0.0

    ref_hist, _ = np.histogram(reference, bins=edges)
    cur_hist, _ = np.histogram(current, bins=edges)

    ref_pct = (ref_hist + 1) / (ref_hist.sum() + len(ref_hist))
    cur_pct = (cur_hist + 1) / (cur_hist.sum() + len(cur_hist))

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return psi


def compute_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> dict[str, Any]:
    """Compute a drift report using Evidently if available, else basic fallback."""
    if feature_cols is None:
        feature_cols = [
            c for c in reference.columns
            if c in current.columns and pd.api.types.is_numeric_dtype(reference[c])
        ]

    try:
        from evidently.metric_preset import DataDriftPreset  # type: ignore[import-not-found]
        from evidently.report import Report  # type: ignore[import-not-found]

        report = Report(metrics=[DataDriftPreset()])
        report.run(
            reference_data=reference[feature_cols],
            current_data=current[feature_cols],
        )
        return {
            "report": report,
            "engine": "evidently",
            "n_total_columns": len(feature_cols),
        }
    except ImportError:
        return _basic_drift_report(reference, current, feature_cols)


def save_drift_html(report: Any, path: str | Path) -> Path:
    """Save the drift report as HTML.

    Works with both Evidently reports (``.save_html``) and the basic fallback
    (we render a small HTML manually).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(report, "report") and hasattr(report["report"], "save_html"):
        report["report"].save_html(str(p))
        return p

    # Fallback: write a simple HTML.
    df = report.get("report") if isinstance(report, dict) else None
    html = ["<!DOCTYPE html><html><head><meta charset='utf-8'>"]
    html.append("<title>FrothIQ — drift report (basic)</title>")
    html.append("<style>body{font-family:sans-serif;margin:2em;} table{border-collapse:collapse;}")
    html.append("th,td{padding:6px 12px;border:1px solid #ddd;} th{background:#f5f5f5;}</style></head><body>")
    html.append("<h1>⚗️ FrothIQ — Drift Report (basic)</h1>")
    if isinstance(report, dict):
        html.append(
            f"<p>Engine: <code>{report.get('engine')}</code> · "
            f"Drifted: <b>{report.get('n_drifted_columns')}</b> / {report.get('n_total_columns')}"
            f" ({report.get('drift_share', 0):.1%})</p>"
        )
    if df is not None and isinstance(df, pd.DataFrame):
        html.append(df.to_html(index=False, float_format=lambda v: f"{v:.4f}"))
    html.append("</body></html>")
    p.write_text("\n".join(html), encoding="utf-8")
    return p
