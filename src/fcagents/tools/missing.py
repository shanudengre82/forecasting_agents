"""Deterministic tools for finding and filling missing values on a fixed-frequency grid."""

from __future__ import annotations

import numpy as np
import pandas as pd

FILL_METHODS = {
    "none": "leave missing",
    "ffill": "forward fill",
    "bfill": "backward fill",
    "interpolate": "interpolate in time",
    "zero": "zero",
    "seasonal": "seasonal naive (value one season earlier)",
}
LONG_GAP_STEPS = 24  # used when no seasonal period is known
MAX_MISSING_SHARE = 0.3


def full_grid(series: pd.Series, freq: str) -> tuple[pd.Series, dict]:
    """Put ``series`` on its complete grid: duplicate timestamps are averaged, missing timestamps become NaN.

    Timestamps off the grid (at most a few percent for a series judged regular) are dropped and counted.
    """
    s = series[series.index.notna()].sort_index()
    n_dup = int(s.index.duplicated().sum())
    if n_dup:
        s = s.groupby(level=0).mean()
    grid = pd.date_range(s.index[0], s.index[-1], freq=freq)
    out = s.reindex(grid)
    out.name = series.name
    return out, {"n_duplicates_merged": n_dup, "n_off_grid_dropped": int((~s.index.isin(grid)).sum())}


def _runs(mask: pd.Series, max_runs: int = 5) -> list[dict]:
    if not mask.any():
        return []
    pos = np.flatnonzero(mask.to_numpy())
    run_ids = np.cumsum(np.r_[1, np.diff(pos) != 1])
    stamps = pd.Series(mask.index[pos], index=run_ids)
    runs = stamps.groupby(level=0).agg(["first", "last", "size"]).sort_values("size", ascending=False)
    return [{"from": r["first"].isoformat(), "to": r["last"].isoformat(), "length": int(r["size"])}
            for _, r in runs.head(max_runs).iterrows()]


def missing_report(series: pd.Series, freq: str) -> dict:
    grid, cleaning = full_grid(series, freq)
    observed_stamps = series.index.dropna().unique()
    missing = grid.isna()
    n_missing_ts = int((~grid.index.isin(observed_stamps)).sum())
    runs = _runs(missing)
    valid = grid.notna().to_numpy()
    lead = int(np.argmax(valid)) if valid.any() else len(grid)
    trail = int(np.argmax(valid[::-1])) if valid.any() else len(grid)
    return {
        "freq": freq,
        "n_expected": int(len(grid)),
        "n_observed": int(valid.sum()),
        "n_missing_timestamps": n_missing_ts,
        "n_nan_values": int(missing.sum()) - n_missing_ts,
        "n_missing": int(missing.sum()),
        "missing_share": round(float(missing.mean()), 4) if len(grid) else 0.0,
        "longest_run": runs[0]["length"] if runs else 0,
        "gaps": runs,
        "n_leading_missing": lead,
        "n_trailing_missing": trail,
        **cleaning,
    }


def pick_seasonal_period(periods: list[int] | None, n: int) -> int | None:
    """Shortest candidate period with at least two full seasons of data."""
    usable = [p for p in periods or [] if n >= 2 * p]
    return min(usable) if usable else None


def recommend_fill(report: dict, agg: str | None, seasonal_period: int | None) -> str:
    if not report["n_missing"]:
        return "none"
    if agg in ("count", "sum"):
        return "zero"  # an empty bin means no events, not an unknown value
    if seasonal_period and report["longest_run"] > 1:
        return "seasonal"
    return "interpolate"


def long_gap_threshold(seasonal_period: int | None) -> int:
    return seasonal_period or LONG_GAP_STEPS


def _seasonal(s: pd.Series, period: int) -> pd.Series:
    vals = s.to_numpy(dtype=float, copy=True)
    for i in np.flatnonzero(np.isnan(vals)):  # ascending, so filled values feed later gaps
        if i >= period and not np.isnan(vals[i - period]):
            vals[i] = vals[i - period]
    return pd.Series(vals, index=s.index, name=s.name).interpolate(method="time", limit_direction="both")


def fill_missing(series: pd.Series, freq: str, method: str = "interpolate",
                 seasonal_period: int | None = None) -> tuple[pd.Series, dict]:
    if method not in FILL_METHODS:
        raise ValueError(f"Unknown fill method '{method}'")
    grid = full_grid(series, freq)[0].astype(float)
    if method == "none":
        out = grid
    elif method == "ffill":
        out = grid.ffill()
    elif method == "bfill":
        out = grid.bfill()
    elif method == "interpolate":
        out = grid.interpolate(method="time", limit_direction="both")
    elif method == "zero":
        out = grid.fillna(0.0)
    else:
        if not seasonal_period:
            raise ValueError("Seasonal filling needs a seasonal period")
        out = _seasonal(grid, seasonal_period)
    filled = grid.isna() & out.notna()
    return out, {
        "method": method,
        "method_label": FILL_METHODS[method],
        "seasonal_period": seasonal_period if method == "seasonal" else None,
        "n_filled": int(filled.sum()),
        "n_still_missing": int(out.isna().sum()),
        "n_points": int(len(out)),
    }
