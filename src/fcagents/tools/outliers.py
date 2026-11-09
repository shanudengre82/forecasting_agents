"""Robust outlier detection on a fixed-frequency series, and the user's keep / remove / replace choices."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fcagents.tools.missing import full_grid

THRESHOLDS = {3.0: "High (|z| > 3)", 3.5: "Normal (|z| > 3.5)", 5.0: "Low (|z| > 5)"}
DEFAULT_THRESHOLD = 3.5
SUGGEST_REPLACE_Z = 5.0  # this far out is almost surely an error, so the rules suggest replacing it
ACTIONS = {"keep": "Keep", "remove": "Remove", "replace": "Replace"}
MIN_POINTS = 14


def _extend_edges(trend: pd.Series, span: int) -> pd.Series:
    """Fill the NaN ends of a centred trend by a straight line through its first / last ``span`` known points."""
    vals = trend.to_numpy(dtype=float).copy()
    pos = np.arange(len(vals))
    known = np.flatnonzero(~np.isnan(vals))
    if len(known) < 2:
        return trend
    for part, edge in ((known[:span], pos < known[0]), (known[-span:], pos > known[-1])):
        if len(part) >= 2 and edge.any():
            slope, icept = np.polyfit(part, vals[part], 1)
            vals[edge] = icept + slope * pos[edge]
    return pd.Series(vals, index=trend.index)


def _moving_average(s: pd.Series, period: int) -> pd.Series:
    """Centred moving average over exactly one season (2×MA for even periods): removes the seasonal pattern and
    keeps a linear trend unchanged."""
    if period % 2:
        return s.rolling(period, center=True).mean()
    return s.rolling(period).mean().rolling(2).mean().shift(-(period // 2))


def _decompose(s: pd.Series, period: int | None) -> pd.Series:
    if not period:
        return _extend_edges(s.rolling(7, center=True).median(), 7)
    trend = _extend_edges(_moving_average(s, period), period)
    phase = pd.Series(np.arange(len(s)) % period, index=s.index)
    seasonal = (s - trend).groupby(phase).median()
    seasonal -= seasonal.mean()
    return trend + phase.map(seasonal)


def expected_values(series: pd.Series, freq: str, period: int | None, threshold: float = DEFAULT_THRESHOLD,
                    protected: pd.DatetimeIndex | None = None, passes: int = 3) -> pd.Series:
    """Trend + seasonal profile on the full grid. Gaps are interpolated for the fit only; holiday-affected points
    are set to their expected value on every pass. Far-off points are swapped for their expected value and the
    fit is repeated, first with a loose cut (2× the threshold) so a spike's neighbours aren't swapped too, so
    neither outliers nor holidays bend it."""
    s, _ = full_grid(series, freq)
    s = s.astype(float)
    if period and len(s) < 3 * period:
        period = None
    hold = s.index.isin(protected) if protected is not None and len(protected) else np.zeros(len(s), bool)
    work = s.interpolate(limit_direction="both")
    expected = _decompose(work, period)
    for i in range(passes):
        work = work.where(~hold, expected)  # holiday-affected points follow the fit instead of bending it
        resid = work - expected
        scale = 1.4826 * float((resid - resid.median()).abs().median()) or float(resid.std() or 1.0)
        work = work.where((resid / scale).abs() <= (2 * threshold if i == 0 else threshold), expected)
        expected = _decompose(work, period)
    return expected


def detect_outliers(series: pd.Series, freq: str, period: int | None, threshold: float = DEFAULT_THRESHOLD,
                    protected: pd.DatetimeIndex | None = None) -> tuple[pd.DataFrame, pd.Series, dict]:
    """(table of flagged points, expected values for the chart, facts).

    Points whose robust z-score exceeds ``threshold`` are flagged, except ``protected`` timestamps (days the
    Holidays agent found to be moved by a holiday); those are counted but never flagged.
    """
    s, _ = full_grid(series, freq)
    expected = expected_values(series, freq, period, threshold, protected)
    resid = s - expected
    valid = resid.dropna()
    scale = 1.4826 * float((valid - valid.median()).abs().median()) if len(valid) else 0.0
    scale = scale or float(valid.std() or 1.0)
    z = resid / scale
    over = z.abs() > threshold
    is_protected = s.index.isin(protected) if protected is not None and len(protected) else np.zeros(len(s), bool)
    flagged = over & ~is_protected
    table = pd.DataFrame({
        "value": s[flagged], "expected": expected[flagged].round(3), "z": z[flagged].round(2),
    })
    table["suggestion"] = np.where(table["z"].abs() >= SUGGEST_REPLACE_Z, "replace", "keep")
    table.index.name = "timestamp"
    facts = {
        "threshold": threshold, "seasonal_period": period, "scale": round(scale, 4),
        "n_checked": int(z.notna().sum()), "n_flagged": int(flagged.sum()),
        "n_protected_skipped": int((over & is_protected).sum()),
        "n_suggest_replace": int((table["suggestion"] == "replace").sum()),
        "largest_z": float(z[flagged].abs().max()) if flagged.any() else 0.0,
        "flagged": [{"timestamp": t.isoformat(), "z": float(r["z"])} for t, r in table.head(20).iterrows()],
    }
    return table, expected, facts


def apply_outlier_actions(series: pd.Series, table: pd.DataFrame,
                          actions: tuple[tuple[str, str, float | None], ...]) -> tuple[pd.Series, dict]:
    """Apply the user's choices. ``actions``: (timestamp ISO, keep | remove | replace, value or None).

    remove → the value becomes missing (the Missing Values agent fills it); replace → the given value, or the
    expected value when none is given. Choices for timestamps that are no longer flagged are ignored.
    """
    out = series.copy()
    counts = {"keep": 0, "remove": 0, "replace": 0}
    for ts_iso, action, value in actions:
        ts = pd.Timestamp(ts_iso)
        if ts not in table.index or action not in counts:
            continue
        counts[action] += 1
        if action == "keep":
            continue
        new = np.nan if action == "remove" else float(table.at[ts, "expected"] if value is None else value)
        if ts in out.index:
            out.loc[ts] = new
        elif action == "replace":
            out.loc[ts] = new
            out = out.sort_index()
    handled = sum(counts.values())
    return out, {"n_kept": counts["keep"], "n_removed": counts["remove"], "n_replaced": counts["replace"],
                 "n_undecided": int(len(table)) - handled}


def suggested_actions(table: pd.DataFrame) -> tuple[tuple[str, str, None], ...]:
    return tuple((t.isoformat(), str(r["suggestion"]), None) for t, r in table.iterrows())


def bulk_actions(table: pd.DataFrame, action: str) -> tuple[tuple[str, str, None], ...]:
    """Same choice for every flagged point; ``suggested`` uses each point's rule suggestion."""
    if action == "suggested":
        return suggested_actions(table)
    return tuple((t.isoformat(), action, None) for t in table.index)
