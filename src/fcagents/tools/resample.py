"""Deterministic tools for converting an irregular series onto a fixed grid between 1 hour and 1 week."""

from __future__ import annotations

import pandas as pd
from pandas.tseries.frequencies import to_offset

FREQ_OPTIONS = {  # ordered fine → coarse
    "h": "hourly",
    "2h": "every 2 hours",
    "3h": "every 3 hours",
    "4h": "every 4 hours",
    "6h": "every 6 hours",
    "12h": "every 12 hours",
    "D": "daily",
    "2D": "every 2 days",
    "3D": "every 3 days",
    "W": "weekly",
}
AGGREGATIONS = ["mean", "sum", "median", "min", "max", "first", "last", "count"]
FILL_METHODS = {"none": "leave missing", "ffill": "forward fill", "interpolate": "interpolate", "zero": "zero"}

MIN_FREQ = pd.Timedelta(hours=1)
MAX_FREQ = pd.Timedelta(weeks=1)
MIN_POINTS = 30
MAX_EMPTY_SHARE_RECOMMEND = 0.2
SPARSE_EMPTY_SHARE = 0.5


def freq_span(alias: str) -> pd.Timedelta:
    off = to_offset(alias)
    if isinstance(off, pd.offsets.Week):
        return pd.Timedelta(weeks=off.n)
    if isinstance(off, pd.offsets.Day):
        return pd.Timedelta(days=off.n)
    try:
        return pd.Timedelta(off)
    except ValueError as e:  # calendar offsets such as "MS" have no fixed length
        raise ValueError(f"'{alias}' is not a fixed-length frequency") from e


def validate_freq(alias: str) -> None:
    span = freq_span(alias)
    if not MIN_FREQ <= span <= MAX_FREQ:
        raise ValueError(f"Frequency '{alias}' ({span}) is outside the allowed range 1h–1W")


def _bins(series: pd.Series, freq: str):
    return series.resample(freq, label="left", closed="left")


def _effective_agg(series: pd.Series, agg: str) -> str:
    # Timestamps without values (event logs) can only be counted.
    return "count" if series.isna().all() else agg


def resample_options(series: pd.Series) -> list[dict]:
    """Points per bin size for every allowed frequency, so the user can see the trade-off."""
    idx = series.index.dropna()
    counts_src = pd.Series(1, index=idx)
    rows = []
    for freq, label in FREQ_OPTIONS.items():
        counts = _bins(counts_src, freq).count()
        n_bins, n_points = len(counts), int((counts > 0).sum())
        rows.append({
            "freq": freq,
            "label": label,
            "n_bins": n_bins,
            "n_points": n_points,
            "empty_share": round(1 - n_points / n_bins, 4) if n_bins else 1.0,
            "enough_points": n_points >= MIN_POINTS,
        })
    return rows


def recommend_freq(options: list[dict]) -> str:
    """Finest frequency with enough points and few empty bins; degrade gracefully otherwise."""
    enough = [o for o in options if o["enough_points"]]
    dense = [o for o in enough if o["empty_share"] <= MAX_EMPTY_SHARE_RECOMMEND]
    if dense:
        return dense[0]["freq"]
    if enough:
        return enough[0]["freq"]
    return options[0]["freq"]


def resample_series(series: pd.Series, freq: str, agg: str = "mean", fill: str = "none") -> tuple[pd.Series, dict]:
    validate_freq(freq)
    if agg not in AGGREGATIONS:
        raise ValueError(f"Unknown aggregation '{agg}'")
    if fill not in FILL_METHODS:
        raise ValueError(f"Unknown fill method '{fill}'")

    s = series[series.index.notna()].sort_index()
    agg = _effective_agg(s, agg)
    counts = _bins(pd.Series(1, index=s.index), freq).count()
    has_data = counts > 0
    out = counts.astype(float) if agg == "count" else _bins(s, freq).agg(agg)

    n_bins, n_points = len(out), int(has_data.sum())
    if fill == "none":
        out = out[has_data]
    elif fill == "ffill":
        out = out.where(has_data).ffill()
    elif fill == "interpolate":
        out = out.where(has_data).interpolate(method="time")
    elif fill == "zero":
        out = out.where(has_data).fillna(0.0)
    out.name = series.name

    return out, {
        "freq": freq,
        "label": FREQ_OPTIONS.get(freq, freq),
        "agg": agg,
        "fill": fill,
        "n_input": int(len(s)),
        "n_bins": n_bins,
        "n_points": n_points,
        "n_empty": n_bins - n_points,
        "empty_share": round(1 - n_points / n_bins, 4) if n_bins else 1.0,
        "start": out.index[0].isoformat() if len(out) else None,
        "end": out.index[-1].isoformat() if len(out) else None,
    }
