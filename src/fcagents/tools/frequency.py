"""Deterministic tools for checking time-index regularity and naming its frequency.

Everything returned is JSON-serialisable so it can be handed to an LLM as facts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
from pandas.tseries import offsets

# A grid is accepted only if (almost) every observed timestamp lies on it ...
MIN_ON_GRID = 0.98
# ... and the observations fill a reasonable share of it (otherwise it's a sparse/irregular series).
MIN_COVERAGE = 0.6

_MONTH_ABBR = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
_DOW_ABBR = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]


def _clean_index(index) -> tuple[pd.DatetimeIndex, dict]:
    # pandas 3 may default to "us"/"s" resolution; normalise so asi8 is always nanoseconds.
    idx = pd.DatetimeIndex(index).as_unit("ns")
    n_nat = int(idx.isna().sum())
    idx = idx.dropna()
    was_sorted = bool(idx.is_monotonic_increasing)
    idx = idx.sort_values()
    n_dup = int(idx.duplicated().sum())
    idx = idx.unique()
    return idx, {"n_raw": len(index), "n_nat": n_nat, "was_sorted": was_sorted, "n_duplicates": n_dup}


def _tick_offset(delta: pd.Timedelta):
    # pandas 3 maps Timedelta("1D") to "24h"; keep whole days as "D" so aliases read naturally.
    day = pd.Timedelta(days=1)
    if delta >= day and delta % day == pd.Timedelta(0):
        n = delta // day
        return to_offset("D" if n == 1 else f"{n}D")
    try:
        return to_offset(delta)
    except ValueError:
        return None


def _candidate_aliases(idx: pd.DatetimeIndex, deltas: pd.TimedeltaIndex) -> list[str]:
    cands: list[str] = []
    strict = pd.infer_freq(idx)
    if strict:
        cands.append(strict)

    mode_delta = deltas.value_counts().idxmax()
    gcd_ns = int(np.gcd.reduce(deltas.asi8))
    for d in (mode_delta, pd.Timedelta(gcd_ns, unit="ns")):
        off = _tick_offset(d)
        if off is not None:
            cands.append(off.freqstr)

    first = idx[0]
    month = _MONTH_ABBR[first.month - 1]
    cands += [
        "B",
        f"W-{_DOW_ABBR[first.dayofweek]}",
        "MS",
        "ME",
        f"QS-{month}",
        f"QE-{month}",
        f"YS-{month}",
        f"YE-{month}",
    ]
    return list(dict.fromkeys(cands))  # dedupe, keep order


def _fit_grid(idx: pd.DatetimeIndex, alias: str) -> dict | None:
    try:
        grid = pd.date_range(idx[0], idx[-1], freq=alias)
    except (ValueError, OverflowError):
        return None
    if len(grid) < 2 or len(grid) > 50 * len(idx):
        return None
    on_grid = float(idx.isin(grid).mean())
    coverage = float(min(1.0, len(idx) / len(grid)))
    return {"alias": alias, "on_grid": on_grid, "coverage": coverage, "grid_len": len(grid)}


def _largest_gap(idx: pd.DatetimeIndex, deltas: pd.TimedeltaIndex) -> dict:
    i = int(np.argmax(deltas.asi8))
    return {"from": idx[i].isoformat(), "to": idx[i + 1].isoformat(), "duration": str(deltas[i])}


def analyze_regularity(index) -> dict:
    """Decide whether a time index has a fixed frequency.

    status: regular | regular_with_gaps | irregular | insufficient_data
    """
    idx, facts = _clean_index(index)
    facts["n_unique"] = len(idx)
    if len(idx) < 3:
        return {**facts, "status": "insufficient_data", "freq": None, "coverage": None,
                "on_grid": None, "n_missing": None}

    deltas = pd.TimedeltaIndex(np.diff(idx.asi8))
    vc = deltas.value_counts()
    facts.update(
        {
            "start": idx[0].isoformat(),
            "end": idx[-1].isoformat(),
            "delta_min": str(deltas.min()),
            "delta_median": str(deltas.median()),
            "delta_max": str(deltas.max()),
            "mode_delta": str(vc.index[0]),
            "mode_delta_share": round(float(vc.iloc[0] / len(deltas)), 4),
            "n_distinct_deltas": int(len(vc)),
            "largest_gap": _largest_gap(idx, deltas),
        }
    )

    fits = [f for a in _candidate_aliases(idx, deltas) if (f := _fit_grid(idx, a))]
    accepted = [f for f in fits if f["on_grid"] >= MIN_ON_GRID and f["coverage"] >= MIN_COVERAGE]
    if not accepted:
        best = max(fits, key=lambda f: (f["on_grid"] * f["coverage"]), default=None)
        return {**facts, "status": "irregular", "freq": None,
                "coverage": round(best["coverage"], 4) if best else None,
                "on_grid": round(best["on_grid"], 4) if best else None,
                "n_missing": None, "closest_grid": best["alias"] if best else None}

    # Coarsest grid that still contains the data (e.g. prefer "B" over "D" for business-day data).
    best = min(accepted, key=lambda f: (f["grid_len"], -f["on_grid"]))
    regular = best["on_grid"] == 1.0 and best["coverage"] == 1.0
    return {
        **facts,
        "status": "regular" if regular else "regular_with_gaps",
        "freq": best["alias"],
        "coverage": round(best["coverage"], 4),
        "on_grid": round(best["on_grid"], 4),
        "n_missing": int(best["grid_len"] - round(best["on_grid"] * len(idx))),
    }


_CALENDAR = [
    # (offset type, singular label, plural unit, seasonal periods for n == 1)
    (offsets.BusinessDay, "business daily", "business days", [5, 260]),
    (offsets.Week, "weekly", "weeks", [52]),
    ((offsets.MonthBegin, offsets.MonthEnd), "monthly", "months", [12]),
    ((offsets.QuarterBegin, offsets.QuarterEnd), "quarterly", "quarters", [4]),
    ((offsets.YearBegin, offsets.YearEnd), "yearly", "years", []),
]
_TICK_UNITS = [  # (Timedelta, singular label, unit word)
    (pd.Timedelta(days=1), "daily", "days"),
    (pd.Timedelta(hours=1), "hourly", "hours"),
    (pd.Timedelta(minutes=1), "every minute", "minutes"),
    (pd.Timedelta(seconds=1), "every second", "seconds"),
]
_SEASON_SPANS = [pd.Timedelta(hours=1), pd.Timedelta(days=1), pd.Timedelta(weeks=1), pd.Timedelta(days=365)]
MAX_SEASONAL_PERIOD = 1000


def _label_and_periods(alias: str) -> tuple[str, list[int]]:
    off = to_offset(alias)
    for types, singular, unit, periods in _CALENDAR:
        if isinstance(off, types):
            if off.n == 1:
                return singular, periods
            return f"every {off.n} {unit}", [p // off.n for p in periods if p % off.n == 0 and p // off.n >= 2]

    step = pd.Timedelta(days=off.n) if isinstance(off, offsets.Day) else pd.Timedelta(off)
    label = f"every {step}"
    for unit_td, singular, unit in _TICK_UNITS:
        if step % unit_td == pd.Timedelta(0):
            n = step // unit_td
            label = singular if n == 1 else f"every {n} {unit}"
            break
    periods = []
    for span in _SEASON_SPANS:
        if span % step == pd.Timedelta(0):
            p = span // step
            if 2 <= p <= MAX_SEASONAL_PERIOD:
                periods.append(int(p))
    return label, periods


def describe_frequency(index, alias: str, max_gaps: int = 5) -> dict:
    """Describe a known frequency: human label, seasonal periods, missing timestamps and gap runs."""
    idx, _ = _clean_index(index)
    grid = pd.date_range(idx[0], idx[-1], freq=alias)
    missing = grid.difference(idx)
    label, periods = _label_and_periods(alias)

    gaps = []
    if len(missing):
        pos = grid.get_indexer(missing)
        run_ids = np.cumsum(np.r_[1, np.diff(pos) != 1])
        runs = pd.Series(missing, index=run_ids).groupby(level=0).agg(["first", "last", "size"])
        runs = runs.sort_values("size", ascending=False).head(max_gaps)
        gaps = [{"from": r["first"].isoformat(), "to": r["last"].isoformat(), "length": int(r["size"])}
                for _, r in runs.iterrows()]

    return {
        "freq": alias,
        "label": label,
        "seasonal_periods": periods,
        "expected_points": len(grid),
        "observed_points": len(idx),
        "n_missing": len(missing),
        "coverage": round(len(idx) / len(grid), 4) if len(grid) else None,
        "gaps": gaps,
        "missing_sample": [t.isoformat() for t in missing[:20]],
    }


__all__ = ["analyze_regularity", "describe_frequency", "MIN_ON_GRID", "MIN_COVERAGE"]
