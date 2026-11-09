"""Deterministic tools for choosing what to forecast: one series or several."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

TARGET_HINTS = re.compile(
    r"(^|_)(y|target|label|sales|demand|load|load_mw|value|qty|quantity|volume|count|revenue|close|units|orders|"
    r"passengers|reading|consumption|traffic|visits|visitors|kwh|mw)($|_)", re.I)
MAX_SERIES = 50


def numeric_columns(df: pd.DataFrame, exclude: set[str]) -> list[str]:
    return [str(c) for c in df.columns if str(c) not in exclude and pd.api.types.is_numeric_dtype(df[c])
            and not pd.api.types.is_bool_dtype(df[c])]


def target_candidates(df: pd.DataFrame, exclude: set[str]) -> list[dict]:
    out = []
    for c in numeric_columns(df, exclude):
        out.append({"column": c, "name_hint": bool(TARGET_HINTS.search(c)),
                    "missing_share": round(float(df[c].isna().mean()), 4), "n_unique": int(df[c].nunique())})
    return out


def choose_targets(candidates: list[dict], user: tuple[str, ...] = ()) -> tuple[list[str], str]:
    """(target columns, source). User choice wins; else hinted columns, else the first numeric column."""
    names = [c["column"] for c in candidates]
    chosen = [t for t in user if t in names]
    if chosen:
        return chosen, "user"
    usable = [c for c in candidates if c["n_unique"] > 1] or candidates
    hinted = [c["column"] for c in usable if c["name_hint"]]
    if hinted:
        return hinted, "name"
    return ([usable[0]["column"]], "first") if usable else ([], "none")


def choose_series(df: pd.DataFrame, id_col: str | None, user: tuple | None = None,
                  limit: int = MAX_SERIES) -> tuple[list, int, str]:
    """(series ids, number available, source). Long format: all series, longest first, at most ``limit``."""
    if not id_col:
        return [None], 1, "none"
    sizes = df[id_col].value_counts()
    if user:
        wanted = {str(u) for u in user}
        chosen = [i for i in sizes.index if str(i) in wanted]
        if chosen:
            return chosen, int(len(sizes)), "user"
    return list(sizes.index[:limit]), int(len(sizes)), "all"


def series_key(col: str | None, sid, n_targets: int, has_id: bool) -> str:
    if has_id and n_targets > 1:
        return f"{col} · {sid}"
    if has_id:
        return str(sid)
    return str(col) if col else "series"


def build_series(df: pd.DataFrame, parsed_time: pd.Series, col: str | None, id_col: str | None,
                 sid) -> tuple[pd.Series, pd.DatetimeIndex, np.ndarray]:
    """(observed series, horizon timestamps, bool mask over df's rows marking the horizon).

    Rows after the last observed value of the target are the forecast horizon: they are cut from the series (they
    are not missing data) and their timestamps are kept for future covariates.
    """
    mine = np.ones(len(df), dtype=bool) if id_col is None else (df[id_col] == sid).to_numpy()
    times = pd.DatetimeIndex(parsed_time.to_numpy()[mine])
    horizon_mask = np.zeros(len(df), dtype=bool)
    if col is None:
        return pd.Series(np.nan, index=times), pd.DatetimeIndex([]), horizon_mask
    values = df[col].to_numpy(dtype=float)[mine]
    s = pd.Series(values, index=times, name=col)
    observed = ~np.isnan(values)
    if not observed.any():
        return s, pd.DatetimeIndex([]), horizon_mask
    last = times[observed].max()
    after = (times > last) & ~observed
    horizon_mask[np.flatnonzero(mine)[after]] = True
    horizon = pd.DatetimeIndex(times[after]).dropna().unique().sort_values()
    return s[~after], horizon, horizon_mask
