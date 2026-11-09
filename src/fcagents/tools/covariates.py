"""Deterministic tools for sorting the non-target columns into past, future and static covariates."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

ROLES = {"past": "Past covariate (observed up to now)", "future": "Future covariate (known in advance)",
         "static": "Static covariate (one value per series)", "ignore": "Not used"}
PLANNED = re.compile(r"(promo|price|discount|holiday|event|planned|plan|forecast|scheduled|campaign|budget|"
                     r"closed|open)", re.I)
CALENDAR = re.compile(r"(^|_)(dow|weekday|day_?of_?week|is_?weekend|weekend|month|week|hour|quarter|year|"
                      r"day_?of_?year)($|_)", re.I)
MAX_MISSING = 0.5
FUTURE_MIN_COVERAGE = 0.5


def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s)


def classify_column(df: pd.DataFrame, col: str, id_col: str | None, horizon_rows: pd.Series | None,
                    is_date: bool = False) -> dict:
    """Detected role of one column, with the reason and the numbers behind it."""
    s = df[col]
    mask = horizon_rows.to_numpy() if horizon_rows is not None else np.zeros(len(df), dtype=bool)
    observed, future = s[~mask], s[mask]
    info = {"column": str(col), "kind": "numeric" if _is_numeric(s) else "categorical",
            "missing_share": round(float(observed.isna().mean()), 4) if len(observed) else 1.0,
            "n_unique": int(s.nunique()),
            "future_coverage": round(float(future.notna().mean()), 4) if len(future) else None}

    def out(role, reason):
        return info | {"detected": role, "reason": reason}

    if is_date:
        return out("ignore", "another date column")
    if info["missing_share"] >= MAX_MISSING:
        return out("ignore", f"{info['missing_share']:.0%} of values are missing")
    if info["n_unique"] <= 1:
        return out("ignore", "the same value in every row")
    if info["kind"] == "categorical" and info["n_unique"] > max(20, 0.5 * len(df)):
        return out("ignore", "looks like an id or free text (almost every value differs)")
    if id_col and df.groupby(id_col)[col].nunique(dropna=True).max() <= 1:
        return out("static", f"one value per series in '{id_col}'")
    if info["future_coverage"] is not None and info["future_coverage"] >= FUTURE_MIN_COVERAGE:
        return out("future", f"known for {info['future_coverage']:.0%} of the future rows")
    if CALENDAR.search(str(col)):
        return out("future", "calendar feature, known in advance")
    if PLANNED.search(str(col)):
        return out("future", "the name suggests it is planned in advance (prices, promotions, events)")
    return out("past", "only known up to the present")


def classify_covariates(df: pd.DataFrame, columns: list[str], id_col: str | None, horizon_rows: pd.Series | None,
                        date_cols: set[str], overrides: dict[str, str]) -> list[dict]:
    rows = []
    for col in columns:
        r = classify_column(df, col, id_col, horizon_rows, is_date=col in date_cols)
        r["role"] = overrides.get(col, r["detected"]) if overrides.get(col) in ROLES else r["detected"]
        r["source"] = "user" if r["role"] != r["detected"] or col in overrides else "rules"
        rows.append(r)
    return rows


def covariate_frame(raw: pd.DataFrame, parsed_time: pd.Series, id_col: str | None, series_id, roles: dict[str, str],
                    freq: str, index: pd.DatetimeIndex, horizon: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """Covariates of one series on its final grid, extended over the forecast horizon.

    Values are aggregated per bin (numeric: mean, categorical: last). Past covariates stop at the last observed
    timestamp; future covariates continue over the horizon; static ones repeat their single value.
    """
    cols = [c for c, r in roles.items() if r in ("past", "future", "static")]
    end = index.max()
    if horizon is not None and len(horizon):
        end = max(end, horizon.max())
    grid = pd.date_range(index.min(), end, freq=freq) if len(index) else index
    out = pd.DataFrame(index=grid.union(index))
    if not cols:
        return out
    mine = np.ones(len(raw), dtype=bool) if id_col is None else (raw[id_col] == series_id).to_numpy()
    sub = raw[mine].assign(_t=parsed_time.to_numpy()[mine]).dropna(subset=["_t"]).set_index("_t").sort_index()
    for col in cols:
        s = sub[col]
        if _is_numeric(s):
            binned = s.astype(float).resample(freq).mean()
        else:
            binned = s.resample(freq).last()
        values = binned.reindex(out.index)
        if roles[col] == "static":
            values = pd.Series(s.dropna().iloc[0] if s.notna().any() else None, index=out.index)
        else:
            values = values.ffill()
            if roles[col] == "past":
                values[values.index > index.max()] = None
        out[col] = values
    return out
