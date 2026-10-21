"""Deterministic tools for finding and parsing the time column of a DataFrame."""

from __future__ import annotations

import re
import warnings

import pandas as pd

NAME_HINTS = re.compile(r"(^|_)(date|time|timestamp|datetime|ds|dt|period|day|month|year|week|ts)($|_)", re.I)
DATE_LIKE = re.compile(r"\d{1,4}[-/.:T ]\d{1,2}|[A-Za-z]{3,9}[ -]\d{1,4}|\d{1,2}[ -][A-Za-z]{3,9}")
MIN_PARSE_RATE = 0.8
AMBIGUITY_MARGIN = 0.1
SAMPLE_SIZE = 500

# Plausible numeric encodings of time, checked only for columns whose name hints at time.
_EPOCH_S = (1e8, 4.2e9)
_EPOCH_MS = (1e11, 4.2e12)
_YYYYMMDD = (1.8e7, 2.2e7)


def _numeric_method(s: pd.Series) -> str | None:
    lo, hi = s.min(), s.max()
    if (s % 1 != 0).any():
        return None
    if _YYYYMMDD[0] <= lo and hi <= _YYYYMMDD[1]:
        return "yyyymmdd"
    if _EPOCH_S[0] <= lo and hi <= _EPOCH_S[1]:
        return "epoch_s"
    if _EPOCH_MS[0] <= lo and hi <= _EPOCH_MS[1]:
        return "epoch_ms"
    return None


def parse_time_column(df: pd.DataFrame, col: str, method: str) -> pd.Series:
    """Parse ``df[col]`` to datetimes using a method reported by :func:`detect_time_column`."""
    s = df[col]
    if method == "datetime":
        return pd.to_datetime(s)
    if method == "epoch_s":
        return pd.to_datetime(s, unit="s")
    if method == "epoch_ms":
        return pd.to_datetime(s, unit="ms")
    if method == "yyyymmdd":
        return pd.to_datetime(s.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        if method == "string":
            return pd.to_datetime(s, errors="coerce")
        if method == "string_dayfirst":
            return pd.to_datetime(s, errors="coerce", dayfirst=True)
        return pd.to_datetime(s, errors="coerce", format="mixed")


def _score_column(df: pd.DataFrame, col: str) -> dict | None:
    s = df[col].dropna()
    if s.empty:
        return None
    name_hint = bool(NAME_HINTS.search(str(col)))

    if pd.api.types.is_datetime64_any_dtype(s):
        method, parse_rate = "datetime", 1.0
    elif pd.api.types.is_bool_dtype(s):
        return None
    elif pd.api.types.is_numeric_dtype(s):
        method = _numeric_method(s) if name_hint else None
        if method is None:
            return None
        parse_rate = float(parse_time_column(df.loc[s.index], col, method).notna().mean())
    else:
        sample = s.astype(str).head(SAMPLE_SIZE)
        looks_like_date = sample.str.contains(DATE_LIKE)
        if looks_like_date.mean() < MIN_PARSE_RATE:
            return None
        sub = pd.DataFrame({col: sample})
        # Strict single-format parsing first (month-first, then day-first, e.g. 13/01/2024);
        # free-form "mixed" parsing is the last resort because it silently mixes conventions.
        for method in ("string", "string_dayfirst", "string_mixed"):
            parsed = parse_time_column(sub, col, method)
            if parsed.notna().mean() >= MIN_PARSE_RATE:
                break
        parse_rate = float((parsed.notna() & looks_like_date).mean())

    if parse_rate < MIN_PARSE_RATE:
        return None
    parsed = parse_time_column(df, col, method).dropna()
    monotonic = bool(parsed.is_monotonic_increasing or parsed.is_monotonic_decreasing)
    unique_ratio = float(parsed.nunique() / len(parsed)) if len(parsed) else 0.0
    score = 0.6 * parse_rate + 0.25 * name_hint + 0.1 * monotonic + 0.05 * unique_ratio
    return {
        "column": str(col),
        "method": method,
        "parse_rate": round(parse_rate, 4),
        "name_hint": name_hint,
        "monotonic": monotonic,
        "unique_ratio": round(unique_ratio, 4),
        "score": round(score, 4),
    }


def _detect_id_col(df: pd.DataFrame, time_col: str) -> str | None:
    """A low-cardinality text column under which timestamps repeat suggests long-format multi-series data."""
    if not df[time_col].duplicated().any():
        return None
    best = None
    for col in df.columns:
        if col == time_col or pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        n = df[col].nunique()
        if 2 <= n <= max(2, len(df) // 2) and not df[[col, time_col]].duplicated().any():
            if best is None or n < best[1]:
                best = (str(col), n)
    return best[0] if best else None


def detect_time_column(df: pd.DataFrame) -> dict:
    """Rank columns by how likely they are the time index; also report targets and a series-id column."""
    candidates = [c for col in df.columns if (c := _score_column(df, col))]
    candidates.sort(key=lambda c: c["score"], reverse=True)

    time_col = candidates[0]["column"] if candidates else None
    ambiguous = len(candidates) > 1 and candidates[0]["score"] - candidates[1]["score"] < AMBIGUITY_MARGIN
    id_col = _detect_id_col(df, time_col) if time_col else None
    target_cols = [
        str(c) for c in df.columns
        if c != time_col and pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
        and c not in {cand["column"] for cand in candidates}
    ]
    n_series = int(df[id_col].nunique()) if id_col else 1
    return {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "columns": [str(c) for c in df.columns],
        "time_col": time_col,
        "ambiguous": bool(ambiguous),
        "candidates": candidates,
        "target_cols": target_cols,
        "id_col": id_col,
        "n_series": n_series,
    }


__all__ = ["detect_time_column", "parse_time_column"]
