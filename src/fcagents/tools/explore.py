"""Deterministic multi-format data import and exploratory data analysis (profiling, anomalies, relationships)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SUPPORTED_EXTENSIONS = (".csv", ".tsv", ".json", ".xlsx", ".xls", ".parquet")
CORR_METHODS = {"pearson": "Pearson (linear)", "spearman": "Spearman (rank / monotonic)"}
HIGH_MISSING_PCT = 20.0
HIGH_SKEW_THRESHOLD = 2.0
STRONG_CORR_THRESHOLD = 0.5


def load_dataset(path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load a tabular dataset from CSV, TSV, JSON (or JSONL), Excel (.xlsx/.xls), or Parquet."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file format '{ext or '(none)'}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}.")

    if ext == ".csv":
        df = pd.read_csv(p)
    elif ext == ".tsv":
        df = pd.read_csv(p, sep="\t")
    elif ext == ".json":
        try:
            df = pd.read_json(p)
        except ValueError:
            df = pd.read_json(p, lines=True)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(p)
    elif ext == ".parquet":
        df = pd.read_parquet(p)
    else:  # pragma: no cover
        raise ValueError(f"Unsupported file format '{ext}'.")

    meta = {
        "filename": p.name,
        "format": ext.lstrip("."),
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "memory_kb": round(float(df.memory_usage(deep=True).sum()) / 1024.0, 1),
    }
    return df, meta


def _column_kind(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if pd.api.types.is_numeric_dtype(s):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    return "categorical"


def _iqr_outlier_count(values: pd.Series) -> int:
    if len(values) < 4:
        return 0
    q1 = float(values.quantile(0.25))
    q3 = float(values.quantile(0.75))
    iqr = q3 - q1
    if iqr <= 0 or not np.isfinite(iqr):
        return 0
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return int(((values < lo) | (values > hi)).sum())


def profile_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute a per-column statistical & quality profile (types, missingness, distribution, IQR anomalies)."""
    n_rows = len(df)
    rows: list[dict[str, Any]] = []
    for col in df.columns:
        s = df[col]
        kind = _column_kind(s)
        missing = int(s.isna().sum())
        missing_pct = round(100.0 * missing / n_rows, 1) if n_rows else 0.0
        unique = int(s.nunique(dropna=True))
        entry: dict[str, Any] = {
            "column": str(col),
            "kind": kind,
            "missing": missing,
            "missing_pct": missing_pct,
            "unique": unique,
            "mean": None,
            "std": None,
            "min": None,
            "median": None,
            "max": None,
            "skew": None,
            "iqr_anomalies": 0,
        }
        if kind == "numeric":
            vals = pd.to_numeric(s, errors="coerce").dropna()
            if len(vals):
                skew_val = float(vals.skew()) if len(vals) > 2 else 0.0
                entry |= {
                    "mean": round(float(vals.mean()), 4),
                    "std": round(float(vals.std()), 4) if len(vals) > 1 else 0.0,
                    "min": round(float(vals.min()), 4),
                    "median": round(float(vals.median()), 4),
                    "max": round(float(vals.max()), 4),
                    "skew": round(skew_val, 2) if np.isfinite(skew_val) else 0.0,
                    "iqr_anomalies": _iqr_outlier_count(vals),
                }
        rows.append(entry)
    return pd.DataFrame(rows)


def detect_anomalies(df: pd.DataFrame, profile: pd.DataFrame) -> dict[str, Any]:
    """Summarize structural and statistical anomalies across the raw dataset."""
    n_rows = len(df)
    n_duplicates = int(df.duplicated().sum()) if n_rows else 0
    duplicate_pct = round(100.0 * n_duplicates / n_rows, 1) if n_rows else 0.0

    constant_cols = profile.loc[profile["unique"] <= 1, "column"].tolist() if len(profile) else []
    high_missing_cols = profile.loc[profile["missing_pct"] >= HIGH_MISSING_PCT, "column"].tolist() if len(profile) else []
    any_missing_cols = profile.loc[profile["missing"] > 0, "column"].tolist() if len(profile) else []

    skewed_cols: list[str] = []
    iqr_cols: dict[str, int] = {}
    if len(profile):
        for row in profile.itertuples(index=False):
            if row.skew is not None and pd.notna(row.skew) and abs(float(row.skew)) >= HIGH_SKEW_THRESHOLD:
                skewed_cols.append(row.column)
            if int(row.iqr_anomalies) > 0:
                iqr_cols[row.column] = int(row.iqr_anomalies)

    return {
        "n_duplicate_rows": n_duplicates,
        "duplicate_pct": duplicate_pct,
        "constant_cols": constant_cols,
        "high_missing_cols": high_missing_cols,
        "any_missing_cols": any_missing_cols,
        "skewed_cols": skewed_cols,
        "iqr_anomaly_cols": iqr_cols,
        "total_iqr_anomalies": int(sum(iqr_cols.values())),
    }


def compute_relationships(
    df: pd.DataFrame,
    *,
    method: str = "pearson",
    max_cols: int = 20,
) -> dict[str, Any]:
    """Compute numeric correlation matrix and strongest pairwise relationships."""
    method = method if method in CORR_METHODS else "pearson"
    num_cols = [str(c) for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    # Keep columns with non-zero variance so correlations are well-defined.
    valid_cols = [c for c in num_cols if df[c].nunique(dropna=True) > 1][:max_cols]

    if len(valid_cols) < 2:
        return {"method": method, "corr_matrix": pd.DataFrame(), "top_pairs": [], "n_numeric_compared": len(valid_cols)}

    corr = df[valid_cols].corr(method=method)
    pairs: list[dict[str, Any]] = []
    for i, c1 in enumerate(valid_cols):
        for c2 in valid_cols[i + 1 :]:
            val = corr.loc[c1, c2]
            if pd.isna(val):
                continue
            r = round(float(val), 3)
            if abs(r) >= STRONG_CORR_THRESHOLD:
                strength = "very strong" if abs(r) >= 0.8 else "strong"
                pairs.append({"col_a": c1, "col_b": c2, "r": r, "strength": strength})
    pairs.sort(key=lambda item: -abs(item["r"]))
    return {
        "method": method,
        "corr_matrix": corr,
        "top_pairs": pairs[:10],
        "n_numeric_compared": len(valid_cols),
    }


def explore_dataset(df: pd.DataFrame, *, corr_method: str = "pearson") -> tuple[dict[str, Any], dict[str, Any]]:
    """Run full deterministic EDA on ``df``. Returns ``(json_facts, ui_extras)``."""
    n_rows, n_cols = int(df.shape[0]), int(df.shape[1])
    if n_rows == 0 or n_cols == 0:
        return (
            {"n_rows": n_rows, "n_cols": n_cols, "empty": True},
            {"profile_table": pd.DataFrame(), "corr_matrix": pd.DataFrame()},
        )

    profile = profile_columns(df)
    anomalies = detect_anomalies(df, profile)
    rels = compute_relationships(df, method=corr_method)

    kind_counts = profile["kind"].value_counts().to_dict()
    total_missing_cells = int(profile["missing"].sum())
    missing_cell_pct = round(100.0 * total_missing_cells / (n_rows * n_cols), 2)

    facts: dict[str, Any] = {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "empty": False,
        "memory_kb": round(float(df.memory_usage(deep=True).sum()) / 1024.0, 1),
        "column_kinds": {str(k): int(v) for k, v in kind_counts.items()},
        "total_missing_cells": total_missing_cells,
        "missing_cell_pct": missing_cell_pct,
        "corr_method": rels["method"],
        "top_pairs": rels["top_pairs"],
        **anomalies,
    }
    extras = {
        "profile_table": profile,
        "corr_matrix": rels["corr_matrix"],
    }
    return facts, extras
