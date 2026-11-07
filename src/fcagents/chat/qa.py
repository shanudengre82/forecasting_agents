"""Answers about the series in hand: a deterministic stats tool, template answers, LLM phrasing for the rest."""

from __future__ import annotations

import json
import re

import pandas as pd

from fcagents.core.agent import SYSTEM_PROMPT
from fcagents.llm.backend import LLMBackend
from fcagents.tools.frequency import frequency_label


def series_stats(series: pd.Series | None, freq: str | None) -> dict:
    """Facts about the current series; this is all the chat may state."""
    if series is None or series.empty:
        return {"n_points": 0}
    s = series[series.index.notna()].sort_index()
    values = s.dropna()
    stats = {
        "n_points": int(len(s)),
        "start": s.index[0].isoformat(),
        "end": s.index[-1].isoformat(),
        "frequency": freq or "not fixed (irregular)",
        "n_missing_values": int(s.isna().sum()),
    }
    if freq:
        label, periods = frequency_label(freq)
        stats["frequency"] = f"{freq} ({label})"  # tiny models misread bare aliases such as 'D'
        stats["candidate_seasonal_periods"] = periods
        grid = pd.date_range(s.index[0], s.index[-1], freq=freq)
        stats["n_missing_timestamps"] = int(len(grid.difference(s.index)))
    if len(values):
        stats |= {
            "mean": round(float(values.mean()), 4),
            "std": round(float(values.std()), 4) if len(values) > 1 else 0.0,
            "min": round(float(values.min()), 4),
            "min_at": values.idxmin().isoformat(),
            "max": round(float(values.max()), 4),
            "max_at": values.idxmax().isoformat(),
            "last_value": round(float(values.iloc[-1]), 4),
        }
    return stats


def _missing(st: dict, f: dict) -> str:
    if "n_filled" in f:
        return (f"Before filling, {f['n_missing']} of {f['n_expected']} values were missing ({f['missing_share']:.1%}; "
                f"{f['n_missing_timestamps']} missing timestamps, {f['n_nan_values']} empty values). "
                f"{f['n_filled']} were filled and {f['n_still_missing']} are still missing.")
    ts = st.get("n_missing_timestamps")
    return (f"The current series has {st['n_missing_values']} empty values"
            + (f" and {ts} missing timestamps on the '{st['frequency']}' grid." if ts is not None
               else "; the timestamps are not on a fixed grid yet, so missing timestamps can't be counted."))


# (pattern, answer builder from (stats, page facts)); first match wins.
_TEMPLATES = [
    (r"\bfill(ed|s)?\b|\bimput", lambda st, f: (
        f"{f['n_filled']} values were filled with {f['method_label']}." if "n_filled" in f else None)),
    (r"\bmissing\b|\bgaps?\b|\bnan\b|\bempty\b|\bnull\b", _missing),
    (r"\bfrequency\b|\binterval\b|\bspacing\b|\bhow often\b|\bgranularity\b",
     lambda st, f: f"The current frequency is {st['frequency']}."),
    (r"\bmean\b|\baverage\b|\bavg\b",
     lambda st, f: f"The mean is {st['mean']} (std {st['std']})." if "mean" in st else None),
    (r"\bmax\w*\b|\bhighest\b|\bpeak\b|\blargest\b",
     lambda st, f: f"The maximum is {st['max']}, at {st['max_at']}." if "max" in st else None),
    (r"\bmin\w*\b|\blowest\b|\bsmallest\b",
     lambda st, f: f"The minimum is {st['min']}, at {st['min_at']}." if "min" in st else None),
    (r"\b(start|begin\w*)\b|\bfirst (date|timestamp)\b",
     lambda st, f: f"The series starts at {st['start']}." if "start" in st else None),
    (r"\b(end|ends|last (date|timestamp)|latest|until)\b",
     lambda st, f: f"The series ends at {st['end']}." if "end" in st else None),
    (r"\bhow many\b|\bnumber of\b|\blength\b|\bhow long\b|\bpoints\b|\brows\b|\bobservations\b",
     lambda st, f: f"The current series has {st['n_points']} points from {st.get('start')} to {st.get('end')}."),
]


def template_answer(question: str, stats: dict, page_facts: dict) -> str | None:
    for pattern, build in _TEMPLATES:
        if re.search(pattern, question, re.I):
            try:
                return build(stats, page_facts)
            except KeyError:
                return None
    return None


def llm_answer(question: str, stats: dict, page_facts: dict, llm: LLMBackend) -> str:
    prompt = (
        f"Facts about the current time series (JSON): {json.dumps(stats, default=str)}\n"
        f"Results of the agents on this page (JSON): {json.dumps(page_facts, default=str)}\n"
        f"Question: {question}\n"
        "Answer using only these facts. If they do not contain the answer, say that you don't know."
    )
    return llm.generate(SYSTEM_PROMPT, prompt, max_new_tokens=160)
