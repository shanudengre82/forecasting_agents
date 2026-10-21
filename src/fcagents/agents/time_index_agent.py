from __future__ import annotations

import numpy as np
import pandas as pd

from fcagents.core.agent import BaseAgent, Decision
from fcagents.core.state import PipelineState
from fcagents.tools.time_index import detect_time_column, parse_time_column


class TimeIndexAgent(BaseAgent):
    name = "time_index"
    title = "Time Index Agent"
    task = "Identify which column holds the timestamps and which column is the value to forecast."
    scope = ("Only discuss which columns were identified. Do NOT talk about sampling frequency, regularity, "
             "trends or seasonality: later agents check those and they are unknown here.")

    summary = ("Finds the column that holds the timestamps, the numeric column to forecast and, for long-format "
               "data, the column that identifies each series.")
    runs_when = "Always (first agent)."
    inputs = ("Uploaded or sample CSV (any column names)",)
    outputs = ("Time column and how it was parsed (ISO, day-first, epoch s/ms, YYYYMMDD, mixed)",
               "Ranked time-column candidates with parse rate and score",
               "Target column", "Series-id column and number of series (long format)")
    decisions = {
        "found": "One clear time column; the pipeline continues.",
        "ambiguous": "Several plausible time columns; the best one is used, but you should confirm it.",
        "not_found": "No column parses as dates; the pipeline stops.",
    }
    user_controls = ("Time column", "Target column", "Series to analyse (long format)")
    tools = ("tools.time_index.detect_time_column", "tools.time_index.parse_time_column")

    def run_tools(self, state: PipelineState) -> dict:
        rep = detect_time_column(state.raw)
        by_col = {c["column"]: c for c in rep["candidates"]}
        if state.time_col in by_col:  # user override
            rep["time_col"], rep["ambiguous"], rep["overridden"] = state.time_col, False, True
        if rep["time_col"] is None:
            return rep

        state.time_col = rep["time_col"]
        state.time_method = by_col[state.time_col]["method"]
        if state.target_col not in rep["target_cols"]:
            state.target_col = rep["target_cols"][0] if rep["target_cols"] else None
        state.id_col = rep["id_col"]

        df = state.raw.assign(_t=parse_time_column(state.raw, state.time_col, state.time_method))
        if state.id_col:
            sizes = df[state.id_col].value_counts()
            if state.series_id not in sizes.index:
                state.series_id = sizes.index[0]  # default to the longest series
            df = df[df[state.id_col] == state.series_id]
        values = df[state.target_col].to_numpy() if state.target_col else np.full(len(df), np.nan)
        state.series = pd.Series(values, index=pd.DatetimeIndex(df["_t"]), name=state.target_col)

        rep.update(
            target_col=state.target_col,
            series_id=None if state.series_id is None else str(state.series_id),
            n_points=int(len(state.series)),
            n_unparseable=int(state.series.index.isna().sum()),
        )
        return rep

    def decide(self, facts: dict) -> Decision:
        if facts["time_col"] is None:
            return Decision("error", "not_found", 0.0,
                            "No column could be parsed as timestamps.",
                            "Upload data with a date/time column (e.g. ISO dates like 2024-01-31).")
        top = facts["candidates"][0] if not facts.get("overridden") else \
            next(c for c in facts["candidates"] if c["column"] == facts["time_col"])
        parts = [f"Time column '{facts['time_col']}' (parsed via {top['method']}, "
                 f"{top['parse_rate']:.0%} of values parse)."]
        parts.append(f"Target column: '{facts['target_col']}'." if facts["target_col"]
                     else "No numeric target column found; only the timestamps will be analysed.")
        if facts["id_col"]:
            parts.append(f"Long format with {facts['n_series']} series in '{facts['id_col']}'; "
                         f"analysing series '{facts['series_id']}'.")
        if facts["ambiguous"]:
            others = ", ".join(c["column"] for c in facts["candidates"][1:])
            return Decision("warning", "ambiguous", top["score"], " ".join(parts) +
                            f" Other plausible time columns: {others}.",
                            "Confirm the time column in the sidebar, then check regularity.")
        return Decision("ok", "found", min(1.0, top["score"]), " ".join(parts),
                        "Check whether the timestamps follow a fixed frequency.")

    def prompt_facts(self, facts: dict) -> dict:
        keys = ["n_rows", "columns", "time_col", "target_col", "id_col", "n_series", "ambiguous"]
        return {k: facts.get(k) for k in keys} | {
            "candidates": [{k: c[k] for k in ("column", "parse_rate", "score")} for c in facts["candidates"][:3]]
        }
