from __future__ import annotations

from fcagents.core.agent import BaseAgent, Decision
from fcagents.core.state import PipelineState
from fcagents.tools.time_index import detect_time_column, parse_time_column


class TimeIndexAgent(BaseAgent):
    name = "time_index"
    title = "Time Index Agent"
    task = "Identify which column holds the timestamps and, for long-format data, which column identifies each series."
    scope = ("Only discuss which columns were identified. Do NOT talk about the target, sampling frequency, "
             "regularity, trends or seasonality: later agents check those and they are unknown here.")

    summary = ("Finds the column that holds the timestamps and, for long-format data, the column that identifies "
               "each series.")
    runs_when = "Always (first agent)."
    inputs = ("Uploaded or sample CSV (any column names)",)
    outputs = ("Time column and how it was parsed (ISO, day-first, epoch s/ms, YYYYMMDD, mixed)",
               "Ranked time-column candidates with parse rate and score",
               "Series-id column and number of series (long format)")
    decisions = {
        "found": "One clear time column; the pipeline continues.",
        "ambiguous": "Several plausible time columns; the best one is used, but you should confirm it.",
        "not_found": "No column parses as dates; the pipeline stops.",
    }
    user_controls = ("Time column",)
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
        state.id_col = rep["id_col"]
        state.parsed_time = parse_time_column(state.raw, state.time_col, state.time_method)
        rep["n_unparseable"] = int(state.parsed_time.isna().sum())
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
        if facts["id_col"]:
            parts.append(f"Long format: {facts['n_series']} series identified by '{facts['id_col']}'.")
        if facts["ambiguous"]:
            others = ", ".join(c["column"] for c in facts["candidates"][1:])
            return Decision("warning", "ambiguous", top["score"], " ".join(parts) +
                            f" Other plausible time columns: {others}.",
                            "Confirm the time column, then choose what to forecast.")
        return Decision("ok", "found", min(1.0, top["score"]), " ".join(parts),
                        "Choose what to forecast (Target agent).")

    def prompt_facts(self, facts: dict) -> dict:
        keys = ["n_rows", "columns", "time_col", "id_col", "n_series", "ambiguous"]
        return {k: facts.get(k) for k in keys} | {
            "candidates": [{k: c[k] for k in ("column", "parse_rate", "score")} for c in facts["candidates"][:3]]
        }
