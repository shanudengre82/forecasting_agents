from __future__ import annotations

import numpy as np
import pandas as pd

from fcagents.core.agent import BaseAgent, Decision
from fcagents.core.state import CHILD_PREFS, PipelineState
from fcagents.tools.targets import MAX_SERIES, build_series, choose_series, choose_targets, series_key, \
    target_candidates
from fcagents.tools.time_index import detect_time_column


class TargetAgent(BaseAgent):
    name = "target"
    title = "Target Agent"
    task = "Decide which values will be forecast: one time series, or several at once."
    scope = ("Only discuss which series are forecast and why. Do NOT talk about frequency, gaps, covariates or "
             "seasonality: other agents check those.")

    summary = ("Chooses what to forecast: one value column or several, and in long-format data one series per id "
               "(all of them by default). Rows after a series' last observed value are kept apart as the forecast "
               "horizon.")
    runs_when = "After a time column is found."
    inputs = ("Numeric columns and the series-id column", "Your choice of target columns and series")
    outputs = ("Target columns and series to forecast", "Single or multiple series",
               "Forecast-horizon rows (future rows without a target value)")
    decisions = {
        "single": "One series is forecast.",
        "multiple": "Several series are forecast at once; every later agent runs on each of them.",
        "capped": f"More than {MAX_SERIES} series; the {MAX_SERIES} longest are used.",
        "timestamps_only": "No numeric column to forecast; only the timestamps are analysed.",
    }
    user_controls = ("Target columns", "Series to forecast (long format)")
    tools = ("tools.targets.target_candidates", "tools.targets.choose_targets", "tools.targets.choose_series",
             "tools.targets.build_series")

    def applies(self, state: PipelineState) -> str | None:
        return None if state.time_col else "Skipped: no time column."

    def run_tools(self, state: PipelineState) -> dict:
        df = state.raw
        others = {c["column"] for c in detect_time_column(df)["candidates"]}
        exclude = {state.time_col, state.id_col, *others} - {None}
        cands = target_candidates(df, exclude)
        targets, source = choose_targets(cands, state.targets)
        ids, n_available, id_source = choose_series(df, state.id_col, state.series_ids)
        has_id = state.id_col is not None
        prefs = {k: getattr(state, k) for k in CHILD_PREFS}
        children, horizon = {}, np.zeros(len(df), dtype=bool)
        for col in targets or [None]:
            for sid in ids:
                s, future, mask = build_series(df, state.parsed_time, col, state.id_col, sid)
                horizon |= mask
                key = series_key(col, sid, len(targets), has_id)
                children[key] = PipelineState(raw=df, time_col=state.time_col, time_method=state.time_method,
                                              target_col=col, id_col=state.id_col, series_id=sid, key=key,
                                              series=s, horizon=future, **prefs)
        horizon_rows = pd.Series(horizon, index=df.index)
        state.children, state.horizon_rows = children, horizon_rows
        sizes = {k: int(c.series.notna().sum()) for k, c in children.items()}
        return {
            "candidates": cands, "targets": targets, "target_source": source,
            "id_col": state.id_col, "n_series_available": n_available, "series_source": id_source,
            "series": list(children), "n_series": len(children), "max_series": MAX_SERIES,
            "mode": "multiple" if len(children) > 1 else "single",
            "n_horizon_rows": int(horizon_rows.sum()),
            "horizon_steps": {k: int(len(c.horizon)) for k, c in children.items() if len(c.horizon)},
            "observed_points": sizes,
            "not_chosen": [c["column"] for c in cands if c["column"] not in targets],
        }

    def decide(self, f: dict) -> Decision:
        if not f["targets"]:
            return Decision("warning", "timestamps_only", 0.5,
                            "No numeric column to forecast; only the timestamps will be analysed.",
                            "Upload data with a numeric value column to forecast.")
        why = {"user": "your choice", "name": "the column name says it is a value to forecast",
               "first": "first numeric column", "none": ""}[f["target_source"]]
        cols = ", ".join(f"'{t}'" for t in f["targets"])
        parts = [f"Target column{'s' if len(f['targets']) > 1 else ''}: {cols} ({why})."]
        if f["id_col"]:
            parts.append(f"{f['n_series']} of {f['n_series_available']} series in '{f['id_col']}' are forecast"
                         + (" (your choice)." if f["series_source"] == "user" else "."))
        if f["n_horizon_rows"]:
            parts.append(f"{f['n_horizon_rows']} rows after the last observed value have no target: they are the "
                         "forecast horizon, not missing data.")
        if f["not_chosen"]:
            parts.append(f"Other numeric columns ({', '.join(f['not_chosen'][:6])}) go to the Covariates agent.")
        summary = " ".join(parts)
        if f["n_series_available"] > f["max_series"] and f["series_source"] == "all":
            return Decision("warning", "capped", 0.8, summary + f" Only the {f['max_series']} longest series are "
                            "used, to keep the app fast.", "Pick the series you need under Series to forecast.")
        if f["mode"] == "multiple":
            return Decision("ok", "multiple", 1.0, summary,
                            f"Every later agent runs on each of the {f['n_series']} series.")
        return Decision("ok", "single", 1.0, summary, "Check which other columns can serve as covariates.")

    def prompt_facts(self, f: dict) -> dict:
        keys = ["targets", "target_source", "id_col", "n_series", "n_series_available", "mode", "n_horizon_rows",
                "not_chosen"]
        return {k: f.get(k) for k in keys} | {"series": f.get("series", [])[:8]}
