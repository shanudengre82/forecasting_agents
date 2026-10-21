from __future__ import annotations

from fcagents.core.agent import BaseAgent, Decision
from fcagents.core.state import PipelineState
from fcagents.tools.frequency import analyze_regularity


class RegularityAgent(BaseAgent):
    name = "regularity"
    title = "Regularity Agent"
    task = "Decide whether the time series is sampled at a fixed frequency."
    scope = "Do NOT discuss seasonality or models; only whether the spacing of timestamps is fixed."

    summary = ("Checks whether the timestamps follow a fixed frequency, including calendar frequencies "
               "(business days, month/quarter/year starts and ends), and flags duplicates, gaps and unsorted data.")
    runs_when = "After a time column is found."
    inputs = ("Parsed timestamps of one series",)
    outputs = ("Status and detected frequency alias", "Coverage and share of timestamps on the grid",
               "Missing-timestamp count, duplicate count, largest gap", "Spacing statistics (min/median/max delta)")
    decisions = {
        "regular": "Every timestamp sits on a fixed grid with no gaps.",
        "regular_with_gaps": "Fixed grid, but some timestamps are missing.",
        "irregular": "No fixed grid fits; the Resampling agent runs next.",
        "insufficient_data": "Fewer than 3 distinct timestamps; the pipeline stops.",
    }
    tools = ("tools.frequency.analyze_regularity",)

    def run_tools(self, state: PipelineState) -> dict:
        rep = analyze_regularity(state.series.index)
        state.freq = rep["freq"]
        return rep

    def decide(self, f: dict) -> Decision:
        dup = (f" {f['n_duplicates']} duplicate timestamps were found and must be aggregated."
               if f["n_duplicates"] else "")
        status = f["status"]
        if status == "insufficient_data":
            return Decision("error", status, 0.0,
                            f"Only {f['n_unique']} distinct timestamps; at least 3 are needed.",
                            "Provide a longer series.")
        if status == "regular":
            return Decision("warning" if dup else "ok", status, 1.0,
                            f"Fixed frequency: every timestamp sits on a '{f['freq']}' grid with no gaps.{dup}",
                            "Name the frequency and derive seasonal periods.")
        if status == "regular_with_gaps":
            return Decision("warning", status, f["on_grid"] * f["coverage"],
                            f"Fixed frequency '{f['freq']}' but {f['n_missing']} timestamps are missing "
                            f"(coverage {f['coverage']:.1%}).{dup}",
                            "Name the frequency, then fill the missing timestamps (gap-imputation agent).")
        closest = f.get("closest_grid")
        return Decision("warning", "irregular", 1 - (f["on_grid"] or 0) * (f["coverage"] or 0),
                        f"No fixed frequency: {f['n_distinct_deltas']} distinct spacings, most common "
                        f"{f['mode_delta']} covers only {f['mode_delta_share']:.0%} of intervals.{dup}",
                        "Resample onto a fixed grid between 1h and 1W (resampling agent)"
                        + (f"; closest natural fit: '{closest}'" if closest else "") + ".")

    def prompt_facts(self, f: dict) -> dict:
        keys = ["status", "freq", "coverage", "n_missing", "n_duplicates", "mode_delta",
                "mode_delta_share", "n_distinct_deltas", "largest_gap"]
        return {k: f.get(k) for k in keys}
