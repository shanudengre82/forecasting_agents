from __future__ import annotations

from fcagents.core.agent import BaseAgent, Decision
from fcagents.core.state import PipelineState
from fcagents.tools.missing import (
    FILL_METHODS,
    MAX_MISSING_SHARE,
    fill_missing,
    full_grid,
    long_gap_threshold,
    missing_report,
    pick_seasonal_period,
    recommend_fill,
)


class MissingValuesAgent(BaseAgent):
    name = "missing_values"
    per_series = True
    title = "Missing Values Agent"
    task = "Find missing timestamps and values on the fixed-frequency grid and fill them."
    scope = ("Only discuss the missing values and the fill method. Filled values are estimates, not observations; "
             "do not claim they are real data.")

    summary = ("Puts the series on its complete fixed-frequency grid, counts missing timestamps and empty values, "
               "and fills them with your chosen method or a rule-based recommendation.")
    runs_when = "When a fixed frequency is known and the series has values (after the Frequency agent)."
    inputs = ("Fixed-frequency series (regular, or resampled with empty bins dropped)",
              "Your fill method (page control or chat)")
    outputs = ("Missing timestamps, missing values and the share of the grid they cover",
               "Longest gap runs, duplicates merged, off-grid timestamps dropped",
               "Fill method and whether it was yours or auto-recommended",
               "The complete, filled series (replaces the previous one for later agents)")
    decisions = {
        "complete": "Nothing is missing; the series is ready as is.",
        "filled": "Every missing value was filled.",
        "long_gaps": "Filled, but a gap is longer than one season (or 24 steps) or over "
                     f"{MAX_MISSING_SHARE:.0%} of the grid was missing, so the filled values are guesswork.",
        "left_missing": "Some values are still missing (method 'leave missing'); most forecasting models need a "
                        "complete series.",
    }
    user_controls = ("Fill method: auto, " + ", ".join(FILL_METHODS.values()) + " (control or chat)",)
    tools = ("tools.missing.missing_report", "tools.missing.recommend_fill", "tools.missing.fill_missing")

    def applies(self, state: PipelineState) -> str | None:
        if state.freq is None or state.series is None:
            return "Skipped: no fixed frequency is known yet."
        if state.series.isna().all():
            return "Skipped: the series has timestamps only, no values to fill."
        return None

    def run_tools(self, state: PipelineState) -> dict:
        report = missing_report(state.series, state.freq)
        period = pick_seasonal_period(state.fact("frequency", "seasonal_periods"), report["n_expected"])
        agg = state.fact("resampling", "agg")
        recommended = recommend_fill(report, agg, period)
        user_choice = state.fill_method not in (None, "auto")
        method = state.fill_method if user_choice else recommended
        if method == "seasonal" and not period:
            method = "interpolate"  # too short for two full seasons

        state.series, fill_facts = fill_missing(state.series, state.freq, method, period)
        return report | fill_facts | {
            "fill_source": "user" if user_choice else "auto",
            "recommended_method": recommended,
            "seasonal_period_available": period,
            "long_gap_threshold": long_gap_threshold(period),
        }

    def run(self, state, llm):
        before = full_grid(state.series, state.freq)[0]
        result = super().run(state, llm)
        result.extras["before"] = before  # lets the UI mark which points were filled
        result.extras["filled_index"] = before.index[before.isna() & state.series.notna()]
        return result

    def decide(self, f: dict) -> Decision:
        src = "your choice" if f["fill_source"] == "user" else "auto-selected"
        cleaned = "".join([
            f" {f['n_duplicates_merged']} duplicate timestamps were averaged." if f["n_duplicates_merged"] else "",
            f" {f['n_off_grid_dropped']} off-grid timestamps were dropped." if f["n_off_grid_dropped"] else "",
        ])
        if not f["n_missing"]:
            return Decision("ok", "complete", 1.0,
                            f"All {f['n_expected']} timestamps on the '{f['freq']}' grid have values; nothing to fill."
                            + cleaned, "The series is ready for the next agents.")
        found = (f"{f['n_missing']} of {f['n_expected']} values are missing ({f['missing_share']:.1%}): "
                 f"{f['n_missing_timestamps']} missing timestamps and {f['n_nan_values']} empty values. "
                 f"Longest gap: {f['longest_run']} steps.{cleaned}")
        how = f" Fill method: {f['method_label']} ({src})"
        if f["n_still_missing"]:
            return Decision("warning", "left_missing", 1 - f["n_still_missing"] / f["n_expected"],
                            found + how + f"; {f['n_still_missing']} values are still missing.",
                            f"Pick a fill method (recommended: {FILL_METHODS[f['recommended_method']]}).")
        summary = found + how + f"; {f['n_filled']} values filled."
        if f["longest_run"] > f["long_gap_threshold"] or f["missing_share"] > MAX_MISSING_SHARE:
            return Decision("warning", "long_gaps", 1 - f["missing_share"], summary,
                            "Check the filled stretches in the chart. Consider a coarser frequency or a shorter "
                            "history that avoids the longest gap.")
        return Decision("ok", "filled", 1 - f["missing_share"], summary,
                        "The series is complete and ready for the next agents.")

    def prompt_facts(self, f: dict) -> dict:
        keys = ["freq", "n_expected", "n_missing", "n_missing_timestamps", "n_nan_values", "missing_share",
                "longest_run", "method", "method_label", "fill_source", "recommended_method", "n_filled",
                "n_still_missing"]
        return {k: f.get(k) for k in keys}
