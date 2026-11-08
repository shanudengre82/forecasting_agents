from __future__ import annotations

from fcagents.core.agent import BaseAgent, Decision
from fcagents.core.state import PipelineState
from fcagents.tools.missing import pick_seasonal_period
from fcagents.tools.outliers import (
    ACTIONS,
    MIN_POINTS,
    SUGGEST_REPLACE_Z,
    THRESHOLDS,
    apply_outlier_actions,
    detect_outliers,
)


class OutliersAgent(BaseAgent):
    name = "outliers"
    title = "Outliers Agent"
    task = "Flag values that are far from what the trend and seasonal pattern predict."
    scope = ("Only discuss the flagged points. A flagged point is a candidate; only the user decides whether it "
             "is an error. Holiday-affected days are never flagged.")

    summary = ("Compares every value with what the trend and seasonal pattern predict and flags the ones that are "
               "far off. You decide per point: keep it, remove it (the Missing Values agent then fills the gap) "
               "or replace it with another value. Nothing changes until you choose.")
    runs_when = "When a fixed frequency is known and the series has values (after the Holidays agent)."
    inputs = ("Fixed-frequency series", "Holiday-affected days from the Holidays agent",
              "Sensitivity and your keep / remove / replace choices (page or chat)")
    outputs = ("Flagged points with their value, expected value and robust z-score",
               f"A suggestion per point (replace when |z| ≥ {SUGGEST_REPLACE_Z:g}, otherwise keep)",
               "Holiday-affected points that were left alone",
               "The series with your choices applied (removed points become gaps for the Missing Values agent)")
    decisions = {
        "none_found": "No value is far from the expected pattern.",
        "flagged": "Some values look like outliers and you haven't decided about all of them yet; they are "
                   "kept until you do.",
        "handled": "You decided about every flagged point; your choices are applied.",
    }
    user_controls = ("Sensitivity: " + ", ".join(THRESHOLDS.values()),
                     "Per point: " + " / ".join(ACTIONS.values()) + " with a replacement value",
                     "All at once: keep all, remove all, replace all, apply suggestions (buttons or chat)")
    tools = ("tools.outliers.detect_outliers", "tools.outliers.apply_outlier_actions")

    def applies(self, state: PipelineState) -> str | None:
        if state.freq is None or state.series is None:
            return "Skipped: no fixed frequency is known yet."
        if state.series.notna().sum() < MIN_POINTS:
            return f"Skipped: fewer than {MIN_POINTS} values to judge what is normal."
        return None

    def run_tools(self, state: PipelineState) -> dict:
        period = pick_seasonal_period(state.fact("frequency", "seasonal_periods"), len(state.series))
        table, expected, facts = detect_outliers(state.series, state.freq, period, state.outlier_threshold,
                                                 state.protected_index)
        state.series, applied = apply_outlier_actions(state.series, table, state.outlier_actions)
        choices = {a[0]: (a[1], a[2]) for a in state.outlier_actions}
        table["action"] = [choices.get(t.isoformat(), ("undecided", None))[0] for t in table.index]
        table["replacement"] = [choices.get(t.isoformat(), (None, None))[1] for t in table.index]
        self._table, self._expected = table, expected
        return facts | applied

    def run(self, state, llm):
        result = super().run(state, llm)
        result.extras.update(table=self._table, expected=self._expected, protected=state.protected_index)
        return result

    def decide(self, f: dict) -> Decision:
        sens = f"|z| > {f['threshold']:g}"
        skipped = (f" {f['n_protected_skipped']} points were far off but holiday-affected, so not flagged."
                   if f["n_protected_skipped"] else "")
        if not f["n_flagged"]:
            return Decision("ok", "none_found", 1.0,
                            f"Checked {f['n_checked']} values; none is far from the expected pattern ({sens})."
                            + skipped, "Nothing to clean; continue to Missing values.")
        found = (f"{f['n_flagged']} of {f['n_checked']} values look like outliers ({sens}); the largest is "
                 f"{f['largest_z']:.1f} robust standard deviations off.{skipped}")
        done = (f" Your choices: {f['n_kept']} kept, {f['n_removed']} removed, {f['n_replaced']} replaced.")
        if f["n_undecided"]:
            return Decision("warning", "flagged", 1 - f["n_undecided"] / f["n_flagged"] * 0.5,
                            found + (done if f["n_undecided"] < f["n_flagged"] else "")
                            + f" {f['n_undecided']} still undecided (kept for now).",
                            f"Decide per point below, or apply the suggestions ({f['n_suggest_replace']} to "
                            "replace, the rest to keep).")
        return Decision("ok", "handled", 1.0, found + done,
                        "Removed points become gaps that the Missing Values agent fills next."
                        if f["n_removed"] else "Continue to Missing values.")

    def prompt_facts(self, f: dict) -> dict:
        keys = ["threshold", "n_checked", "n_flagged", "n_protected_skipped", "largest_z", "n_suggest_replace",
                "n_kept", "n_removed", "n_replaced", "n_undecided"]
        return {k: f.get(k) for k in keys} | {"flagged": f.get("flagged", [])[:5]}
