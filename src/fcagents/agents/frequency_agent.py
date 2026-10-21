from __future__ import annotations

import pandas as pd

from fcagents.core.agent import BaseAgent, Decision
from fcagents.core.state import PipelineState
from fcagents.tools.frequency import describe_frequency


class FrequencyAgent(BaseAgent):
    name = "frequency"
    title = "Frequency Agent"
    task = "Name the sampling frequency of a regular time series and its likely seasonal periods."
    scope = "Seasonal periods listed are only candidates to test, not confirmed seasonality."

    def run_tools(self, state: PipelineState) -> dict:
        return describe_frequency(state.series.index, state.freq)

    def run(self, state, llm):
        result = super().run(state, llm)
        idx = state.series.index.dropna().unique().sort_values()
        result.extras["missing_index"] = pd.date_range(idx[0], idx[-1], freq=result.facts["freq"]).difference(idx)
        return result

    def decide(self, f: dict) -> Decision:
        periods = ", ".join(map(str, f["seasonal_periods"])) or "none (too coarse)"
        summary = (f"The series is {f['label']} (pandas alias '{f['freq']}'). Expected {f['expected_points']} "
                   f"points, observed {f['observed_points']} (coverage {f['coverage']:.1%}). "
                   f"Candidate seasonal periods: {periods}.")
        test = f"test seasonality at periods {periods}" if f["seasonal_periods"] else "check for trend"
        if f["n_missing"]:
            longest = f["gaps"][0]
            summary += f" Longest gap: {longest['length']} steps from {longest['from']}."
            return Decision("warning", f["freq"], f["coverage"], summary,
                            f"Impute the {f['n_missing']} missing timestamps, then {test}.")
        return Decision("ok", f["freq"], 1.0, summary, test[0].upper() + test[1:] + ".")

    def prompt_facts(self, f: dict) -> dict:
        keys = ["freq", "label", "seasonal_periods", "expected_points", "observed_points", "n_missing", "coverage"]
        return {k: f.get(k) for k in keys} | {"longest_gaps": f["gaps"][:2]}
