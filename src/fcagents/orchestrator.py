"""Runs agents in stages (one per app page) with rule-based routing, yielding results so the UI can stream them."""

from __future__ import annotations

from collections.abc import Iterator

import pandas as pd

from fcagents.agents.frequency_agent import FrequencyAgent
from fcagents.agents.holidays_agent import HolidaysAgent
from fcagents.agents.missing_values_agent import MissingValuesAgent
from fcagents.agents.outliers_agent import OutliersAgent
from fcagents.agents.regularity_agent import RegularityAgent
from fcagents.agents.resampling_agent import ResamplingAgent
from fcagents.agents.time_index_agent import TimeIndexAgent
from fcagents.core.agent import AgentResult, BaseAgent, skipped
from fcagents.core.state import PipelineState
from fcagents.llm.backend import LLMBackend, NullBackend

STAGES: dict[str, list[BaseAgent]] = {
    "data": [TimeIndexAgent(), RegularityAgent()],
    "resampling": [ResamplingAgent(), FrequencyAgent()],
    "holidays": [HolidaysAgent()],
    "outliers": [OutliersAgent()],
    "missing": [MissingValuesAgent()],
}
STAGE_TITLES = {"data": "Data", "resampling": "Resampling", "holidays": "Holidays", "outliers": "Outliers",
                "missing": "Missing values"}
PIPELINE: list[BaseAgent] = [a for agents in STAGES.values() for a in agents]


def _route(agent: BaseAgent, result: AgentResult) -> str | None:
    """Return a reason to stop the pipeline after ``result``, or None to continue."""
    if result.status == "error":
        return f"Stopped: {agent.title} could not proceed."
    if agent.name == "resampling" and result.decision == "too_few_points":
        return "Skipped: too few data points after resampling; choose a finer frequency first."
    return None


def initial_state(df: pd.DataFrame, time_col: str | None = None, target_col: str | None = None,
                  series_id: str | None = None) -> PipelineState:
    return PipelineState(raw=df, time_col=time_col, target_col=target_col, series_id=series_id)


def run_stage(stage: str, state: PipelineState, llm: LLMBackend | None = None,
              **prefs) -> Iterator[tuple[AgentResult, PipelineState]]:
    """Run one stage on a fork of ``state`` (the input is left untouched, so a page can re-run its stage)."""
    llm = llm or NullBackend()
    state = state.fork(**prefs)
    for agent in STAGES[stage]:
        if state.stopped:
            result = skipped(agent, state.stopped)
        elif skip_reason := agent.applies(state):
            result = skipped(agent, skip_reason)
        else:
            result = agent.run(state, llm)
            state.stopped = _route(agent, result)
        yield result, state


def run_pipeline(
    df: pd.DataFrame,
    llm: LLMBackend | None = None,
    time_col: str | None = None,
    target_col: str | None = None,
    series_id: str | None = None,
    resample_freq: str = "auto",
    resample_agg: str = "mean",
    fill_method: str = "auto",
    **more_prefs,
) -> Iterator[tuple[AgentResult, PipelineState]]:
    """All stages in one go (tests, scripts and the container smoke test). ``more_prefs``: holiday_country,
    holiday_subdiv, holiday_types, holiday_window, outlier_threshold, outlier_actions."""
    state = initial_state(df, time_col, target_col, series_id)
    prefs = {"resample_freq": resample_freq, "resample_agg": resample_agg, "fill_method": fill_method,
             **more_prefs}
    for stage in STAGES:
        for result, state in run_stage(stage, state, llm, **prefs):
            yield result, state
