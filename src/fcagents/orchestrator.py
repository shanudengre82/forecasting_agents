"""Runs agents in sequence with rule-based routing, yielding results so the UI can stream them."""

from __future__ import annotations

from collections.abc import Iterator

import pandas as pd

from fcagents.agents.frequency_agent import FrequencyAgent
from fcagents.agents.regularity_agent import RegularityAgent
from fcagents.agents.resampling_agent import ResamplingAgent
from fcagents.agents.time_index_agent import TimeIndexAgent
from fcagents.core.agent import AgentResult, BaseAgent, skipped
from fcagents.core.state import PipelineState
from fcagents.llm.backend import LLMBackend, NullBackend

PIPELINE: list[BaseAgent] = [TimeIndexAgent(), RegularityAgent(), ResamplingAgent(), FrequencyAgent()]


def _route(agent: BaseAgent, result: AgentResult) -> str | None:
    """Return a reason to stop the pipeline after ``result``, or None to continue."""
    if result.status == "error":
        return f"Stopped: {agent.title} could not proceed."
    if agent.name == "resampling" and result.decision == "too_few_points":
        return "Skipped: too few data points after resampling; choose a finer frequency first."
    return None


def run_pipeline(
    df: pd.DataFrame,
    llm: LLMBackend | None = None,
    time_col: str | None = None,
    target_col: str | None = None,
    series_id: str | None = None,
    resample_freq: str = "auto",
    resample_agg: str = "mean",
    resample_fill: str = "none",
) -> Iterator[tuple[AgentResult, PipelineState]]:
    llm = llm or NullBackend()
    state = PipelineState(raw=df, time_col=time_col, target_col=target_col, series_id=series_id,
                          resample_freq=resample_freq, resample_agg=resample_agg, resample_fill=resample_fill)
    stop_reason = None
    for agent in PIPELINE:
        if stop_reason:
            result = skipped(agent, stop_reason)
        elif skip_reason := agent.applies(state):
            result = skipped(agent, skip_reason)
        else:
            result = agent.run(state, llm)
            stop_reason = _route(agent, result)
        yield result, state
