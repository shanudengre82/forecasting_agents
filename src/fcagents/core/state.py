from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from fcagents.core.agent import AgentResult


@dataclass
class PipelineState:
    """Shared blackboard passed from agent to agent."""

    raw: pd.DataFrame
    time_col: str | None = None
    time_method: str | None = None
    target_col: str | None = None
    id_col: str | None = None
    series_id: str | None = None
    series: pd.Series | None = None  # target values indexed by parsed timestamps (one series)
    freq: str | None = None  # fixed frequency of ``series`` once known (detected or resampled)
    resample_freq: str = "auto"  # user preferences for the resampling agent
    resample_agg: str = "mean"
    resample_fill: str = "none"
    findings: dict[str, AgentResult] = field(default_factory=dict)

    def fact(self, agent: str, key: str, default=None):
        res = self.findings.get(agent)
        return res.facts.get(key, default) if res else default
