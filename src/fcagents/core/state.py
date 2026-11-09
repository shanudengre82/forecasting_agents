from __future__ import annotations

from dataclasses import dataclass, field, replace

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
    resample_freq: str = "auto"  # user preferences, set from page controls or the chat
    resample_agg: str = "mean"
    fill_method: str = "auto"
    holiday_country: str = "US"
    holiday_subdiv: str | None = None  # state / region code, None = whole country
    holiday_types: tuple[str, ...] = ("official",)  # official and/or unofficial
    holiday_window: int = 7  # days before and after each holiday to check
    outlier_threshold: float = 3.5
    outlier_actions: tuple[tuple[str, str, float | None], ...] = ()  # (timestamp ISO, keep|remove|replace, value)
    protected_index: pd.DatetimeIndex | None = None  # holiday-affected timestamps: never flagged as outliers
    covariates: pd.DataFrame | None = None  # suggested holiday covariates, aligned to ``series``
    stopped: str | None = None  # why the pipeline stopped; later stages skip every agent with this reason
    findings: dict[str, AgentResult] = field(default_factory=dict)

    def fork(self, **prefs) -> PipelineState:
        """Copy for the next stage. Series are replaced, never mutated, so a shallow copy is enough."""
        return replace(self, findings=dict(self.findings), **prefs)

    def fact(self, agent: str, key: str, default=None):
        res = self.findings.get(agent)
        return res.facts.get(key, default) if res else default
