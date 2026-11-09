from __future__ import annotations

from dataclasses import dataclass, field, replace

import pandas as pd

from fcagents.core.agent import AgentResult

# Preferences a series (child) state inherits from the dataset (parent) state when a stage starts.
CHILD_PREFS = ("resample_freq", "resample_agg", "fill_method", "holiday_country", "holiday_subdiv", "holiday_types",
               "holiday_window", "outlier_threshold")


@dataclass
class PipelineState:
    """Shared blackboard passed from agent to agent.

    The dataset-level state (Time Index, Target and Covariates agents) holds one child state per series to
    forecast in ``children``; every per-series agent runs on each child. A child holds exactly one series.
    """

    raw: pd.DataFrame
    time_col: str | None = None
    time_method: str | None = None
    parsed_time: pd.Series | None = None  # raw[time_col] parsed to datetimes, aligned to raw's rows
    target_col: str | None = None  # child: the column this series comes from
    id_col: str | None = None
    series_id: str | None = None  # child: the id of this series in long-format data
    key: str | None = None  # child: name of this series in the app ("store_1", "sales", "sales · store_1")
    series: pd.Series | None = None  # child: target values indexed by parsed timestamps
    freq: str | None = None  # child: fixed frequency of ``series`` once known (detected or resampled)
    horizon: pd.DatetimeIndex | None = None  # child: future timestamps (rows after the last observed target)
    # user preferences, set from the app's controls or the chat
    targets: tuple[str, ...] = ()  # value columns to forecast; () = let the Target agent choose
    series_ids: tuple | None = None  # long format: series to forecast; None = all
    covariate_overrides: tuple[tuple[str, str], ...] = ()  # (column, past | future | static | ignore)
    resample_freq: str = "auto"
    resample_agg: str = "mean"
    fill_method: str = "auto"
    holiday_country: str = "US"
    holiday_subdiv: str | None = None  # state / region code, None = whole country
    holiday_types: tuple[str, ...] = ("official",)  # official and/or unofficial
    holiday_window: int = 7  # days before and after each holiday to check
    outlier_threshold: float = 3.5
    # (series key, timestamp ISO, keep|remove|replace, value). On a child: (timestamp ISO, action, value).
    # 3-tuples given to the dataset state apply to every series.
    outlier_actions: tuple[tuple, ...] = ()
    # results
    covariate_roles: dict[str, str] = field(default_factory=dict)  # column -> past | future | static
    horizon_rows: pd.Series | None = None  # bool per raw row: after the last observed target of its series
    protected_index: pd.DatetimeIndex | None = None  # holiday-affected timestamps: never flagged as outliers
    covariates: pd.DataFrame | None = None  # suggested holiday covariates, aligned to ``series``
    stopped: str | None = None  # why the pipeline stopped; later stages skip every agent with this reason
    findings: dict[str, AgentResult] = field(default_factory=dict)
    children: dict[str, PipelineState] = field(default_factory=dict)

    def fork(self, **prefs) -> PipelineState:
        """Copy for the next stage. Series are replaced, never mutated, so a shallow copy is enough."""
        new = replace(self, findings=dict(self.findings), **prefs)
        child_prefs = {k: v for k, v in prefs.items() if k in CHILD_PREFS}
        new.children = {}
        for key, child in self.children.items():
            extra = {}
            if "outlier_actions" in prefs:
                extra["outlier_actions"] = tuple(
                    a if len(a) == 3 else a[1:] for a in prefs["outlier_actions"] if len(a) == 3 or a[0] == key)
            new.children[key] = child.fork(**child_prefs, **extra)
        return new

    def fact(self, agent: str, key: str, default=None):
        res = self.findings.get(agent)
        return res.facts.get(key, default) if res else default

    def child(self, key: str | None = None) -> PipelineState | None:
        """The series ``key``, or the first one."""
        if not self.children:
            return None
        return self.children.get(key) or next(iter(self.children.values()))
