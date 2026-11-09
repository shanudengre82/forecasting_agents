"""One visitor's app state: data, preferences, cached stage results and chat. No UI framework in here."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from fcagents.chat import respond
from fcagents.chat.actions import DEFAULTS, HOLIDAY_TYPE_CHOICES, PAGE_KEYS, Action
from fcagents.llm.backend import LLMBackend, NullBackend
from fcagents.orchestrator import STAGES, initial_state, run_stage
from fcagents.tools.holidays import holiday_covariates
from fcagents.tools.outliers import bulk_actions
from fcagents.tools.time_index import detect_time_column

STAGE_ORDER = list(STAGES)
# Settings each stage's agents read. The outliers chat sets "outlier_bulk", which becomes outlier_actions.
STAGE_PREFS = {**PAGE_KEYS, "outliers": ("outlier_threshold", "outlier_actions")}
SOURCE_LABELS = {"rule": "rules", "llm-action": "LLM proposal, validated by rules; confirm with Apply",
                 "llm-answer": "LLM, from computed facts only"}


def native(key: str, value, ctx: dict):
    """Chat values are strings; convert them to what the pipeline stores."""
    if key == "series_id" and value is not None:  # ids may be ints in the data
        return next((s for s in ctx.get("series_ids", []) if str(s) == str(value)), value)
    if key == "holiday_types" and isinstance(value, str):
        return HOLIDAY_TYPE_CHOICES[value]
    if key == "holiday_window" and value is not None:
        return int(value)
    if key == "outlier_threshold" and value is not None:
        return float(value)
    return value


def default_prefs() -> dict:
    out = {k: native(k, v, {}) for k, v in DEFAULTS.items() if k != "outlier_bulk"}
    return out | {"outlier_actions": ()}


@dataclass
class Session:
    df: pd.DataFrame | None = None
    data_key: str = ""
    llm: LLMBackend = field(default_factory=NullBackend)
    prefs: dict = field(default_factory=default_prefs)
    stages: dict = field(default_factory=dict)  # stage -> (key, results, state)
    chat: dict = field(default_factory=dict)  # stage -> [{"role", "content", "source"?, "stats"?}]
    pending: dict = field(default_factory=dict)  # stage -> [Action] proposed by the LLM, awaiting Apply
    _ctx: tuple | None = None  # (data_key, data context)

    # ----------------------------------------------------------------- data

    def set_data(self, df: pd.DataFrame | None, data_key: str) -> None:
        """A new dataset resets preferences, cached stages and chats."""
        if data_key != self.data_key:
            self.data_key, self.prefs, self.stages, self.chat, self.pending = data_key, default_prefs(), {}, {}, {}
            self._ctx = None
        self.df = df

    def data_context(self) -> dict:
        """What the data offers, for the Data page controls and the chat."""
        if self.df is None:
            return {"time_cols": [], "target_cols": [], "series_ids": [], "id_col": None, "ambiguous": False}
        if self._ctx is None or self._ctx[0] != self.data_key:
            rep = detect_time_column(self.df)
            ids = self.df[rep["id_col"]].value_counts().index.tolist() if rep["id_col"] else []
            self._ctx = (self.data_key, {"time_cols": [c["column"] for c in rep["candidates"]],
                                         "target_cols": rep["target_cols"], "series_ids": ids,
                                         "id_col": rep["id_col"], "ambiguous": rep["ambiguous"]})
        return self._ctx[1]

    # ----------------------------------------------------------------- stages

    def ensure_stage(self, stage: str) -> tuple[tuple, list, object]:
        """(key, results, state) for ``stage``; upstream stages run first. Re-runs only when an input changed."""
        i = STAGE_ORDER.index(stage)
        if i == 0:
            up_key, up_state = self.data_key, initial_state(self.df)
        else:
            up_key, _, up_state = self.ensure_stage(STAGE_ORDER[i - 1])
        stage_prefs = {k: self.prefs[k] for k in STAGE_PREFS[stage]}
        key = (up_key, tuple(stage_prefs.items()), self.llm.name)
        cached = self.stages.get(stage)
        if cached and cached[0] == key:
            return cached
        results = []
        for r, state in run_stage(stage, up_state, self.llm, **stage_prefs):
            results.append(r)
        self.stages[stage] = (key, results, state)
        return self.stages[stage]

    def stage_status(self, stage: str) -> str:
        """ok | warning | error | skipped | pending (not run yet, or stale because an earlier step changed)."""
        entry = self.stages.get(stage)
        i = STAGE_ORDER.index(stage)
        if entry is None or (i and self.stages.get(STAGE_ORDER[i - 1], (None,))[0] != entry[0][0]):
            return "pending"
        statuses = {r.status for r in entry[1]}
        return next((s for s in ("error", "warning", "ok") if s in statuses), "skipped")

    def stage_result(self, stage: str, agent: str):
        entry = self.stages.get(stage)
        return next((r for r in entry[1] if r.agent == agent and r.status != "skipped"), None) if entry else None

    def outlier_table(self) -> pd.DataFrame | None:
        r = self.stage_result("outliers", "outliers")
        return r.extras.get("table") if r else None

    def final_covariates(self, index: pd.DatetimeIndex) -> pd.DataFrame | None:
        """Suggested holiday covariates for ``index`` (e.g. the filled series), or None if none were suggested."""
        r = self.stage_result("holidays", "holidays")
        if r is None or r.extras.get("covariates") is None:
            return None
        return holiday_covariates(index, r.extras["calendar"], r.extras["effects"])

    # ----------------------------------------------------------------- chat

    def page_context(self, stage: str) -> dict:
        ctx = self.data_context()
        if stage == "holidays":
            from fcagents.tools.holidays import subdivisions

            ctx = ctx | {"subdivisions": subdivisions(self.prefs["holiday_country"]),
                         "holiday_country": self.prefs["holiday_country"]}
        return ctx

    def send(self, stage: str, text: str) -> None:
        """Chat on a page: commands change this page's settings, questions are answered from facts."""
        text = (text or "").strip()
        if not text or self.df is None:
            return
        _, results, state = self.ensure_stage(stage)
        agents = {a.name: a for a in STAGES[stage]}
        facts = {r.agent: agents[r.agent].prompt_facts(r.facts) for r in results if r.status != "skipped"}
        ctx = self.page_context(stage)
        reply = respond(text, stage, ctx, state.series, state.freq, facts, self.llm)
        if reply.source == "llm-action":
            self.pending[stage] = reply.actions  # applied only after the user confirms
        else:
            self.pending.pop(stage, None)
            self.apply(reply.actions, ctx)
        self.chat.setdefault(stage, []).extend([
            {"role": "user", "content": text},
            {"role": "assistant", "content": reply.text, "source": reply.source, "stats": reply.stats}])

    def resolve(self, stage: str, accept: bool) -> None:
        actions = self.pending.pop(stage, [])
        if not actions:
            return
        if accept:
            self.apply(actions, self.page_context(stage))
        text = ("Applied: " + "; ".join(a.describe() for a in actions) + ".") if accept else "Dismissed; nothing changed."
        self.chat.setdefault(stage, []).append({"role": "assistant", "content": text, "source": "rule"})

    def apply(self, actions: list[Action], ctx: dict) -> None:
        for a in actions:
            if a.key == "outlier_bulk":  # one choice for every point flagged right now
                self.ensure_stage("outliers")
                table = self.outlier_table()
                self.prefs["outlier_actions"] = () if a.value is None or table is None else bulk_actions(table, a.value)
            else:
                self.prefs[a.key] = native(a.key, a.value, ctx)
