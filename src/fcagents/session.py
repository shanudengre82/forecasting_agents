"""One visitor's app state: data, preferences, cached stage results and chat. No UI framework in here."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from fcagents.chat import respond
from fcagents.chat.actions import DEFAULTS, HOLIDAY_TYPE_CHOICES, PAGE_KEYS, Action
from fcagents.core.multi import series_prompt_facts
from fcagents.llm.backend import LLMBackend, NullBackend
from fcagents.orchestrator import STAGES, initial_state, run_stage
from fcagents.tools.covariates import covariate_frame
from fcagents.tools.holidays import holiday_covariates
from fcagents.tools.outliers import bulk_actions
from fcagents.tools.time_index import detect_time_column

STAGE_ORDER = list(STAGES)
# Settings each stage's agents read (the chat's page keys are translated by Session.apply).
STAGE_PREFS = {"data": ("time_col", "targets", "series_ids", "covariate_overrides"),
               "resampling": ("resample_freq", "resample_agg"),
               "holidays": ("holiday_country", "holiday_subdiv", "holiday_types", "holiday_window"),
               "outliers": ("outlier_threshold", "outlier_actions"),
               "missing": ("fill_method",)}
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
    out = {k: native(k, v, {}) for k, v in DEFAULTS.items()
           if k not in ("outlier_bulk", "target_col", "targets", "series_id", "series_scope")}
    return out | {"targets": (), "series_ids": None, "covariate_overrides": (), "outlier_actions": (), "focus": None}


def series_result(result, key: str | None):
    """The result of one series: a combined result's per-series entry, or the result itself (one series)."""
    if result is None:
        return None
    per = result.extras.get("per_series")
    if per:
        return per.get(key) or next(iter(per.values()))
    return result


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
        """What the data offers, for the Data controls and the chat."""
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

    def series_keys(self, stage: str = "data") -> list[str]:
        entry = self.ensure_stage(stage) if self.df is not None else None
        return list(entry[2].children) if entry else []

    def focus(self, state=None) -> str | None:
        """The series shown in detail: the user's pick if it is forecast, else the first series."""
        keys = list(state.children) if state is not None else self.series_keys()
        return self.prefs["focus"] if self.prefs["focus"] in keys else (keys[0] if keys else None)

    def stage_result(self, stage: str, agent: str, key: str | None = None):
        """``agent``'s result for series ``key`` (the focus series by default), or None if it did not run."""
        entry = self.stages.get(stage)
        if not entry:
            return None
        combined = next((r for r in entry[1] if r.agent == agent), None)
        r = series_result(combined, key or self.focus(entry[2]))
        return r if r is not None and r.status != "skipped" else None

    def outlier_table(self, key: str | None = None) -> pd.DataFrame | None:
        r = self.stage_result("outliers", "outliers", key)
        return r.extras.get("table") if r else None

    def set_outlier_actions(self, key: str, actions: tuple) -> None:
        """Replace the choices for series ``key``; other series keep theirs."""
        others = tuple(a for a in self.prefs["outlier_actions"] if a[0] != key)
        self.prefs["outlier_actions"] = others + tuple((key, *a) for a in actions)

    def outlier_actions(self, key: str) -> tuple:
        return tuple(a[1:] for a in self.prefs["outlier_actions"] if a[0] == key)

    def bulk_outliers(self, choice: str | None) -> None:
        """One choice for every flagged point of every series (None clears all choices)."""
        if choice is None:
            self.prefs["outlier_actions"] = ()
            return
        _, _, state = self.ensure_stage("outliers")
        out = []
        for key in state.children:
            table = self.outlier_table(key)
            if table is not None:
                out += [(key, *a) for a in bulk_actions(table, choice)]
        self.prefs["outlier_actions"] = tuple(out)

    def export_frame(self) -> pd.DataFrame | None:
        """Every series after the last step, long format, with covariates and the forecast-horizon rows."""
        _, results, state = self.ensure_stage("missing")
        _, data_results, data_state = self.ensure_stage("data")
        frames = []
        for key, child in state.children.items():
            if child.series is None or child.freq is None or child.stopped:
                continue
            index = child.series.index
            cov = covariate_frame(data_state.raw, data_state.parsed_time, data_state.id_col, child.series_id,
                                  data_state.covariate_roles, child.freq, index, child.horizon)
            frame = pd.DataFrame({"value": child.series}).reindex(cov.index).join(cov)
            hol = self.stage_result("holidays", "holidays", key)
            if hol is not None and hol.extras.get("covariates") is not None:
                frame = frame.join(holiday_covariates(frame.index, hol.extras["calendar"], hol.extras["effects"]))
            frame = frame.rename_axis("timestamp").reset_index()
            frame.insert(0, "series", key)
            frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else None

    # ----------------------------------------------------------------- chat

    def page_context(self, stage: str) -> dict:
        ctx = self.data_context()
        if stage == "data" and self.df is not None:
            entry = self.stages.get("data")
            ctx = ctx | {"targets": list(self.prefs["targets"]) or (
                [entry[2].findings["target"].facts["targets"][0]] if entry and "target" in entry[2].findings
                and entry[2].findings["target"].facts.get("targets") else [])}
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
        facts = {r.agent: series_prompt_facts(agents[r.agent], r) for r in results if r.status != "skipped"}
        child = state.child(self.focus(state))
        ctx = self.page_context(stage)
        reply = respond(text, stage, ctx, child.series if child else None, child.freq if child else None, facts,
                        self.llm)
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
            if a.key == "outlier_bulk":
                self.bulk_outliers(a.value)
            elif a.key == "target_col":
                self.prefs["targets"] = () if a.value is None else (a.value,)
            elif a.key == "targets":
                self.prefs["targets"] = () if a.value is None else tuple(a.value.split(","))
            elif a.key == "series_scope":
                self.prefs["series_ids"] = None
            elif a.key == "series_id":
                sid = native("series_id", a.value, ctx)
                keys = self.series_keys()
                match = next((k for k in keys if k == str(sid) or k.endswith(f" · {sid}")), None)
                if match is None and sid is not None:  # not forecast yet: add it
                    current = self.prefs["series_ids"] or ()
                    self.prefs["series_ids"] = (*current, sid) if current else (sid,)
                    match = str(sid)
                self.prefs["focus"] = match
            else:
                self.prefs[a.key] = native(a.key, a.value, ctx)
