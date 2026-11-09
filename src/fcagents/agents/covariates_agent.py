from __future__ import annotations

from fcagents.core.agent import BaseAgent, Decision
from fcagents.core.state import PipelineState
from fcagents.tools.covariates import ROLES, classify_covariates
from fcagents.tools.time_index import detect_time_column


class CovariatesAgent(BaseAgent):
    name = "covariates"
    title = "Covariates Agent"
    task = "Decide which other columns can help the forecast, and whether each is a past, future or static covariate."
    scope = ("Only discuss the covariate roles and their reasons. Past covariates are known only up to now, future "
             "covariates are known in advance, static ones are fixed per series. Do NOT discuss frequency or gaps.")

    summary = ("Sorts every column that is not the time, id or target into past covariates (known only up to "
               "now), future covariates (known in advance: values on future rows, calendar or planned columns such "
               "as prices and promotions), static covariates (one value per series) or unused.")
    runs_when = "After the targets are chosen."
    inputs = ("Columns other than time, series id and targets", "Forecast-horizon rows from the Target agent",
              "Your role per column")
    outputs = ("Role and reason per column", "Lists of past, future and static covariates")
    decisions = {
        "found": "At least one column can be used as a covariate.",
        "none": "No other usable column; the model uses the target's own history (plus holiday covariates).",
    }
    user_controls = ("Role per column (past / future / static / not used)",)
    tools = ("tools.covariates.classify_covariates", "tools.covariates.covariate_frame")

    def applies(self, state: PipelineState) -> str | None:
        return None if state.children else "Skipped: no series to forecast."

    def run_tools(self, state: PipelineState) -> dict:
        df = state.raw
        targets = {c.target_col for c in state.children.values()}
        dates = {c["column"] for c in detect_time_column(df)["candidates"]} - {state.time_col}
        columns = [str(c) for c in df.columns if str(c) not in {state.time_col, state.id_col} | targets]
        rows = classify_covariates(df, columns, state.id_col, state.horizon_rows, dates,
                                   dict(state.covariate_overrides))
        state.covariate_roles = {r["column"]: r["role"] for r in rows if r["role"] != "ignore"}
        by_role = {role: [r["column"] for r in rows if r["role"] == role] for role in ROLES}
        return {"columns": rows, "past": by_role["past"], "future": by_role["future"], "static": by_role["static"],
                "ignored": by_role["ignore"], "n_overridden": sum(r["source"] == "user" for r in rows),
                "n_horizon_rows": int(state.horizon_rows.sum()) if state.horizon_rows is not None else 0}

    def decide(self, f: dict) -> Decision:
        def names(cols):
            return ", ".join(f"'{c}'" for c in cols) or "none"

        if not (f["past"] or f["future"] or f["static"]):
            why = f" Not used: {names(f['ignored'])}." if f["ignored"] else ""
            return Decision("ok", "none", 1.0, "No other column can serve as a covariate." + why,
                            "The model will learn from the target's own history (plus holiday covariates).")
        parts = [f"Past covariates: {names(f['past'])}.", f"Future covariates: {names(f['future'])}.",
                 f"Static covariates: {names(f['static'])}."]
        if f["ignored"]:
            parts.append(f"Not used: {names(f['ignored'])}.")
        if f["n_overridden"]:
            parts.append(f"{f['n_overridden']} role(s) set by you.")
        step = ("Future covariates must be supplied for every step you forecast."
                if f["future"] else "Past covariates only help models that use lagged inputs.")
        return Decision("ok", "found", 1.0, " ".join(parts), step)

    def prompt_facts(self, f: dict) -> dict:
        return {k: f.get(k) for k in ("past", "future", "static", "ignored", "n_horizon_rows")} | {
            "reasons": {r["column"]: r["reason"] for r in f.get("columns", [])[:10]}}
