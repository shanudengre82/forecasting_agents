from __future__ import annotations

from fcagents.core.agent import BaseAgent, Decision
from fcagents.core.state import PipelineState
from fcagents.tools.holidays import (
    EFFECT_Z,
    HOLIDAY_TYPES,
    MIN_OCCURRENCES,
    countries,
    holiday_calendar,
    holiday_covariates,
    holiday_impact,
    protected_index,
)

MIN_OCCURRENCES_OVERALL = 10  # an overall shift alone (no single holiday stands out) needs this many holidays


class HolidaysAgent(BaseAgent):
    name = "holidays"
    title = "Holidays Agent"
    task = "Check whether public or other holidays move the series, and suggest holiday covariates if they do."
    scope = ("Only discuss holidays and their effect. Holiday covariates are suggestions for a forecasting model; "
             "do not claim they are already in the data.")

    summary = ("Compares each holiday, and the days around it, with a same-weekday baseline to find out whether "
               "holidays move the series. If they do, it suggests holiday covariates for the forecasting model "
               "and tells the Outliers agent to leave those days alone.")
    runs_when = "When a fixed frequency of one day or finer is known and the series has values (after Frequency)."
    inputs = ("Fixed-frequency series (daily or finer)",
              "Country and state, official and/or unofficial holidays, days around each holiday (page or chat)")
    outputs = ("Holidays in the data and whether they move the series (overall test and per holiday)",
               "Effects on the days before and after a holiday (e.g. a shopping run-up)",
               "Suggested 0/1 covariates: is_holiday plus one column per detected effect (CSV download)",
               "Holiday-affected days, which the Outliers agent never flags")
    decisions = {
        "impact": "Holidays move the series. Use the suggested holiday covariates in the forecasting model.",
        "window_impact": "The holidays themselves look normal, but the days around them move. Use the "
                         "before/after covariates.",
        "no_impact": "No holiday stands out from ordinary days; holiday covariates are not needed.",
        "too_few_holidays": f"Fewer than {MIN_OCCURRENCES} holidays fall inside the data, too few to judge.",
        "not_applicable": "The frequency is coarser than a day (weeks, months), so single days can't be seen.",
    }
    user_controls = ("Country (default United States) and state / region",
                     "Holiday types: " + ", ".join(HOLIDAY_TYPES.values()),
                     "Days before and after each holiday to check (0–14, default 7)")
    tools = ("tools.holidays.holiday_calendar", "tools.holidays.holiday_impact",
             "tools.holidays.holiday_covariates")

    def applies(self, state: PipelineState) -> str | None:
        if state.freq is None or state.series is None:
            return "Skipped: no fixed frequency is known yet."
        if state.series.isna().all():
            return "Skipped: the series has timestamps only, no values to compare."
        return None

    def run_tools(self, state: PipelineState) -> dict:
        idx = state.series.index
        cal = holiday_calendar(state.holiday_country, state.holiday_subdiv, state.holiday_types, idx.min(), idx.max())
        facts, extras = holiday_impact(state.series, state.freq, cal, state.holiday_window,
                                       agg=state.fact("resampling", "agg"))
        facts |= {"country": state.holiday_country, "country_name": countries().get(state.holiday_country),
                  "subdivision": state.holiday_subdiv, "holiday_types": list(state.holiday_types)}
        self._calendar, self._extras = cal, extras
        effects = extras.get("effects")
        if facts.get("applicable") and effects is not None and len(effects):
            state.covariates = holiday_covariates(idx, cal, effects)
        else:
            state.covariates = None
        state.protected_index = protected_index(idx, facts.get("protected_dates", []))
        facts["covariates"] = [] if state.covariates is None else list(state.covariates.columns)
        facts["n_protected"] = int(len(state.protected_index))
        return facts

    def run(self, state, llm):
        result = super().run(state, llm)
        result.extras.update(self._extras, calendar=self._calendar, covariates=state.covariates)
        return result

    def decide(self, f: dict) -> Decision:
        where = f["country_name"] or f["country"]
        if f["subdivision"]:
            where += f" ({f['subdivision']})"
        types = " and ".join(f["holiday_types"]) or "no"
        if not f.get("applicable"):
            return Decision("ok", "not_applicable", 1.0,
                            "Holiday effects need daily or finer data; this series is coarser.",
                            "Holiday covariates are not needed at this frequency.")
        n = f["n_occurrences"]
        if n < MIN_OCCURRENCES:
            return Decision("ok", "too_few_holidays", 0.5,
                            f"Only {n} {types} holidays in {where} fall inside the data, too few to judge.",
                            "Add more history, include unofficial holidays, or check the country.")
        named, window = f["holidays_with_effect"], f["window_effects"]
        found = (f"Checked {n} {types} holidays in {where}, with {f['window']} days before and after each. "
                 f"On holidays the series is on average {f['effect_day0_abs_z']:.1f} robust standard deviations "
                 f"from normal (p = {f['p_day0']:.3f}).")
        cov = f"Suggested covariates: {', '.join(f['covariates'])}."
        if named or (f["overall_significant"] and n >= MIN_OCCURRENCES_OVERALL):
            moved = f" Holidays that move it every year: {', '.join(named)}." if named else ""
            around = (f" {len(window)} effects on days around holidays, e.g. "
                      f"{self._offset_text(window[0])}." if window else "")
            confidence = min(1.0, 0.6 + 0.05 * f["n_effects"])
            return Decision("ok", "impact", confidence, found + moved + around,
                            cov + " Holiday-affected days won't be flagged as outliers." if f["covariates"]
                            else "Add an is_holiday covariate to the forecasting model.")
        if window:
            return Decision("ok", "window_impact", 0.7,
                            found + f" The days around holidays move, e.g. {self._offset_text(window[0])}.", cov)
        return Decision("ok", "no_impact", 1 - min(f["p_day0"], 1) * 0.5 if f["p_day0"] is not None else 0.8,
                        found + f" No holiday moves it by {EFFECT_Z:g} or more every year.",
                        "Holiday covariates are not needed.")

    @staticmethod
    def _offset_text(e: dict) -> str:
        o = e["offset"]
        when = f"{abs(o)} day{'s' if abs(o) > 1 else ''} {'before' if o < 0 else 'after'}"
        return f"{when} {e['holiday']} ({'up' if e['mean_z'] > 0 else 'down'})"

    def prompt_facts(self, f: dict) -> dict:
        keys = ["country_name", "subdivision", "holiday_types", "window", "n_occurrences", "effect_day0_abs_z",
                "p_day0", "holidays_with_effect", "top_holidays", "covariates"]
        out = {k: f.get(k) for k in keys}
        out["window_effects"] = (f.get("window_effects") or [])[:5]
        return out
