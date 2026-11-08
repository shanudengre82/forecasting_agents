from pathlib import Path

import pandas as pd
import pytest

from fcagents.orchestrator import run_pipeline
from fcagents.tools.holidays import (
    countries,
    holiday_calendar,
    holiday_covariates,
    holiday_impact,
    protected_index,
    subdivisions,
)

SAMPLES = Path(__file__).resolve().parents[1] / "data" / "samples"


def _sales() -> pd.Series:
    df = pd.read_csv(SAMPLES / "daily_sales_holidays.csv")
    return pd.Series(df["sales"].to_numpy(), index=pd.to_datetime(df["date"]))


def _results(name: str, **prefs) -> dict:
    return {r.agent: r for r, _ in run_pipeline(pd.read_csv(SAMPLES / name), **prefs)}


def test_calendar_types_country_and_state():
    official = holiday_calendar("US", None, ("official",), "2024-01-01", "2024-12-31")
    both = holiday_calendar("US", None, ("official", "unofficial"), "2024-01-01", "2024-12-31")
    assert "Christmas Day" in official.to_numpy() and "Valentine's Day" not in official.to_numpy()
    assert "Valentine's Day" in both.to_numpy()
    california = holiday_calendar("US", "CA", ("official",), "2024-01-01", "2024-12-31")
    assert len(california) > len(official)  # Cesar Chavez Day etc.
    assert holiday_calendar("US", None, (), "2024-01-01", "2024-12-31").empty
    assert countries()["DE"] == "Germany" and subdivisions("US")["CA"] == "California"


def test_detects_planted_holiday_effects():
    s = _sales()
    cal = holiday_calendar("US", None, ("official",), s.index[0], s.index[-1])
    facts, extras = holiday_impact(s, "D", cal, 7)
    assert facts["overall_significant"] and facts["p_day0"] < 0.05
    assert {"Christmas Day", "Thanksgiving Day", "Independence Day"} <= set(facts["holidays_with_effect"])
    window = {(e["holiday"], e["offset"]) for e in facts["window_effects"]}
    assert ("Thanksgiving Day", 1) in window  # Black Friday
    assert {("Christmas Day", -2), ("Christmas Day", -3)} <= window  # shopping run-up
    cov = holiday_covariates(s.index, cal, extras["effects"])
    assert cov.index.equals(s.index)
    assert cov.loc["2024-11-29", "holiday_thanksgiving_day_p1"] == 1  # the day after, not before
    assert cov.loc["2024-12-23", "holiday_christmas_day_m2"] == 1
    assert cov["is_holiday"].sum() == len(cal)


@pytest.mark.parametrize("name", ["daily_sales_with_gaps.csv", "business_day_stock.csv"])
def test_no_effect_where_none_was_planted(name):
    res = _results(name)["holidays"]
    assert res.decision == "no_impact" and not res.facts["holidays_with_effect"] and not res.facts["covariates"]


def test_coarse_and_short_series():
    assert _results("monthly_passengers.csv")["holidays"].decision == "not_applicable"
    assert _results("hourly_energy.csv")["holidays"].decision == "too_few_holidays"


def test_agent_protects_affected_days_and_window_setting():
    res = _results("daily_sales_holidays.csv")
    h = res["holidays"]
    assert h.decision == "impact" and "is_holiday" in h.facts["covariates"]
    assert "2024-12-25" in h.facts["protected_dates"] and "2024-11-29" in h.facts["protected_dates"]
    narrow = _results("daily_sales_holidays.csv", holiday_window=0)["holidays"]
    assert narrow.facts["window_effects"] == [] and narrow.decision == "impact"


def test_protected_index_maps_days_to_timestamps():
    idx = pd.date_range("2024-12-24", periods=72, freq="h")
    assert len(protected_index(idx, ["2024-12-25"])) == 24
    assert len(protected_index(idx, [])) == 0
