"""The app's per-visitor state, without any UI: stages, settings, chat and confirmations."""

import pytest

from fcagents.chat.actions import Action
from fcagents.session import Session
from fcagents.webui import load_data


def _session(sample: str) -> Session:
    s = Session()
    load_data(s, "Sample dataset", sample, None)
    return s


def _decisions(s: Session, stage: str) -> dict[str, str]:
    return {r.agent: r.decision for r in s.ensure_stage(stage)[1]}


@pytest.mark.parametrize("sample, stage, agent, decision", [
    ("hourly_energy.csv", "data", "regularity", "regular"),
    ("irregular_sensor.csv", "data", "regularity", "irregular"),
    ("hourly_energy.csv", "resampling", "frequency", "h"),
    ("irregular_sensor.csv", "resampling", "resampling", "resampled"),
    ("daily_sales_with_gaps.csv", "missing", "missing_values", "filled"),
    ("hourly_energy.csv", "missing", "missing_values", "complete"),
    ("daily_sales_holidays.csv", "holidays", "holidays", "impact"),
    ("daily_sales_holidays.csv", "outliers", "outliers", "flagged"),
])
def test_each_step_runs_its_agents(sample, stage, agent, decision):
    assert _decisions(_session(sample), stage)[agent] == decision


def test_stage_reruns_only_when_an_input_changes():
    s = _session("hourly_energy.csv")
    first = s.ensure_stage("resampling")
    assert s.ensure_stage("resampling") is first
    s.prefs["fill_method"] = "zero"  # a later stage's setting
    assert s.ensure_stage("resampling") is first
    s.prefs["resample_freq"] = "D"
    assert s.ensure_stage("resampling") is not first


def test_chat_changes_resampling_and_too_few_points_stops():
    s = _session("irregular_sensor.csv")
    s.send("resampling", "resample to 6 hours with sum")
    assert (s.prefs["resample_freq"], s.prefs["resample_agg"]) == ("6h", "sum")
    assert _decisions(s, "resampling")["frequency"] == "6h"
    s.send("resampling", "weekly")
    assert _decisions(s, "resampling")["resampling"] == "too_few_points"


def test_chat_question_is_answered_from_stats():
    s = _session("hourly_energy.csv")
    s.send("data", "how many points are there?")
    assert "1008 points" in s.chat["data"][-1]["content"]


def test_chat_setting_survives_moving_between_steps():
    s = _session("daily_sales_with_gaps.csv")
    s.send("missing", "fill with zeros")
    s.ensure_stage("data")
    assert s.prefs["fill_method"] == "zero"
    assert "zero (your choice)" in s.ensure_stage("missing")[1][0].summary


@pytest.mark.parametrize("accept", [True, False])
def test_llm_proposal_waits_for_confirmation(accept):
    s = _session("daily_sales_with_gaps.csv")
    s.pending["missing"] = [Action("fill_method", "zero")]  # as the LLM router would leave it
    assert s.prefs["fill_method"] == "auto"
    s.resolve("missing", accept)
    assert (s.prefs["fill_method"] == "zero") == accept
    assert not s.pending
    assert s.chat["missing"][-1]["content"].startswith("Applied" if accept else "Dismissed")


def test_new_data_resets_settings_and_chat():
    s = _session("daily_sales_with_gaps.csv")
    s.send("missing", "fill with zeros")
    load_data(s, "Sample dataset", "hourly_energy.csv", None)
    assert s.prefs["fill_method"] == "auto" and not s.chat and not s.stages


def test_upload_is_keyed_by_content(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    a.write_text("ts,y\n2024-01-01,1\n2024-01-02,2\n")
    b.write_text("ts,y\n2024-01-01,5\n2024-01-02,6\n")
    s = Session()
    load_data(s, "Upload CSV", None, str(a))
    key = s.data_key
    load_data(s, "Upload CSV", None, str(b))
    assert s.data_key != key and s.df["y"].tolist() == [5, 6]
    load_data(s, "Upload CSV", None, None)
    assert s.df is None


def test_holiday_country_chat_resets_region():
    s = _session("daily_sales_holidays.csv")
    s.prefs["holiday_subdiv"] = "CA"
    s.send("holidays", "use Germany")
    assert s.prefs["holiday_country"] == "DE" and s.prefs["holiday_subdiv"] is None


def test_outliers_bulk_then_missing_fills_removed_points():
    s = _session("daily_sales_holidays.csv")
    s.send("outliers", "apply the suggestions")
    assert _decisions(s, "outliers")["outliers"] == "handled"
    assert sum(a == "replace" for _, a, _ in s.prefs["outlier_actions"]) == 6
    s.send("outliers", "remove all outliers")
    assert _decisions(s, "outliers")["outliers"] == "handled"
    assert _decisions(s, "missing")["missing_values"] == "filled"


def test_stage_status_goes_stale_when_an_earlier_step_changes():
    s = _session("hourly_energy.csv")
    s.ensure_stage("resampling")
    assert s.stage_status("resampling") in ("ok", "warning")
    assert s.stage_status("missing") == "pending"
    s.prefs["time_col"] = "something-else"  # any change to the Data step's settings
    s.ensure_stage("data")
    assert s.stage_status("resampling") == "pending"
