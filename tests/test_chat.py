import numpy as np
import pandas as pd
import pytest

from fcagents.chat import respond
from fcagents.chat.actions import Action, validate
from fcagents.chat.llm_router import parse_router_reply
from fcagents.chat.parser import parse_command
from fcagents.chat.qa import series_stats, template_answer

CTX = {"time_cols": ["ds"], "target_cols": ["y", "temp"], "series_ids": ["store_1", "store_2"]}


@pytest.mark.parametrize("page, text, expected", [
    ("resampling", "resample to 6 hours with sum", {"resample_freq": "6h", "resample_agg": "sum"}),
    ("resampling", "daily", {"resample_freq": "D"}),
    ("resampling", "24 hours, average", {"resample_freq": "D", "resample_agg": "mean"}),
    ("resampling", "use the max per 2 days", {"resample_freq": "2D", "resample_agg": "max"}),
    ("resampling", "could you resample to weekly?", {"resample_freq": "W"}),
    ("resampling", "take the last value every 12h", {"resample_freq": "12h", "resample_agg": "last"}),
    ("resampling", "reset", {"resample_freq": "auto", "resample_agg": "mean"}),
    ("missing", "fill with linear interpolation", {"fill_method": "interpolate"}),
    ("missing", "forward fill the gaps", {"fill_method": "ffill"}),
    ("missing", "leave them missing", {"fill_method": "none"}),
    ("missing", "use the recommended method", {"fill_method": "auto"}),
    ("missing", "copy the value from the previous period", {"fill_method": "seasonal"}),
    ("missing", "put nothing in the holes", {"fill_method": "none"}),
    ("data", "use temp as target", {"target_col": "temp"}),
    ("data", "switch to store_2", {"series_id": "store_2"}),
])
def test_parser_commands(page, text, expected):
    res = parse_command(text, page, CTX)
    assert not res.is_question and not res.errors
    assert {a.key: a.value for a in res.actions} == expected


@pytest.mark.parametrize("text", ["monthly please", "aggregate to 15 min", "5 hours"])
def test_parser_refuses_unsupported_frequencies(text):
    res = parse_command(text, "resampling", CTX)
    assert not res.actions and res.errors


@pytest.mark.parametrize("text", ["how many gaps?", "What is the mean", "show the frequency"])
def test_questions_are_not_commands(text):
    assert parse_command(text, "resampling", CTX).is_question


def test_validate_scopes_actions_to_page():
    assert validate("resampling", Action("resample_freq", "6h"), CTX) is None
    assert validate("missing", Action("resample_freq", "6h"), CTX)
    assert validate("resampling", Action("resample_freq", "MS"), CTX)


@pytest.mark.parametrize("reply, expected", [
    ('{"set": "resample_freq", "value": "4h"}', Action("resample_freq", "4h")),
    ('Sure! {"set": "resample_agg", "value": "median"} done', Action("resample_agg", "median")),
    ('{"set": "resample_freq", "value": "MS"}', None),  # outside 1h–1W
    ('{"set": "fill_method", "value": "zero"}', None),  # not this page
    ('{"set": "question"}', None),
    ("I think you should resample", None),
    ("{not json}", None),
])
def test_llm_router_reply_is_validated(reply, expected):
    assert parse_router_reply(reply, "resampling", CTX) == expected


class FakeLLM:
    name, available = "fake", True

    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    def generate(self, system, user, max_new_tokens=120):
        self.prompts.append(user)
        return self.reply


def _series():
    idx = pd.date_range("2024-01-01", periods=10, freq="D").delete([4])
    return pd.Series(np.arange(9.0), index=idx)


def test_stats_and_templates():
    st = series_stats(_series(), "D")
    assert st["n_points"] == 9 and st["n_missing_timestamps"] == 1 and st["max"] == 8.0
    assert "8.0" in template_answer("what is the highest value?", st, {})
    assert "1 missing timestamps" in template_answer("how many gaps are there?", st, {})
    filled = {"n_filled": 3, "method_label": "zero"}
    assert template_answer("how many values were filled?", st, filled) == "3 values were filled with zero."


def test_respond_rule_action_without_llm():
    reply = respond("resample to 6h", "resampling", CTX, _series(), "D", {})
    assert reply.actions == [Action("resample_freq", "6h")] and reply.source == "rule"


def test_respond_uses_llm_only_when_rules_fail():
    llm = FakeLLM('{"set": "resample_agg", "value": "median"}')
    reply = respond("go for the middle value per bin", "resampling", CTX, _series(), "D", {}, llm)
    assert reply.actions == [Action("resample_agg", "median")] and reply.source == "llm-action"
    respond("resample to 6h", "resampling", CTX, _series(), "D", {}, llm)
    assert len(llm.prompts) == 1  # rules handled the second prompt


def test_respond_free_question_grounded_in_stats():
    llm = FakeLLM("It looks like a steady upward trend.")
    facts = {"frequency": {"freq": "D"}}
    reply = respond("does this look like a trend?", "resampling", CTX, _series(), "D", facts, llm)
    assert reply.source == "llm-answer" and reply.text.startswith("It looks")
    assert '"n_points": 9' in llm.prompts[0] and '"freq": "D"' in llm.prompts[0]


def test_respond_without_llm_shows_stats():
    reply = respond("does this look like a trend?", "resampling", CTX, _series(), "D", {})
    assert reply.stats and reply.stats["n_points"] == 9 and not reply.actions


@pytest.mark.slow
def test_real_llm_chat():
    from fcagents.llm.backend import TransformersBackend

    llm = TransformersBackend()
    s = _series()
    reply = respond("Is the series long enough to see a weekly pattern? Explain briefly.", "resampling", CTX, s, "D",
                    {}, llm)
    assert reply.source == "llm-answer" and len(reply.text) > 20
    print("\nanswer:", reply.text)
    proposal = respond("bins should keep their middle reading", "resampling", CTX, s, "D", {}, llm)
    print("router:", proposal)
    for a in proposal.actions:  # whatever the model proposes must have passed validation
        assert validate("resampling", a, CTX) is None


@pytest.mark.parametrize("text, expected", [
    ("use Germany", {"holiday_country": "DE", "holiday_subdiv": None}),
    ("switch to the UK", {"holiday_country": "GB", "holiday_subdiv": None}),
    ("only California", {"holiday_subdiv": "CA"}),
    ("include unofficial holidays", {"holiday_types": "both"}),
    ("official only", {"holiday_types": "official"}),
    ("check 3 days around holidays", {"holiday_window": "3"}),
    ("all states", {"holiday_subdiv": None}),
])
def test_holiday_commands(text, expected):
    from fcagents.tools.holidays import subdivisions

    ctx = {"subdivisions": subdivisions("US"), "holiday_country": "US"}
    res = parse_command(text, "holidays", ctx)
    assert {a.key: a.value for a in res.actions} == expected
    assert all(validate("holidays", a, ctx) is None for a in res.actions)


@pytest.mark.parametrize("text, expected", [
    ("apply the suggestions", {"outlier_bulk": "suggested"}),
    ("remove all outliers", {"outlier_bulk": "remove"}),
    ("keep them all", {"outlier_bulk": "keep"}),
    ("replace all points", {"outlier_bulk": "replace"}),
    ("be more sensitive", {"outlier_threshold": "3.0"}),
    ("set the threshold to 5", {"outlier_threshold": "5.0"}),
])
def test_outlier_commands(text, expected):
    res = parse_command(text, "outliers", {})
    assert {a.key: a.value for a in res.actions} == expected


def test_holiday_window_and_threshold_limits():
    assert parse_command("check 20 days", "holidays", {}).errors
    assert parse_command("threshold 4", "outliers", {}).errors
