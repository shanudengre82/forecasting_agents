"""Gradio views: every step renders for every sample, HTML is escaped, LLM backends follow the rules."""

import re
from types import SimpleNamespace

import pandas as pd
import pytest

from fcagents import webui
from fcagents.llm import backend
from fcagents.session import STAGE_ORDER, Session


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    monkeypatch.setenv("USE_LLM", "0")


def _session(sample: str) -> Session:
    s = Session()
    webui.load_data(s, "Sample dataset", sample, None)
    return s


def test_app_builds():
    demo = webui.build_app()
    assert demo.title == "Forecasting Agents"


@pytest.mark.parametrize("sample", webui.sample_names())
def test_every_step_renders(sample):
    s = _session(sample)
    for stage in STAGE_ORDER:
        out = webui.render(s, stage)
        assert set(webui.VIEW_KEYS[stage]) <= set(out)
        assert re.search(r'data-decision="[^"]+"', out["cards"])
        assert 'aria-current="step"' in out["strip"]


def test_no_data_asks_for_data():
    out = webui.render(Session(), "data")
    assert "Pick a sample dataset" in out["cards"]


def test_strip_shows_status_lamps():
    s = _session("hourly_energy.csv")
    strip = webui.render(s, "data")["strip"]
    assert "1. Data" in strip and 'aria-label="Checked"' in strip and 'aria-label="Not run yet"' in strip


def test_card_html_is_escaped():
    r = SimpleNamespace(agent="x", decision="<b>x</b>", status="ok", title="T<script>", confidence=0.5,
                        elapsed_s=0.1, summary="Uses `col` and **bold** <img src=x>", explanation="<i>llm</i>",
                        next_step="", facts={"a": "<b>"}, prompt="<p>")
    html = webui.card_html(r)
    assert "<script>" not in html and "<img" not in html and "<i>llm" not in html
    assert "<code>col</code>" in html and "<strong>bold</strong>" in html


def test_chat_messages_show_source_and_stats():
    s = _session("hourly_energy.csv")
    s.send("data", "how many points are there?")
    msgs = webui.render(s, "data")["chat"]
    assert msgs[0] == {"role": "user", "content": "how many points are there?"}
    assert "rules" in msgs[1]["content"]


def test_downloads_are_csv_files():
    s = _session("daily_sales_holidays.csv")
    out = webui.render(s, "missing")
    path = out["download"]["value"]
    frame = pd.read_csv(path)
    assert frame.columns[0] == "timestamp" and len(frame) > 100
    assert ("covariate" in out["note"]) == (len(frame.columns) > 2)


def test_outlier_editor_round_trip():
    s = _session("daily_sales_holidays.csv")
    s.ensure_stage("outliers")
    table = s.outlier_table()
    view = webui.outlier_view_table(table)
    view.loc[0, "action"] = "remove"
    view.loc[1, "action"] = "Replace"
    view.loc[1, "replacement"] = 42.0
    view.loc[2, "action"] = "nonsense"
    actions, problems = webui.outlier_actions_from_view(view, table)
    assert actions[0] == (view.loc[0, "timestamp"], "remove", None)
    assert actions[1] == (view.loc[1, "timestamp"], "replace", 42.0)
    assert len(problems) == 1 and "nonsense" in problems[0]
    s.prefs["outlier_actions"] = actions
    facts = s.ensure_stage("outliers")[1][0].facts
    assert facts["n_undecided"] == len(table) - 2


def test_replace_with_unchanged_expected_value_uses_expected():
    s = _session("daily_sales_holidays.csv")
    s.ensure_stage("outliers")
    table = s.outlier_table()
    view = webui.outlier_view_table(table)
    view.loc[0, "action"] = "Replace"
    actions, _ = webui.outlier_actions_from_view(view, table)
    assert actions == ((view.loc[0, "timestamp"], "replace", None),)


# --------------------------------------------------------------------------- language model choice


def _configure(s, kind, token="", **kw):
    args = dict(use=True, kind=kind, local_model="", url="http://x", server_model=None, api_model="org/model",
                provider="auto", token=token) | kw
    return webui.configure_llm(s, **args)


def test_hosted_llm_never_falls_back_to_the_app_token(monkeypatch):
    monkeypatch.setenv("SPACE_ID", "user/space")
    monkeypatch.setenv("HF_TOKEN", "hf_app_owner_secret")
    s = Session()
    status = _configure(s, "api")
    assert s.llm.name == "none" and "Paste your own" in status


def test_hosted_llm_uses_typed_token_and_model(monkeypatch):
    seen = []

    class StubBackend(backend.NullBackend):
        def __init__(self, model_id, token, provider="auto"):
            seen.append((model_id, token, provider))
            self.name = f"hf-api:{model_id}"

    monkeypatch.setattr(backend, "InferenceAPIBackend", StubBackend)
    monkeypatch.setenv("SPACE_ID", "user/space")
    monkeypatch.setenv("HF_TOKEN", "hf_app_owner_secret")
    s = Session()
    _configure(s, "api", token="hf_user_token", api_model="org/my-model")
    assert s.llm.name == "hf-api:org/my-model" and seen == [("org/my-model", "hf_user_token", "auto")]
    _configure(s, "api", token="hf_user_token", api_model="org/my-model")
    assert len(seen) == 1  # same settings: the session keeps its client


def test_api_backend_is_ignored_outside_a_space(monkeypatch):
    monkeypatch.delenv("SPACE_ID", raising=False)
    s = Session()
    _configure(s, "api", token="hf_user_token")
    assert s.llm.name == "none"


def test_backend_choices_local_vs_space(monkeypatch):
    monkeypatch.delenv("SPACE_ID", raising=False)
    monkeypatch.setenv("LLM_BACKEND", "api")
    assert webui.backend_choices() == ["local", "server"] and webui.default_backend() == "local"
    monkeypatch.setenv("SPACE_ID", "user/space")
    monkeypatch.delenv("SPACES_ZERO_GPU", raising=False)
    assert webui.backend_choices() == ["local", "api"]
    monkeypatch.setenv("SPACES_ZERO_GPU", "true")
    monkeypatch.setenv("LLM_BACKEND", "zerogpu")
    assert webui.backend_choices() == ["zerogpu", "api"] and webui.default_backend() == "zerogpu"


def test_local_server_uses_chosen_model(monkeypatch):
    class StubServer(backend.NullBackend):
        def __init__(self, base_url, model_id, api_key=None):
            self.name = f"server:{model_id}"

    monkeypatch.setattr(backend, "LocalServerBackend", StubServer)
    s = Session()
    _configure(s, "server", server_model="qwen2.5:0.5b")
    assert s.llm.name == "server:qwen2.5:0.5b"


def test_unreachable_local_server_falls_back_to_rules():
    s = Session()
    status = _configure(s, "server", server_model=None)
    assert s.llm.name == "none" and "Can't reach an LLM server" in status


def test_failed_local_model_falls_back_to_rules(monkeypatch):
    def boom(model_id):
        raise OSError("no network")

    monkeypatch.setattr(webui, "_local_llm", boom)
    s = Session()
    status = _configure(s, "local")
    assert s.llm.name == "none" and "no network" in status


def test_llm_off_uses_rules():
    s = Session()
    _configure(s, "server", use=False, server_model="m")
    assert s.llm.name == "none"
