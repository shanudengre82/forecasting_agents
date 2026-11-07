from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    monkeypatch.setenv("USE_LLM", "0")


def _app(sample: str) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    next(s for s in at.sidebar.selectbox if s.label == "Sample").set_value(sample).run()
    assert not at.exception
    return at


def _decisions(at) -> str:
    return " ".join(m.value for m in at.markdown if m.value.startswith("**Decision:**"))


def _page(at, name: str) -> AppTest:
    at.switch_page(f"views/{name}.py").run()
    assert not at.exception
    return at


def test_language_model_is_first_in_sidebar():
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert at.sidebar.header[0].value == "Language model"


@pytest.mark.parametrize("sample, page, expected", [
    ("hourly_energy.csv", "data", "`regular`"),
    ("irregular_sensor.csv", "data", "`irregular`"),
    ("hourly_energy.csv", "resampling", "`h`"),
    ("irregular_sensor.csv", "resampling", "`resampled`"),
    ("daily_sales_with_gaps.csv", "missing", "`filled`"),
    ("hourly_energy.csv", "missing", "`complete`"),
])
def test_each_page_runs_its_agents(sample, page, expected):
    at = _page(_app(sample), page)
    assert expected in _decisions(at)


def test_chat_changes_resampling_and_warns_on_too_few_points():
    at = _page(_app("irregular_sensor.csv"), "resampling")
    at.chat_input[0].set_value("resample to 6 hours with sum").run()
    assert not at.exception
    assert "`6h`" in _decisions(at)
    controls = {s.label: s.value for s in at.main.selectbox}
    assert controls["Target frequency (1h – 1W)"] == "6h" and controls["Aggregate values per bin"] == "sum"
    at.chat_input[0].set_value("weekly").run()
    assert at.warning and "finer frequency" in at.warning[0].value


def test_chat_question_is_answered_from_stats():
    at = _page(_app("hourly_energy.csv"), "data")
    at.chat_input[0].set_value("how many points are there?").run()
    assert not at.exception
    assert any("1008 points" in m.value for m in at.markdown)


def test_chat_sets_fill_method_and_choice_survives_page_switch():
    at = _page(_app("daily_sales_with_gaps.csv"), "missing")
    at.chat_input[0].set_value("fill with zeros").run()
    assert any("zero (your choice)" in m.value for m in at.markdown)
    _page(at, "data")
    _page(at, "missing")
    assert any("zero (your choice)" in m.value for m in at.markdown)


def test_agents_page_lists_every_agent():
    from fcagents.orchestrator import PIPELINE

    at = _page(AppTest.from_file(APP, default_timeout=30).run(), "agents")
    headers = " ".join(h.value for h in at.subheader)
    for a in PIPELINE:
        assert a.title in headers


@pytest.mark.parametrize("button, applied", [("apply_missing", True), ("dismiss_missing", False)])
def test_llm_proposal_waits_for_confirmation(button, applied):
    from fcagents.chat.actions import Action

    at = _page(_app("daily_sales_with_gaps.csv"), "missing")
    at.session_state["chat"]["missing"] = []
    at.session_state["pending"]["missing"] = [Action("fill_method", "zero")]  # as the LLM router would leave it
    at.run()
    assert not any("zero (your choice)" in m.value for m in at.markdown)  # nothing applied yet
    at.button(key=button).click().run()
    assert not at.exception
    assert any("zero (your choice)" in m.value for m in at.markdown) == applied
    assert not at.session_state["pending"]


def _sidebar_caption(at) -> str:
    return next(c.value for c in at.sidebar.caption if c.value.startswith("Backend:"))


def test_hosted_llm_never_falls_back_to_the_app_token(monkeypatch):
    monkeypatch.setenv("USE_LLM", "1")
    monkeypatch.setenv("LLM_BACKEND", "api")
    monkeypatch.setenv("HF_TOKEN", "hf_app_owner_secret")
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert _sidebar_caption(at) == "Backend: `none`"
    assert any("Paste your own" in i.value for i in at.sidebar.info)


def test_hosted_llm_uses_typed_token_and_model(monkeypatch):
    import fcagents.llm.backend as backend

    seen = []

    class StubBackend(backend.NullBackend):
        def __init__(self, model_id, token, provider="auto"):
            seen.append((model_id, token, provider))
            self.name = f"hf-api:{model_id}"

    monkeypatch.setattr(backend, "InferenceAPIBackend", StubBackend)
    monkeypatch.setenv("USE_LLM", "1")
    monkeypatch.setenv("LLM_BACKEND", "api")
    monkeypatch.setenv("HF_TOKEN", "hf_app_owner_secret")
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.sidebar.text_input(key="hf_token").set_value("hf_user_token").run()
    at.sidebar.text_input(key="api_model").set_value("org/my-model").run()
    assert not at.exception
    assert _sidebar_caption(at) == "Backend: `hf-api:org/my-model`"
    assert seen and all(tok == "hf_user_token" for _, tok, _ in seen)
