import re
from pathlib import Path

import pytest
import streamlit as st
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
    """Decisions shown in the agent cards, as `code` words (read from the card header's data attribute)."""
    return " ".join(f"`{d}`" for h in at.get("html") for d in re.findall(r'data-decision="([^"]*)"', h.value))


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
    monkeypatch.setenv("SPACE_ID", "user/space")
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
    monkeypatch.setenv("SPACE_ID", "user/space")
    monkeypatch.setenv("HF_TOKEN", "hf_app_owner_secret")
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.sidebar.text_input(key="hf_token").set_value("hf_user_token").run()
    at.sidebar.text_input(key="api_model").set_value("org/my-model").run()
    assert not at.exception
    assert _sidebar_caption(at) == "Backend: `hf-api:org/my-model`"
    assert seen and all(tok == "hf_user_token" for _, tok, _ in seen)


def test_step_strip_shows_status_lamps():
    at = _page(_app("hourly_energy.csv"), "data")
    strip = next(h.value for h in at.get("html") if "fc-strip" in h.value)
    assert 'aria-current="step"' in strip and "1. Data" in strip
    assert 'aria-label="Checked"' in strip and 'aria-label="Not run yet"' in strip


def test_card_header_is_escaped():
    from types import SimpleNamespace

    from fcagents.ui import result_header_html

    r = SimpleNamespace(decision="<b>x</b>", status="ok", title="T<script>", confidence=0.5, elapsed_s=0.1)
    html = result_header_html(r)
    assert "<script>" not in html and "&lt;b&gt;" in html


def _backend_radio(at):
    return next(r for r in at.sidebar.radio if r.label == "Backend")


def test_local_run_offers_no_hugging_face_token_option(monkeypatch):
    monkeypatch.delenv("SPACE_ID", raising=False)
    monkeypatch.setenv("USE_LLM", "1")
    monkeypatch.setenv("LLM_BACKEND", "api")  # ignored outside a Space
    import fcagents.llm.backend as backend

    class StubLocal(backend.NullBackend):
        def __init__(self, model_id):
            self.name = model_id

    monkeypatch.setattr(backend, "TransformersBackend", StubLocal)  # don't load the real CPU model
    st.cache_resource.clear()
    at = AppTest.from_file(APP, default_timeout=30).run()
    st.cache_resource.clear()
    assert _sidebar_caption(at).startswith("Backend: `Qwen/")  # LLM_BACKEND=api fell back to the local model
    radio = _backend_radio(at)
    assert radio.options == ["Local model (in-app, CPU)", "Local LLM server (Ollama, vLLM, LM Studio)"]
    assert not [t for t in at.sidebar.text_input if t.key == "hf_token"]


def test_space_offers_hugging_face_api(monkeypatch):
    monkeypatch.setenv("SPACE_ID", "user/space")
    monkeypatch.setenv("USE_LLM", "1")
    monkeypatch.setenv("LLM_BACKEND", "api")
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert _backend_radio(at).options == ["Local model (in-app, CPU)", "Hugging Face API (your token)"]


def test_local_server_backend_uses_listed_model(monkeypatch):
    import fcagents.llm.backend as backend

    class StubServer(backend.NullBackend):
        def __init__(self, base_url, model_id, api_key=None):
            self.name = f"server:{model_id}"

    monkeypatch.delenv("SPACE_ID", raising=False)
    monkeypatch.setenv("USE_LLM", "1")
    monkeypatch.setenv("LLM_BACKEND", "server")
    monkeypatch.setenv("LOCAL_LLM_MODEL", "qwen2.5:0.5b")
    monkeypatch.setattr(backend, "list_server_models", lambda url, timeout=2.0: ["llama3.2", "qwen2.5:0.5b"])
    monkeypatch.setattr(backend, "LocalServerBackend", StubServer)
    st.cache_data.clear()
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert _sidebar_caption(at) == "Backend: `server:qwen2.5:0.5b`"


def test_unreachable_local_server_falls_back_to_rules(monkeypatch):
    import fcagents.llm.backend as backend

    monkeypatch.delenv("SPACE_ID", raising=False)
    monkeypatch.setenv("USE_LLM", "1")
    monkeypatch.setenv("LLM_BACKEND", "server")
    monkeypatch.setattr(backend, "list_server_models", lambda url, timeout=2.0: [])
    st.cache_data.clear()  # the model list is cached across reruns
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert _sidebar_caption(at) == "Backend: `none`"
    assert any("Can't reach an LLM server" in w.value for w in at.sidebar.warning)


def test_holidays_page_runs_and_country_change_resets_state():
    at = _page(_app("daily_sales_holidays.csv"), "holidays")
    assert "`impact`" in _decisions(at)
    at.chat_input[0].set_value("use Germany").run()
    assert not at.exception
    assert at.session_state["prefs"]["holiday_country"] == "DE"
    assert at.session_state["prefs"]["holiday_subdiv"] is None


def test_outliers_page_bulk_buttons_change_decision():
    at = _page(_app("daily_sales_holidays.csv"), "outliers")
    assert "`flagged`" in _decisions(at)
    at.button[next(i for i, b in enumerate(at.button) if b.label == "Apply suggestions")].click().run()
    assert not at.exception
    assert "`handled`" in _decisions(at)
    actions = at.session_state["prefs"]["outlier_actions"]
    assert len(actions) >= 6  # the six planted outliers, plus any borderline points
    assert sum(a == "replace" for _, a, _ in actions) == 6


def test_outliers_chat_removes_all_and_missing_page_fills_them():
    at = _page(_app("daily_sales_holidays.csv"), "outliers")
    at.chat_input[0].set_value("remove all outliers").run()
    assert not at.exception and "`handled`" in _decisions(at)
    _page(at, "missing")
    assert "`filled`" in _decisions(at)
