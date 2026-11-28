"""Tests for backend-agnostic observability: NullTracer default, custom Tracer, Langfuse adapter, UI hiding, and PII privacy."""

from contextlib import contextmanager

import pytest

from fcagents import observability as obs
from fcagents.guardrails.checks import GuardrailConfig
from fcagents.guardrails.wrapper import GuardedLLMBackend
from fcagents.observability.base import NullTracer, ObservationHandle, sanitize
from fcagents.observability.langfuse import LangfuseTracer
from fcagents.session import Session
from fcagents.webui import _agents_note


class FakeObservation:
    def __init__(self, as_type, name, **kwargs):
        self.as_type = as_type
        self.name = name
        self.kwargs = kwargs
        self.updates: list[dict] = []
        self.scores: list[dict] = []

    def update(self, **kwargs):
        self.updates.append(kwargs)

    def score(self, **kwargs):
        self.scores.append(kwargs)


class FakeLangfuseClient:
    def __init__(self):
        self.observations: list[FakeObservation] = []
        self.flushed = False

    @contextmanager
    def start_as_current_observation(self, as_type="span", name="", **kwargs):
        o = FakeObservation(as_type, name, **kwargs)
        self.observations.append(o)
        yield o

    def flush(self):
        self.flushed = True


class CustomTracer:
    name = "custom-otel"
    enabled = True
    dashboard_url = "https://traces.example.com"

    def __init__(self):
        self.events: list[tuple[str, str]] = []

    @contextmanager
    def span(self, name, *, input=None, metadata=None, session_id=None, tags=None):
        self.events.append(("span", name))
        yield ObservationHandle(None)

    @contextmanager
    def generation(self, name, *, model, input=None, model_parameters=None, metadata=None):
        self.events.append(("generation", name))
        yield ObservationHandle(None)

    def flush(self):
        return None


class FakeLLM:
    name = "fake-llm"
    available = True
    last_error = None

    def generate(self, system, user, max_new_tokens=120):
        return "Reply mentioning secret hf_1234567890abcdef1234 and alice@example.com."


@pytest.fixture(autouse=True)
def _reset_observability(monkeypatch):
    obs.reset_tracer()
    for key in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST",
                "LANGFUSE_BASE_URL", "LANGFUSE_ENABLED", "TRACING_BACKEND", "TRACING_DASHBOARD_URL"):
        monkeypatch.delenv(key, raising=False)
    yield
    obs.reset_tracer()


def test_default_is_null_tracer_and_ui_hides_dashboard():
    assert isinstance(obs.get_tracer(), NullTracer)
    assert obs.is_enabled() is False
    assert obs.dashboard_url() is None

    note = _agents_note(Session())
    assert "Langfuse" not in note
    assert "Tracing:" not in note

    # No-op context managers work without errors
    with obs.trace_span("test:span", input="hello") as span:
        span.update(output="world")
        span.score(name="s", value=1)
    with obs.trace_generation("test:gen", model="m", input="prompt") as gen:
        gen.update(output="reply")
    obs.flush()


def test_env_vars_without_langfuse_sdk_falls_back_to_null(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test1234567890123456")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test1234567890123456")
    monkeypatch.setattr("fcagents.observability.langfuse.langfuse_sdk_available", lambda: False)

    assert isinstance(obs.get_tracer(), NullTracer)
    assert obs.is_enabled() is False
    assert "Tracing:" not in _agents_note(Session())


def test_custom_tracer_is_langfuse_agnostic():
    custom = CustomTracer()
    obs.set_tracer(custom)

    assert obs.is_enabled() is True
    assert obs.tracer_name() == "custom-otel"
    assert obs.dashboard_url() == "https://traces.example.com"

    guarded = GuardedLLMBackend(FakeLLM(), config=GuardrailConfig(use_nemo=False))
    guarded.generate("sys", "Analyze the daily series.")
    assert custom.events == [
        ("span", "guardrails:input"),
        ("generation", "llm:generate"),
        ("span", "guardrails:output"),
    ]

    note = _agents_note(Session())
    assert "custom-otel" in note and "https://traces.example.com" in note
    assert "Langfuse" not in note


def test_langfuse_tracer_records_spans_generations_scores_without_raw_pii(monkeypatch):
    fake_client = FakeLangfuseClient()
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test1234567890123456")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test1234567890123456")
    monkeypatch.setenv("LANGFUSE_HOST", "http://localhost:3000")
    obs.set_tracer(LangfuseTracer(client=fake_client))

    assert obs.is_enabled() is True
    assert obs.dashboard_url() == "http://localhost:3000"
    assert "http://localhost:3000" in _agents_note(Session())

    guarded = GuardedLLMBackend(FakeLLM(), config=GuardrailConfig(use_nemo=False))
    out = guarded.generate("sys", "Analyze series for bob@example.com with SSN 123-45-6789.")

    assert "[EMAIL]" in out and "[SECRET]" in out
    names = [(o.as_type, o.name) for o in fake_client.observations]
    assert names == [
        ("span", "guardrails:input"),
        ("generation", "llm:generate"),
        ("span", "guardrails:output"),
    ]

    in_obs, _gen_obs, out_obs = fake_client.observations
    assert any(s["name"] == "pii_detected_input" for s in in_obs.scores)
    assert any(s["name"] == "pii_detected_output" for s in out_obs.scores)

    serialized = repr([(o.kwargs, o.updates, o.scores) for o in fake_client.observations])
    for sensitive in ("bob@example.com", "123-45-6789", "alice@example.com", "hf_1234567890abcdef1234"):
        assert sensitive not in serialized
    assert "[EMAIL]" in sanitize("bob@example.com")
