"""Tests for Langfuse observability: no-op mode, mocked spans/generations/scores, and PII privacy."""

from contextlib import contextmanager

import pytest

from fcagents.guardrails.checks import GuardrailConfig
from fcagents.guardrails.wrapper import GuardedLLMBackend
from fcagents.observability import langfuse as lf


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
        obs = FakeObservation(as_type, name, **kwargs)
        self.observations.append(obs)
        yield obs

    def flush(self):
        self.flushed = True


class FakeLLM:
    name = "fake-llm"
    available = True
    last_error = None

    def generate(self, system, user, max_new_tokens=120):
        return "Reply mentioning secret hf_1234567890abcdef1234 and alice@example.com."


@pytest.fixture(autouse=True)
def _reset_langfuse(monkeypatch):
    lf.reset_client()
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_ENABLED", raising=False)
    yield
    lf.reset_client()


def test_disabled_by_default_and_noop_safe():
    assert lf.is_enabled() is False
    with lf.trace_span("test:span", input="hello") as span:
        span.update(output="world")
        span.score(name="s", value=1)
    with lf.trace_generation("test:gen", model="m", input="prompt") as gen:
        gen.update(output="reply")
    lf.flush()


def test_enabled_records_spans_generations_scores_without_raw_pii(monkeypatch):
    fake_client = FakeLangfuseClient()
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test1234567890123456")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test1234567890123456")
    monkeypatch.setattr(lf, "_get_client", lambda: fake_client)

    assert lf.is_enabled() is True

    guarded = GuardedLLMBackend(FakeLLM(), config=GuardrailConfig(use_nemo=False))
    out = guarded.generate("sys", "Analyze series for bob@example.com with SSN 123-45-6789.")

    assert "[EMAIL]" in out and "[SECRET]" in out
    names = [(o.as_type, o.name) for o in fake_client.observations]
    assert names == [
        ("span", "guardrails:input"),
        ("generation", "llm:generate"),
        ("span", "guardrails:output"),
    ]

    # Verify PII scores recorded on both input and output guardrail spans
    in_obs, gen_obs, out_obs = fake_client.observations
    assert any(s["name"] == "pii_detected_input" for s in in_obs.scores)
    assert any(s["name"] == "pii_detected_output" for s in out_obs.scores)

    # Verify raw PII/secrets never appear anywhere in recorded Langfuse payloads
    serialized = repr([
        (o.kwargs, o.updates, o.scores) for o in fake_client.observations
    ])
    for sensitive in ("bob@example.com", "123-45-6789", "alice@example.com", "hf_1234567890abcdef1234"):
        assert sensitive not in serialized
