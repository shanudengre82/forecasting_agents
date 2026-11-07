"""The hosted LLM backend must only ever use the token the user typed in, never the app's own."""

from types import SimpleNamespace

import huggingface_hub
import pytest

from fcagents.llm.backend import InferenceAPIBackend

USER_TOKEN = "hf_user_token_123"


class FakeClient:
    instances: list["FakeClient"] = []

    def __init__(self, model=None, token=None, provider=None):
        self.model, self.token, self.provider = model, token, provider
        self.reply: str | Exception = "  The series is hourly.  "
        FakeClient.instances.append(self)

    def chat_completion(self, messages, max_tokens=None, temperature=None):
        if isinstance(self.reply, Exception):
            raise self.reply
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.reply))])


@pytest.fixture(autouse=True)
def _fake_client(monkeypatch):
    FakeClient.instances.clear()
    monkeypatch.setattr(huggingface_hub, "InferenceClient", FakeClient)
    monkeypatch.setenv("HF_TOKEN", "hf_app_owner_secret")


@pytest.mark.parametrize("token", ["", "   ", None])
def test_requires_user_token_even_if_app_has_one(token):
    with pytest.raises(ValueError, match="token"):
        InferenceAPIBackend("some/model", token)
    assert not FakeClient.instances


def test_uses_the_user_token_not_the_environment():
    llm = InferenceAPIBackend("meta-llama/Llama-3.1-8B-Instruct", f" {USER_TOKEN} ", provider="together")
    client = FakeClient.instances[-1]
    assert client.token == USER_TOKEN
    assert (client.model, client.provider) == ("meta-llama/Llama-3.1-8B-Instruct", "together")
    assert llm.name == "hf-api:meta-llama/Llama-3.1-8B-Instruct"
    assert USER_TOKEN not in repr(llm) and USER_TOKEN not in llm.name


def test_generate_returns_stripped_text():
    llm = InferenceAPIBackend("m", USER_TOKEN)
    assert llm.generate("sys", "user") == "The series is hourly."
    assert llm.last_error is None


def test_errors_degrade_to_empty_text_without_leaking_token():
    llm = InferenceAPIBackend("m", USER_TOKEN)
    FakeClient.instances[-1].reply = RuntimeError(f"401 Unauthorized for token {USER_TOKEN}")
    assert llm.generate("sys", "user") == ""
    assert "401" in llm.last_error and USER_TOKEN not in llm.last_error
