"""The hosted LLM backend must only ever use the token the user typed in, never the app's own."""

from types import SimpleNamespace

import huggingface_hub
import pytest

from fcagents.llm.backend import InferenceAPIBackend, LocalServerBackend, list_server_models

USER_TOKEN = "hf_user_token_123"


class FakeClient:
    instances: list["FakeClient"] = []

    def __init__(self, model=None, token=None, provider=None, base_url=None, api_key=None):
        self.model, self.token, self.provider = model, token, provider
        self.base_url, self.api_key = base_url, api_key
        self.calls = []
        self.reply: str | Exception = "  The series is hourly.  "
        FakeClient.instances.append(self)

    def chat_completion(self, messages, max_tokens=None, temperature=None, model=None):
        self.calls.append(model)
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


def test_local_server_never_sends_the_host_hf_token():
    llm = LocalServerBackend("http://localhost:11434/v1/", "qwen2.5:0.5b")
    client = FakeClient.instances[-1]
    assert client.base_url == "http://localhost:11434/v1"
    assert client.api_key == "local" and client.token is None
    assert llm.generate("sys", "user") == "The series is hourly."
    assert client.calls == ["qwen2.5:0.5b"] and llm.name == "server:qwen2.5:0.5b"


def test_local_server_errors_degrade_to_empty_text():
    llm = LocalServerBackend("http://localhost:11434/v1", "m")
    FakeClient.instances[-1].reply = ConnectionError("refused")
    assert llm.generate("sys", "user") == "" and "refused" in llm.last_error


def test_list_server_models_handles_unreachable_and_non_http_urls():
    assert list_server_models("http://127.0.0.1:9/v1", timeout=0.5) == []
    assert list_server_models("file:///etc/passwd") == []
