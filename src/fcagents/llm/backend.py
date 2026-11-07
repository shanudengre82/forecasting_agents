"""Pluggable LLM backends. Heavy imports (torch/transformers) happen lazily."""

from __future__ import annotations

import os
from typing import Protocol

DEFAULT_MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")  # local CPU model
DEFAULT_API_MODEL = os.environ.get("HF_API_MODEL", "Qwen/Qwen2.5-7B-Instruct")  # hosted model (user's token)
DEFAULT_BACKEND = os.environ.get("LLM_BACKEND", "local")  # "local" | "api"
API_PROVIDERS = ["auto", "hf-inference", "together", "novita", "fireworks-ai", "groq", "featherless-ai",
                 "cerebras", "nscale", "scaleway"]


class LLMBackend(Protocol):
    name: str
    available: bool

    def generate(self, system: str, user: str, max_new_tokens: int = 120) -> str: ...


class NullBackend:
    """No LLM: agents fall back to their deterministic summaries."""

    name = "none"
    available = False

    def generate(self, system: str, user: str, max_new_tokens: int = 120) -> str:
        return ""


class TransformersBackend:
    """Small instruct model on CPU via Hugging Face transformers."""

    available = True

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, threads: int | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if threads:
            torch.set_num_threads(threads)
        self.name = model_id
        self._torch = torch
        # token=False: never send the host's HF_TOKEN / cached login. Local models must be public.
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=False)
        # float32 is the safe/fast choice on CPUs without native bf16 support.
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32, token=False)
        self.model.eval()

    def generate(self, system: str, user: str, max_new_tokens: int = 120) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt")
        with self._torch.inference_mode():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                repetition_penalty=1.05, pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = out[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


class InferenceAPIBackend:
    """Hosted chat model via Hugging Face Inference Providers, paid for with the *user's* token.

    The token must be passed explicitly. There is deliberately no fallback to HF_TOKEN, a cached
    `hf auth login` or a Space secret, so the app owner's credentials are never used for LLM calls.
    """

    available = True

    def __init__(self, model_id: str, token: str, provider: str = "auto"):
        token = (token or "").strip()
        if not token:
            raise ValueError("A Hugging Face token from the user is required for the hosted LLM.")
        model_id = (model_id or "").strip()
        if not model_id:
            raise ValueError("A model id is required.")
        from huggingface_hub import InferenceClient

        self.name = f"hf-api:{model_id}"
        self.model_id, self.provider = model_id, provider
        self.last_error: str | None = None
        self._token = token
        self._client = InferenceClient(model=model_id, token=token, provider=provider)

    def __repr__(self) -> str:  # never leak the token in logs or tracebacks
        return f"InferenceAPIBackend(model_id={self.model_id!r}, provider={self.provider!r}, token=***)"

    def generate(self, system: str, user: str, max_new_tokens: int = 120) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            out = self._client.chat_completion(messages, max_tokens=max_new_tokens, temperature=0.0)
            self.last_error = None
            return (out.choices[0].message.content or "").strip()
        except Exception as e:  # bad token, model not served by the provider, quota…
            self.last_error = f"{type(e).__name__}: {e}".replace(self._token, "***")[:500]
            return ""
