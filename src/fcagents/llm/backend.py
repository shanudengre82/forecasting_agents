"""Pluggable LLM backends. Heavy imports (torch/transformers) happen lazily."""

from __future__ import annotations

import os
from typing import Protocol

DEFAULT_MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")


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
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # float32 is the safe/fast choice on CPUs without native bf16 support.
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
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
