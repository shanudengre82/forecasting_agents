"""The app's own model on a Hugging Face ZeroGPU Space.

app.py imports this module at startup, before anything else touches torch: ZeroGPU needs at least one
``@spaces.GPU`` function registered at startup, and weights loaded at module scope (moved to "cuda" eagerly;
ZeroGPU streams them onto a GPU for each decorated call). Each call uses the visitor's ZeroGPU quota.
"""

from __future__ import annotations

import spaces  # must be imported before torch

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from fcagents.llm.backend import DEFAULT_MODEL_ID

MODEL_ID = DEFAULT_MODEL_ID
LOAD_ERROR: str | None = None
try:
    # token=False: never send the Space's HF_TOKEN. The model must be public.
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=False)
    _model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, token=False).to("cuda")
    _model.eval()
except Exception as e:  # no network, unknown model…: the app still runs, with rule-based summaries
    LOAD_ERROR = f"{type(e).__name__}: {e}"[:500]


@spaces.GPU(duration=30)
def _generate(system: str, user: str, max_new_tokens: int) -> str:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    text = _tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = _tokenizer(text, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        out = _model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.05,
                              pad_token_id=_tokenizer.eos_token_id)
    return _tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


class ZeroGPUBackend:
    available = True
    name = f"zerogpu:{MODEL_ID}"

    def __init__(self):
        self.last_error: str | None = None

    def __deepcopy__(self, memo):  # shared, holds no per-visitor data
        return self

    def generate(self, system: str, user: str, max_new_tokens: int = 120) -> str:
        try:
            text = _generate(system, user, max_new_tokens)
            self.last_error = None
            return text
        except Exception as e:  # GPU quota used up, queue timeout…
            self.last_error = f"{type(e).__name__}: {e}"[:500]
            return ""


def backend() -> ZeroGPUBackend:
    if LOAD_ERROR:
        raise RuntimeError(f"{MODEL_ID} did not load: {LOAD_ERROR}")
    return ZeroGPUBackend()
