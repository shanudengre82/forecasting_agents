"""NVIDIA NeMo Guardrails runner with custom LLMBackend adapter and graceful fallback."""

from __future__ import annotations

import asyncio
import concurrent.futures
import importlib.util
from pathlib import Path
from typing import Any

from fcagents.guardrails.checks import GuardrailConfig, GuardrailVerdict, check_input, check_output
from fcagents.llm.backend import LLMBackend

NEMO_CONFIG_DIR = Path(__file__).resolve().parent / "nemo_config"


def nemo_available() -> bool:
    """True when the optional ``nemoguardrails`` package is installed."""
    return importlib.util.find_spec("nemoguardrails") is not None


def _run_coro_sync(coro):
    """Run an async coroutine from sync code even when an event loop is already running (e.g. Gradio)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def _build_langchain_adapter(backend: LLMBackend, system_holder: list[str], max_tokens_holder: list[int]):
    """Build a minimal LangChain BaseLLM wrapper around ``fcagents.llm.backend.LLMBackend``."""
    from langchain_core.language_models.llms import LLM

    class FcAgentsLangChainLLM(LLM):
        @property
        def _llm_type(self) -> str:
            return "fcagents_llm"

        def _call(self, prompt: str, stop: list[str] | None = None, run_manager: Any = None, **kwargs: Any) -> str:
            return backend.generate(system_holder[0], prompt, max_new_tokens=max_tokens_holder[0])

    return FcAgentsLangChainLLM()


class NemoGuardrailsRunner:
    """Runs LLM calls through NVIDIA NeMo Guardrails Colang rails, falling back cleanly if unavailable."""

    def __init__(self, backend: LLMBackend, config: GuardrailConfig | None = None):
        self.backend = backend
        self.config = config or GuardrailConfig.from_env()
        self._system_holder = [""]
        self._max_tokens_holder = [120]
        self._rails = None
        self._init_error: str | None = None
        self.active = False

        if self.config.use_nemo and nemo_available():
            self._init_rails()

    def _init_rails(self) -> None:
        try:
            from nemoguardrails import LLMRails, RailsConfig
            from nemoguardrails.llm.helpers import get_llm_instance_wrapper
            from nemoguardrails.llm.providers import register_llm_provider

            adapter = _build_langchain_adapter(self.backend, self._system_holder, self._max_tokens_holder)
            provider = get_llm_instance_wrapper(llm_instance=adapter, llm_type="fcagents_llm")
            register_llm_provider("fcagents_llm", provider)

            rails_config = RailsConfig.from_path(str(NEMO_CONFIG_DIR))
            self._rails = LLMRails(rails_config)
            self.active = True
        except Exception as e:  # incompatible langchain/nemo version, missing optional dep…
            self._init_error = f"{type(e).__name__}: {e}"[:300]
            self._rails = None
            self.active = False

    def generate(
        self,
        system: str,
        user: str,
        max_new_tokens: int = 120,
        *,
        mode: str = "agent",
    ) -> tuple[str, GuardrailVerdict, GuardrailVerdict]:
        """Run input guardrails, LLM generation (via NeMo rails when active), and output guardrails."""
        in_verdict = check_input(user, mode=mode, config=self.config)
        sys_verdict = check_input(system, mode="agent", config=self.config)
        if not in_verdict.allowed:
            return "", in_verdict, GuardrailVerdict(allowed=True, text="", engine=in_verdict.engine)

        redacted_system = sys_verdict.text
        redacted_user = in_verdict.text

        raw_reply = ""
        engine = "builtin"
        if self.active and self._rails is not None:
            engine = "nemo"
            self._system_holder[0] = redacted_system
            self._max_tokens_holder[0] = max_new_tokens
            try:
                messages = [
                    {"role": "context", "content": {"fcagents_mode": mode, "fcagents_system_prompt": redacted_system}},
                    {"role": "user", "content": redacted_user},
                ]
                res = _run_coro_sync(self._rails.generate_async(messages=messages))
                raw_reply = (res.get("content", "") if isinstance(res, dict) else str(res or "")).strip()
            except Exception:
                # Never break forecasting explanations if NeMo's runtime raises; fall back to direct guarded call.
                engine = "builtin"
                raw_reply = self.backend.generate(redacted_system, redacted_user, max_new_tokens=max_new_tokens)
        else:
            raw_reply = self.backend.generate(redacted_system, redacted_user, max_new_tokens=max_new_tokens)

        in_verdict.engine = engine
        out_verdict = check_output(raw_reply, system_prompt=redacted_system, config=self.config)
        out_verdict.engine = engine
        final_text = out_verdict.text if out_verdict.allowed else ""
        return final_text, in_verdict, out_verdict
