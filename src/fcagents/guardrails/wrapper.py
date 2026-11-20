"""LLMBackend wrapper combining PII redaction, deterministic + NeMo guardrails, and Langfuse tracing."""

from __future__ import annotations

import time
from typing import Any

from fcagents.guardrails.checks import GuardrailConfig, GuardrailMode, GuardrailVerdict
from fcagents.guardrails.nemo import NemoGuardrailsRunner
from fcagents.llm.backend import LLMBackend
from fcagents.observability.langfuse import trace_generation, trace_span


class GuardedLLMBackend:
    """Wraps any ``LLMBackend`` with input/output PII redaction, guardrails (built-in + NeMo), and Langfuse traces."""

    def __init__(self, inner: LLMBackend, config: GuardrailConfig | None = None):
        self.inner = inner
        self.config = config or GuardrailConfig.from_env()
        self._runner = NemoGuardrailsRunner(inner, self.config)
        self.last_guardrail_event: dict[str, Any] | None = None

    @property
    def name(self) -> str:
        return self.inner.name

    @property
    def available(self) -> bool:
        return self.inner.available

    @property
    def last_error(self) -> str | None:
        return getattr(self.inner, "last_error", None)

    @property
    def _session_key(self):
        return getattr(self.inner, "_session_key", None)

    @_session_key.setter
    def _session_key(self, value) -> None:
        setattr(self.inner, "_session_key", value)

    @property
    def engine_label(self) -> str:
        if not self.config.enabled:
            return "off"
        return "NeMo + PII" if self._runner.active else "built-in + PII"

    def __repr__(self) -> str:
        return f"GuardedLLMBackend(inner={self.inner!r}, engine={self.engine_label!r})"

    def generate(
        self,
        system: str,
        user: str,
        max_new_tokens: int = 120,
        *,
        mode: GuardrailMode = "agent",
    ) -> str:
        if not self.available:
            return ""

        t0 = time.perf_counter()
        self.last_guardrail_event = None

        with trace_span(
            "guardrails:input",
            metadata={"mode": mode, "backend": self.name, "nemo_active": self._runner.active},
        ) as in_span:
            # Execute the guarded pipeline (input check -> LLM -> output check).
            # We trace the LLM generation separately below using only redacted I/O.
            from fcagents.guardrails.checks import check_input, check_output

            in_verdict = check_input(user, mode=mode, config=self.config)
            sys_verdict = check_input(system, mode="agent", config=self.config)
            pii_total = sum(in_verdict.pii_counts.values()) + sum(sys_verdict.pii_counts.values())
            in_span.update(
                output={
                    "allowed": in_verdict.allowed,
                    "flags": in_verdict.flags,
                    "pii_counts": in_verdict.pii_counts,
                },
                level="WARNING" if (not in_verdict.allowed or pii_total) else None,
                status_message=in_verdict.blocked_reason,
            )
            if pii_total:
                in_span.score(name="pii_detected_input", value=pii_total, comment=str(in_verdict.pii_counts))
            if not in_verdict.allowed:
                in_span.score(name="guardrail_blocked", value=1, comment=in_verdict.blocked_reason)
                self.last_guardrail_event = {
                    "phase": "input",
                    "blocked": True,
                    "reason": in_verdict.blocked_reason,
                    "flags": in_verdict.flags,
                    "pii_counts": in_verdict.pii_counts,
                }
                return ""

        redacted_system = sys_verdict.text
        redacted_user = in_verdict.text

        with trace_generation(
            "llm:generate",
            model=self.name,
            model_parameters={"max_new_tokens": max_new_tokens, "temperature": 0.0},
            input=[
                {"role": "system", "content": redacted_system},
                {"role": "user", "content": redacted_user},
            ],
            metadata={"mode": mode, "guardrail_engine": self.engine_label},
        ) as gen_obs:
            if self._runner.active:
                final_text, _in_v, out_verdict = self._runner.generate(
                    redacted_system, redacted_user, max_new_tokens=max_new_tokens, mode=mode
                )
            else:
                raw_reply = self.inner.generate(redacted_system, redacted_user, max_new_tokens=max_new_tokens)
                out_verdict = check_output(raw_reply, system_prompt=redacted_system, config=self.config)
                final_text = out_verdict.text if out_verdict.allowed else ""

            elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
            err = self.last_error
            gen_obs.update(
                output=final_text,
                metadata={"elapsed_ms": elapsed_ms, "output_flags": out_verdict.flags},
                level="ERROR" if err else None,
                status_message=err,
            )

        with trace_span(
            "guardrails:output",
            metadata={"mode": mode, "backend": self.name, "engine": out_verdict.engine},
        ) as out_span:
            out_pii_total = sum(out_verdict.pii_counts.values())
            out_span.update(
                output={
                    "allowed": out_verdict.allowed,
                    "flags": out_verdict.flags,
                    "pii_counts": out_verdict.pii_counts,
                },
                level="WARNING" if (not out_verdict.allowed or out_pii_total) else None,
                status_message=out_verdict.blocked_reason,
            )
            if out_pii_total:
                out_span.score(name="pii_detected_output", value=out_pii_total, comment=str(out_verdict.pii_counts))
            if not out_verdict.allowed:
                out_span.score(name="guardrail_blocked", value=1, comment=out_verdict.blocked_reason)
                self.last_guardrail_event = {
                    "phase": "output",
                    "blocked": True,
                    "reason": out_verdict.blocked_reason,
                    "flags": out_verdict.flags,
                    "pii_counts": out_verdict.pii_counts,
                }
                return ""

        if pii_total or out_verdict.flags:
            self.last_guardrail_event = {
                "phase": "completed",
                "blocked": False,
                "input_pii_counts": in_verdict.pii_counts,
                "output_pii_counts": out_verdict.pii_counts,
                "flags": in_verdict.flags + out_verdict.flags,
            }
        return final_text


def wrap_llm(llm: LLMBackend, config: GuardrailConfig | None = None) -> LLMBackend:
    """Wrap an available ``LLMBackend`` in ``GuardedLLMBackend`` (idempotent; leaves NullBackend untouched)."""
    if not llm.available or isinstance(llm, GuardedLLMBackend):
        return llm
    return GuardedLLMBackend(llm, config=config)
