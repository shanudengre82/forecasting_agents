"""Custom NVIDIA NeMo Guardrails actions bridging into fcagents deterministic PII and safety checks."""

from __future__ import annotations

from typing import Any

from fcagents.guardrails.checks import check_input, check_output

try:
    from nemoguardrails.actions import action
except ImportError:  # pragma: no cover - allows importing/testing actions even without nemoguardrails installed
    def action(*_args, **_kwargs):
        def _decorator(fn):
            return fn
        return _decorator


@action(is_system_action=True)
async def fcagents_input_rail(user_message: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
    """NeMo input rail action: redacts PII and checks prompt injection / length / chat topic."""
    mode = (context or {}).get("fcagents_mode", "agent")
    verdict = check_input(user_message or "", mode=mode)
    return {
        "allowed": verdict.allowed,
        "text": verdict.text,
        "reason": verdict.blocked_reason,
        "pii_counts": verdict.pii_counts,
        "flags": verdict.flags,
    }


@action(is_system_action=True)
async def fcagents_output_rail(bot_message: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
    """NeMo output rail action: redacts PII/secrets, blocks system-prompt leaks, and caps length."""
    system_prompt = (context or {}).get("fcagents_system_prompt", "")
    verdict = check_output(bot_message or "", system_prompt=system_prompt)
    return {
        "allowed": verdict.allowed,
        "text": verdict.text,
        "reason": verdict.blocked_reason,
        "pii_counts": verdict.pii_counts,
        "flags": verdict.flags,
    }
