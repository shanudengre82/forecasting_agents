"""PII redaction and LLM input/output guardrails (built-in + optional NVIDIA NeMo Guardrails)."""

from fcagents.guardrails.checks import (
    GuardrailConfig,
    GuardrailMode,
    GuardrailVerdict,
    check_input,
    check_output,
)
from fcagents.guardrails.nemo import NemoGuardrailsRunner, nemo_available
from fcagents.guardrails.pii import PIIMatch, PIIScanResult, redact_pii
from fcagents.guardrails.wrapper import GuardedLLMBackend, wrap_llm

__all__ = [
    "GuardrailConfig",
    "GuardrailMode",
    "GuardrailVerdict",
    "GuardedLLMBackend",
    "NemoGuardrailsRunner",
    "PIIMatch",
    "PIIScanResult",
    "check_input",
    "check_output",
    "nemo_available",
    "redact_pii",
    "wrap_llm",
]
