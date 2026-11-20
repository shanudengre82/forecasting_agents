"""Deterministic input and output guardrails for LLM prompts, chat messages, and completions."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Literal

from fcagents.guardrails.pii import redact_pii

GuardrailMode = Literal["agent", "chat_router", "chat_qa", "chat_user"]


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


@dataclass(frozen=True)
class GuardrailConfig:
    enabled: bool = True
    use_nemo: bool = True
    max_input_chars: int = 12000
    max_chat_chars: int = 1500
    max_output_chars: int = 2000
    enforce_topic_in_chat: bool = True

    @classmethod
    def from_env(cls) -> GuardrailConfig:
        return cls(
            enabled=_env_bool("GUARDRAILS_ENABLED", True),
            use_nemo=_env_bool("NEMO_GUARDRAILS_ENABLED", True),
            max_input_chars=_env_int("GUARDRAILS_MAX_INPUT_CHARS", 12000),
            max_chat_chars=_env_int("GUARDRAILS_MAX_CHAT_CHARS", 1500),
            max_output_chars=_env_int("GUARDRAILS_MAX_OUTPUT_CHARS", 2000),
            enforce_topic_in_chat=_env_bool("GUARDRAILS_ENFORCE_TOPIC", True),
        )


@dataclass
class GuardrailVerdict:
    allowed: bool
    text: str
    blocked_reason: str | None = None
    pii_counts: dict[str, int] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    engine: str = "builtin"


_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(?:ignore|disregard|forget)\s+(?:all\s+|any\s+)?(?:previous|prior|above|earlier)\s+(?:instructions|prompts|rules|directions)\b", re.I),
    re.compile(r"\b(?:reveal|show|print|repeat|leak|output|dump)\s+(?:me\s+)?(?:your\s+|the\s+)?(?:system\s+prompt|hidden\s+instructions|initial\s+instructions)\b", re.I),
    re.compile(r"\b(?:override|bypass|disable)\s+(?:all\s+)?(?:safety|guardrails|rules|restrictions|filters)\b", re.I),
    re.compile(r"\b(?:you\s+are\s+now|act\s+as|pretend\s+to\s+be|switch\s+to)\s+(?:dan\b|developer\s+mode|jailbroken|an?\s+unrestricted)\b", re.I),
    re.compile(r"(?:<\|im_start\|>|<\|im_end\|>|<\|system\|>|\[INST\]|\[/INST\]|<<SYS>>)", re.I),
]

# Forecasting / time-series domain signals (if present, chat message is on-topic).
_DOMAIN_SIGNALS = re.compile(
    r"\b(?:series|time\s*series|forecast\w*|predict\w*|trend|season\w*|freq\w*|resampl\w*|aggregat\w*|"
    r"bin\w*|interpolat\w*|imput\w*|fill\w*|missing|gap\w*|nan|null|empty|outlier\w*|anomal\w*|spike\w*|"
    r"holiday\w*|calendar|country|state|region|subdiv\w*|covariate\w*|target\w*|column\w*|col\b|"
    r"timestamp\w*|date\w*|time|hour\w*|day\w*|daily|week\w*|month\w*|year\w*|period\w*|grid|regular\w*|"
    r"irregular|mean|average|avg|median|sum|max\w*|min\w*|std|variance|count|point\w*|row\w*|data\w*|"
    r"dataset|csv|chart|plot|graph|value\w*|number\w*|threshold|sensitiv\w*|z-?score|keep|remove|replace|"
    r"apply|reset|switch|store|sales|demand|temp\w*|energy|stock|passenger\w*|sensor)\b",
    re.I,
)

# Clearly off-topic / unsafe intents when no domain signal is present.
_OFF_TOPIC_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(?:recipe|cook|bake|ingredients?\s+for)\b", re.I),
    re.compile(r"\b(?:write|compose|tell)\s+(?:me\s+)?(?:a\s+)?(?:poem|story|song|haiku|joke|essay|novel)\b", re.I),
    re.compile(r"\b(?:hack|exploit|malware|ransomware|phishing|sql\s+injection|ddos|keylogger)\b", re.I),
    re.compile(r"\b(?:medical\s+advice|diagnos\w*|prescription|legal\s+advice|lawsuit|tax\s+evasion)\b", re.I),
    re.compile(r"\b(?:stock\s+tip|buy\s+crypto|investment\s+advice|gambl\w*)\b", re.I),
    re.compile(r"\b(?:capital\s+of|who\s+won\s+the\s+(?:world\s+cup|election)|president\s+of)\b", re.I),
    re.compile(r"\b(?:write|create|build)\s+(?:a\s+)?(?:python|javascript|java|c\+\+|html|css|bash)\s+(?:script|program|app|website|bot)\b", re.I),
]

_HTML_TAG = re.compile(r"<(?:script|iframe|object|embed)\b[^>]*>.*?</(?:script|iframe|object|embed)>", re.I | re.S)

REFUSAL_INJECTION = (
    "I can't follow instructions that try to override my guidelines or inspect system prompts. "
    "Ask me about the current time series or pipeline settings instead."
)
REFUSAL_OFF_TOPIC = (
    "I can only help with the time series in this pipeline (data, resampling, holidays, outliers, "
    "and missing values)."
)
REFUSAL_LENGTH = "That message is too long for the assistant. Please shorten it and try again."
REFUSAL_LEAK = "The model response was withheld because it echoed internal instructions."


def _has_prompt_injection(text: str) -> bool:
    return any(p.search(text) for p in _INJECTION_PATTERNS)


def _is_off_topic(text: str) -> bool:
    if _DOMAIN_SIGNALS.search(text):
        return False
    return any(p.search(text) for p in _OFF_TOPIC_PATTERNS)


def check_input(
    text: str,
    *,
    mode: GuardrailMode = "agent",
    config: GuardrailConfig | None = None,
) -> GuardrailVerdict:
    """Redact PII and apply input guardrails (length, prompt injection, and chat topic scoping)."""
    cfg = config or GuardrailConfig.from_env()
    if not cfg.enabled:
        return GuardrailVerdict(allowed=True, text=text, engine="disabled")

    flags: list[str] = []
    pii = redact_pii(text)
    redacted = pii.redacted_text
    if pii.has_pii:
        flags.append("pii_redacted")

    limit = cfg.max_chat_chars if mode == "chat_user" else cfg.max_input_chars
    if len(redacted) > limit:
        flags.append("length_exceeded")
        return GuardrailVerdict(
            allowed=False,
            text=REFUSAL_LENGTH,
            blocked_reason=f"Input length {len(redacted)} exceeds limit {limit}.",
            pii_counts=pii.counts,
            flags=flags,
        )

    if _has_prompt_injection(redacted):
        flags.append("prompt_injection")
        return GuardrailVerdict(
            allowed=False,
            text=REFUSAL_INJECTION,
            blocked_reason="Prompt injection or jailbreak pattern detected.",
            pii_counts=pii.counts,
            flags=flags,
        )

    if mode == "chat_user" and cfg.enforce_topic_in_chat and _is_off_topic(redacted):
        flags.append("off_topic")
        return GuardrailVerdict(
            allowed=False,
            text=REFUSAL_OFF_TOPIC,
            blocked_reason="Query is outside the time-series forecasting domain.",
            pii_counts=pii.counts,
            flags=flags,
        )

    return GuardrailVerdict(allowed=True, text=redacted, pii_counts=pii.counts, flags=flags)


def _leaked_system_prompt(output: str, system_prompt: str) -> bool:
    if not system_prompt or len(system_prompt) < 40:
        return False
    # Check for verbatim echo of a substantial slice of the system prompt.
    snippet = system_prompt.strip()[:60].lower()
    return snippet in output.lower()


def check_output(
    text: str,
    *,
    system_prompt: str = "",
    config: GuardrailConfig | None = None,
) -> GuardrailVerdict:
    """Redact PII/secrets, block system-prompt leaks, strip unsafe HTML, and enforce output length."""
    cfg = config or GuardrailConfig.from_env()
    if not cfg.enabled or not text:
        return GuardrailVerdict(allowed=True, text=text or "", engine="disabled" if not cfg.enabled else "builtin")

    flags: list[str] = []
    pii = redact_pii(text)
    cleaned = pii.redacted_text
    if pii.has_pii:
        flags.append("pii_redacted")
        if "SECRET" in pii.counts:
            flags.append("secret_scrubbed")

    if _leaked_system_prompt(cleaned, system_prompt):
        flags.append("system_prompt_leak")
        return GuardrailVerdict(
            allowed=False,
            text=REFUSAL_LEAK,
            blocked_reason="Model output echoed the system prompt.",
            pii_counts=pii.counts,
            flags=flags,
        )

    stripped = _HTML_TAG.sub("", cleaned).strip()
    if stripped != cleaned.strip():
        flags.append("html_stripped")
        cleaned = stripped

    if len(cleaned) > cfg.max_output_chars:
        flags.append("output_truncated")
        cleaned = cleaned[: cfg.max_output_chars].rstrip() + "…"

    return GuardrailVerdict(allowed=True, text=cleaned, pii_counts=pii.counts, flags=flags)
