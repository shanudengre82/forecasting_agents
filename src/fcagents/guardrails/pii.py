"""Lightweight PII & secret detector/redactor tuned to avoid false positives on time-series data."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field


@dataclass(frozen=True)
class PIIMatch:
    entity_type: str
    start: int
    end: int
    placeholder: str


@dataclass(frozen=True)
class PIIScanResult:
    redacted_text: str
    matches: list[PIIMatch] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def has_pii(self) -> bool:
        return bool(self.matches)


def _luhn_valid(digits: str) -> bool:
    """Return True if ``digits`` (13–19 ASCII digits, not all identical) passes the Luhn checksum."""
    if not (13 <= len(digits) <= 19) or not digits.isdigit() or len(set(digits)) == 1:
        return False
    total = 0
    for i, ch in enumerate(reversed(digits)):
        n = int(ch)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def _iban_valid(raw: str) -> bool:
    """ISO 7064 mod-97-10 IBAN check."""
    cleaned = re.sub(r"\s+", "", raw).upper()
    if not (15 <= len(cleaned) <= 34) or not re.match(r"^[A-Z]{2}\d{2}[A-Z0-9]+$", cleaned):
        return False
    rearranged = cleaned[4:] + cleaned[:4]
    expanded = "".join(str(ord(c) - 55) if c.isalpha() else c for c in rearranged)
    return int(expanded) % 97 == 1


# Ordered by priority (higher priority wins on overlapping spans).
_PATTERNS: list[tuple[str, str, re.Pattern[str], callable | None]] = [
    # API keys, tokens, JWTs
    (
        "SECRET",
        "[SECRET]",
        re.compile(
            r"(?:"
            r"\bhf_[A-Za-z0-9]{16,}\b"
            r"|\bsk-(?:proj-|lf-|ant-)?[A-Za-z0-9_-]{16,}\b"
            r"|\bpk-lf-[A-Za-z0-9_-]{16,}\b"
            r"|\bgh[pousr]_[A-Za-z0-9]{20,}\b"
            r"|\bAKIA[0-9A-Z]{16}\b"
            r"|\bBearer\s+[A-Za-z0-9._~+/-]{20,}=*"
            r"|\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"
            r")"
        ),
        None,
    ),
    # Email addresses
    (
        "EMAIL",
        "[EMAIL]",
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        None,
    ),
    # US Social Security Numbers (excludes 4-digit ISO years via word boundary + 3-2-4 structure)
    (
        "SSN",
        "[SSN]",
        re.compile(r"(?<!\d)(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}(?!\d)"),
        None,
    ),
    # Credit cards (13–19 digits with optional spaces/hyphens, Luhn verified, not part of a float)
    (
        "CREDIT_CARD",
        "[CREDIT_CARD]",
        re.compile(r"(?<![\d.])\d(?:[ -]?\d){12,18}(?![\d.])"),
        lambda m: _luhn_valid(re.sub(r"[ -]", "", m.group(0))),
    ),
    # IBAN (mod-97 verified)
    (
        "IBAN",
        "[IBAN]",
        re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){2,7}(?:[ ]?[A-Z0-9]{1,4})?\b"),
        lambda m: _iban_valid(m.group(0)),
    ),
    # IPv4 (4 dot-separated octets 0–255; floats only have 1 dot, semver has 3 parts)
    (
        "IP_ADDRESS",
        "[IP_ADDRESS]",
        re.compile(
            r"(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}"
            r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(?![\d.])"
        ),
        None,
    ),
    # IPv6 (full 8-group or compressed with ::; excludes HH:MM:SS timestamps)
    (
        "IP_ADDRESS",
        "[IP_ADDRESS]",
        re.compile(
            r"(?<![0-9A-Fa-f:])(?:"
            r"(?:[0-9A-Fa-f]{1,4}:){7}[0-9A-Fa-f]{1,4}"
            r"|(?:[0-9A-Fa-f]{1,4}:){1,6}:[0-9A-Fa-f]{1,4}(?::[0-9A-Fa-f]{1,4}){0,5}"
            r"|::(?:[0-9A-Fa-f]{1,4}:){0,5}[0-9A-Fa-f]{1,4}"
            r")(?![0-9A-Fa-f:])"
        ),
        None,
    ),
    # Phone numbers (explicitly structured so ISO dates YYYY-MM-DD and bare numbers do not match)
    (
        "PHONE",
        "[PHONE]",
        re.compile(
            r"(?<![\d+])(?:"
            r"\+\d{1,3}[\s.-]?(?:\(\d{1,4}\)|\d{1,4})[\s.-]?\d{3,4}[\s.-]?\d{3,4}"
            r"|\(\d{3}\)[\s.-]?\d{3}[\s.-]\d{4}"
            r"|\d{3}[\s.-]\d{3}[\s.-]\d{4}"
            r")(?!\d)"
        ),
        None,
    ),
]


def redact_pii(text: str) -> PIIScanResult:
    """Detect and mask PII/secrets in ``text`` with typed placeholders (e.g. ``[EMAIL]``)."""
    if not text:
        return PIIScanResult(redacted_text=text or "")

    candidates: list[tuple[int, int, int, str, str]] = []
    for priority, (entity_type, placeholder, pattern, validator) in enumerate(_PATTERNS):
        for m in pattern.finditer(text):
            if validator is not None and not validator(m):
                continue
            candidates.append((m.start(), -(m.end() - m.start()), priority, entity_type, placeholder))

    if not candidates:
        return PIIScanResult(redacted_text=text)

    # Sort by start ascending, longest match first, then priority.
    candidates.sort()
    chosen: list[PIIMatch] = []
    last_end = -1
    for start, neg_len, _prio, entity_type, placeholder in candidates:
        end = start - neg_len
        if start < last_end:
            continue
        chosen.append(PIIMatch(entity_type=entity_type, start=start, end=end, placeholder=placeholder))
        last_end = end

    parts: list[str] = []
    cursor = 0
    for match in chosen:
        parts.append(text[cursor:match.start])
        parts.append(match.placeholder)
        cursor = match.end
    parts.append(text[cursor:])

    counts = dict(Counter(m.entity_type for m in chosen))
    return PIIScanResult(redacted_text="".join(parts), matches=chosen, counts=counts)
