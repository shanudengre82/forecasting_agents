"""Rule-based reading of chat prompts: commands become validated actions, everything else is a question."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from fcagents.chat.actions import DEFAULTS, PAGE_KEYS, Action
from fcagents.tools.resample import FREQ_OPTIONS, MAX_FREQ, MIN_FREQ, freq_span

QUESTION_START = re.compile(
    r"^(how|what|which|when|where|why|who|is|are|was|were|does|do|did|should|explain|describe|"
    r"summari[sz]e|tell|list|show)\b", re.I)
COMMAND_VERB = re.compile(
    r"\b(resample|aggregate|agg|convert|switch|change|set|use|fill|impute|interpolate|make|group|bin|try|apply|"
    r"pick|choose|select|reset|undo)\b", re.I)
POLITE = re.compile(r"^(can|could|would|will) you (please )?", re.I)

_SPAN_TO_ALIAS = {freq_span(a): a for a in FREQ_OPTIONS}
_UNITS = {"h": "hours", "hr": "hours", "hrs": "hours", "hour": "hours", "hours": "hours", "hourly": "hours",
          "d": "days", "day": "days", "days": "days", "w": "weeks", "wk": "weeks", "wks": "weeks", "week": "weeks",
          "weeks": "weeks", "min": "minutes", "mins": "minutes", "minute": "minutes", "minutes": "minutes",
          "t": "minutes", "s": "seconds", "sec": "seconds", "secs": "seconds", "second": "seconds",
          "seconds": "seconds"}
_NUM_UNIT = re.compile(r"\b(\d+)\s*-?\s*(" + "|".join(sorted(_UNITS, key=len, reverse=True)) + r")\b", re.I)
_FREQ_WORDS = [  # (pattern, alias)
    (r"twice (?:a|per) day|twice daily|half[- ]da(?:y|ily)", "12h"),
    (r"hourly|every hour|per hour|each hour|by (?:the )?hour|an hour\b", "h"),
    (r"daily|every day|per day|each day|by (?:the )?day|a day\b", "D"),
    (r"weekly|every week|per week|each week|by (?:the )?week|a week\b", "W"),
    (r"\b(?:to|into|per|by) (?:an? |one )?(?:hour)\b", "h"),
    (r"\b(?:to|into|per|by) (?:an? |one )?(?:day)\b", "D"),
    (r"\b(?:to|into|per|by) (?:an? |one )?(?:week)\b", "W"),
]
_OUT_OF_RANGE = re.compile(r"\b(minutely|monthly|months?|quarterly|quarters?|yearly|annual(?:ly)?|years?)\b", re.I)
_AUTO = re.compile(r"\b(auto\w*|recommend\w*|default)\b", re.I)
_RESET = re.compile(r"\b(reset|undo|start over|defaults)\b", re.I)

_AGG_WORDS = [
    ("count", r"count|number of (?:events|observations|readings|rows)"),
    ("mean", r"mean|average|avg"),
    ("median", r"median"),
    ("sum", r"sum|summed|total|add(?:ing)? up"),
    ("min", r"min|minimum|lowest"),
    ("max", r"max|maximum|highest|peak"),
    ("first", r"(?:(?:with|using|use|take|keep|by)\s+(?:the\s+)?first)|first (?:value|observation|reading)"),
    ("last", r"(?:(?:with|using|use|take|keep|by)\s+(?:the\s+)?last)|last (?:value|observation|reading)"),
]
_FILL_WORDS = [
    ("none", r"leave (?:\w+ )?(?:missing|empty|as is)|don'?t fill|do not fill|no fill\w*|keep (?:\w+ )?missing|"
             r"(?:put|fill(?: in)?) nothing|leave (?:them|it) alone"),
    ("seasonal", r"season\w*|same (?:time|hour|day) last|previous (?:period|season|cycle|week|day)|"
                 r"last (?:period|cycle)"),
    ("interpolate", r"interpolat\w*|linear\w*"),
    ("ffill", r"forward[- ]?fill\w*|ffill|forward|carry \w* ?forward|previous value|last known"),
    ("bfill", r"backward[- ]?fill\w*|bfill|backward|back[- ]?fill\w*|next value"),
    ("zero", r"zeros?|with 0\b|to 0\b"),
]


@dataclass
class ParseResult:
    actions: list[Action] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    is_question: bool = False


def is_question(text: str) -> bool:
    t = text.strip()
    if POLITE.match(t) and COMMAND_VERB.search(t):
        return False  # "could you resample to daily?" is a command
    return t.endswith("?") or bool(QUESTION_START.match(t))


def _span_alias(n: int, unit: str) -> tuple[str | None, str | None]:
    span = pd.Timedelta(**{_UNITS[unit.lower()]: n})
    if not MIN_FREQ <= span <= MAX_FREQ:
        return None, f"{n} {unit} is outside the supported range of 1 hour to 1 week."
    if span not in _SPAN_TO_ALIAS:
        return None, (f"{n} {unit} is not one of the offered frequencies: " + ", ".join(FREQ_OPTIONS) + ".")
    return _SPAN_TO_ALIAS[span], None


def parse_frequency(text: str) -> tuple[str | None, str | None, str]:
    """Return (alias, error, text with the frequency phrase removed)."""
    if m := _NUM_UNIT.search(text):
        alias, err = _span_alias(int(m.group(1)), m.group(2))
        return alias, err, text[:m.start()] + text[m.end():]
    for pattern, alias in _FREQ_WORDS:
        if m := re.search(pattern, text, re.I):
            return alias, None, text[:m.start()] + text[m.end():]
    if m := _OUT_OF_RANGE.search(text):
        return None, f"'{m.group(1)}' is outside the supported range of 1 hour to 1 week.", text
    return None, None, text


def _first_match(words: list[tuple[str, str]], text: str) -> str | None:
    hits = [(m.start(), value) for value, pattern in words if (m := re.search(rf"\b(?:{pattern})", text, re.I))]
    return min(hits)[1] if hits else None


def _mentions(name: str, text: str) -> bool:
    return re.search(rf"(?<![\w]){re.escape(str(name))}(?![\w])", text, re.I) is not None


def _parse_data(text: str, ctx: dict) -> list[Action]:
    actions = []
    time_cols, target_cols = ctx.get("time_cols", []), ctx.get("target_cols", [])
    wants_time = re.search(r"\b(time|date|timestamp|index)\b", text, re.I)
    for col in dict.fromkeys([*time_cols, *target_cols]):
        if not _mentions(col, text):
            continue
        if col in time_cols and (wants_time or col not in target_cols):
            actions.append(Action("time_col", col))
        elif col in target_cols:
            actions.append(Action("target_col", col))
    for sid in ctx.get("series_ids", []):
        if _mentions(sid, text):
            actions.append(Action("series_id", str(sid)))
            break
    return list({a.key: a for a in actions}.values())


def parse_command(text: str, page: str, ctx: dict | None = None) -> ParseResult:
    ctx = ctx or {}
    if is_question(text):
        return ParseResult(is_question=True)
    if _RESET.search(text):
        return ParseResult([Action(k, DEFAULTS[k]) for k in PAGE_KEYS[page]])

    res = ParseResult()
    if page == "data":
        res.actions = _parse_data(text, ctx)
    elif page == "resampling":
        alias, err, rest = parse_frequency(text)
        if alias is None and not err and _AUTO.search(text):
            alias = "auto"
        if alias:
            res.actions.append(Action("resample_freq", alias))
        if err:
            res.errors.append(err)
        if agg := _first_match(_AGG_WORDS, rest):
            res.actions.append(Action("resample_agg", agg))
    elif page == "missing":
        method = _first_match(_FILL_WORDS, text)
        if method is None and _AUTO.search(text):
            method = "auto"
        if method:
            res.actions.append(Action("fill_method", method))
    return res
