"""Fallback when the rule parser finds nothing: the LLM proposes one action as JSON, which is then validated."""

from __future__ import annotations

import json
import re

from fcagents.chat.actions import PAGE_KEYS, Action, allowed_values, validate
from fcagents.llm.backend import LLMBackend

ROUTER_SYSTEM = "You convert a user message into a JSON settings change. Reply with one JSON object and nothing else."
_JSON = re.compile(r"\{.*?\}", re.S)
MEANINGS = {
    "resample_freq": "size of each time bin",
    "resample_agg": "how the values inside one bin are combined",
    "fill_method": "how missing values are filled",
    "time_col": "column holding the timestamps",
    "target_col": "column to forecast",
    "targets": "several columns to forecast, comma-separated",
    "series_id": "which series to show in detail",
    "series_scope": "forecast all series",
    "holiday_country": "country whose holidays are checked (ISO code)",
    "holiday_subdiv": "state or region code within that country",
    "holiday_types": "which holidays: official, unofficial or both",
    "holiday_window": "days before and after each holiday to check",
    "outlier_threshold": "outlier sensitivity as a z-score cut-off (lower flags more points)",
    "outlier_bulk": "one choice for every flagged outlier",
}


def _values(key: str, ctx: dict) -> str:
    vals = allowed_values(key, ctx)
    return json.dumps(vals) if len(vals) <= 30 else json.dumps(vals[:30])[:-1] + ", ...]"


def build_router_prompt(text: str, page: str, ctx: dict) -> str:
    lines = [f'- "{k}" = {MEANINGS[k]}; one of {_values(k, ctx)}' for k in PAGE_KEYS[page]]
    example = PAGE_KEYS[page][0]
    return (
        "Settings you may change:\n" + "\n".join(lines) + "\n"
        f'If the message asks to change a setting, reply like {{"set": "{example}", "value": "..."}} using only '
        'the values listed above. If the message is a question or asks for anything else, reply {"set": "question"}.\n'
        f"Message: {text}"
    )


def parse_router_reply(reply: str, page: str, ctx: dict) -> Action | None:
    """Validated action from the model's reply; None for questions, junk or anything not allowed."""
    m = _JSON.search(reply or "")
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or obj.get("set") not in PAGE_KEYS[page] or not isinstance(obj.get("value"), str):
        return None
    action = Action(obj["set"], obj["value"])
    return None if validate(page, action, ctx) else action


def route_with_llm(text: str, page: str, ctx: dict, llm: LLMBackend) -> Action | None:
    reply = llm.generate(ROUTER_SYSTEM, build_router_prompt(text, page, ctx), max_new_tokens=40)
    return parse_router_reply(reply, page, ctx)
