"""Page chat: rules first, then a validated LLM proposal; answers are grounded in computed stats."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from fcagents.chat.actions import PAGE_KEYS, Action, validate
from fcagents.chat.llm_router import route_with_llm
from fcagents.chat.parser import parse_command
from fcagents.chat.qa import llm_answer, series_stats, template_answer
from fcagents.llm.backend import LLMBackend, NullBackend

HINTS = {
    "data": "Try: “use temp as the target”, “switch to store_2”, “how many points are there?”",
    "resampling": "Try: “resample to 6 hours with sum”, “daily, max per day”, “what is the mean?”",
    "holidays": "Try: “use Germany”, “only California”, “include unofficial holidays”, “check 3 days around”",
    "outliers": "Try: “apply the suggestions”, “remove all outliers”, “be more sensitive”, “keep all”",
    "missing": "Try: “fill gaps by interpolation”, “use seasonal filling”, “how many values were filled?”",
}


@dataclass
class ChatReply:
    text: str
    actions: list[Action] = field(default_factory=list)
    source: str = "rule"  # rule | llm-action (needs confirmation) | llm-answer
    stats: dict | None = None  # shown as a table when no template or LLM could answer


def respond(text: str, page: str, ctx: dict, series: pd.Series | None, freq: str | None,
            page_facts: dict[str, dict], llm: LLMBackend | None = None) -> ChatReply:
    """``page_facts`` maps agent name → its prompt facts for the agents on this page."""
    llm = llm or NullBackend()
    parsed = parse_command(text, page, ctx)
    actions = [a for a in parsed.actions if not validate(page, a, ctx)]
    errors = parsed.errors + [e for a in parsed.actions if (e := validate(page, a, ctx))]
    if actions:
        return ChatReply("Done: " + "; ".join(a.describe() for a in actions) + "." + "".join(f" {e}" for e in errors),
                         actions)
    if errors:
        return ChatReply(" ".join(errors))

    if not parsed.is_question and llm.available and PAGE_KEYS[page]:
        if action := route_with_llm(text, page, ctx, llm):  # a small model can misread intent: user confirms
            return ChatReply(f"The LLM suggests: {action.describe()}. Apply it?", [action], "llm-action")

    stats = series_stats(series, freq)
    flat = {k: v for facts in page_facts.values() for k, v in facts.items()}
    if answer := template_answer(text, stats, flat):
        return ChatReply(answer)
    if llm.available:
        return ChatReply(llm_answer(text, stats, page_facts, llm), source="llm-answer")
    return ChatReply("I can answer questions about points, missing values, frequency, mean, min/max and start/end. "
                     "Turn on the language model for free-form questions. Here is what I know about the series:",
                     stats=stats)
