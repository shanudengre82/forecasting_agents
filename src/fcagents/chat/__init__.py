"""Page chat: rules first, then a validated LLM proposal; answers are grounded in computed stats."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from fcagents.chat.actions import PAGE_KEYS, Action, validate
from fcagents.chat.llm_router import route_with_llm
from fcagents.chat.parser import parse_command
from fcagents.chat.qa import llm_answer, series_stats, template_answer
from fcagents.guardrails.checks import check_input
from fcagents.guardrails.wrapper import wrap_llm
from fcagents.llm.backend import LLMBackend, NullBackend
from fcagents.observability import trace_span

HINTS = {
    "explore": "Try: “use spearman correlation”, “inspect sales”, “compare temp and demand”",
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
    verdict = check_input(text, mode="chat_user")
    if not verdict.allowed:
        with trace_span("guardrails:chat_input", metadata={"page": page, "flags": verdict.flags}) as span:
            span.update(
                output={"allowed": False, "flags": verdict.flags},
                level="WARNING",
                status_message=verdict.blocked_reason,
            )
            span.score(name="guardrail_blocked", value=1, comment=verdict.blocked_reason)
        return ChatReply(verdict.text, source="guardrail")

    safe_text = verdict.text
    llm = wrap_llm(llm or NullBackend())
    parsed = parse_command(safe_text, page, ctx)
    actions = [a for a in parsed.actions if not validate(page, a, ctx)]
    errors = parsed.errors + [e for a in parsed.actions if (e := validate(page, a, ctx))]
    if actions:
        return ChatReply("Done: " + "; ".join(a.describe() for a in actions) + "." + "".join(f" {e}" for e in errors),
                         actions)
    if errors:
        return ChatReply(" ".join(errors))

    if not parsed.is_question and llm.available and PAGE_KEYS[page]:
        if action := route_with_llm(safe_text, page, ctx, llm):  # a small model can misread intent: user confirms
            return ChatReply(f"The LLM suggests: {action.describe()}. Apply it?", [action], "llm-action")

    stats = series_stats(series, freq)
    flat = {k: v for facts in page_facts.values() for k, v in facts.items()}
    if answer := template_answer(safe_text, stats, flat):
        return ChatReply(answer)
    if llm.available:
        answer = llm_answer(safe_text, stats, page_facts, llm)
        if answer:
            return ChatReply(answer, source="llm-answer")
        return ChatReply("The model could not produce a safe answer for that question. Here is what I know:",
                         source="guardrail", stats=stats)
    return ChatReply("I can answer questions about points, missing values, frequency, mean, min/max and start/end. "
                     "Turn on the language model for free-form questions. Here is what I know about the series:",
                     stats=stats)
