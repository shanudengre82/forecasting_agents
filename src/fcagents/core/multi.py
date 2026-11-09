"""Run a per-series agent on every series and fold the results into one card.

One series: the agent's own result, unchanged. Several: the worst status wins, the summary counts decisions and
quotes the series that needs the most attention, and the LLM explains the combined facts in one call.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Callable

from fcagents.core.agent import SYSTEM_PROMPT, AgentResult, BaseAgent, skipped
from fcagents.llm.backend import LLMBackend, NullBackend

SEVERITY = {"error": 3, "warning": 2, "ok": 1, "skipped": 0}
MAX_PROMPT_SERIES = 8  # keep prompts small for tiny models


def run_per_series(agent: BaseAgent, state, llm: LLMBackend,
                   route: Callable[[BaseAgent, AgentResult], str | None]) -> AgentResult:
    children = state.children
    many = len(children) > 1
    results: dict[str, AgentResult] = {}
    for key, child in children.items():
        if child.stopped:
            r = skipped(agent, child.stopped)
        elif reason := agent.applies(child):
            r = skipped(agent, reason)
        else:
            r = agent.run(child, NullBackend() if many else llm)
            child.stopped = route(agent, r)
        results[key] = r
    result = results[next(iter(results))] if not many else combine(agent, results, llm)
    if many:
        result.extras["per_series"] = results
    state.findings[agent.name] = result
    if children and all(c.stopped for c in children.values()):
        state.stopped = next(iter(children.values())).stopped
    return result


def series_prompt_facts(agent: BaseAgent, result: AgentResult) -> dict:
    """What the LLM and the chat see of a result (combined results carry their own compact facts)."""
    if "per_series" in result.facts:
        return result.facts
    return agent.prompt_facts(result.facts) if result.status != "skipped" else {}


def combine(agent: BaseAgent, results: dict[str, AgentResult], llm: LLMBackend) -> AgentResult:
    t0 = time.perf_counter()
    ran = {k: r for k, r in results.items() if r.status != "skipped"}
    first = next(iter(results.values()))
    if not ran:
        return skipped(agent, f"{first.summary} (all {len(results)} series)")
    worst_key = max(ran, key=lambda k: (SEVERITY[ran[k].status], -list(ran).index(k)))
    worst = ran[worst_key]
    counts = Counter(r.decision for r in results.values())
    count_text = ", ".join(f"{n} × {d}" for d, n in counts.most_common())
    summary = f"{len(results)} series: {count_text}."
    if len(set(r.decision for r in ran.values())) > 1 or worst.status != "ok":
        summary += f" Needs most attention: {worst_key}. {worst.summary}"
    else:
        summary += f" For example {worst_key}: {worst.summary}"
    facts = {
        "n_series": len(results),
        "decisions": dict(counts),
        "worst_series": worst_key,
        "per_series": {k: agent.prompt_facts(r.facts) for k, r in list(ran.items())[:MAX_PROMPT_SERIES]},
    }
    result = AgentResult(
        agent=agent.name, title=agent.title, status=worst.status, decision=worst.decision,
        confidence=round(sum(r.confidence for r in ran.values()) / len(ran), 3), facts=facts, summary=summary,
        next_step=worst.next_step,
    )
    if llm.available:
        result.prompt = (
            f"Task: {agent.task} The same check ran on {len(results)} series.\n"
            f"Facts per series (JSON): {json.dumps(facts, default=str)}\n"
            f"Verdict from rules: {summary}\nSuggested next step: {worst.next_step}\n{agent.scope}\n"
            "Summarise across the series in plain language: what most series show, which series differ and why "
            "it matters for forecasting. Mention nothing that is not in the facts above."
        )
        result.explanation = llm.generate(SYSTEM_PROMPT, result.prompt, max_new_tokens=160)
    result.elapsed_s = round(sum(r.elapsed_s for r in results.values()) + time.perf_counter() - t0, 2)
    return result
