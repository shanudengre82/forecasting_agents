"""Agent contract: deterministic tools produce facts, rules decide, the LLM explains."""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from fcagents.llm.backend import LLMBackend

if TYPE_CHECKING:
    from fcagents.core.state import PipelineState

SYSTEM_PROMPT = (
    "You are a careful time series analyst helping build a forecasting pipeline. "
    "Use ONLY the facts given; never invent numbers, frequencies or properties that are not stated. "
    "Answer in 2-3 short plain-language sentences."
)


@dataclass
class AgentResult:
    agent: str
    title: str
    status: str  # ok | warning | error | skipped
    decision: str
    confidence: float
    facts: dict
    summary: str  # deterministic, template-based
    next_step: str
    explanation: str | None = None  # LLM commentary, if an LLM was used
    prompt: str | None = None
    elapsed_s: float = 0.0
    extras: dict = field(default_factory=dict)  # non-JSON payloads for the UI (never sent to the LLM)


@dataclass
class Decision:
    status: str
    decision: str
    confidence: float
    summary: str
    next_step: str


class BaseAgent(ABC):
    name: str = "base"
    title: str = "Base agent"
    task: str = ""  # one-line description given to the LLM
    scope: str = ""  # what the LLM must NOT talk about (small models drift into other agents' topics)

    def applies(self, state: PipelineState) -> str | None:
        """Return a reason to skip this agent for the current state, or None to run it."""
        return None

    @abstractmethod
    def run_tools(self, state: PipelineState) -> dict:
        """Compute facts with deterministic tools; may update ``state``."""

    @abstractmethod
    def decide(self, facts: dict) -> Decision:
        """Rule-based verdict on the facts."""

    def prompt_facts(self, facts: dict) -> dict:
        """Subset of facts shown to the LLM (keep it small for tiny models)."""
        return facts

    def build_prompt(self, facts: dict, d: Decision) -> str:
        return (
            f"Task: {self.task}\n"
            f"Facts (JSON): {json.dumps(self.prompt_facts(facts), default=str)}\n"
            f"Verdict from rules: {d.decision}. {d.summary}\n"
            f"Suggested next step: {d.next_step}\n"
            f"{self.scope}\n"
            "Restate the verdict in plain language and say why it matters for forecasting. "
            "Mention nothing that is not in the facts above."
        )

    def run(self, state: PipelineState, llm: LLMBackend) -> AgentResult:
        t0 = time.perf_counter()
        facts = self.run_tools(state)
        d = self.decide(facts)
        prompt = explanation = None
        if llm.available:
            prompt = self.build_prompt(facts, d)
            explanation = llm.generate(SYSTEM_PROMPT, prompt)
        result = AgentResult(
            agent=self.name, title=self.title, status=d.status, decision=d.decision,
            confidence=round(d.confidence, 3), facts=facts, summary=d.summary, next_step=d.next_step,
            explanation=explanation, prompt=prompt, elapsed_s=round(time.perf_counter() - t0, 2),
        )
        state.findings[self.name] = result
        return result


def skipped(agent: BaseAgent, reason: str) -> AgentResult:
    return AgentResult(agent=agent.name, title=agent.title, status="skipped", decision="skipped",
                       confidence=0.0, facts={}, summary=reason, next_step="")
