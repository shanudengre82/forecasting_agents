"""Describe the pipeline's agents from their own metadata (app "Agents" tab and README section)."""

from __future__ import annotations

from fcagents.orchestrator import PIPELINE

PRINCIPLE = ("Deterministic tools compute every fact → rules make every decision → a small LLM only explains. "
             "No routing decision ever depends on LLM output.")

STOP, NEXT = "stop", "next"
# Pipeline flow: (from, to, condition). Mirrors orchestrator routing and each agent's applies().
EDGES = [
    ("data", "time_index", ""),
    ("time_index", "regularity", "found / ambiguous"),
    ("time_index", STOP, "not_found"),
    ("regularity", "resampling", "irregular"),
    ("regularity", "frequency", "regular / with gaps"),
    ("regularity", STOP, "insufficient_data"),
    ("resampling", "frequency", "resampled / sparse"),
    ("resampling", STOP, "too_few_points"),
    ("frequency", NEXT, ""),
]


def _labels() -> dict[str, str]:
    return {"data": "CSV data", **{a.name: a.title for a in PIPELINE},
            STOP: "Stop & ask user", NEXT: "Next agents (planned)"}


def agent_catalog() -> list[dict]:
    return [
        {
            "step": i,
            "name": a.name,
            "title": a.title,
            "summary": a.summary,
            "runs_when": a.runs_when,
            "inputs": list(a.inputs),
            "outputs": list(a.outputs),
            "decisions": dict(a.decisions),
            "user_controls": list(a.user_controls),
            "tools": list(a.tools),
        }
        for i, a in enumerate(PIPELINE, start=1)
    ]


def pipeline_dot() -> str:
    labels = _labels()
    lines = ['digraph pipeline {', '  rankdir=LR; bgcolor="transparent";',
             '  node [shape=box, style="rounded", fontname="sans-serif", fontsize=11];',
             '  edge [fontname="sans-serif", fontsize=9];',
             '  data [shape=note];', f'  {STOP} [style="rounded,dashed"];', f'  {NEXT} [style="rounded,dotted"];']
    lines += [f'  {k} [label="{v}"];' for k, v in labels.items()]
    lines += [f'  {s} -> {d}' + (f' [label="{c}"]' if c else "") + ";" for s, d, c in EDGES]
    return "\n".join(lines + ["}"])


def pipeline_mermaid() -> str:
    labels = _labels()
    lines = ["flowchart LR"]
    lines += [f'  {k}["{v}"]' for k, v in labels.items()]
    lines += [f"  {s} -->|{c}| {d}" if c else f"  {s} --> {d}" for s, d, c in EDGES]
    return "\n".join(lines)


def _cell(items: list[str]) -> str:
    return "<br>".join(items) if items else "—"


def catalog_markdown() -> str:
    out = [f"**Design principle:** {PRINCIPLE}", "", "```mermaid", pipeline_mermaid(), "```", ""]
    for a in agent_catalog():
        out += [
            f"### {a['step']}. {a['title']}",
            "",
            a["summary"],
            "",
            "| Property | Details |",
            "|---|---|",
            f"| Runs when | {a['runs_when']} |",
            f"| Inputs | {_cell(a['inputs'])} |",
            f"| Outputs | {_cell(a['outputs'])} |",
            f"| User controls | {_cell(a['user_controls'])} |",
            f"| Tools | {_cell([f'`{t}`' for t in a['tools']])} |",
            "",
            "| Decision | Meaning |",
            "|---|---|",
            *[f"| `{k}` | {v} |" for k, v in a["decisions"].items()],
            "",
        ]
    return "\n".join(out).rstrip() + "\n"
