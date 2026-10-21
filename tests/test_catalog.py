import sys
from pathlib import Path

import pandas as pd
import pytest

from fcagents.catalog import EDGES, agent_catalog, catalog_markdown, pipeline_dot, pipeline_mermaid
from fcagents.orchestrator import PIPELINE, run_pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from render_agents_md import render  # noqa: E402


@pytest.mark.parametrize("entry", agent_catalog(), ids=lambda e: e["name"])
def test_metadata_complete(entry):
    for key in ("summary", "runs_when", "inputs", "outputs", "decisions", "tools"):
        assert entry[key], f"{entry['name']} is missing {key}"


def test_catalog_order_matches_pipeline():
    assert [e["name"] for e in agent_catalog()] == [a.name for a in PIPELINE]


def test_every_agent_in_diagrams():
    dot, mermaid = pipeline_dot(), pipeline_mermaid()
    edge_nodes = {n for s, d, _ in EDGES for n in (s, d)}
    for a in PIPELINE:
        assert a.name in edge_nodes and a.title in dot and a.title in mermaid


def test_decisions_seen_on_samples_are_documented():
    documented = {a.name: set(a.decisions) for a in PIPELINE}
    for csv in sorted((ROOT / "data" / "samples").glob("*.csv")):
        df = pd.read_csv(csv)
        for freq in ("auto", "W"):
            for r, _ in run_pipeline(df, resample_freq=freq):
                if r.status == "skipped":
                    continue
                if r.agent == "frequency":
                    assert "<alias>" in documented["frequency"]
                else:
                    assert r.decision in documented[r.agent], (csv.name, r.agent, r.decision)


def test_readme_in_sync():
    text = (ROOT / "README.md").read_text()
    assert render(text) == text, "README agents section is stale: run scripts/render_agents_md.py"
    assert catalog_markdown() in text
