---
title: Forecasting Agents
emoji: 📈
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: Small-LLM agents that prepare time series for forecasting
---

# Forecasting Agents

A demo of LLM-assisted agents that together form an end-to-end time series forecasting pipeline.

**Design rule:** deterministic tools (pandas/numpy) compute every fact, rule-based logic
makes the decision, and a small LLM (default `Qwen/Qwen2.5-0.5B-Instruct`, CPU) explains
what it means and suggests the next step. A tiny model never has to compute anything.

## Agents so far

| # | Agent | Question it answers |
|---|-------|---------------------|
| 1 | Time Index | Which column holds the timestamps? Which is the target? Is the data long-format multi-series? |
| 2 | Regularity | Is the series sampled at a fixed frequency (`regular`, `regular_with_gaps`, `irregular`)? |
| 3 | Resampling | *Irregular data only.* Converts it onto a fixed grid (1h–1W) using your choice of frequency, aggregation and empty-bin handling. Warns, and stops, if fewer than 30 data points result, and lists finer frequencies that would give enough. |
| 4 | Frequency | Which frequency (pandas alias + label), how many timestamps are missing, which seasonal periods are candidates? |

The orchestrator (`src/fcagents/orchestrator.py`) runs them in sequence. Agents can opt out
via `applies(state)` (e.g. Resampling is skipped for series that already have a fixed frequency).

## Run locally

```bash
uv sync --extra llm                      # drop --extra llm for a rule-only install
uv run streamlit run app.py
uv run pytest                            # fast tests (no model)
uv run pytest -m slow                    # also loads the real LLM
```

Env vars: `MODEL_ID` (default model), `USE_LLM=0` (start with the LLM toggle off).

## Layout

```
app.py                     Streamlit UI (rendering only)
src/fcagents/tools/        deterministic tools (time_index.py, frequency.py)
src/fcagents/agents/       one agent per pipeline step
src/fcagents/core/         BaseAgent / AgentResult / PipelineState
src/fcagents/llm/          LLM backends (transformers on CPU, or none)
scripts/make_samples.py    regenerates data/samples/*.csv
```

## Adding an agent

Subclass `BaseAgent`, then:

1. Implement `run_tools(state)`, which returns JSON facts.
2. Implement `decide(facts)`, which returns a `Decision`.
3. Optionally implement `prompt_facts(facts)` to trim what the LLM sees.

Finally, add the agent to `PIPELINE` in the orchestrator. Override `applies(state)` if it only runs in some situations, and add a routing rule if it can stop the pipeline.
