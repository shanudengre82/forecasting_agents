"""Streamlit UI for the forecasting-agents demo. All logic lives in src/fcagents; this file only renders."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from fcagents.llm.backend import DEFAULT_MODEL_ID, NullBackend  # noqa: E402
from fcagents.orchestrator import PIPELINE, run_pipeline  # noqa: E402
from fcagents.tools.resample import AGGREGATIONS, FILL_METHODS, FREQ_OPTIONS, MIN_POINTS  # noqa: E402
from fcagents.tools.time_index import detect_time_column  # noqa: E402

SAMPLES = ROOT / "data" / "samples"
MODELS = list(dict.fromkeys([DEFAULT_MODEL_ID, "Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"]))
STATUS_ICON = {"ok": "✅", "warning": "⚠️", "error": "❌", "skipped": "⏭️"}
SERIES_COLOR = "#2a78d6"
MISSING_COLOR = "#fab219"
RAW_COLOR = "#9a9993"  # muted: context only, the resampled line is the subject

st.set_page_config(page_title="Forecasting Agents", page_icon="📈", layout="wide")


@st.cache_resource(show_spinner="Loading language model (first run downloads it)…")
def load_llm(model_id: str):
    from fcagents.llm.backend import TransformersBackend

    return TransformersBackend(model_id)


@st.cache_data
def read_csv(source) -> pd.DataFrame:
    return pd.read_csv(source)


def resampling_prefs() -> dict:
    st.header("Resampling")
    st.caption("Used only when the timestamps are irregular.")
    freq_opts = ["auto"] + list(FREQ_OPTIONS)
    freq = st.selectbox("Target frequency (1h – 1W)", freq_opts,
                        format_func=lambda f: "Auto (recommended)" if f == "auto" else f"{f} · {FREQ_OPTIONS[f]}",
                        help=f"Finer frequencies give more data points. At least {MIN_POINTS} are needed.")
    agg = st.selectbox("Aggregate values per bin", AGGREGATIONS,
                       help="Event logs without a numeric value are always counted.")
    fill = st.selectbox("Empty bins", list(FILL_METHODS), format_func=FILL_METHODS.get)
    return {"resample_freq": freq, "resample_agg": agg, "resample_fill": fill}


def sidebar() -> tuple[pd.DataFrame | None, str, object, dict]:
    with st.sidebar:
        st.header("Data")
        mode = st.radio("Source", ["Sample dataset", "Upload CSV"], horizontal=True, label_visibility="collapsed")
        if mode == "Sample dataset":
            name = st.selectbox("Sample", sorted(p.name for p in SAMPLES.glob("*.csv")))
            df, key = read_csv(SAMPLES / name), name
        else:
            up = st.file_uploader("CSV file", type=["csv"])
            df, key = (read_csv(up), up.name) if up else (None, "")

        prefs = resampling_prefs()

        st.header("Language model")
        use_llm = st.toggle("Explain results with an LLM", value=os.environ.get("USE_LLM", "1") == "1",
                            help="Tools compute every fact; the LLM only interprets them.")
        llm = NullBackend()
        if use_llm:
            model_id = st.selectbox("Model", MODELS)
            try:
                llm = load_llm(model_id)
            except Exception as e:  # missing torch, no network, OOM…
                st.error(f"Could not load {model_id}: {e}. Falling back to rule-based summaries.")
        st.caption(f"Backend: `{llm.name}`")
    return df, key, llm, prefs


def overrides(df: pd.DataFrame) -> dict:
    rep = detect_time_column(df)
    with st.expander("Overrides (optional)", expanded=rep["ambiguous"]):
        cols = st.columns(3)
        time_opts = [c["column"] for c in rep["candidates"]]
        time_col = cols[0].selectbox("Time column", time_opts or ["—"], disabled=not time_opts)
        target_opts = rep["target_cols"]
        target_col = cols[1].selectbox("Target column", target_opts or ["—"], disabled=not target_opts)
        series_id = None
        if rep["id_col"]:
            ids = df[rep["id_col"]].value_counts().index.tolist()
            series_id = cols[2].selectbox(f"Series ({rep['id_col']})", ids)
    return {"time_col": time_col if time_opts else None,
            "target_col": target_col if target_opts else None,
            "series_id": series_id}


def render_result(r) -> None:
    with st.container(border=True):
        head, badge = st.columns([5, 2])
        head.subheader(f"{STATUS_ICON.get(r.status, '')} {r.title}")
        badge.markdown(f"**Decision:** `{r.decision}`  \n**Confidence:** {r.confidence:.0%} · {r.elapsed_s}s")
        st.write(r.summary)
        if r.explanation:
            st.info(r.explanation, icon="🤖")
        if r.agent == "resampling" and r.facts:
            render_resampling_details(r)
        if r.next_step:
            st.markdown(f"**Next step →** {r.next_step}")
        if r.facts:
            with st.expander("Facts & prompt"):
                st.json(r.facts, expanded=False)
                if r.prompt:
                    st.code(r.prompt, language="text", wrap_lines=True)


def render_resampling_details(r) -> None:
    f = r.facts
    if r.decision == "too_few_points":
        st.warning(f"Only **{f['n_points']}** data points at **{f['chosen_freq']}**; at least {MIN_POINTS} "
                   f"are needed. {r.next_step} Change *Target frequency* in the sidebar and run again.",
                   icon="⚠️")
    table = pd.DataFrame(f["options"])
    table["≥ 30 points"] = table["enough_points"].map({True: "✅ yes", False: "⚠️ no"})
    table["chosen"] = table["freq"].map(
        lambda x: "◀ chosen" if x == f["chosen_freq"] else ("recommended" if x == f["recommended_freq"] else ""))
    table = table[["freq", "label", "n_points", "n_bins", "empty_share", "≥ 30 points", "chosen"]]
    st.caption("Points per candidate frequency (points = bins that contain observations)")
    st.dataframe(table, hide_index=True, width="stretch",
                 column_config={"empty_share": st.column_config.ProgressColumn(
                     "empty bins", format="percent", min_value=0.0, max_value=1.0)})


def render_chart(state, results) -> None:
    s = state.series
    if s is None or s.empty:
        return
    s = s[~s.index.isna()].sort_index()
    s = s.groupby(level=0).mean()  # collapse duplicate timestamps for display
    fig = go.Figure()
    resampled = next((r for r in results if r.agent == "resampling" and r.status != "skipped"), None)
    if resampled is not None and resampled.facts["agg"] != "count":
        raw = resampled.extras["raw_series"].dropna()
        fig.add_trace(go.Scatter(x=raw.index, y=raw.values, mode="markers", name="raw observations",
                                 marker=dict(color=RAW_COLOR, size=4, opacity=0.6)))
    y = s.values if s.notna().any() else [1] * len(s)
    name = state.target_col or "timestamps"
    if resampled is not None:
        name = f"{resampled.facts['agg']} per {resampled.facts['chosen_freq']}"
    fig.add_trace(go.Scatter(x=s.index, y=y, mode="lines+markers" if len(s) < 60 else "lines", name=name,
                             line=dict(color=SERIES_COLOR, width=2), marker=dict(size=8)))
    missing = next((r.extras.get("missing_index") for r in results if r.agent == "frequency"), None)
    if missing is not None and len(missing):
        y0 = float(pd.Series(y).min())
        fig.add_trace(go.Scatter(x=missing[:5000], y=[y0] * min(len(missing), 5000), mode="markers",
                                 name=f"Missing timestamp ({len(missing)})",
                                 marker=dict(color=MISSING_COLOR, symbol="x-thin-open", size=10,
                                             line=dict(width=2, color=MISSING_COLOR))))
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                      legend=dict(orientation="h", y=1.08, x=0), showlegend=len(fig.data) > 1)
    title = state.target_col or "Timestamps"
    if state.series_id is not None:
        title += f" · series {state.series_id}"
    if resampled is not None:
        title += f" · resampled to {resampled.facts['chosen_freq']} ({resampled.facts['agg']})"
    st.markdown(f"**{title}**")
    st.plotly_chart(fig, width="stretch")


def main() -> None:
    st.title("📈 Forecasting Agents")
    st.caption("Small-LLM agents that prepare a time series for forecasting. "
               "Deterministic tools compute the facts → rules decide → the LLM explains.")
    st.markdown(" → ".join(f"**{a.title}**" for a in PIPELINE) + " → *more agents coming*")

    df, key, llm, prefs = sidebar()
    if df is None:
        st.info("Pick a sample dataset or upload a CSV in the sidebar.")
        return

    st.dataframe(df.head(8), width="stretch")
    opts = overrides(df)

    run_key = (key, tuple(opts.items()), tuple(prefs.items()), llm.name)
    if st.button("▶ Run agents", type="primary"):
        results, state = [], None
        progress = st.empty()
        pipeline = run_pipeline(df, llm, **opts, **prefs)
        for i, agent in enumerate(PIPELINE):
            progress.caption(f"Running {agent.title} ({i + 1}/{len(PIPELINE)})…")
            r, state = next(pipeline)
            results.append(r)
            render_result(r)
        progress.empty()
        st.session_state["run"] = (run_key, results, state)
        render_chart(state, results)
    elif st.session_state.get("run", (None,))[0] == run_key:
        _, results, state = st.session_state["run"]
        for r in results:
            render_result(r)
        render_chart(state, results)


main()
