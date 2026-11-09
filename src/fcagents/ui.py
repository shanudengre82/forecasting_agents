"""Streamlit building blocks shared by the step pages in views/. Rendering only; logic lives in the agents."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fcagents import theme
from fcagents.chat import HINTS, respond
from fcagents.chat.actions import DEFAULTS, HOLIDAY_TYPE_CHOICES, PAGE_KEYS
from fcagents.orchestrator import STAGE_TITLES, STAGES, initial_state, run_stage
from fcagents.tools.holidays import holiday_covariates
from fcagents.tools.outliers import bulk_actions
from fcagents.tools.resample import MIN_POINTS
from fcagents.tools.time_index import detect_time_column

STAGE_ORDER = list(STAGES)
PAGE_FILES = {"data": "views/data.py", "resampling": "views/resampling.py", "holidays": "views/holidays.py",
              "outliers": "views/outliers.py", "missing": "views/missing.py"}
# Settings each stage's agents read. The outliers page chat sets "outlier_bulk", which becomes outlier_actions.
STAGE_PREFS = {**PAGE_KEYS, "outliers": ("outlier_threshold", "outlier_actions")}

# --------------------------------------------------------------------------- app context


def set_app(df: pd.DataFrame | None, data_key: str, llm) -> None:
    """Called by app.py on every run. A new dataset resets preferences, cached stages and chats."""
    ss = st.session_state
    if ss.get("data_key") != data_key:
        ss.update(data_key=data_key, prefs=default_prefs(), stages={}, chat={}, pending={})
    ss["_df"], ss["_llm"] = df, llm


def app_df() -> pd.DataFrame | None:
    return st.session_state.get("_df")


def app_llm():
    return st.session_state["_llm"]


def default_prefs() -> dict:
    out = {k: _native(k, v, {}) for k, v in DEFAULTS.items() if k != "outlier_bulk"}
    return out | {"outlier_actions": ()}


def prefs() -> dict:
    return st.session_state.setdefault("prefs", default_prefs())


def require_data() -> pd.DataFrame:
    df = app_df()
    if df is None:
        st.info("Pick a sample dataset or upload a CSV in the sidebar.")
        st.stop()
    return df


@st.cache_data(show_spinner=False)
def _detect(df: pd.DataFrame) -> dict:
    return detect_time_column(df)


def data_context(df: pd.DataFrame) -> dict:
    """What the data offers for the Data page controls and the chat."""
    rep = _detect(df)
    ids = df[rep["id_col"]].value_counts().index.tolist() if rep["id_col"] else []
    return {"time_cols": [c["column"] for c in rep["candidates"]], "target_cols": rep["target_cols"],
            "series_ids": ids, "id_col": rep["id_col"], "ambiguous": rep["ambiguous"]}


# --------------------------------------------------------------------------- stages


def ensure_stage(stage: str) -> tuple[tuple, list, object]:
    """(key, results, state) for ``stage``; upstream stages run first. Re-runs only when an input changed."""
    ss = st.session_state
    i = STAGE_ORDER.index(stage)
    if i == 0:
        up_key, up_state = ss["data_key"], initial_state(app_df())
    else:
        up_key, _, up_state = ensure_stage(STAGE_ORDER[i - 1])
    stage_prefs = {k: prefs()[k] for k in STAGE_PREFS[stage]}
    key = (up_key, tuple(stage_prefs.items()), app_llm().name)
    cached = ss["stages"].get(stage)
    if cached and cached[0] == key:
        return cached
    results = []
    with st.spinner(f"Running the {STAGE_TITLES[stage]} agents…"):
        for r, state in run_stage(stage, up_state, app_llm(), **stage_prefs):
            results.append(r)
    ss["stages"][stage] = (key, results, state)
    return ss["stages"][stage]


def _stage_status(stage: str) -> str:
    """ok | warning | error | skipped | pending (not run yet, or stale because an earlier step changed)."""
    entry = st.session_state.get("stages", {}).get(stage)
    i = STAGE_ORDER.index(stage)
    if entry is None or (i and st.session_state["stages"].get(STAGE_ORDER[i - 1], (None,))[0] != entry[0][0]):
        return "pending"
    statuses = {r.status for r in entry[1]}
    return next((s for s in ("error", "warning", "ok") if s in statuses), "skipped")


def page_header(stage: str, intro: str) -> None:
    """Signal-flow strip (one status lamp per step), then the step's title and question."""
    current = STAGE_ORDER.index(stage)
    items = []
    for i, s in enumerate(STAGE_ORDER):
        status = _stage_status(s)
        cls = " ".join(c for c in ("is-done" if i < current else "", "is-current" if i == current else "") if c)
        aria = ' aria-current="step"' if i == current else ""
        items.append(f'<li class="{cls}"{aria}>{theme.lamp(status)}'
                     f'<span class="fc-step-name">{i + 1}. {theme.esc(STAGE_TITLES[s])}</span>'
                     f'<span class="fc-step-state">{theme.LAMP[status][1]}</span></li>')
    st.html(f'<nav class="fc-strip" aria-label="Pipeline steps"><ol>{"".join(items)}</ol></nav>')
    st.header(STAGE_TITLES[stage], anchor=False)
    st.caption(intro)


def nav_buttons(stage: str) -> None:
    i = STAGE_ORDER.index(stage)
    left, _, right = st.columns([3, 2, 3])
    if i > 0 and left.button(f"Back to {STAGE_TITLES[STAGE_ORDER[i - 1]]}", icon=":material/arrow_back:",
                             width="stretch"):
        st.switch_page(PAGE_FILES[STAGE_ORDER[i - 1]])
    if i < len(STAGE_ORDER) - 1 and right.button(f"Next: {STAGE_TITLES[STAGE_ORDER[i + 1]]}", type="primary",
                                                  icon=":material/arrow_forward:", width="stretch"):
        st.switch_page(PAGE_FILES[STAGE_ORDER[i + 1]])


# --------------------------------------------------------------------------- controls and chat


def _sync(key: str) -> None:
    prefs()[key] = st.session_state[f"w_{key}"]


def pref_selectbox(label: str, key: str, options: list, container=st, **kw):
    """A selectbox bound to ``prefs()[key]``, which outlives page switches and is also set by the chat."""
    current = prefs()[key]
    st.session_state[f"w_{key}"] = current if current in options else options[0]
    return container.selectbox(label, options, key=f"w_{key}", on_change=_sync, args=(key,), **kw)


def _page_facts(stage: str, results: list) -> dict[str, dict]:
    agents = {a.name: a for a in STAGES[stage]}
    return {r.agent: agents[r.agent].prompt_facts(r.facts) for r in results if r.status != "skipped"}


def _native(key: str, value, ctx: dict):
    """Chat values are strings; convert them to what the pipeline stores."""
    if key == "series_id" and value is not None:  # ids may be ints in the data
        return next((s for s in ctx["series_ids"] if str(s) == value), value)
    if key == "holiday_types" and isinstance(value, str):
        return HOLIDAY_TYPE_CHOICES[value]
    if key == "holiday_window" and value is not None:
        return int(value)
    if key == "outlier_threshold" and value is not None:
        return float(value)
    return value


def chat_panel(stage: str, ctx: dict, entry) -> None:
    """Chat at the top of a page: commands change this page's controls, questions are answered from facts."""
    _, results, state = entry
    history = st.session_state["chat"].setdefault(stage, [])
    with st.container(border=True, key=f"fcchat_{stage}"):
        st.markdown("**Chat with this step**")
        st.caption(HINTS[stage])
        if history:
            box = st.container(height=220)
            for m in history:
                with box.chat_message(m["role"]):
                    st.markdown(m["content"])
                    if m.get("stats"):
                        st.dataframe(pd.Series(m["stats"], name="value").astype(str), width="stretch")
                    if m.get("source"):
                        st.caption({"rule": "rules", "llm-action": "LLM proposal, validated by rules; confirm below",
                                    "llm-answer": "LLM, from computed facts only"}[m["source"]])
        pending = st.session_state["pending"].get(stage)
        if pending:
            apply_col, dismiss_col, _ = st.columns([1, 1, 4])
            apply_col.button("Apply", key=f"apply_{stage}", type="primary", icon=":material/check:",
                             on_click=_resolve, args=(stage, ctx, True))
            dismiss_col.button("Dismiss", key=f"dismiss_{stage}", icon=":material/close:", on_click=_resolve,
                               args=(stage, ctx, False))
        prompt = st.chat_input("Ask about the series, or tell me what to change…", key=f"chat_input_{stage}")
    if not prompt:
        return
    with st.spinner("Thinking…"):
        reply = respond(prompt, stage, ctx, state.series, state.freq, _page_facts(stage, results), app_llm())
    if reply.source == "llm-action":
        st.session_state["pending"][stage] = reply.actions  # applied only after the user confirms
    else:
        st.session_state["pending"].pop(stage, None)
        _apply(reply.actions, ctx)
    history += [{"role": "user", "content": prompt},
                {"role": "assistant", "content": reply.text, "source": reply.source, "stats": reply.stats}]
    st.rerun()


def _apply(actions, ctx: dict) -> None:
    for a in actions:
        if a.key == "outlier_bulk":  # one choice for every point flagged right now
            table = outlier_table()
            prefs()["outlier_actions"] = () if a.value is None or table is None else bulk_actions(table, a.value)
        else:
            prefs()[a.key] = _native(a.key, a.value, ctx)


def stage_result(stage: str, agent: str):
    entry = st.session_state.get("stages", {}).get(stage)
    return next((r for r in entry[1] if r.agent == agent and r.status != "skipped"), None) if entry else None


def outlier_table() -> pd.DataFrame | None:
    r = stage_result("outliers", "outliers")
    return r.extras.get("table") if r else None


def final_covariates(index: pd.DatetimeIndex) -> pd.DataFrame | None:
    """Suggested holiday covariates for ``index`` (e.g. the filled series), or None if none were suggested."""
    r = stage_result("holidays", "holidays")
    if r is None or r.extras.get("covariates") is None:
        return None
    return holiday_covariates(index, r.extras["calendar"], r.extras["effects"])


def _resolve(stage: str, ctx: dict, accept: bool) -> None:
    actions = st.session_state["pending"].pop(stage, [])
    if accept:
        _apply(actions, ctx)
    text = ("Applied: " + "; ".join(a.describe() for a in actions) + ".") if accept else "Dismissed; nothing changed."
    st.session_state["chat"][stage].append({"role": "assistant", "content": text, "source": "rule"})


# --------------------------------------------------------------------------- results and charts


def result_header_html(r) -> str:
    """Channel header: status lamp and title, then the decision / confidence / time readouts."""
    pct = max(0, min(100, round(r.confidence * 100)))
    return (
        f'<div class="fc-card-head" data-decision="{theme.esc(r.decision)}">'
        f'<div class="fc-card-name">{theme.lamp(r.status)}<h3 class="fc-card-title">{theme.esc(r.title)}</h3></div>'
        f'<dl class="fc-readouts">'
        f'<div><dt>Decision</dt><dd>{theme.esc(r.decision)}</dd></div>'
        f'<div><dt>Confidence</dt><dd><span class="fc-meter" aria-hidden="true"><span style="width:{pct}%">'
        f'</span></span>{pct}%</dd></div>'
        f'<div><dt>Time</dt><dd>{r.elapsed_s:.2f} s</dd></div>'
        f'</dl></div>'
    )


def render_result(r) -> None:
    with st.container(border=True, key=f"fccard_{r.agent}"):
        st.html(result_header_html(r))
        st.write(r.summary)
        if r.explanation:  # LLM text: always escaped, never rendered as HTML
            st.html(f'<div class="fc-llm"><small>Explained by the language model, from the facts below</small>'
                    f'<p>{theme.esc(r.explanation)}</p></div>')
        if r.agent == "resampling" and r.facts:
            render_resampling_details(r)
        if r.next_step:
            st.markdown(f"**Next step:** {r.next_step}")
        if r.facts:
            with st.expander("Facts & prompt"):
                st.json(r.facts, expanded=False)
                if r.prompt:
                    st.code(r.prompt, language="text", wrap_lines=True)


def render_resampling_details(r) -> None:
    f = r.facts
    if r.decision == "too_few_points":
        st.warning(f"Only **{f['n_points']}** data points at **{f['chosen_freq']}**; at least {MIN_POINTS} "
                   f"are needed. {r.next_step} Change *Target frequency* above, or type it in the chat.",
                   icon=":material/warning:")
    table = pd.DataFrame(f["options"])
    table["≥ 30 points"] = table["enough_points"].map({True: "yes", False: "no"})
    table["chosen"] = table["freq"].map(
        lambda x: "chosen" if x == f["chosen_freq"] else ("recommended" if x == f["recommended_freq"] else ""))
    table = table[["freq", "label", "n_points", "n_bins", "empty_share", "≥ 30 points", "chosen"]]
    st.caption("Points per candidate frequency (points = bins that contain observations)")
    st.dataframe(table, hide_index=True, width="stretch",
                 column_config={"empty_share": st.column_config.ProgressColumn(
                     "empty bins", format="percent", min_value=0.0, max_value=1.0)})


def plot_series(title: str, series: pd.Series | None, name: str, raw: pd.Series | None = None,
                missing: pd.DatetimeIndex | None = None, filled: pd.DatetimeIndex | None = None,
                marks: list[tuple[str, pd.Series, dict]] | None = None, expected: pd.Series | None = None) -> None:
    """``marks``: extra point layers as (legend name, values indexed by timestamp, plotly marker dict)."""
    if series is None or series.empty:
        return
    s = series[~series.index.isna()].sort_index()
    s = s.groupby(level=0).mean()  # collapse duplicate timestamps for display
    p = theme.palette()
    fig = go.Figure()
    if raw is not None:
        raw = raw.dropna()
        fig.add_trace(go.Scatter(x=raw.index, y=raw.values, mode="markers", name="raw observations",
                                 marker=dict(color=p["raw"], size=4, opacity=0.6)))
    if expected is not None:
        fig.add_trace(go.Scatter(x=expected.index, y=expected.values, mode="lines", name="expected",
                                 line=dict(color=p["raw"], width=1, dash="dot")))
    y = s.values if s.notna().any() else [1] * len(s)
    fig.add_trace(go.Scatter(x=s.index, y=y, mode="lines+markers" if len(s) < 60 else "lines", name=name,
                             line=dict(color=p["trace"], width=2), marker=dict(size=6)))
    if filled is not None and len(filled):
        fig.add_trace(go.Scatter(x=filled, y=s.reindex(filled).values, mode="markers",
                                 name=f"Filled value ({len(filled)})",
                                 marker=dict(color=p["estimate"], size=8, line=dict(width=1.5, color=p["panel"]))))
    if missing is not None and len(missing):
        y0 = float(pd.Series(y).min())
        fig.add_trace(go.Scatter(x=missing[:5000], y=[y0] * min(len(missing), 5000), mode="markers",
                                 name=f"Missing timestamp ({len(missing)})",
                                 marker=dict(color=p["warn"], symbol="x-thin-open", size=10,
                                             line=dict(width=2, color=p["warn"]))))
    for mark_name, pts, marker in marks or []:
        if len(pts):
            fig.add_trace(go.Scatter(x=pts.index, y=pts.values, mode="markers", name=f"{mark_name} ({len(pts)})",
                                     marker=marker))
    fig.update_layout(**theme.plotly_layout(), showlegend=len(fig.data) > 1)
    st.markdown(f"**{title}**")
    st.plotly_chart(fig, width="stretch", theme=None)


def series_title(state, suffix: str = "") -> str:
    title = state.target_col or "Timestamps"
    if state.series_id is not None:
        title += f" · series {state.series_id}"
    return title + suffix
