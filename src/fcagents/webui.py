"""Gradio UI: sidebar (language model, data) and one tab per pipeline step. Rendering only; state lives in
``fcagents.session.Session`` and every decision in the agents."""

from __future__ import annotations

import functools
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

import gradio as gr
import pandas as pd
import plotly.graph_objects as go

from fcagents import theme
from fcagents.catalog import catalog_markdown
from fcagents.chat import HINTS
from fcagents.llm import backend as llm_backend
from fcagents.orchestrator import STAGE_TITLES
from fcagents.session import SOURCE_LABELS, STAGE_ORDER, Session
from fcagents.tools.holidays import HOLIDAY_TYPES, MAX_WINDOW, countries, subdivisions, unofficial_categories
from fcagents.tools.missing import FILL_METHODS
from fcagents.tools.outliers import ACTIONS, THRESHOLDS, bulk_actions
from fcagents.tools.resample import AGGREGATIONS, FREQ_OPTIONS, MIN_POINTS

ROOT = Path(__file__).resolve().parents[2]
SAMPLES = ROOT / "data" / "samples"
TOKEN_URL = "https://huggingface.co/settings/tokens"
BACKEND_LABELS = {"zerogpu": "ZeroGPU model (free, on this Space)", "local": "Local model (in-app, CPU)",
                  "server": "Local LLM server (Ollama, vLLM, LM Studio)", "api": "Hugging Face API (your token)"}
INTROS = {
    "data": "Which column is time, what to forecast, and is the spacing fixed?",
    "resampling": "Irregular timestamps are binned onto a fixed grid. Regular series pass through unless you pick "
                  "a frequency, e.g. to aggregate hourly data to daily.",
    "holidays": "Do holidays, or the days around them, move the series? If so, the forecasting model should know "
                "about them.",
    "outliers": "Which values are far from what the trend and seasonal pattern predict? You decide what happens to "
                "each one; nothing changes until you do.",
    "missing": "Every timestamp on the grid gets a value. Filled values are estimates and are marked in the chart.",
}
NO_DATA = '<div class="fc-card">Pick a sample dataset or upload a CSV in the sidebar.</div>'
OUTLIER_LABELS = {"undecided": "Undecided", **ACTIONS}

# --------------------------------------------------------------------------- data and LLM


def sample_names() -> list[str]:
    return sorted(p.name for p in SAMPLES.glob("*.csv"))


@functools.lru_cache(maxsize=16)
def _read_sample(name: str) -> pd.DataFrame:
    return pd.read_csv(SAMPLES / name)


def load_data(sess: Session, mode: str, sample: str | None, upload: str | None) -> None:
    if mode == "Upload CSV":
        if not upload:
            sess.set_data(None, "")
            return
        raw = Path(upload).read_bytes()
        sess.set_data(pd.read_csv(upload), f"upload:{Path(upload).name}:{hashlib.sha256(raw).hexdigest()[:16]}")
    elif sample:
        sess.set_data(_read_sample(sample).copy(), sample)


def backend_choices() -> list[str]:
    """A Space offers its ZeroGPU model (if it has one) and hosted models with the visitor's own token;
    local runs serve every model locally."""
    if llm_backend.on_hf_space():
        return (["zerogpu"] if llm_backend.zerogpu_available() else ["local"]) + ["api"]
    return ["local", "server"]


def default_backend() -> str:
    keys, wanted = backend_choices(), os.environ.get("LLM_BACKEND", llm_backend.DEFAULT_BACKEND)
    if wanted in keys:
        return wanted
    return keys[0]


@functools.lru_cache(maxsize=2)
def _local_llm(model_id: str):
    """Local model, shared by all visitors. Holds no credentials, so a global cache is safe."""
    return llm_backend.TransformersBackend(model_id)


def configure_llm(sess: Session, use: bool, kind: str, local_model: str, url: str, server_model: str | None,
                  api_model: str, provider: str, token: str) -> str:
    """Set ``sess.llm`` from the sidebar; return the status line. The API token lives only in this session."""
    llm, note = llm_backend.NullBackend(), ""
    if use:
        try:
            if kind == "zerogpu":
                llm = llm_backend.zerogpu_backend()
            elif kind == "local":
                llm = _local_llm((local_model or llm_backend.DEFAULT_MODEL_ID).strip())
            elif kind == "server":
                if server_model:
                    llm = llm_backend.LocalServerBackend(url, server_model)
                else:
                    note = (f"Can't reach an LLM server at {url}. Start Ollama, vLLM or LM Studio, or check the URL. "
                            "Until then, agents use rule-based summaries.")
            elif kind == "api" and llm_backend.on_hf_space():
                token = (token or "").strip()
                if not token:
                    note = (f"Paste your own [Hugging Face token]({TOKEN_URL}) to use the hosted model. "
                            "Until then, agents use rule-based summaries.")
                else:
                    key = (api_model, provider, hashlib.sha256(token.encode()).hexdigest())
                    if getattr(sess.llm, "_session_key", None) == key:
                        llm = sess.llm
                    else:
                        llm = llm_backend.InferenceAPIBackend(api_model, token, provider)
                        llm._session_key = key
        except Exception as e:  # missing torch, no network, OOM, bad model id…
            note = f"Could not set up the language model: {e}. Falling back to rule-based summaries."
            llm = llm_backend.NullBackend()
    sess.llm = llm
    return llm_status(sess, note)


def llm_status(sess: Session, note: str = "") -> str:
    out = f"Backend: `{sess.llm.name}`"
    if note:
        out = f"{note}\n\n{out}"
    if err := getattr(sess.llm, "last_error", None):
        out += f"\n\n**Last LLM call failed:** {err}"
    return out


# --------------------------------------------------------------------------- HTML pieces


def _inline_md(text) -> str:
    """Escape, then render the two inline Markdown forms agent texts use: `code` and **bold**."""
    out = theme.esc(text)
    out = re.sub(r"`([^`]+)`", r"<code>\1</code>", out)
    return re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", out)


def strip_html(sess: Session, stage: str) -> str:
    """Signal-flow strip (one status lamp per step), then the step's title and question."""
    current = STAGE_ORDER.index(stage)
    items = []
    for i, s in enumerate(STAGE_ORDER):
        status = sess.stage_status(s)
        cls = " ".join(c for c in ("is-done" if i < current else "", "is-current" if i == current else "") if c)
        aria = ' aria-current="step"' if i == current else ""
        items.append(f'<li class="{cls}"{aria}>{theme.lamp(status)}'
                     f'<span class="fc-step-name">{i + 1}. {theme.esc(STAGE_TITLES[s])}</span>'
                     f'<span class="fc-step-state">{theme.LAMP[status][1]}</span></li>')
    return (f'<nav class="fc-strip" aria-label="Pipeline steps"><ol>{"".join(items)}</ol></nav>'
            f'<h2>{theme.esc(STAGE_TITLES[stage])}</h2><p class="fc-intro">{theme.esc(INTROS[stage])}</p>')


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


def card_html(r) -> str:
    parts = [result_header_html(r), f"<p>{_inline_md(r.summary)}</p>"]
    if r.explanation:  # LLM text: always escaped, never rendered as HTML
        parts.append(f'<div class="fc-llm"><small>Explained by the language model, from the facts below</small>'
                     f'<p>{theme.esc(r.explanation)}</p></div>')
    if r.agent == "resampling" and r.decision == "too_few_points" and r.facts:
        f = r.facts
        parts.append(f'<p class="fc-warn">Only <strong>{f["n_points"]}</strong> data points at '
                     f'<strong>{theme.esc(f["chosen_freq"])}</strong>; at least {MIN_POINTS} are needed. '
                     f'{_inline_md(r.next_step)} Change <em>Target frequency</em> above, or type it in the chat.</p>')
    if r.next_step:
        parts.append(f'<p class="fc-next"><strong>Next step:</strong> {_inline_md(r.next_step)}</p>')
    if r.facts:
        facts = theme.esc(json.dumps(r.facts, indent=2, default=str))
        prompt = f"<pre>{theme.esc(r.prompt)}</pre>" if r.prompt else ""
        parts.append(f"<details><summary>Facts &amp; prompt</summary><pre>{facts}</pre>{prompt}</details>")
    return f'<div class="fc-card">{"".join(parts)}</div>'


def cards_html(results: list) -> str:
    return "".join(card_html(r) for r in results)


def chat_messages(sess: Session, stage: str) -> list[dict]:
    out = []
    for m in sess.chat.get(stage, []):
        text = m["content"]
        if m.get("stats"):
            text += "\n\n| | value |\n|---|---|\n" + "\n".join(f"| {k} | {v} |" for k, v in m["stats"].items())
        if m.get("source"):
            text += f"\n\n<sub>{SOURCE_LABELS[m['source']]}</sub>"
        out.append({"role": m["role"], "content": text})
    return out


# --------------------------------------------------------------------------- charts


def series_figure(series: pd.Series | None, name: str, raw: pd.Series | None = None,
                  missing: pd.DatetimeIndex | None = None, filled: pd.DatetimeIndex | None = None,
                  marks: list[tuple[str, pd.Series, dict]] | None = None,
                  expected: pd.Series | None = None, title: str = "") -> go.Figure | None:
    """``marks``: extra point layers as (legend name, values indexed by timestamp, plotly marker dict)."""
    if series is None or series.empty:
        return None
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
    if title:
        fig.update_layout(title=dict(text=title, x=0, font=dict(size=14)), margin=dict(t=64))
    return fig


def series_title(state, suffix: str = "") -> str:
    title = state.target_col or "Timestamps"
    if state.series_id is not None:
        title += f" · series {state.series_id}"
    return title + suffix


def _csv_file(df: pd.DataFrame, name: str) -> str:
    path = Path(tempfile.mkdtemp(prefix="fcagents-")) / name
    df.to_csv(path, index_label="timestamp")
    return str(path)


# --------------------------------------------------------------------------- per-step views
# Each returns {name: value or gr.update} for that step's components (see Ui.outputs).


def _first(choices: list, current):
    return current if current in choices else (choices[0] if choices else None)


def view_data(sess: Session, results, state) -> dict:
    ctx = sess.data_context()
    p = sess.prefs
    missing = None
    reg = state.findings.get("regularity")
    if reg is not None and reg.facts.get("freq") and state.series is not None:
        idx = state.series.index.dropna().unique().sort_values()
        missing = pd.date_range(idx[0], idx[-1], freq=reg.facts["freq"]).difference(idx)
    sid_label = f"Series ({ctx['id_col']})" if ctx["id_col"] else "Series"
    return {
        "time_col": gr.update(choices=ctx["time_cols"], value=_first(ctx["time_cols"], p["time_col"]),
                              visible=bool(ctx["time_cols"])),
        "target_col": gr.update(choices=ctx["target_cols"], value=_first(ctx["target_cols"], p["target_col"]),
                                visible=bool(ctx["target_cols"])),
        "series_id": gr.update(choices=[(str(s), str(s)) for s in ctx["series_ids"]], label=sid_label,
                               value=str(_first(ctx["series_ids"], p["series_id"])) if ctx["series_ids"] else None,
                               visible=bool(ctx["series_ids"])),
        "head": gr.update(value=sess.df.head(8), visible=True),
        "plot": series_figure(state.series, state.target_col or "timestamps", missing=missing,
                              title=series_title(state)),
    }


def view_resampling(sess: Session, results, state) -> dict:
    p = sess.prefs
    resampled = next((r for r in results if r.agent == "resampling" and r.status != "skipped"), None)
    freq_result = next((r for r in results if r.agent == "frequency" and r.status != "skipped"), None)
    missing = freq_result.extras.get("missing_index") if freq_result else None
    table = gr.update(value=None, visible=False)
    if resampled is not None:
        f = resampled.facts
        raw = resampled.extras["raw_series"] if f["agg"] != "count" else None
        fig = series_figure(state.series, f"{f['agg']} per {f['chosen_freq']}", raw=raw, missing=missing,
                            title=series_title(state, f" · resampled to {f['chosen_freq']} ({f['agg']})"))
        if f.get("options"):
            t = pd.DataFrame(f["options"])
            t["≥ 30 points"] = t["enough_points"].map({True: "yes", False: "no"})
            t["chosen"] = t["freq"].map(lambda x: "chosen" if x == f["chosen_freq"]
                                        else ("recommended" if x == f["recommended_freq"] else ""))
            t["empty bins"] = (t["empty_share"] * 100).round(1).astype(str) + "%"
            table = gr.update(value=t[["freq", "label", "n_points", "n_bins", "empty bins", "≥ 30 points", "chosen"]],
                              visible=True)
    else:
        fig = series_figure(state.series, state.target_col or "timestamps", missing=missing,
                            title=series_title(state))
    return {"resample_freq": p["resample_freq"], "resample_agg": p["resample_agg"], "options": table, "plot": fig}


def holiday_profile_figure(r) -> go.Figure:
    profile = r.extras["profile"]
    pal = theme.palette()
    fig = go.Figure(go.Bar(
        x=profile["offset"], y=profile["mean_abs_z"],
        marker_color=[pal["trace"] if e else pal["raw"] for e in profile["has_effect"]],
        hovertemplate="%{x} days: %{y:.2f}<extra></extra>"))
    fig.add_hline(y=r.facts["ordinary_abs_z"], line=dict(color=pal["warn"], dash="dot", width=1.5),
                  annotation_text="ordinary day", annotation_position="top right")
    fig.update_layout(**theme.plotly_layout(height=300))
    fig.update_layout(title=dict(text="How far from normal the days around holidays are (blue: a holiday moves "
                                      "the series on that day every year)", x=0, font=dict(size=14)),
                      margin=dict(t=64))
    fig.update_xaxes(title_text="Days from the holiday (negative = before)", dtick=1)
    fig.update_yaxes(title_text="Distance from normal (mean |z|)")
    return fig


def view_holidays(sess: Session, results, state) -> dict:
    p = sess.prefs
    names = countries()
    subdivs = subdivisions(p["holiday_country"])
    has_unofficial = bool(unofficial_categories(p["holiday_country"]))
    out = {
        "holiday_country": p["holiday_country"],
        "holiday_subdiv": gr.update(
            choices=[(f"All of {names[p['holiday_country']]}", "")] + [(f"{n} ({c})", c) for c, n in subdivs.items()],
            value=p["holiday_subdiv"] if p["holiday_subdiv"] in subdivs else "", interactive=bool(subdivs)),
        "holiday_window": p["holiday_window"],
        "type_official": "official" in p["holiday_types"],
        "type_unofficial": gr.update(value="unofficial" in p["holiday_types"], interactive=has_unofficial),
        "profile": None, "plot": None,
        "occurrences": gr.update(value=None, visible=False), "covariates": gr.update(value=None, visible=False),
        "download": gr.update(value=None, visible=False),
    }
    r = results[0]
    ex = r.extras if r.status != "skipped" else {}
    profile = ex.get("profile")
    if profile is None or not len(profile):
        return out
    daily, pal = ex["daily"], theme.palette()
    hol = ex["occurrences"].set_index("date")["value"]
    prot = state.protected_index
    around = daily.reindex(prot.normalize().unique()).dropna().drop(hol.index, errors="ignore") if prot is not None \
        else pd.Series(dtype=float)
    occ = ex["occurrences"].assign(date=lambda d: d["date"].dt.date.astype(str))
    occ = occ[["date", "holiday", "value", "expected", "change_pct", "z"]].round(
        {"value": 1, "expected": 1, "change_pct": 1, "z": 2}).rename(columns={"change_pct": "change %"})
    out |= {
        "profile": holiday_profile_figure(r),
        "plot": series_figure(daily, "per day", title=series_title(state, " · per day"), marks=[
            ("Holiday", hol, dict(color=pal["trace"], size=10, symbol="diamond",
                                  line=dict(width=1.5, color=pal["panel"]))),
            ("Day affected by a holiday", around, dict(color=pal["trace"], size=7, symbol="circle-open",
                                                       line=dict(width=1.5)))]),
        "occurrences": gr.update(value=occ, visible=True),
    }
    cov = ex.get("covariates")
    if cov is not None:
        out["covariates"] = gr.update(value=cov.sum().rename("days marked").rename_axis("covariate").reset_index(),
                                      visible=True)
        out["download"] = gr.update(value=_csv_file(cov, f"holiday_covariates_{r.facts['country']}.csv"),
                                    visible=True)
    return out


def outlier_view_table(table: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": [pd.Timestamp(t).isoformat() for t in table.index],
        "value": table["value"].round(3).to_numpy(), "expected": table["expected"].round(3).to_numpy(),
        "z": table["z"].round(1).to_numpy(), "suggestion": table["suggestion"].map(ACTIONS).to_numpy(),
        "action": table["action"].map(OUTLIER_LABELS).to_numpy(),
        "replacement": [round(float(e if v is None or pd.isna(v) else v), 3)
                        for v, e in zip(table["replacement"], table["expected"])],
    })


def outlier_actions_from_view(view: pd.DataFrame, table: pd.DataFrame) -> tuple[tuple, list[str]]:
    """The user's edited table → (outlier_actions, problems). Unknown actions count as undecided."""
    back = {v.lower(): k for k, v in OUTLIER_LABELS.items()} | {k: k for k in OUTLIER_LABELS}
    expected = {pd.Timestamp(t).isoformat(): e for t, e in zip(table.index, table["expected"])}
    actions, problems = [], []
    for row in view.itertuples(index=False):
        ts, typed = str(row.timestamp), str(row.action).strip().lower()
        act = back.get(typed)
        if act is None:
            problems.append(f"“{row.action}” at {ts} is not keep, remove, replace or undecided.")
            continue
        if act == "undecided" or ts not in expected:
            continue
        value = None
        if act == "replace":
            try:
                rep = float(row.replacement)
                value = None if abs(rep - round(float(expected[ts]), 3)) < 1e-9 else rep
            except (TypeError, ValueError):
                problems.append(f"Replacement at {ts} is not a number; the expected value is used.")
        actions.append((ts, act, value))
    return tuple(actions), problems


def view_outliers(sess: Session, results, state) -> dict:
    out = {"outlier_threshold": sess.prefs["outlier_threshold"], "plot": None,
           "editor": gr.update(value=None, visible=False), "bulk": gr.update(visible=False)}
    r = results[0]
    if r.status == "skipped":
        return out
    table: pd.DataFrame = r.extras["table"]
    pal = theme.palette()
    prot = r.extras.get("protected")
    series = state.series
    marks = [("Flagged", table["value"], dict(color=pal["error"], size=11, symbol="x-thin-open",
                                              line=dict(width=2.5, color=pal["error"])))]
    if prot is not None and len(prot):
        marks.append(("Holiday-affected, not flagged", series.reindex(prot).dropna(),
                      dict(color=pal["trace"], size=7, symbol="circle-open", line=dict(width=1.5))))
    changed = table.index[table["action"].isin(["remove", "replace"])]
    if len(changed):
        marks.append(("Your change", series.reindex(changed).dropna(),
                      dict(color=pal["estimate"], size=9, line=dict(width=1.5, color=pal["panel"]))))
    out["plot"] = series_figure(series, state.target_col or "value", marks=marks, expected=r.extras["expected"],
                                title=series_title(state, " · after your choices"))
    if len(table):
        out["editor"] = gr.update(value=outlier_view_table(table), visible=True)
        out["bulk"] = gr.update(visible=True)
    return out


def view_missing(sess: Session, results, state) -> dict:
    out = {"fill_method": sess.prefs["fill_method"], "plot": None, "download": gr.update(value=None, visible=False),
           "note": ""}
    r = results[0]
    if r.status == "skipped":
        return out
    out["plot"] = series_figure(state.series, state.target_col or "value", filled=r.extras["filled_index"],
                                title=series_title(state, f" · {r.facts['method_label']}"))
    frame = state.series.to_frame()
    cov = sess.final_covariates(frame.index)
    if cov is not None:
        frame = frame.join(cov)
        out["note"] = f"The download includes {len(cov.columns)} holiday covariate columns from the Holidays step."
    name = f"{state.target_col or 'series'}_{state.freq}_prepared.csv"
    out["download"] = gr.update(value=_csv_file(frame, name), visible=True)
    return out


VIEWS = {"data": view_data, "resampling": view_resampling, "holidays": view_holidays, "outliers": view_outliers,
         "missing": view_missing}
# Components each step's view fills besides the shared ones (strip, chat, apply, dismiss, cards).
VIEW_KEYS = {
    "data": ("time_col", "target_col", "series_id", "head", "plot"),
    "resampling": ("resample_freq", "resample_agg", "options", "plot"),
    "holidays": ("holiday_country", "holiday_subdiv", "holiday_window", "type_official", "type_unofficial",
                 "profile", "plot", "occurrences", "covariates", "download"),
    "outliers": ("outlier_threshold", "bulk", "plot", "editor"),
    "missing": ("fill_method", "note", "plot", "download"),
}


def render(sess: Session, stage: str) -> dict[str, object]:
    """Everything a step tab shows, by component name. Runs the step's agents if an input changed."""
    pending = bool(sess.pending.get(stage))
    out = {"chat": chat_messages(sess, stage), "apply": gr.update(visible=pending),
           "dismiss": gr.update(visible=pending)}
    if sess.df is None:
        empty = {k: gr.update() for k in VIEW_KEYS[stage]} | {"plot": None}
        return out | empty | {"strip": strip_html(sess, stage), "cards": NO_DATA}
    _, results, state = sess.ensure_stage(stage)
    out |= VIEWS[stage](sess, results, state)
    return out | {"strip": strip_html(sess, stage), "cards": cards_html(results)}


# --------------------------------------------------------------------------- layout and events


class Ui:
    """Component registry: ``tabs[stage][name]`` → component."""

    def __init__(self):
        self.tabs: dict[str, dict[str, gr.components.Component]] = {}

    def outputs(self, stage: str) -> list:
        return list(self.tabs[stage].values())

    def updates(self, stage: str, sess: Session) -> dict:
        values = render(sess, stage)
        return {comp: values[name] for name, comp in self.tabs[stage].items() if name in values}


def _step_tab(ui: Ui, stage: str, build_controls) -> dict:
    c: dict[str, gr.components.Component] = {}
    c["strip"] = gr.HTML()
    with gr.Group():
        c["chat"] = gr.Chatbot(label="Chat with this step", height=220, placeholder=HINTS[stage],
                               buttons=["copy"])
        with gr.Row():
            c["msg"] = gr.Textbox(placeholder="Ask about the series, or tell me what to change…", show_label=False,
                                  scale=6, submit_btn=True)
        with gr.Row():
            c["apply"] = gr.Button("Apply", variant="primary", visible=False, size="sm")
            c["dismiss"] = gr.Button("Dismiss", visible=False, size="sm")
    build_controls(c)
    c["cards"] = gr.HTML()
    c["plot"] = gr.Plot(show_label=False)
    ui.tabs[stage] = {k: v for k, v in c.items() if k != "msg"}
    return c


def build_app() -> gr.Blocks:
    ui = Ui()
    names = countries()
    with gr.Blocks(title="Forecasting Agents") as demo:
        sess = gr.State(lambda: Session())  # a factory: one fresh Session per visitor
        current = gr.State("data")

        # ------------------------------------------------------------- sidebar
        with gr.Sidebar(open=True, width=340):
            gr.Markdown("## Language model")
            use_llm = gr.Checkbox(label="Use an LLM", value=os.environ.get("USE_LLM", "1") == "1",
                                  info="Explains agent results and answers chat questions. Tools compute every "
                                       "fact; the LLM never makes a decision.")
            keys = backend_choices()
            kind = gr.Radio([(BACKEND_LABELS[k], k) for k in keys], value=default_backend(), label="Backend")
            local_model = gr.Textbox(llm_backend.DEFAULT_MODEL_ID, label="Model",
                                     info="Any public Hugging Face chat model that fits on CPU.",
                                     visible=default_backend() == "local")
            server_url = gr.Textbox(llm_backend.DEFAULT_SERVER_URL, label="Server URL",
                                    info="OpenAI-compatible endpoint, e.g. Ollama http://localhost:11434/v1.",
                                    visible=default_backend() == "server")
            server_model = gr.Dropdown([], label="Model", visible=default_backend() == "server")
            api_model = gr.Textbox(llm_backend.DEFAULT_API_MODEL, label="Model",
                                   info="Any chat model served by Hugging Face Inference Providers.",
                                   visible=default_backend() == "api")
            provider = gr.Dropdown(llm_backend.API_PROVIDERS, value=llm_backend.API_PROVIDERS[0], label="Provider",
                                   visible=default_backend() == "api")
            token = gr.Textbox(label="Your Hugging Face token", type="password", placeholder="hf_…",
                               info="Fine-grained token with “Make calls to Inference Providers”. Calls are billed "
                                    "to your account. It stays in this browser session and is never stored.",
                               visible=default_backend() == "api")
            status = gr.Markdown()
            gr.Markdown("---\n## Data")
            mode = gr.Radio(["Sample dataset", "Upload CSV"], value="Sample dataset", show_label=False)
            samples = sample_names()
            sample = gr.Dropdown(samples, value=samples[0] if samples else None, label="Sample")
            upload = gr.File(label="CSV file", file_types=[".csv"], type="filepath", visible=False)
            gr.Markdown("<small>Tools compute the facts, rules make the decisions, and the language model explains "
                        "them.</small>")

        gr.HTML('<p class="fc-brand">Forecasting Agents</p>')
        with gr.Tabs(selected="data") as tabs:
            tab_objs = {}

            def data_controls(c):
                with gr.Accordion("Columns", open=True):
                    with gr.Row():
                        c["time_col"] = gr.Dropdown(label="Time column", visible=False)
                        c["target_col"] = gr.Dropdown(label="Target column", visible=False)
                        c["series_id"] = gr.Dropdown(label="Series", visible=False)
                c["head"] = gr.Dataframe(label="First rows", interactive=False, visible=False, max_height=300)

            def resampling_controls(c):
                with gr.Row():
                    c["resample_freq"] = gr.Dropdown(
                        [("Auto (recommended)", "auto")] + [(f"{f} · {n}", f) for f, n in FREQ_OPTIONS.items()],
                        label="Target frequency (1h – 1W)",
                        info=f"Finer frequencies give more data points. At least {MIN_POINTS} are needed.")
                    c["resample_agg"] = gr.Dropdown(AGGREGATIONS, label="Aggregate values per bin",
                                                    info="Event logs without a numeric value are always counted.")

            def resampling_after(c):
                c["options"] = gr.Dataframe(label="Points per candidate frequency (points = bins that contain "
                                                  "observations)", interactive=False, visible=False)

            def holidays_controls(c):
                with gr.Row():
                    c["holiday_country"] = gr.Dropdown([(f"{n} ({k})", k) for k, n in names.items()], label="Country")
                    c["holiday_subdiv"] = gr.Dropdown(label="State / region",
                                                      info="Regional holidays are added for the chosen state.")
                    c["holiday_window"] = gr.Slider(0, MAX_WINDOW, step=1, label="Days before and after each holiday",
                                                    info="Effects such as a shopping run-up before Christmas show up "
                                                         "on these days.")
                with gr.Row():
                    c["type_official"] = gr.Checkbox(label=HOLIDAY_TYPES["official"])
                    c["type_unofficial"] = gr.Checkbox(label=HOLIDAY_TYPES["unofficial"])

            def outliers_controls(c):
                c["outlier_threshold"] = gr.Dropdown([(v, k) for k, v in THRESHOLDS.items()], label="Sensitivity",
                                                     info="Lower cut-offs flag more points. Holiday-affected days "
                                                          "are never flagged.")

            def missing_controls(c):
                c["fill_method"] = gr.Dropdown(
                    [("Auto (recommended)", "auto")] + [(v, k) for k, v in FILL_METHODS.items()], label="Fill method",
                    info="Auto: zero for counts and sums, seasonal when two full seasons exist, otherwise "
                         "interpolation.")

            built = {}
            for i, stage in enumerate(STAGE_ORDER):
                with gr.Tab(f"{i + 1}. {STAGE_TITLES[stage]}", id=stage) as tab:
                    tab_objs[stage] = tab
                    controls = {"data": data_controls, "resampling": resampling_controls,
                                "holidays": holidays_controls, "outliers": outliers_controls,
                                "missing": missing_controls}[stage]
                    c = _step_tab(ui, stage, controls)
                    if stage == "resampling":
                        resampling_after(c)
                    elif stage == "holidays":
                        c["profile"] = gr.Plot(show_label=False)
                        with gr.Row():
                            c["occurrences"] = gr.Dataframe(label="Holidays in the data", interactive=False,
                                                            visible=False, max_height=320, scale=3)
                            with gr.Column(scale=2):
                                c["covariates"] = gr.Dataframe(label="Suggested covariates", interactive=False,
                                                               visible=False, max_height=260)
                                c["download"] = gr.DownloadButton("Download holiday covariates (CSV)", visible=False)
                    elif stage == "outliers":
                        with gr.Column(visible=False) as bulk:
                            gr.Markdown("**Decide per point.** Keep: the value stays. Remove: it becomes a gap that "
                                        "the Missing Values agent fills. Replace: it gets the replacement value (the "
                                        "expected value unless you type another). Edit the *action* column "
                                        "(keep / remove / replace / undecided) or use the buttons.")
                            with gr.Row():
                                c["btn_suggested"] = gr.Button("Apply suggestions", variant="primary", size="sm")
                                c["btn_keep"] = gr.Button("Keep all", size="sm")
                                c["btn_clear"] = gr.Button("Clear choices", size="sm")
                        c["bulk"] = bulk
                        c["editor"] = gr.Dataframe(interactive=True, visible=False, max_height=360,
                                                   static_columns=[0, 1, 2, 3, 4])
                    elif stage == "missing":
                        c["note"] = gr.Markdown()
                        c["download"] = gr.DownloadButton("Download prepared series (CSV)", visible=False)
                    with gr.Row():
                        if i > 0:
                            c["back"] = gr.Button(f"← Back to {STAGE_TITLES[STAGE_ORDER[i - 1]]}")
                        if i < len(STAGE_ORDER) - 1:
                            c["next"] = gr.Button(f"Next: {STAGE_TITLES[STAGE_ORDER[i + 1]]} →", variant="primary")
                    built[stage] = c
                    for k in ("back", "next", "btn_suggested", "btn_keep", "btn_clear"):
                        ui.tabs[stage].pop(k, None)
                    for k in VIEW_KEYS[stage]:
                        ui.tabs[stage][k] = c[k]
            with gr.Tab("Agents", id="agents"):
                agents_backend = gr.Markdown()
                gr.Markdown(catalog_markdown())

        # ------------------------------------------------------------- events
        def const(value):
            return lambda: value

        def show(stage):
            def fn(s: Session):
                return ui.updates(stage, s)
            return fn

        def refresh_current(s: Session, cur: str):
            if cur not in ui.tabs:
                return {agents_backend: _agents_note(s)}
            return ui.updates(cur, s) | {agents_backend: _agents_note(s)}

        all_outputs = [agents_backend] + [comp for st_ in STAGE_ORDER for comp in ui.outputs(st_)]

        llm_inputs = [sess, use_llm, kind, local_model, server_url, server_model, api_model, provider, token]

        def on_llm(s, use, k, lm, url, sm, am, prov, tok):
            return configure_llm(s, use, k, lm, url, sm, am, prov, tok)

        def on_kind(k, url):
            models = llm_backend.list_server_models(url) if k == "server" else []
            wanted = os.environ.get("LOCAL_LLM_MODEL", llm_backend.DEFAULT_SERVER_MODEL)
            return (gr.update(visible=k == "local"), gr.update(visible=k == "server"),
                    gr.update(visible=k == "server", choices=models, value=_first(models, wanted)),
                    gr.update(visible=k == "api"), gr.update(visible=k == "api"), gr.update(visible=k == "api"))

        kind_outputs = [local_model, server_url, server_model, api_model, provider, token]
        kind.input(on_kind, [kind, server_url], kind_outputs).then(on_llm, llm_inputs, status).then(
            refresh_current, [sess, current], all_outputs)
        server_url.submit(on_kind, [kind, server_url], kind_outputs).then(on_llm, llm_inputs, status).then(
            refresh_current, [sess, current], all_outputs)
        for comp in (use_llm, server_model, provider):
            comp.input(on_llm, llm_inputs, status).then(refresh_current, [sess, current], all_outputs)
        for comp in (local_model, api_model, token):
            comp.blur(on_llm, llm_inputs, status).then(refresh_current, [sess, current], all_outputs)
            comp.submit(on_llm, llm_inputs, status).then(refresh_current, [sess, current], all_outputs)

        def on_data(s, m, smp, up):
            load_data(s, m, smp, up)

        mode.input(lambda m: (gr.update(visible=m == "Sample dataset"), gr.update(visible=m == "Upload CSV")),
                   mode, [sample, upload]).then(on_data, [sess, mode, sample, upload]).then(
            refresh_current, [sess, current], all_outputs)
        for comp, ev in ((sample, "input"), (upload, "change")):
            getattr(comp, ev)(on_data, [sess, mode, sample, upload]).then(
                refresh_current, [sess, current], all_outputs)

        demo.load(on_kind, [kind, server_url], kind_outputs).then(on_llm, llm_inputs, status).then(
            on_data, [sess, mode, sample, upload]).then(const("data"), None, current).then(
            show("data"), sess, ui.outputs("data"))

        def go_to(target):
            return lambda: (gr.Tabs(selected=target), target)

        for stage, tab in tab_objs.items():
            tab.select(const(stage), None, current).then(show(stage), sess, ui.outputs(stage))
        for i, stage in enumerate(STAGE_ORDER):
            c = built[stage]
            outs = ui.outputs(stage)
            if i > 0:
                prev = STAGE_ORDER[i - 1]
                c["back"].click(go_to(prev), None, [tabs, current]).then(show(prev), sess, ui.outputs(prev))
            if i < len(STAGE_ORDER) - 1:
                nxt = STAGE_ORDER[i + 1]
                c["next"].click(go_to(nxt), None, [tabs, current]).then(show(nxt), sess, ui.outputs(nxt))

            def on_msg(s, text, st_=stage):
                s.send(st_, text)
                return ""

            c["msg"].submit(on_msg, [sess, c["msg"]], c["msg"]).then(show(stage), sess, outs).then(
                lambda s: llm_status(s), sess, status)

            def on_resolve(s, accept, st_=stage):
                s.resolve(st_, accept)

            c["apply"].click(lambda s, st_=stage: on_resolve(s, True, st_), sess).then(show(stage), sess, outs)
            c["dismiss"].click(lambda s, st_=stage: on_resolve(s, False, st_), sess).then(show(stage), sess, outs)

            def bind_pref(comp, key, st_=stage, convert=lambda v: v):
                def fn(s, v):
                    s.prefs[key] = convert(v)
                comp.input(fn, [sess, comp]).then(show(st_), sess, ui.outputs(st_))

            if stage == "data":
                ids = lambda s: s.data_context()["series_ids"]  # noqa: E731
                bind_pref(c["time_col"], "time_col")
                bind_pref(c["target_col"], "target_col")

                def set_series(s, v):
                    s.prefs["series_id"] = next((x for x in ids(s) if str(x) == str(v)), v)
                c["series_id"].input(set_series, [sess, c["series_id"]]).then(show(stage), sess, outs)
            elif stage == "resampling":
                bind_pref(c["resample_freq"], "resample_freq")
                bind_pref(c["resample_agg"], "resample_agg")
            elif stage == "holidays":
                def set_country(s, v):
                    s.prefs.update(holiday_country=v, holiday_subdiv=None)
                c["holiday_country"].input(set_country, [sess, c["holiday_country"]]).then(show(stage), sess, outs)
                bind_pref(c["holiday_subdiv"], "holiday_subdiv", convert=lambda v: v or None)
                c["holiday_window"].release(lambda s, v: s.prefs.update(holiday_window=int(v)),
                                            [sess, c["holiday_window"]]).then(show(stage), sess, outs)

                def set_types(s, official, unofficial):
                    s.prefs["holiday_types"] = tuple(t for t, on in (("official", official),
                                                                     ("unofficial", unofficial)) if on)
                for box in (c["type_official"], c["type_unofficial"]):
                    box.input(set_types, [sess, c["type_official"], c["type_unofficial"]]).then(
                        show(stage), sess, outs)
            elif stage == "outliers":
                bind_pref(c["outlier_threshold"], "outlier_threshold", convert=float)

                def bulk(s, choice):
                    table = s.outlier_table()
                    s.prefs["outlier_actions"] = () if choice is None or table is None else bulk_actions(table, choice)
                c["btn_suggested"].click(lambda s: bulk(s, "suggested"), sess).then(show(stage), sess, outs)
                c["btn_keep"].click(lambda s: bulk(s, "keep"), sess).then(show(stage), sess, outs)
                c["btn_clear"].click(lambda s: bulk(s, None), sess).then(show(stage), sess, outs)

                def edited(s, view):
                    table = s.outlier_table()
                    if table is None or view is None or not len(view):
                        return
                    actions, problems = outlier_actions_from_view(pd.DataFrame(view), table)
                    for msg in problems:
                        gr.Warning(msg)
                    s.prefs["outlier_actions"] = actions
                c["editor"].input(edited, [sess, c["editor"]]).then(show(stage), sess, outs)
            elif stage == "missing":
                bind_pref(c["fill_method"], "fill_method")
    return demo


def _agents_note(sess: Session) -> str:
    return (f"LLM backend in use: `{sess.llm.name}`. It explains results and answers chat questions from computed "
            "facts; it never makes a decision. Chat changes it proposes are validated by rules.")
