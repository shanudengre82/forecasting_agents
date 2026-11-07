"""Visual tokens for the "instrument" look, shared by the page CSS and the Plotly charts.

Colours mirror .streamlit/config.toml. Meaning is fixed everywhere: trace = observed data, estimate = filled
values, warn = missing timestamps / needs attention, ok / error = agent status.
"""

from __future__ import annotations

import html

import streamlit as st

LIGHT = {
    "bench": "#E9EDF0", "panel": "#FFFFFF", "ink": "#17222C", "muted": "#5B6976", "rule": "#C5CFD7",
    "grid": "#DCE3E8", "trace": "#1B5FA8", "estimate": "#7A4FC4", "raw": "#8796A3",
    "ok": "#2F7D5B", "warn": "#B7860B", "error": "#C0392B", "idle": "#AEB9C2",
}
DARK = {
    "bench": "#151B21", "panel": "#1D252D", "ink": "#E3E9EE", "muted": "#9AA8B4", "rule": "#33404B",
    "grid": "#28323C", "trace": "#5FA2E8", "estimate": "#A88BEA", "raw": "#6F7E8B",
    "ok": "#4FB386", "warn": "#E0B23A", "error": "#E8675A", "idle": "#4A5864",
}
FONT = "Archivo, system-ui, sans-serif"

# Agent status → lamp colour token and the words a screen reader (and the legend) uses.
LAMP = {
    "ok": ("ok", "Checked"),
    "warning": ("warn", "Needs attention"),
    "error": ("error", "Stopped"),
    "skipped": ("idle", "Skipped"),
    "pending": ("idle", "Not run yet"),
}


def is_dark() -> bool:
    try:
        return st.context.theme.type == "dark"
    except Exception:  # bare mode / tests
        return False


def palette() -> dict[str, str]:
    return DARK if is_dark() else LIGHT


def esc(value) -> str:
    return html.escape(str(value), quote=True)


def lamp(status: str) -> str:
    token, label = LAMP.get(status, LAMP["pending"])
    return f'<span class="fc-lamp" style="--lamp: var(--fc-{token})" role="img" aria-label="{label}"></span>'


def inject_css() -> None:
    p = palette()
    tokens = "".join(f"--fc-{k}: {v};" for k, v in p.items())
    st.html(f"<style>:root {{ {tokens} }}\n{_CSS}</style>")


_CSS = """
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .fc-strip, .fc-card-title, .fc-readouts dt, .fc-brand {
  font-stretch: 75%;
}
.stApp h1, .stApp h2 { letter-spacing: -0.01em; }
.fc-num, .fc-readouts dd, .stApp [data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }
.stApp :focus-visible { outline: 2px solid var(--fc-trace) !important; outline-offset: 2px; }

.fc-brand { margin: 0; font-size: 0.95rem; font-weight: 600; color: var(--fc-muted); }

/* Signal-flow strip: one lamp per pipeline step, a solid line up to the current step. */
.fc-strip ol { list-style: none; display: flex; margin: 0.25rem 0 0.75rem; padding: 0; gap: 0; }
.fc-strip li { flex: 1 1 0; min-width: 0; margin: 0; padding: 0; position: relative; }
.fc-strip li + li::before {
  content: ""; position: absolute; top: 8px; right: calc(100% + 6px); width: calc(100% - 30px);
  border-top: 2px dashed var(--fc-rule);
}
.fc-strip li.is-done + li::before {
  border-top-style: solid; border-color: var(--fc-trace);
}
.fc-strip .fc-lamp { width: 18px; height: 18px; position: relative; z-index: 1; }
.fc-strip li.is-current .fc-lamp { box-shadow: 0 0 0 3px var(--fc-bench), 0 0 0 5px var(--fc-trace); }
.fc-step-name { display: block; margin-top: 0.4rem; font-size: 1rem; font-weight: 600; line-height: 1.2; }
.fc-step-state { display: block; font-size: 0.8rem; color: var(--fc-muted); }
.fc-strip li:not(.is-current) .fc-step-name { font-weight: 450; color: var(--fc-muted); }

.fc-lamp {
  display: inline-block; flex: none; width: 12px; height: 12px; border-radius: 999px;
  background: var(--lamp); box-shadow: inset 0 0 0 1px rgba(0, 0, 0, 0.12);
}

/* Agent channel card */
[class*="st-key-fccard"], [class*="st-key-fcchat"] { background: var(--fc-panel); }
.fc-card-head { display: flex; flex-wrap: wrap; align-items: baseline; gap: 0.5rem 1.5rem; }
.fc-card-name { display: flex; align-items: center; gap: 0.6rem; flex: 1 1 14rem; min-width: 0; }
.fc-card-title { margin: 0 !important; padding: 0 !important; font-size: 1.3rem !important; font-weight: 650; }
.fc-readouts { display: flex; flex-wrap: wrap; gap: 0.25rem 1.5rem; margin: 0; }
.fc-readouts div { display: flex; flex-direction: column; }
.fc-readouts dt { font-size: 0.8rem; font-weight: 500; color: var(--fc-muted); }
.fc-readouts dd { margin: 0; font-size: 1.05rem; font-weight: 600; }
.fc-meter { display: inline-block; width: 4.5rem; height: 6px; margin-right: 0.4rem; vertical-align: middle;
  background: var(--fc-grid); border-radius: 999px; overflow: hidden; }
.fc-meter span { display: block; height: 100%; background: var(--fc-trace); }
.fc-llm { margin: 0.25rem 0 0.5rem; padding: 0.6rem 0.8rem; border-left: 2px solid var(--fc-rule);
  background: var(--fc-bench); }
.fc-llm small { display: block; color: var(--fc-muted); font-size: 0.8rem; margin-bottom: 0.15rem; }
.fc-llm p { margin: 0; }

@media (max-width: 640px) {
  .fc-step-state { display: none; }
  .fc-step-name { font-size: 0.85rem; }
}
@media (prefers-reduced-motion: reduce) { .stApp * { transition: none !important; animation: none !important; } }
"""


def plotly_layout(height: int = 360) -> dict:
    """Scope-screen look: panel background, fine graticule, Archivo, no chart-junk."""
    p = palette()
    axis = dict(automargin=True, showgrid=True, gridcolor=p["grid"], gridwidth=1, zeroline=False, showline=True,
                linecolor=p["rule"], ticks="outside", tickcolor=p["rule"], ticklen=4,
                minor=dict(showgrid=True, gridcolor=p["grid"], griddash="dot", nticks=5),
                tickfont=dict(color=p["muted"]))
    return dict(
        height=height, margin=dict(l=56, r=12, t=36, b=40), hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=p["panel"],
        font=dict(family=FONT, color=p["ink"], size=13),
        xaxis=axis, yaxis={**axis, "ticks": ""},
        legend=dict(orientation="h", y=1.08, x=0, bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=p["panel"], bordercolor=p["rule"], font=dict(family=FONT, color=p["ink"])),
    )
