"""The "signal bench" look, shared by the Gradio theme, the page CSS and the Plotly charts.

The app reads like a piece of test equipment: data runs along a signal path of agents, and each agent is a
channel with a status lamp and readouts. Meaning is fixed everywhere: trace = observed data and the signal path,
estimate = filled values, warn = missing timestamps / needs attention, ok / error = agent status.
The same tokens drive docs/index.html.
"""

from __future__ import annotations

import html

LIGHT = {
    "bench": "#DCE2E0", "panel": "#F6F8F7", "ink": "#1B292D", "muted": "#5C6B6E", "rule": "#BAC5C3",
    "grid": "#D3DBD9", "trace": "#0B6E8A", "estimate": "#7D4CB3", "raw": "#86969A",
    "ok": "#2C7A4D", "warn": "#B97F08", "error": "#BF3A2B", "idle": "#A9B5B6",
}
DARK = {
    "bench": "#11181A", "panel": "#1A2427", "ink": "#E2EAEC", "muted": "#93A3A6", "rule": "#33433F",
    "grid": "#263336", "trace": "#4CC0DD", "estimate": "#A98BE6", "raw": "#6E7F82",
    "ok": "#55B784", "warn": "#E2B341", "error": "#EC6B5C", "idle": "#46575A",
}
# Charts sit on a transparent ground in both modes, so they use mid-tones that read on light and dark.
CHART = {
    "trace": "#1786A6", "estimate": "#8D5FC6", "raw": "#8A9A9D", "warn": "#C58A0E", "error": "#D0493A",
    "grid": "rgba(120, 140, 142, 0.22)", "rule": "rgba(120, 140, 142, 0.55)", "ink": "#6B7C80",
    "muted": "#7E8E91", "panel": "rgba(0, 0, 0, 0)",
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


def palette() -> dict[str, str]:
    """Chart colours (mid-tones that work in light and dark mode)."""
    return CHART


def esc(value) -> str:
    return html.escape(str(value), quote=True)


def lamp(status: str) -> str:
    token, label = LAMP.get(status, LAMP["pending"])
    return f'<span class="fc-lamp" style="--lamp: var(--fc-{token})" role="img" aria-label="{label}"></span>'


def gradio_theme():
    """Gradio theme: bench-coloured ground, panel faces, the trace colour as the only accent, no shadows."""
    import gradio as gr

    L, D = LIGHT, DARK
    font = [gr.themes.Font("Archivo"), "system-ui", "sans-serif"]
    return gr.themes.Base(font=font, font_mono=font, radius_size=gr.themes.sizes.radius_sm,
                          text_size=gr.themes.sizes.text_md, spacing_size=gr.themes.sizes.spacing_md).set(
        body_background_fill=L["bench"], body_background_fill_dark=D["bench"],
        body_text_color=L["ink"], body_text_color_dark=D["ink"],
        body_text_color_subdued=L["muted"], body_text_color_subdued_dark=D["muted"],
        background_fill_primary=L["panel"], background_fill_primary_dark=D["panel"],
        background_fill_secondary=L["bench"], background_fill_secondary_dark=D["bench"],
        border_color_primary=L["rule"], border_color_primary_dark=D["rule"],
        border_color_accent=L["trace"], border_color_accent_dark=D["trace"],
        color_accent=L["trace"], color_accent_soft=L["grid"], color_accent_soft_dark=D["grid"],
        link_text_color=L["trace"], link_text_color_dark=D["trace"],
        link_text_color_hover=L["ink"], link_text_color_hover_dark=D["ink"],
        block_background_fill=L["panel"], block_background_fill_dark=D["panel"],
        block_border_color=L["rule"], block_border_color_dark=D["rule"], block_border_width="1px",
        block_shadow="none", block_shadow_dark="none", block_radius="6px",
        block_label_background_fill="transparent", block_label_background_fill_dark="transparent",
        block_label_text_color=L["muted"], block_label_text_color_dark=D["muted"],
        block_title_text_color=L["ink"], block_title_text_color_dark=D["ink"],
        block_title_text_weight="600", block_label_text_weight="500",
        panel_background_fill=L["bench"], panel_background_fill_dark=D["bench"],
        input_background_fill="#E9EEEC", input_background_fill_dark=D["bench"],
        input_border_color=L["rule"], input_border_color_dark=D["rule"],
        input_border_color_focus=L["trace"], input_border_color_focus_dark=D["trace"],
        input_shadow="none", input_shadow_dark="none", input_shadow_focus="none", input_shadow_focus_dark="none",
        input_radius="3px",
        button_primary_background_fill=L["trace"], button_primary_background_fill_dark=D["trace"],
        button_primary_background_fill_hover=L["ink"], button_primary_background_fill_hover_dark=D["ink"],
        button_primary_text_color=L["panel"], button_primary_text_color_dark=D["bench"],
        button_primary_border_color=L["trace"], button_primary_border_color_dark=D["trace"],
        button_secondary_background_fill=L["panel"], button_secondary_background_fill_dark=D["panel"],
        button_secondary_background_fill_hover=L["grid"], button_secondary_background_fill_hover_dark=D["grid"],
        button_secondary_text_color=L["ink"], button_secondary_text_color_dark=D["ink"],
        button_secondary_border_color=L["rule"], button_secondary_border_color_dark=D["rule"],
        button_large_radius="3px", button_small_radius="3px", button_medium_radius="3px",
        checkbox_background_color_selected=L["trace"], checkbox_background_color_selected_dark=D["trace"],
        checkbox_border_color_selected=L["trace"], checkbox_border_color_selected_dark=D["trace"],
        checkbox_border_color_focus=L["trace"], checkbox_border_color_focus_dark=D["trace"],
        checkbox_label_background_fill_selected=L["panel"], checkbox_label_background_fill_selected_dark=D["panel"],
        checkbox_label_border_color_selected=L["trace"], checkbox_label_border_color_selected_dark=D["trace"],
        checkbox_label_shadow="none",
        slider_color=L["trace"], slider_color_dark=D["trace"],
        loader_color=L["trace"], loader_color_dark=D["trace"],
        table_border_color=L["rule"], table_border_color_dark=D["rule"],
        table_even_background_fill=L["panel"], table_even_background_fill_dark=D["panel"],
        table_odd_background_fill=L["bench"], table_odd_background_fill_dark=D["bench"],
        shadow_drop="none", shadow_drop_lg="none", shadow_spread="0px",
    )


_CSS = """
/* ---- type: one family, width used as the voice. Legends condensed, readouts semi-condensed. */
.gradio-container { font-family: Archivo, system-ui, sans-serif; font-variant-numeric: tabular-nums; }
.gradio-container h1, .gradio-container h2, .gradio-container h3, .fc-step-name, .fc-card-title, .fc-brand,
.fc-catalogue { font-stretch: 68%; letter-spacing: -0.005em; }
.gradio-container h2 { font-size: 1.95rem; font-weight: 700; line-height: 1.05; margin: 0.25rem 0 0.35rem; }
.gradio-container .prose h2 { font-size: 1.56rem; }
.gradio-container .prose h3 { font-size: 1.25rem; font-weight: 650; }
.fc-readouts dd, .fc-step-state { font-stretch: 85%; }
.gradio-container :focus-visible { outline: 2px solid var(--fc-trace) !important; outline-offset: 2px; }

/* ---- panels: the two side panels are the bench; the centre is the instrument face. */
.sidebar { background: var(--fc-bench) !important; border-color: var(--fc-rule) !important; }
.sidebar .prose h2 { font-size: 1.25rem; margin: 0.75rem 0 0.25rem; }
#fc-right .prose h3 { font-size: 1.25rem; margin: 1rem 0 0.4rem; }
.fc-brand { margin: 0; font-size: 1.56rem; font-weight: 750; color: var(--fc-ink); }
.fc-centre { max-width: 1180px; }
.fc-intro { color: var(--fc-muted); margin: 0 0 0.5rem; max-width: 62ch; }
.fc-hint { color: var(--fc-muted); font-size: 0.8rem; line-height: 1.45; }
.fc-hint code { font: inherit; font-weight: 650; color: var(--fc-ink); background: none; padding: 0; border: 0; }
#fc-right .placeholder, #fc-right [class*="placeholder"] { text-align: left; align-items: flex-start; justify-content: flex-start; color: var(--fc-muted); padding: 0.75rem 0.9rem; }

/* The tab row is replaced by the signal path below (its buttons drive the hidden tabs). */
#fc-tabs > .tab-wrapper { display: none; }

/* ---- the signal path: the one loud element. A thick trace runs from lamp to lamp; it is solid up to the
   current step and dashed after it; the current lamp is lit. */
.fc-strip { margin: 0.75rem 0 1.25rem; }
.fc-strip ol { list-style: none; display: flex; align-items: flex-start; margin: 0; padding: 0; }
.fc-strip li { flex: 1 1 0; min-width: 0; margin: 0; padding: 0; position: relative; }
.fc-strip li + li::before {
  content: ""; position: absolute; top: 9px; right: calc(100% - 4px); width: calc(100% - 26px); height: 0;
  border-top: 4px dashed var(--fc-rule); transform: translateY(-50%);
}
.fc-strip li.is-done + li::before, .fc-strip li.is-current::before { border-top-style: solid; border-color: var(--fc-trace); }
.fc-strip button {
  all: unset; box-sizing: border-box; display: block; width: 100%; cursor: pointer; padding: 0 0.75rem 0.25rem 0;
  border-radius: 3px;
}
.fc-strip button:focus-visible { outline: 2px solid var(--fc-trace); outline-offset: 4px; }
.fc-strip .fc-lamp { width: 18px; height: 18px; position: relative; z-index: 1;
  box-shadow: 0 0 0 3px var(--fc-bench), inset 0 0 0 1px rgba(0, 0, 0, 0.18); }
.fc-strip li.is-current .fc-lamp {
  box-shadow: 0 0 0 3px var(--fc-bench), 0 0 0 5px var(--fc-trace), 0 0 14px 2px color-mix(in srgb, var(--fc-trace) 45%, transparent);
}
.fc-step-name { display: block; margin-top: 0.55rem; font-size: 1.02rem; font-weight: 600; line-height: 1.15;
  color: var(--fc-muted); }
.fc-step-name, .fc-catalogue { white-space: nowrap; }
.fc-strip li.is-current .fc-step-name { color: var(--fc-ink); font-weight: 750; }
.fc-strip button:hover .fc-step-name { color: var(--fc-ink); text-decoration: underline; text-underline-offset: 0.2em; }
.fc-step-state { display: block; font-size: 0.8rem; color: var(--fc-muted); }
.fc-strip .fc-catalogue-item { flex: 0 0 auto; padding-left: 1.5rem; }
.fc-strip .fc-catalogue-item::before { display: none; }
.fc-catalogue { display: block; margin-top: 1.65rem; font-size: 1rem; font-weight: 600; color: var(--fc-trace); }

.fc-lamp {
  display: inline-block; flex: none; width: 12px; height: 12px; border-radius: 999px;
  background: var(--lamp); box-shadow: inset 0 0 0 1px rgba(0, 0, 0, 0.18);
}

/* ---- agent channels (right panel): a status rail on the left edge, no box shadow. */
.fc-card { --rail: var(--fc-idle); background: var(--fc-panel); border-left: 4px solid var(--rail);
  border-radius: 0 6px 6px 0; padding: 0.75rem 0.9rem 0.8rem; margin: 0 0 0.6rem; color: var(--fc-ink);
  line-height: 1.5; }
.fc-card.is-ok { --rail: var(--fc-ok); }
.fc-card.is-warning { --rail: var(--fc-warn); }
.fc-card.is-error { --rail: var(--fc-error); }
.fc-card p { margin: 0.45rem 0 0; }
.fc-card details { margin-top: 0.55rem; }
.fc-card details > summary { cursor: pointer; color: var(--fc-muted); font-size: 0.9rem; }
.fc-card pre { white-space: pre-wrap; font: 0.78rem/1.45 ui-monospace, SFMono-Regular, Menlo, monospace;
  max-height: 22rem; overflow: auto; background: var(--fc-bench); padding: 0.5rem 0.6rem; border-radius: 3px; }
.fc-card code { font: inherit; font-stretch: 85%; font-weight: 650; }
.fc-card-head { display: flex; flex-wrap: wrap; align-items: baseline; gap: 0.35rem 1.25rem; }
.fc-card-name { display: flex; align-items: center; gap: 0.55rem; flex: 1 1 12rem; min-width: 0; }
.fc-card-title { margin: 0 !important; padding: 0 !important; font-size: 1.25rem !important; font-weight: 700; }
.fc-card-name .fc-lamp { display: none; }  /* the rail carries the status; the lamp stays for screen readers */
.fc-readouts { display: flex; flex-wrap: wrap; gap: 0.15rem 1.25rem; margin: 0; }
.fc-readouts div { display: flex; flex-direction: column; }
.fc-readouts dt { font-size: 0.75rem; color: var(--fc-muted); }
.fc-readouts dd { margin: 0; font-size: 1rem; font-weight: 650; }
.fc-meter { display: inline-block; width: 3.5rem; height: 4px; margin-right: 0.4rem; vertical-align: middle;
  background: var(--fc-grid); border-radius: 2px; overflow: hidden; }
.fc-meter span { display: block; height: 100%; background: var(--fc-trace); }
.fc-next { color: var(--fc-ink); }
.fc-llm { margin: 0.55rem 0 0; padding: 0.5rem 0.7rem; border-left: 2px solid var(--fc-estimate);
  background: var(--fc-bench); border-radius: 0 3px 3px 0; }
.fc-llm small { display: block; color: var(--fc-muted); font-size: 0.75rem; margin-bottom: 0.15rem; }
.fc-llm p { margin: 0; }
.fc-warn { border-left: 3px solid var(--fc-warn); padding: 0.4rem 0.7rem; background: var(--fc-bench); }
.fc-focus { margin-top: 0.6rem; }
.fc-focus > summary { font-weight: 600; }
.fc-focus .fc-card { margin: 0.5rem 0 0; background: var(--fc-bench); }

@media (max-width: 760px) {
  .fc-step-state { display: none; }
  .fc-step-name { font-size: 0.85rem; }
  .fc-strip button { padding-right: 0.35rem; }
  .fc-catalogue { margin-top: 1.4rem; font-size: 0.85rem; }
  .gradio-container h2 { font-size: 1.56rem; }
}
@media (prefers-reduced-motion: reduce) { .gradio-container * { transition: none !important; animation: none !important; } }
"""

# Clicking a step on the signal path selects the matching (hidden) tab, which runs that step.
JS = """
() => {
  document.addEventListener('click', (e) => {
    const step = e.target.closest('[data-fc-tab]');
    if (!step) return;
    const tabs = document.querySelectorAll('#fc-tabs [role=tab]');
    const tab = tabs[Number(step.dataset.fcTab)];
    if (tab) tab.click();
  });
}
"""


def _tokens(p: dict[str, str]) -> str:
    return "".join(f"--fc-{k}: {v};" for k, v in p.items())


def css() -> str:
    """Page CSS for gr.Blocks: colour tokens for light and dark mode, then the component rules."""
    return f":root {{ {_tokens(LIGHT)} }}\n.dark {{ {_tokens(DARK)} }}\n{_CSS}"


def plotly_layout(height: int = 360) -> dict:
    """Scope-screen look on a transparent ground: fine graticule, Archivo, no chart-junk."""
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
