"""Step 2: put the series on a fixed frequency (1h-1W) and name it."""

import streamlit as st

from fcagents import ui
from fcagents.tools.resample import AGGREGATIONS, FREQ_OPTIONS, MIN_POINTS

df = ui.require_data()
ctx = ui.data_context(df)
entry = ui.ensure_stage("resampling")
ui.page_header("resampling", "Irregular timestamps are binned onto a fixed grid. Regular series pass through "
                             "unless you pick a frequency, e.g. to aggregate hourly data to daily.")
ui.chat_panel("resampling", ctx, entry)

left, right = st.columns(2)
ui.pref_selectbox("Target frequency (1h – 1W)", "resample_freq", ["auto", *FREQ_OPTIONS], container=left,
                  format_func=lambda f: "Auto (recommended)" if f == "auto" else f"{f} · {FREQ_OPTIONS[f]}",
                  help=f"Finer frequencies give more data points. At least {MIN_POINTS} are needed.")
ui.pref_selectbox("Aggregate values per bin", "resample_agg", AGGREGATIONS, container=right,
                  help="Event logs without a numeric value are always counted.")
_, results, state = entry

for r in results:
    ui.render_result(r)

resampled = next((r for r in results if r.agent == "resampling" and r.status != "skipped"), None)
freq_result = next((r for r in results if r.agent == "frequency" and r.status != "skipped"), None)
missing = freq_result.extras.get("missing_index") if freq_result else None
if resampled is not None:
    f = resampled.facts
    raw = resampled.extras["raw_series"] if f["agg"] != "count" else None
    ui.plot_series(ui.series_title(state, f" · resampled to {f['chosen_freq']} ({f['agg']})"), state.series,
                   f"{f['agg']} per {f['chosen_freq']}", raw=raw, missing=missing)
else:
    ui.plot_series(ui.series_title(state), state.series, state.target_col or "timestamps", missing=missing)
ui.nav_buttons("resampling")
