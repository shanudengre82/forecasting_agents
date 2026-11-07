"""Step 1: find the time index and check whether the timestamps are regular."""

import pandas as pd
import streamlit as st

from fcagents import ui

df = ui.require_data()
ctx = ui.data_context(df)
entry = ui.ensure_stage("data")
ui.page_header("data", "Which column is time, what to forecast, and is the spacing fixed?")
ui.chat_panel("data", ctx, entry)

with st.expander("Columns", expanded=ctx["ambiguous"]):
    cols = st.columns(3)
    if ctx["time_cols"]:
        ui.pref_selectbox("Time column", "time_col", ctx["time_cols"], container=cols[0])
    if ctx["target_cols"]:
        ui.pref_selectbox("Target column", "target_col", ctx["target_cols"], container=cols[1])
    if ctx["series_ids"]:
        ui.pref_selectbox(f"Series ({ctx['id_col']})", "series_id", ctx["series_ids"], container=cols[2])
_, results, state = entry  # control changes arrive via on_change before this run, so entry is current

st.dataframe(df.head(8), width="stretch")
for r in results:
    ui.render_result(r)

missing = None
reg = state.findings.get("regularity")
if reg is not None and reg.facts.get("freq") and state.series is not None:
    idx = state.series.index.dropna().unique().sort_values()
    missing = pd.date_range(idx[0], idx[-1], freq=reg.facts["freq"]).difference(idx)
ui.plot_series(ui.series_title(state), state.series, state.target_col or "timestamps", missing=missing)
ui.nav_buttons("data")
