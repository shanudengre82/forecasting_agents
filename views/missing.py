"""Step 3: fill missing timestamps and values on the fixed grid."""

import streamlit as st

from fcagents import ui
from fcagents.tools.missing import FILL_METHODS

df = ui.require_data()
ctx = ui.data_context(df)
entry = ui.ensure_stage("missing")
ui.page_header("missing", "Every timestamp on the grid gets a value. Filled values are estimates and are "
                          "marked in the chart.")
ui.chat_panel("missing", ctx, entry)

ui.pref_selectbox("Fill method", "fill_method", ["auto", *FILL_METHODS],
                  format_func=lambda m: "Auto (recommended)" if m == "auto" else FILL_METHODS[m],
                  help="Auto: zero for counts and sums, seasonal when two full seasons exist, otherwise "
                       "interpolation.")
_, results, state = entry

for r in results:
    ui.render_result(r)

r = results[0]
if r.status != "skipped":
    ui.plot_series(ui.series_title(state, f" · {r.facts['method_label']}"), state.series,
                   state.target_col or "value", filled=r.extras["filled_index"])
    name = f"{state.target_col or 'series'}_{state.freq}_prepared.csv"
    out = state.series.to_frame()
    cov = ui.final_covariates(out.index)
    if cov is not None:
        out = out.join(cov)
        st.caption(f"The download includes {len(cov.columns)} holiday covariate columns from the Holidays step.")
    st.download_button("Download prepared series (CSV)", out.to_csv(index_label="timestamp"),
                       file_name=name, mime="text/csv", icon=":material/download:")
ui.nav_buttons("missing")
