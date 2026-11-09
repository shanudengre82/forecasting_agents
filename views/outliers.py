"""Step 4: flag outliers; the user keeps, removes or replaces each one."""

import pandas as pd
import streamlit as st

from fcagents import theme, ui
from fcagents.tools.outliers import ACTIONS, THRESHOLDS, bulk_actions

df = ui.require_data()
ctx = ui.data_context(df)
entry = ui.ensure_stage("outliers")
ui.page_header("outliers", "Which values are far from what the trend and seasonal pattern predict? You decide what "
                           "happens to each one; nothing changes until you do.")
ui.chat_panel("outliers", ctx, entry)

ui.pref_selectbox("Sensitivity", "outlier_threshold", list(THRESHOLDS), format_func=THRESHOLDS.get,
                  help="Lower cut-offs flag more points. Holiday-affected days are never flagged.")
_, results, state = entry

for r in results:
    ui.render_result(r)

r = results[0]
if r.status != "skipped":
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
    ui.plot_series(ui.series_title(state, " · after your choices"), series, state.target_col or "value",
                   marks=marks, expected=r.extras["expected"])

    if len(table):
        st.markdown("**Decide per point**")
        st.caption("Keep: the value stays. Remove: it becomes a gap that the Missing Values agent fills. "
                   "Replace: it gets the replacement value (the expected value unless you type another).")
        b1, b2, b3, _ = st.columns([2, 2, 2, 3])
        if b1.button("Apply suggestions", icon=":material/rule:", type="primary"):
            ui.prefs()["outlier_actions"] = bulk_actions(table, "suggested")
            st.rerun()
        if b2.button("Keep all", icon=":material/check:"):
            ui.prefs()["outlier_actions"] = bulk_actions(table, "keep")
            st.rerun()
        if b3.button("Clear choices", icon=":material/restart_alt:"):
            ui.prefs()["outlier_actions"] = ()
            st.rerun()

        labels = {"undecided": "Undecided", **ACTIONS}
        view = pd.DataFrame({
            "timestamp": table.index, "value": table["value"].to_numpy(), "expected": table["expected"].to_numpy(),
            "z": table["z"].to_numpy(), "suggestion": table["suggestion"].map(ACTIONS).to_numpy(),
            "action": table["action"].map(labels).to_numpy(),
            "replacement": [e if v is None or pd.isna(v) else v
                            for v, e in zip(table["replacement"], table["expected"])],
        })
        edited = st.data_editor(
            view, hide_index=True, width="stretch", key=f"outlier_editor_{len(table)}_{r.facts['threshold']}",
            disabled=["timestamp", "value", "expected", "z", "suggestion"],
            column_config={
                "timestamp": st.column_config.DatetimeColumn("timestamp"),
                "z": st.column_config.NumberColumn("z", format="%.1f", help="Robust standard deviations from expected"),
                "action": st.column_config.SelectboxColumn("Action", options=list(labels.values()), required=True),
                "replacement": st.column_config.NumberColumn("Replacement value",
                                                             help="Used when the action is Replace"),
            })
        back = {v: k for k, v in labels.items()}
        actions = tuple(
            (pd.Timestamp(row.timestamp).isoformat(), back[row.action],
             None if back[row.action] != "replace" or row.replacement == row.expected else float(row.replacement))
            for row in edited.itertuples() if back[row.action] != "undecided")
        if actions != ui.prefs()["outlier_actions"]:
            ui.prefs()["outlier_actions"] = actions
            st.rerun()
ui.nav_buttons("outliers")
