"""Step 3: do holidays move the series? If so, suggest holiday covariates."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fcagents import theme, ui
from fcagents.tools.holidays import HOLIDAY_TYPES, MAX_WINDOW, countries, subdivisions, unofficial_categories

df = ui.require_data()
p = ui.prefs()
names = countries()
subdivs = subdivisions(p["holiday_country"])
ctx = ui.data_context(df) | {"subdivisions": subdivs, "holiday_country": p["holiday_country"]}
entry = ui.ensure_stage("holidays")
ui.page_header("holidays", "Do holidays, or the days around them, move the series? If so, the forecasting model "
                           "should know about them.")
ui.chat_panel("holidays", ctx, entry)


def _set(**values):
    p.update(values)


def _country_changed():
    _set(holiday_country=st.session_state["w_holiday_country"], holiday_subdiv=None)


def _types_changed():
    chosen = tuple(t for t in HOLIDAY_TYPES if st.session_state.get(f"w_type_{t}"))
    _set(holiday_types=chosen)


c1, c2, c3 = st.columns([3, 3, 4])
st.session_state["w_holiday_country"] = p["holiday_country"]
c1.selectbox("Country", list(names), format_func=lambda c: f"{names[c]} ({c})", key="w_holiday_country",
             on_change=_country_changed)
options = ["", *subdivs]  # "" = whole country (a None value would show the empty placeholder)
st.session_state["w_holiday_subdiv"] = p["holiday_subdiv"] if p["holiday_subdiv"] in subdivs else ""
c2.selectbox("State / region", options, key="w_holiday_subdiv",
             format_func=lambda c: f"All of {names[p['holiday_country']]}" if not c else f"{subdivs[c]} ({c})",
             on_change=lambda: _set(holiday_subdiv=st.session_state["w_holiday_subdiv"] or None),
             disabled=not subdivs, help="Regional holidays are added for the chosen state.")
st.session_state["w_holiday_window"] = p["holiday_window"]
c3.slider("Days before and after each holiday", 0, MAX_WINDOW, key="w_holiday_window",
          on_change=lambda: _set(holiday_window=st.session_state["w_holiday_window"]),
          help="Effects such as a shopping run-up before Christmas show up on these days.")
t1, t2, _ = st.columns([2, 3, 3])
has_unofficial = bool(unofficial_categories(p["holiday_country"]))
for col, (t, label) in zip((t1, t2), HOLIDAY_TYPES.items()):
    st.session_state[f"w_type_{t}"] = t in p["holiday_types"]
    col.checkbox(label, key=f"w_type_{t}", on_change=_types_changed,
                 disabled=t == "unofficial" and not has_unofficial,
                 help=None if t == "official" or has_unofficial else "This country's calendar has no unofficial days.")
_, results, state = entry

for r in results:
    ui.render_result(r)

r = results[0]
ex = r.extras if r.status != "skipped" else {}
profile = ex.get("profile")
if profile is not None and len(profile):
    pal = theme.palette()
    fig = go.Figure(go.Bar(
        x=profile["offset"], y=profile["mean_abs_z"],
        marker_color=[pal["trace"] if e else pal["raw"] for e in profile["has_effect"]],
        hovertemplate="%{x} days: %{y:.2f}<extra></extra>"))
    fig.add_hline(y=r.facts["ordinary_abs_z"], line=dict(color=pal["warn"], dash="dot", width=1.5),
                  annotation_text="ordinary day", annotation_position="top right")
    fig.update_layout(**theme.plotly_layout(height=280))
    fig.update_xaxes(title_text="Days from the holiday (negative = before)", dtick=1)
    fig.update_yaxes(title_text="Distance from normal (mean |z|)")
    st.markdown("**How far from normal the days around holidays are**")
    st.caption("Blue bars: at least one holiday moves the series on that day every year.")
    st.plotly_chart(fig, width="stretch", theme=None)

    daily = ex["daily"]
    hol = ex["occurrences"].set_index("date")["value"]
    pal = theme.palette()
    prot = state.protected_index
    around = daily.reindex(prot.normalize().unique()).dropna().drop(hol.index, errors="ignore") if prot is not None \
        else pd.Series(dtype=float)
    ui.plot_series(ui.series_title(state, " · per day"), daily, "per day", marks=[
        ("Holiday", hol, dict(color=pal["trace"], size=10, symbol="diamond", line=dict(width=1.5, color=pal["panel"]))),
        ("Day affected by a holiday", around, dict(color=pal["trace"], size=7, symbol="circle-open",
                                                   line=dict(width=1.5))),
    ])

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Holidays in the data**")
        occ = ex["occurrences"].assign(date=lambda d: d["date"].dt.date)
        st.dataframe(occ[["date", "holiday", "value", "expected", "change_pct", "z"]], hide_index=True,
                     width="stretch", height=300,
                     column_config={"change_pct": st.column_config.NumberColumn("change", format="%.1f%%"),
                                    "value": st.column_config.NumberColumn(format="%.1f"),
                                    "expected": st.column_config.NumberColumn(format="%.1f")})
    with right:
        st.markdown("**Suggested covariates**")
        cov = ex.get("covariates")
        if cov is None:
            st.caption("None: no holiday moves the series.")
        else:
            st.dataframe(cov.sum().rename("days marked").to_frame(), width="stretch", height=300)
            st.download_button("Download holiday covariates (CSV)", cov.to_csv(index_label="timestamp"),
                               file_name=f"holiday_covariates_{r.facts['country']}.csv", mime="text/csv",
                               icon=":material/download:")
ui.nav_buttons("holidays")
