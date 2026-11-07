"""Catalogue of every agent, generated from the agents' own metadata."""

import pandas as pd
import streamlit as st

from fcagents import ui
from fcagents.catalog import PRINCIPLE, agent_catalog, pipeline_dot
from fcagents.orchestrator import PIPELINE
from fcagents.tools.resample import MAX_FREQ, MIN_FREQ, MIN_POINTS

st.header("Agents")
st.markdown(f"**Design principle:** {PRINCIPLE}")
st.graphviz_chart(pipeline_dot(), width="stretch")
c1, c2, c3 = st.columns(3)
c1.metric("Agents in pipeline", len(PIPELINE))
c2.metric("Resampling range", f"{MIN_FREQ.components.hours}h – {MAX_FREQ.days // 7}W")
c3.metric("Min. data points", MIN_POINTS)
st.caption(f"LLM backend in use: `{ui.app_llm().name}`. It explains results and answers chat questions from "
           "computed facts; it never makes a decision. Chat changes it proposes are validated by rules.")

page = None
for a in agent_catalog():
    if a["page"] != page:
        page = a["page"]
        st.markdown(f"#### Page: {page}")
    with st.container(border=True):
        st.subheader(f"{a['step']}. {a['title']}")
        st.write(a["summary"])
        st.markdown(f"**Runs when:** {a['runs_when']}")
        left, right = st.columns(2)
        with left:
            st.markdown("**Inputs**\n" + "\n".join(f"- {x}" for x in a["inputs"]))
            st.markdown("**Outputs**\n" + "\n".join(f"- {x}" for x in a["outputs"]))
        with right:
            controls = a["user_controls"] or ["None; fully automatic"]
            st.markdown("**User controls**\n" + "\n".join(f"- {x}" for x in controls))
            st.markdown("**Tools**\n" + "\n".join(f"- `{x}`" for x in a["tools"]))
        st.markdown("**Decisions**")
        st.dataframe(pd.DataFrame(a["decisions"].items(), columns=["decision", "meaning"]),
                     hide_index=True, width="stretch")
