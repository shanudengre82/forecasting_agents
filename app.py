"""Streamlit entry point: sidebar (language model, data) and step navigation. Pages live in views/."""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from fcagents import ui  # noqa: E402
from fcagents.llm.backend import (  # noqa: E402
    API_PROVIDERS, DEFAULT_API_MODEL, DEFAULT_BACKEND, DEFAULT_MODEL_ID, NullBackend,
)

SAMPLES = ROOT / "data" / "samples"
BACKENDS = {"local": "Local (CPU, free)", "api": "Hugging Face API (your token)"}
TOKEN_URL = "https://huggingface.co/settings/tokens"

st.set_page_config(page_title="Forecasting Agents", page_icon="📈", layout="wide")


@st.cache_resource(show_spinner="Loading language model (first run downloads it)…")
def load_llm(model_id: str):
    """Local model, shared by all visitors. Holds no credentials, so a global cache is safe."""
    from fcagents.llm.backend import TransformersBackend

    return TransformersBackend(model_id)


@st.cache_data
def read_csv(source) -> pd.DataFrame:
    return pd.read_csv(source)


def api_llm(model_id: str, provider: str, token: str):
    """Hosted model billed to the visitor's own token.

    Kept in this browser session's state, never in st.cache_resource (shared by every visitor of the Space)
    and never read from HF_TOKEN or a Space secret.
    """
    from fcagents.llm.backend import InferenceAPIBackend

    key = (model_id, provider, hashlib.sha256(token.encode()).hexdigest())
    ss = st.session_state
    if ss.get("_api_llm_key") != key:
        ss["_api_llm"], ss["_api_llm_key"] = InferenceAPIBackend(model_id, token, provider), key
    return ss["_api_llm"]


def language_model():
    st.header("Language model")
    use_llm = st.toggle("Use an LLM", value=os.environ.get("USE_LLM", "1") == "1",
                        help="Explains agent results and answers chat questions. Tools compute every fact; "
                             "the LLM never makes a decision.")
    llm = NullBackend()
    if use_llm:
        keys, default = list(BACKENDS), os.environ.get("LLM_BACKEND", DEFAULT_BACKEND)
        kind = st.radio("Backend", keys, index=keys.index(default) if default in keys else 0,
                        format_func=BACKENDS.get)
        if kind == "local":
            model_id = st.text_input("Model", value=DEFAULT_MODEL_ID, key="local_model",
                                     help="Any public Hugging Face chat model that fits on CPU, e.g. "
                                          "Qwen/Qwen2.5-0.5B-Instruct or Qwen/Qwen2.5-1.5B-Instruct.").strip()
            try:
                llm = load_llm(model_id)
            except Exception as e:  # missing torch, no network, OOM…
                st.error(f"Could not load {model_id}: {e}. Falling back to rule-based summaries.")
        else:
            model_id = st.text_input("Model", value=DEFAULT_API_MODEL, key="api_model",
                                     help="Any chat model served by Hugging Face Inference Providers, "
                                          "e.g. meta-llama/Llama-3.1-8B-Instruct.").strip()
            provider = st.selectbox("Provider", API_PROVIDERS, key="api_provider")
            token = st.text_input("Your Hugging Face token", type="password", key="hf_token",
                                  placeholder="hf_…",
                                  help=f"Create a fine-grained token with “Make calls to Inference Providers” "
                                       f"at {TOKEN_URL}. Calls are billed to your account. The token stays in "
                                       f"this browser session only and is never stored.").strip()
            if not token:
                st.info(f"Paste your own [Hugging Face token]({TOKEN_URL}) to use the hosted model. "
                        "Until then, agents use rule-based summaries.")
            else:
                try:
                    llm = api_llm(model_id, provider, token)
                except Exception as e:
                    st.error(f"Could not set up {model_id}: {e}. Falling back to rule-based summaries.")
    st.caption(f"Backend: `{llm.name}`")
    if err := getattr(llm, "last_error", None):
        st.warning(f"Last LLM call failed: {err}")
    return llm


def data_source() -> tuple[pd.DataFrame | None, str]:
    st.header("Data")
    mode = st.radio("Source", ["Sample dataset", "Upload CSV"], horizontal=True, label_visibility="collapsed")
    if mode == "Sample dataset":
        name = st.selectbox("Sample", sorted(p.name for p in SAMPLES.glob("*.csv")))
        return read_csv(SAMPLES / name), name
    up = st.file_uploader("CSV file", type=["csv"])
    return (read_csv(up), up.name) if up else (None, "")


with st.sidebar:
    llm = language_model()
    st.divider()
    df, key = data_source()
    st.divider()
    st.caption("Deterministic tools compute the facts → rules decide → the LLM explains.")
ui.set_app(df, key, llm)

nav = st.navigation([
    st.Page("views/data.py", title="1 · Data", icon="🗂️", default=True),
    st.Page("views/resampling.py", title="2 · Resampling", icon="⏱️"),
    st.Page("views/missing.py", title="3 · Missing values", icon="🩹"),
    st.Page("views/agents.py", title="Agents", icon="🧩"),
], position="top")
st.title("📈 Forecasting Agents")
nav.run()
