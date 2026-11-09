# syntax=docker/dockerfile:1
# Forecasting Agents: Gradio app, CPU-only, for plain Docker and docker compose.
# (The Hugging Face Space uses the Gradio SDK with requirements.txt instead.)
#   docker build -t fcagents .                      # full: app + PyTorch + baked-in Qwen2.5-0.5B
#   docker build --target lite -t fcagents:lite .   # lite: no PyTorch; rule summaries or your own LLM server

FROM python:3.12-slim AS base
COPY --from=ghcr.io/astral-sh/uv:0.12.21 /uv /usr/local/bin/uv

# HF Spaces (and good practice) run as an unprivileged uid 1000.
RUN useradd -m -u 1000 user
USER user
WORKDIR /home/user/app

ENV HOME=/home/user \
    PATH=/home/user/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/home/user/.cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    GRADIO_ANALYTICS_ENABLED=False \
    # LLM choice (plain *Variables*, not secrets). On a HF Space (SPACE_ID set by HF) visitors can switch to a
    # hosted model paid with the token they paste; the app never uses HF_TOKEN for LLM calls, so don't set one.
    # Outside a Space the choices are the in-app model or your own server at LOCAL_LLM_URL (OpenAI-compatible).
    LLM_BACKEND=local \
    HF_API_MODEL=Qwen/Qwen2.5-7B-Instruct \
    LOCAL_LLM_URL=http://host.docker.internal:11434/v1 \
    LOCAL_LLM_MODEL=qwen2.5:0.5b

COPY --chown=user pyproject.toml uv.lock README.md ./

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 CMD ["python", "scripts/healthcheck.py"]
CMD ["python", "app.py"]


FROM base AS lite
# No PyTorch here: use your own LLM server (LLM_BACKEND=server), or rule-based summaries (the default).
ENV USE_LLM=0 LLM_BACKEND=server
RUN --mount=type=cache,target=/home/user/.cache/uv,uid=1000 \
    uv sync --frozen --no-dev --no-install-project
COPY --chown=user . .
RUN --mount=type=cache,target=/home/user/.cache/uv,uid=1000 \
    uv sync --frozen --no-dev


FROM base AS full
RUN --mount=type=cache,target=/home/user/.cache/uv,uid=1000 \
    uv sync --frozen --no-dev --extra llm --no-install-project
COPY --chown=user . .
RUN --mount=type=cache,target=/home/user/.cache/uv,uid=1000 \
    uv sync --frozen --no-dev --extra llm
# Bake the default model into the image so the first request doesn't wait for a download.
ARG PRELOAD_MODEL=Qwen/Qwen2.5-0.5B-Instruct
ENV MODEL_ID=${PRELOAD_MODEL}
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('${PRELOAD_MODEL}')"
