# Hugging Face Space (Docker SDK): Streamlit on port 7860, CPU-only.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.21 /uv /usr/local/bin/uv

# HF Spaces run as uid 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/home/user/.cache/huggingface
WORKDIR /home/user/app

# Dependencies first for layer caching.
COPY --chown=user pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --extra llm --no-install-project

COPY --chown=user . .
RUN uv sync --frozen --no-dev --extra llm

# Bake the default model into the image so the first request doesn't wait for a download.
ARG PRELOAD_MODEL=Qwen/Qwen2.5-0.5B-Instruct
RUN .venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('${PRELOAD_MODEL}')"

EXPOSE 7860
# XSRF/CORS off: required for st.file_uploader inside the HF Spaces iframe.
CMD [".venv/bin/streamlit", "run", "app.py", \
     "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true", \
     "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
