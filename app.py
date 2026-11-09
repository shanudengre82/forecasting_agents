"""Gradio entry point. Runs locally (`python app.py`), in Docker, and on a Hugging Face ZeroGPU Space."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

if os.environ.get("SPACES_ZERO_GPU"):
    # ZeroGPU: `spaces` must be imported before torch, and the @spaces.GPU function must exist at startup.
    import spaces  # noqa: F401, E402

    import fcagents.llm.zerogpu  # noqa: F401, E402

from fcagents import theme  # noqa: E402
from fcagents.webui import build_app  # noqa: E402

FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
         '<link href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,400..700&display=swap" '
         'rel="stylesheet">')

demo = build_app()

if __name__ == "__main__":
    demo.launch(css=theme.css(), head=FONTS, footer_links=["gradio", "settings"])
