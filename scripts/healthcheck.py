"""Docker HEALTHCHECK: exit 0 if Streamlit's health endpoint answers (no curl in the slim image)."""

import os
import sys
import urllib.request

port = os.environ.get("STREAMLIT_SERVER_PORT", "7860")
try:
    with urllib.request.urlopen(f"http://localhost:{port}/_stcore/health", timeout=4) as r:
        sys.exit(0 if r.status == 200 else 1)
except Exception:
    sys.exit(1)
