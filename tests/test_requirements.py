"""The Space installs requirements.txt, not pyproject.toml: both must list the same runtime packages."""

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLATFORM = {"gradio", "spaces", "huggingface_hub"}  # provided by Hugging Face Spaces; pinning them breaks builds


def _names(lines) -> set[str]:
    return {re.split(r"[<>=!~\[ ;]", line.strip(), maxsplit=1)[0].lower().replace("-", "_")
            for line in lines if line.strip() and not line.strip().startswith("#")}


def test_requirements_cover_pyproject():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    wanted = _names(project["dependencies"] + project["optional-dependencies"]["llm"]) - PLATFORM
    have = _names((ROOT / "requirements.txt").read_text().splitlines())
    assert wanted == have
    assert not have & PLATFORM
