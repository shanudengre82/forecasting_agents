from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    monkeypatch.setenv("USE_LLM", "0")


@pytest.mark.parametrize("sample, expected", [
    ("hourly_energy.csv", "`h`"),
    ("daily_sales_with_gaps.csv", "`D`"),
    ("irregular_sensor.csv", "`irregular`"),
])
def test_app_runs_pipeline(sample, expected):
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    at.sidebar.selectbox[0].set_value(sample).run()
    at.button[0].click().run()
    assert not at.exception
    decisions = " ".join(m.value for m in at.markdown)
    assert expected in decisions


def _select(at, label, value):
    next(s for s in at.sidebar.selectbox if s.label.startswith(label)).set_value(value)


def test_irregular_sample_shows_resampling_card():
    at = AppTest.from_file(APP, default_timeout=30).run()
    _select(at, "Sample", "irregular_sensor.csv")
    at.run()
    at.button[0].click().run()
    assert not at.exception
    assert any("Resampling Agent" in h.value for h in at.subheader)
    assert not at.warning


def test_weekly_on_short_data_warns():
    at = AppTest.from_file(APP, default_timeout=30).run()
    _select(at, "Sample", "irregular_sensor.csv")
    _select(at, "Target frequency", "W")
    at.run()
    at.button[0].click().run()
    assert not at.exception
    assert at.warning and "finer frequency" in at.warning[0].value
