import numpy as np
import pandas as pd
import pytest

from fcagents.orchestrator import run_pipeline


def _run(df, **kw):
    results = [r for r, _ in run_pipeline(df, **kw)]
    return {r.agent: r for r in results}


def _frame(index, **extra):
    return pd.DataFrame({"date": index.strftime("%Y-%m-%d %H:%M:%S"), "y": np.arange(len(index), dtype=float), **extra})


def test_hourly_regular():
    res = _run(_frame(pd.date_range("2024-01-01", periods=200, freq="h")))
    assert res["time_index"].decision == "found"
    assert res["regularity"].decision == "regular"
    assert res["frequency"].decision == "h"
    assert res["frequency"].status == "ok"
    assert res["frequency"].explanation is None  # NullBackend


def test_daily_with_gaps_reports_missing():
    idx = pd.date_range("2024-01-01", periods=90, freq="D").delete([3, 4, 50])
    res = _run(_frame(idx))
    assert res["regularity"].decision == "regular_with_gaps"
    assert res["frequency"].facts["n_missing"] == 3
    assert len(res["frequency"].extras["missing_index"]) == 3


def _irregular_index(days=14, n=600, seed=1):
    rng = np.random.default_rng(seed)
    secs = rng.uniform(0, days * 86400, n)
    return pd.DatetimeIndex(sorted(pd.Timestamp("2024-01-01") + pd.to_timedelta(secs, unit="s")))


def test_irregular_is_resampled_then_named():
    res = _run(_frame(_irregular_index()))
    assert res["regularity"].decision == "irregular"
    rs = res["resampling"]
    assert rs.decision == "resampled" and rs.facts["freq_source"] == "auto"
    assert rs.facts["n_points"] >= 30
    assert res["frequency"].decision == rs.facts["chosen_freq"]
    assert len(rs.extras["raw_series"]) == 600


def test_user_frequency_respected():
    res = _run(_frame(_irregular_index()), resample_freq="6h", resample_agg="sum")
    rs = res["resampling"]
    assert rs.facts["freq_source"] == "user" and rs.facts["chosen_freq"] == "6h"
    assert rs.facts["agg"] == "sum"
    assert res["frequency"].decision == "6h"


@pytest.mark.parametrize("freq", ["W", "12h"])
def test_too_few_points_warns_and_stops(freq):
    res = _run(_frame(_irregular_index(days=14)), resample_freq=freq)
    rs = res["resampling"]
    assert rs.status == "warning" and rs.decision == "too_few_points"
    assert rs.facts["n_points"] < 30
    finer = rs.facts["finer_options_with_enough_points"]
    assert finer and finer[0]["freq"] == "6h"
    assert "6h" in rs.next_step
    assert res["frequency"].status == "skipped"


def test_too_short_span_even_hourly():
    res = _run(_frame(_irregular_index(days=1, n=40)), resample_freq="h")
    rs = res["resampling"]
    assert rs.decision == "too_few_points"
    assert "too short" in rs.next_step


def test_regular_series_skips_resampling():
    res = _run(_frame(pd.date_range("2024-01-01", periods=100, freq="h")))
    assert res["resampling"].status == "skipped"
    assert res["frequency"].decision == "h"


def test_no_time_column_stops():
    res = _run(pd.DataFrame({"a": np.arange(10.0)}))
    assert res["time_index"].status == "error"
    assert res["regularity"].status == "skipped"
    assert res["frequency"].status == "skipped"


def test_long_format_picks_one_series():
    idx = pd.date_range("2024-01-01", periods=30, freq="D")
    df = pd.concat([_frame(idx, unique_id="a"), _frame(idx[:20], unique_id="b")])
    res = _run(df)
    assert res["time_index"].facts["series_id"] == "a"
    assert res["regularity"].decision == "regular"
    res_b = _run(df, series_id="b")
    assert res_b["time_index"].facts["series_id"] == "b"


@pytest.mark.slow
def test_real_llm_explanation():
    from fcagents.llm.backend import TransformersBackend

    llm = TransformersBackend()
    res = _run(_frame(pd.date_range("2024-01-01", periods=100, freq="h")), llm=llm)
    text = res["frequency"].explanation
    assert text and len(text) > 20
    assert "hour" in text.lower() or "'h'" in text
