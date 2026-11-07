import numpy as np
import pandas as pd
import pytest

from fcagents.orchestrator import initial_state, run_pipeline, run_stage


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
    assert res["missing_values"].decision == "complete"


def test_daily_with_gaps_reports_missing():
    idx = pd.date_range("2024-01-01", periods=90, freq="D").delete([3, 4, 50])
    res = _run(_frame(idx))
    assert res["regularity"].decision == "regular_with_gaps"
    assert res["frequency"].facts["n_missing"] == 3
    assert len(res["frequency"].extras["missing_index"]) == 3
    mv = res["missing_values"]
    assert mv.decision == "filled" and mv.facts["n_filled"] == 3 and mv.facts["fill_source"] == "auto"
    assert len(mv.extras["filled_index"]) == 3


def test_user_fill_method_and_leave_missing():
    idx = pd.date_range("2024-01-01", periods=90, freq="D").delete([3, 4, 50])
    res = _run(_frame(idx), fill_method="zero")
    assert res["missing_values"].facts["method"] == "zero" and res["missing_values"].facts["fill_source"] == "user"
    res = _run(_frame(idx), fill_method="none")
    assert res["missing_values"].decision == "left_missing" and res["missing_values"].status == "warning"


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
    mv = res["missing_values"]
    assert mv.facts["n_missing"] == rs.facts["n_empty"]  # empty bins surface as missing timestamps
    assert mv.decision in ("filled", "complete")


def test_user_frequency_respected():
    res = _run(_frame(_irregular_index()), resample_freq="6h", resample_agg="sum")
    rs = res["resampling"]
    assert rs.facts["freq_source"] == "user" and rs.facts["chosen_freq"] == "6h"
    assert rs.facts["agg"] == "sum"
    assert res["frequency"].decision == "6h"
    hourly = _run(_frame(_irregular_index()), resample_freq="h", resample_agg="sum")["missing_values"]
    assert hourly.facts["n_missing"] > 0 and hourly.facts["method"] == "zero"  # an empty bin sums to zero


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
    assert res["missing_values"].status == "skipped"


def test_too_short_span_even_hourly():
    res = _run(_frame(_irregular_index(days=1, n=40)), resample_freq="h")
    rs = res["resampling"]
    assert rs.decision == "too_few_points"
    assert "too short" in rs.next_step


def test_regular_series_skips_resampling():
    res = _run(_frame(pd.date_range("2024-01-01", periods=100, freq="h")))
    assert res["resampling"].status == "skipped"
    assert res["frequency"].decision == "h"


def test_stages_rerun_from_a_saved_state():
    df = _frame(pd.date_range("2024-01-01", periods=24 * 40, freq="h"))
    data_state = None
    for _, data_state in run_stage("data", initial_state(df)):
        pass
    hourly = {r.agent: r for r, _ in run_stage("resampling", data_state)}
    daily = {r.agent: r for r, _ in run_stage("resampling", data_state, resample_freq="D", resample_agg="sum")}
    assert hourly["resampling"].status == "skipped" and hourly["frequency"].decision == "h"
    assert daily["resampling"].decision == "resampled" and daily["frequency"].decision == "D"
    assert data_state.freq == "h" and len(data_state.series) == 24 * 40  # input state untouched


def test_stop_carries_across_stages():
    state = None
    for _, state in run_stage("data", initial_state(pd.DataFrame({"a": np.arange(10.0)}))):
        pass
    assert state.stopped
    assert all(r.status == "skipped" for r, _ in run_stage("missing", state))


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
