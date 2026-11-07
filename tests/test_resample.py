import numpy as np
import pandas as pd
import pytest

from fcagents.tools.resample import (
    FREQ_OPTIONS,
    MIN_POINTS,
    recommend_freq,
    resample_options,
    resample_series,
    validate_freq,
)


def _irregular(days=14, n=600, seed=0, values=True):
    rng = np.random.default_rng(seed)
    secs = np.sort(rng.uniform(0, days * 86400, n)).astype("int64")
    idx = pd.Timestamp("2024-03-01") + pd.to_timedelta(secs, unit="s")
    vals = rng.normal(20, 2, n) if values else np.full(n, np.nan)
    return pd.Series(vals, index=pd.DatetimeIndex(idx))


@pytest.mark.parametrize("bad", ["30min", "15min", "2W", "MS", "8D"])
def test_out_of_range_rejected(bad):
    with pytest.raises(ValueError):
        validate_freq(bad)


@pytest.mark.parametrize("good", list(FREQ_OPTIONS))
def test_all_options_valid(good):
    validate_freq(good)


def test_mean_and_sum():
    idx = pd.to_datetime(["2024-01-01 00:10", "2024-01-01 00:50", "2024-01-01 01:30", "2024-01-01 03:05"])
    s = pd.Series([1.0, 3.0, 10.0, 7.0], index=idx)
    mean, facts = resample_series(s, "h", "mean")
    assert mean.loc["2024-01-01 00:00"] == 2.0
    assert mean.loc["2024-01-01 01:00"] == 10.0
    assert pd.Timestamp("2024-01-01 02:00") not in mean.index  # empty bin dropped
    assert facts["n_bins"] == 4 and facts["n_points"] == 3 and facts["n_empty"] == 1
    total, _ = resample_series(s, "h", "sum")
    assert total.loc["2024-01-01 00:00"] == 4.0


def test_count_without_target():
    s = _irregular(days=2, n=100, values=False)
    out, facts = resample_series(s, "6h", "count")
    assert facts["agg"] == "count"
    assert out.sum() == 100


def test_nan_values_force_count():
    s = _irregular(days=2, n=50, values=False)
    _, facts = resample_series(s, "6h", "mean")
    assert facts["agg"] == "count"


def test_empty_bins_are_dropped():
    idx = pd.to_datetime(["2024-01-01 00:00", "2024-01-01 03:00", "2024-01-01 04:00"])
    s = pd.Series([1.0, 4.0, 5.0], index=idx)
    out, facts = resample_series(s, "h", "mean")
    assert len(out) == 3 and out.notna().all()  # gaps are left for the Missing Values agent
    assert facts["n_bins"] == 5 and facts["n_points"] == 3 and facts["n_empty"] == 2


def test_options_and_recommendation():
    s = _irregular(days=14, n=600)
    opts = resample_options(s)
    by = {o["freq"]: o for o in opts}
    assert list(by) == list(FREQ_OPTIONS)
    assert by["W"]["n_points"] < MIN_POINTS and not by["W"]["enough_points"]
    assert by["D"]["n_points"] == 14
    rec = recommend_freq(opts)
    assert by[rec]["enough_points"] and by[rec]["empty_share"] <= 0.2
    finer = list(FREQ_OPTIONS)[: list(FREQ_OPTIONS).index(rec)]
    assert all(by[f]["empty_share"] > 0.2 or not by[f]["enough_points"] for f in finer)


def test_recommendation_when_nothing_has_enough_points():
    s = _irregular(days=1, n=10)
    assert recommend_freq(resample_options(s)) == "h"
