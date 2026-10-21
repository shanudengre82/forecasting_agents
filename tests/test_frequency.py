import numpy as np
import pandas as pd
import pytest

from fcagents.tools.frequency import analyze_regularity, describe_frequency


def _drop(idx, positions):
    return idx.delete(positions)


@pytest.mark.parametrize(
    "freq, periods, expected",
    [
        ("h", 200, "h"),
        ("15min", 300, "15min"),
        ("D", 120, "D"),
        ("B", 120, "B"),
        ("W-SUN", 60, "W-SUN"),
        ("MS", 48, "MS"),
        ("ME", 48, "ME"),
    ],
)
def test_regular_series(freq, periods, expected):
    idx = pd.date_range("2022-01-02", periods=periods, freq=freq)
    rep = analyze_regularity(idx)
    assert rep["status"] == "regular"
    assert rep["freq"] == expected
    assert rep["coverage"] == 1.0


def test_quarterly_and_yearly():
    q = analyze_regularity(pd.date_range("2010-01-01", periods=40, freq="QS"))
    assert q["status"] == "regular" and q["freq"].startswith("QS")
    y = analyze_regularity(pd.date_range("1990-12-31", periods=30, freq="YE"))
    assert y["status"] == "regular" and y["freq"].startswith("YE")


def test_daily_with_gaps():
    idx = _drop(pd.date_range("2023-01-01", periods=100, freq="D"), [5, 6, 40, 77])
    rep = analyze_regularity(idx)
    assert rep["status"] == "regular_with_gaps"
    assert rep["freq"] == "D"
    assert rep["n_missing"] == 4


def test_monthly_with_gaps():
    idx = _drop(pd.date_range("2015-01-01", periods=60, freq="MS"), [10, 30])
    rep = analyze_regularity(idx)
    assert rep["status"] == "regular_with_gaps"
    assert rep["freq"] == "MS"


def test_business_days_not_confused_with_daily_gaps():
    idx = _drop(pd.date_range("2023-01-02", periods=200, freq="B"), [17])
    rep = analyze_regularity(idx)
    assert rep["freq"] == "B"
    assert rep["status"] == "regular_with_gaps"


def test_irregular():
    rng = np.random.default_rng(0)
    secs = np.sort(rng.uniform(0, 30 * 86400, size=300)).astype("int64")
    idx = pd.to_datetime("2024-01-01") + pd.to_timedelta(secs, unit="s")
    rep = analyze_regularity(idx)
    assert rep["status"] == "irregular"
    assert rep["freq"] is None


def test_duplicates_and_unsorted():
    idx = pd.date_range("2024-01-01", periods=50, freq="h")
    shuffled = pd.DatetimeIndex(list(idx[::-1]) + [idx[3], idx[10]])
    rep = analyze_regularity(shuffled)
    assert rep["was_sorted"] is False
    assert rep["n_duplicates"] == 2
    assert rep["status"] == "regular"
    assert rep["freq"] == "h"


def test_insufficient_data():
    rep = analyze_regularity(pd.DatetimeIndex(["2024-01-01", "2024-01-02"]))
    assert rep["status"] == "insufficient_data"


def test_describe_frequency_hourly_gaps():
    idx = _drop(pd.date_range("2024-01-01", periods=72, freq="h"), [10, 11, 12, 50])
    d = describe_frequency(idx, "h")
    assert d["label"] == "hourly"
    assert d["n_missing"] == 4
    assert d["seasonal_periods"] == [24, 168]
    assert d["gaps"][0]["length"] == 3


@pytest.mark.parametrize(
    "freq, label, periods",
    [
        ("D", "daily", [7, 365]),
        ("B", "business daily", [5, 260]),
        ("W-SUN", "weekly", [52]),
        ("MS", "monthly", [12]),
        ("QS-JAN", "quarterly", [4]),
        ("YE-DEC", "yearly", []),
        ("15min", "every 15 minutes", [4, 96, 672]),
    ],
)
def test_describe_labels(freq, label, periods):
    idx = pd.date_range("2020-01-01", periods=30, freq=freq)
    d = describe_frequency(idx, freq)
    assert d["label"] == label
    assert d["seasonal_periods"] == periods
