import numpy as np
import pandas as pd
import pytest

from fcagents.tools.missing import (
    FILL_METHODS,
    fill_missing,
    full_grid,
    missing_report,
    pick_seasonal_period,
    recommend_fill,
)


def _hourly_with_gaps():
    idx = pd.date_range("2024-01-01", periods=10, freq="h").delete([3, 4])
    s = pd.Series(np.arange(10.0)[np.r_[0:3, 5:10]], index=idx)
    s.iloc[-2] = np.nan  # an empty value at a present timestamp
    return s


def test_report_counts_missing_timestamps_and_values():
    rep = missing_report(_hourly_with_gaps(), "h")
    assert rep["n_expected"] == 10
    assert rep["n_missing_timestamps"] == 2 and rep["n_nan_values"] == 1 and rep["n_missing"] == 3
    assert rep["longest_run"] == 2 and rep["gaps"][0]["length"] == 2
    assert rep["missing_share"] == 0.3


@pytest.mark.parametrize("method, expected_at_3", [
    ("ffill", 2.0), ("bfill", 5.0), ("interpolate", 3.0), ("zero", 0.0),
])
def test_fill_methods(method, expected_at_3):
    out, facts = fill_missing(_hourly_with_gaps(), "h", method)
    assert len(out) == 10 and out.notna().all()
    assert out.loc["2024-01-01 03:00"] == expected_at_3
    assert facts["n_filled"] == 3 and facts["n_still_missing"] == 0


def test_none_leaves_gaps_on_full_grid():
    out, facts = fill_missing(_hourly_with_gaps(), "h", "none")
    assert len(out) == 10 and facts["n_still_missing"] == 3 and facts["n_filled"] == 0


def test_seasonal_copies_previous_season_and_falls_back():
    idx = pd.date_range("2024-01-01", periods=12, freq="h")
    s = pd.Series([1.0, 2, 3, 4] * 3, index=idx).drop(idx[[1, 9, 10]])
    out, facts = fill_missing(s, "h", "seasonal", seasonal_period=4)
    assert out.loc[idx[9]] == 2.0 and out.loc[idx[10]] == 3.0  # one season earlier
    assert out.loc[idx[1]] == 2.0  # no earlier season: interpolated between 1 and 3
    assert facts["seasonal_period"] == 4
    with pytest.raises(ValueError):
        fill_missing(s, "h", "seasonal")


@pytest.mark.parametrize("alias, periods", [("B", 20), ("MS", 24)])
def test_calendar_frequencies(alias, periods):
    idx = pd.date_range("2022-01-03", periods=periods, freq=alias)
    s = pd.Series(np.arange(periods, dtype=float), index=idx).drop(idx[[5, 6]])
    rep = missing_report(s, alias)
    assert rep["n_expected"] == periods and rep["n_missing_timestamps"] == 2
    out, _ = fill_missing(s, alias, "interpolate")
    assert out.index.equals(idx) and out.notna().all()


def test_duplicates_merged_and_off_grid_dropped():
    idx = pd.to_datetime(["2024-01-01 00:00", "2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 01:30",
                          "2024-01-01 02:00"])
    s = pd.Series([1.0, 3.0, 5.0, 99.0, 7.0], index=idx)
    grid, info = full_grid(s, "h")
    assert list(grid) == [2.0, 5.0, 7.0]
    assert info == {"n_duplicates_merged": 1, "n_off_grid_dropped": 1}


def test_recommend_fill_rules():
    rep = {"n_missing": 3, "longest_run": 2}
    assert recommend_fill({"n_missing": 0, "longest_run": 0}, "mean", 24) == "none"
    assert recommend_fill(rep, "sum", 24) == "zero"
    assert recommend_fill(rep, "count", None) == "zero"
    assert recommend_fill(rep, "mean", 24) == "seasonal"
    assert recommend_fill(rep, None, None) == "interpolate"
    assert recommend_fill({"n_missing": 1, "longest_run": 1}, None, 7) == "interpolate"
    assert pick_seasonal_period([24, 168], 100) == 24
    assert pick_seasonal_period([24, 168], 40) is None


def test_unknown_method_rejected():
    assert "seasonal" in FILL_METHODS
    with pytest.raises(ValueError):
        fill_missing(_hourly_with_gaps(), "h", "magic")
