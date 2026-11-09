from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fcagents.orchestrator import run_pipeline
from fcagents.tools.outliers import apply_outlier_actions, bulk_actions, detect_outliers

SAMPLES = Path(__file__).resolve().parents[1] / "data" / "samples"
PLANTED = ["2023-03-14", "2023-06-08", "2023-09-20", "2024-02-27", "2024-04-17", "2024-08-13"]


def _results(name: str = "daily_sales_holidays.csv", **prefs) -> dict:
    return {r.agent: r for r, _ in run_pipeline(pd.read_csv(SAMPLES / name), **prefs)}


def test_finds_planted_outliers_but_not_holidays():
    res = _results()
    o = res["outliers"]
    table = o.extras["table"]
    assert o.decision == "flagged" and o.status == "warning"
    assert set(pd.DatetimeIndex(PLANTED)) <= set(table.index)
    assert set(table.index[table["suggestion"] == "replace"]) == set(pd.DatetimeIndex(PLANTED))
    for holiday in ["2023-12-25", "2024-11-28", "2024-11-29", "2024-07-04"]:
        assert pd.Timestamp(holiday) not in table.index
    assert o.facts["n_protected_skipped"] > 0


@pytest.mark.parametrize("name", ["hourly_energy.csv", "monthly_passengers.csv", "daily_sales_with_gaps.csv",
                                  "business_day_stock.csv"])
def test_few_false_alarms_on_clean_samples(name):
    o = _results(name)["outliers"]
    assert o.facts["n_flagged"] <= 0.005 * o.facts["n_checked"] + 1


def test_actions_keep_remove_replace():
    s = pd.Series(np.r_[np.ones(30) * 10, 100, np.ones(30) * 10],
                  index=pd.date_range("2024-01-01", periods=61, freq="D"))
    table, _, facts = detect_outliers(s, "D", None, 3.5)
    ts = pd.Timestamp("2024-01-31")
    assert list(table.index) == [ts] and facts["n_flagged"] == 1
    out, f = apply_outlier_actions(s, table, ((ts.isoformat(), "remove", None),))
    assert np.isnan(out[ts]) and f["n_removed"] == 1 and f["n_undecided"] == 0
    out, _ = apply_outlier_actions(s, table, ((ts.isoformat(), "replace", None),))
    assert out[ts] == pytest.approx(10)
    out, _ = apply_outlier_actions(s, table, ((ts.isoformat(), "replace", 42.0),))
    assert out[ts] == 42.0
    out, f = apply_outlier_actions(s, table, ((ts.isoformat(), "keep", None), ("2024-01-02T00:00:00", "remove", None)))
    assert out.equals(s) and f["n_kept"] == 1  # a choice for a point that isn't flagged is ignored


def test_removed_outlier_is_filled_downstream_and_decision_handled():
    table = _results()["outliers"].extras["table"]
    res = _results(outlier_actions=bulk_actions(table, "suggested"))
    assert res["outliers"].decision == "handled"
    assert res["outliers"].facts["n_replaced"] == 6
    res = _results(outlier_actions=bulk_actions(table, "remove"))
    mv = res["missing_values"]
    assert mv.facts["n_filled"] == len(table)
    assert set(table.index) <= set(mv.extras["filled_index"])


def test_sensitivity_changes_how_many_are_flagged():
    low = _results(outlier_threshold=5.0)["outliers"].facts["n_flagged"]
    high = _results(outlier_threshold=3.0)["outliers"].facts["n_flagged"]
    assert low == 6 and high > low
