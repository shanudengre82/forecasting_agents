import numpy as np
import pandas as pd

from fcagents.orchestrator import run_pipeline
from fcagents.tools.targets import MAX_SERIES, build_series, choose_series, choose_targets, target_candidates


def _results(df, **kw):
    return {r.agent: r for r, _ in run_pipeline(df, **kw)}


def test_name_hint_beats_column_order():
    df = pd.DataFrame({"temperature": [1.0, 2, 3], "sales": [4.0, 5, 6]})
    targets, source = choose_targets(target_candidates(df, set()))
    assert (targets, source) == (["sales"], "name")


def test_first_numeric_without_hints_and_user_choice_wins():
    df = pd.DataFrame({"a": [1.0, 2, 3], "b": [4.0, 5, 6]})
    cands = target_candidates(df, set())
    assert choose_targets(cands) == (["a"], "first")
    assert choose_targets(cands, ("b", "missing")) == (["b"], "user")


def test_series_choice_longest_first_and_capped():
    df = pd.DataFrame({"id": ["a"] * 2 + ["b"] * 5 + ["c"] * 3})
    assert choose_series(df, "id") == (["b", "c", "a"], 3, "all")
    assert choose_series(df, "id", ("a",)) == (["a"], 3, "user")
    assert choose_series(df, "id", limit=2)[0] == ["b", "c"]
    assert choose_series(df, None) == ([None], 1, "none")


def test_trailing_rows_without_target_are_the_horizon():
    t = pd.Series(pd.date_range("2024-01-01", periods=6, freq="D"))
    df = pd.DataFrame({"y": [1.0, np.nan, 3, 4, np.nan, np.nan]})
    s, horizon, mask = build_series(df, t, "y", None, None)
    assert len(s) == 4 and np.isnan(s.iloc[1])  # an inner gap stays a gap
    assert list(horizon) == list(t[4:]) and mask.tolist() == [False] * 4 + [True] * 2


def test_target_agent_decisions():
    idx = pd.date_range("2024-01-01", periods=40, freq="D").strftime("%Y-%m-%d")
    one = pd.DataFrame({"date": idx, "sales": np.arange(40.0)})
    assert _results(one)["target"].decision == "single"
    many = pd.concat([one.assign(store=k) for k in "ab"])
    assert _results(many)["target"].decision == "multiple"
    only_time = pd.DataFrame({"date": idx})
    assert _results(only_time)["target"].decision == "timestamps_only"


def test_more_than_max_series_is_capped():
    idx = pd.date_range("2024-01-01", periods=10, freq="D").strftime("%Y-%m-%d")
    df = pd.concat([pd.DataFrame({"date": idx, "id": f"s{i:03d}", "y": np.arange(10.0)})
                    for i in range(MAX_SERIES + 2)])
    target = _results(df)["target"]
    assert target.decision == "capped" and target.facts["n_series"] == MAX_SERIES
