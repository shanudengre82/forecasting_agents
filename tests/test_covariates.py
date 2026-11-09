from pathlib import Path

import numpy as np
import pandas as pd

from fcagents.orchestrator import run_pipeline
from fcagents.tools.covariates import classify_column, covariate_frame

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "samples" / "store_demand_covariates.csv"


def _results(df, **kw):
    out, state = {}, None
    for r, state in run_pipeline(df, **kw):
        out[r.agent] = r
    return out, state


def test_sample_roles():
    res, state = _results(pd.read_csv(SAMPLE))
    f = res["covariates"].facts
    assert f["past"] == ["temperature"] and f["future"] == ["promo", "price"]
    assert f["static"] == ["store_type"] and f["ignored"] == ["note"]
    assert res["target"].facts["n_horizon_rows"] == 42
    assert all(len(c.horizon) == 14 for c in state.children.values())


def test_rules_and_reasons():
    n = 40
    horizon = pd.Series([False] * 30 + [True] * 10)
    df = pd.DataFrame({
        "weekday": np.arange(n) % 7, "humidity": np.r_[np.random.default_rng(0).normal(size=30), [np.nan] * 10],
        "discount": np.r_[np.arange(30) % 2, [np.nan] * 10], "known": np.arange(n, dtype=float),
        "mostly_empty": [1.0] + [np.nan] * (n - 1), "constant": 1.0,
    })
    role = {c: classify_column(df, c, None, horizon)["detected"] for c in df.columns}
    assert role == {"weekday": "future", "humidity": "past", "discount": "future", "known": "future",
                    "mostly_empty": "ignore", "constant": "ignore"}


def test_user_override_wins():
    res, state = _results(pd.read_csv(SAMPLE), covariate_overrides=(("temperature", "future"), ("note", "ignore")))
    f = res["covariates"].facts
    assert "temperature" in f["future"] and f["n_overridden"] == 2
    assert state.covariate_roles["temperature"] == "future"


def test_covariate_frame_extends_future_only():
    res, state = _results(pd.read_csv(SAMPLE))
    child = state.children["north"]
    frame = covariate_frame(state.raw, state.parsed_time, state.id_col, "north", state.covariate_roles,
                            child.freq, child.series.index, child.horizon)
    assert len(frame) == 180 + 14
    future = frame.index > child.series.index.max()
    assert frame.loc[future, "temperature"].isna().all()
    assert frame.loc[future, ["promo", "price"]].notna().all().all()
    assert (frame["store_type"] == "mall").all()
