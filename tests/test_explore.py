"""Unit & integration tests for multi-format data import, EDA profiling, DataExplorationAgent, and explore chat/UI."""

import numpy as np
import pandas as pd
import pytest

from fcagents import webui
from fcagents.agents.exploration_agent import DataExplorationAgent
from fcagents.chat import respond
from fcagents.chat.actions import Action
from fcagents.core.state import PipelineState
from fcagents.llm.backend import NullBackend
from fcagents.session import Session
from fcagents.tools.explore import compute_relationships, detect_anomalies, explore_dataset, load_dataset, profile_columns


def _sample_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    x = np.linspace(0, 10, 50)
    y = 2.5 * x + rng.normal(0, 0.2, 50)
    y[0] = 500.0  # clear IQR anomaly
    df = pd.DataFrame({
        "ds": pd.date_range("2024-01-01", periods=50, freq="D").astype(str),
        "sales": y,
        "temp": x,
        "store": ["A"] * 25 + ["B"] * 25,
        "const_col": ["same"] * 50,
    })
    df.loc[5, "temp"] = np.nan
    # Add one duplicate row
    return pd.concat([df, df.iloc[[1]]], ignore_index=True)


def test_load_dataset_supports_csv_tsv_json_excel_parquet(tmp_path):
    base = pd.DataFrame({"ds": ["2024-01-01", "2024-01-02"], "y": [10.0, 20.5], "cat": ["a", "b"]})

    csv_p = tmp_path / "data.csv"
    tsv_p = tmp_path / "data.tsv"
    json_p = tmp_path / "data.json"
    jsonl_p = tmp_path / "lines.json"
    xlsx_p = tmp_path / "data.xlsx"
    pq_p = tmp_path / "data.parquet"

    base.to_csv(csv_p, index=False)
    base.to_csv(tsv_p, sep="\t", index=False)
    base.to_json(json_p, orient="records")
    base.to_json(jsonl_p, orient="records", lines=True)
    base.to_excel(xlsx_p, index=False)
    base.to_parquet(pq_p, index=False)

    for p, expected_fmt in [
        (csv_p, "csv"),
        (tsv_p, "tsv"),
        (json_p, "json"),
        (jsonl_p, "json"),
        (xlsx_p, "xlsx"),
        (pq_p, "parquet"),
    ]:
        loaded, meta = load_dataset(p)
        assert loaded.shape == (2, 3)
        assert meta["format"] == expected_fmt and meta["rows"] == 2 and meta["cols"] == 3

    bad = tmp_path / "notes.txt"
    bad.write_text("hello")
    with pytest.raises(ValueError, match="Unsupported file format"):
        load_dataset(bad)


def test_profile_anomalies_and_relationships():
    df = _sample_df()
    prof = profile_columns(df)
    assert set(prof["column"]) == {"ds", "sales", "temp", "store", "const_col"}

    anom = detect_anomalies(df, prof)
    assert anom["n_duplicate_rows"] == 1
    assert "const_col" in anom["constant_cols"]
    assert "temp" in anom["any_missing_cols"]
    assert anom["iqr_anomaly_cols"]["sales"] >= 1

    rels = compute_relationships(df, method="spearman")
    assert rels["method"] == "spearman"
    assert not rels["corr_matrix"].empty
    assert any({p["col_a"], p["col_b"]} == {"sales", "temp"} for p in rels["top_pairs"])


def test_exploration_agent_decisions():
    agent = DataExplorationAgent()

    # Empty dataset -> error / empty
    empty_res = agent.run(PipelineState(raw=pd.DataFrame()), NullBackend())
    assert empty_res.status == "error" and empty_res.decision == "empty"

    # Dataset with duplicates / constant col / IQR anomaly / correlation -> warning / findings
    noisy_res = agent.run(PipelineState(raw=_sample_df()), NullBackend())
    assert noisy_res.status == "warning" and noisy_res.decision == "findings"
    assert "profile_table" in noisy_res.extras and "corr_matrix" in noisy_res.extras

    # Clean, uncorrelated, non-skewed dataset -> ok / clean
    clean_df = pd.DataFrame({
        "ds": pd.date_range("2024-01-01", periods=20, freq="D").astype(str),
        "val": np.linspace(1.0, 20.0, 20),
    })
    clean_res = agent.run(PipelineState(raw=clean_df), NullBackend())
    assert clean_res.status == "ok" and clean_res.decision == "clean"


def test_explore_chat_commands_and_ui_view():
    df = _sample_df()
    ctx = {"all_cols": list(df.columns)}

    r1 = respond("use spearman correlation", "explore", ctx, None, None, {})
    assert r1.actions == [Action("corr_method", "spearman")] and r1.source == "rule"

    r2 = respond("inspect sales", "explore", ctx, None, None, {})
    assert r2.actions == [Action("explore_col", "sales")] and r2.source == "rule"

    r3 = respond("compare temp and sales", "explore", ctx, None, None, {})
    assert r3.actions == [Action("rel_x", "temp"), Action("rel_y", "sales")] and r3.source == "rule"

    sess = Session()
    sess.set_data(df, "test_df")
    out = webui.render(sess, "explore")
    assert out["plot"] is not None
    assert out["corr_plot"] is not None
    assert out["rel_plot"] is not None
