import numpy as np
import pandas as pd

from fcagents.tools.time_index import detect_time_column, parse_time_column


def _values(n):
    return np.arange(n, dtype=float)


def test_string_dates_with_name_hint():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=30).strftime("%Y-%m-%d"),
            "sales": _values(30),
            "store": ["A"] * 30,
        }
    )
    rep = detect_time_column(df)
    assert rep["time_col"] == "date"
    assert rep["ambiguous"] is False
    assert rep["target_cols"] == ["sales"]
    parsed = parse_time_column(df, "date", rep["candidates"][0]["method"])
    assert parsed.iloc[0] == pd.Timestamp("2024-01-01")


def test_datetime_dtype_without_name_hint():
    df = pd.DataFrame({"x": pd.date_range("2024-01-01", periods=10, freq="h"), "y": _values(10)})
    rep = detect_time_column(df)
    assert rep["time_col"] == "x"


def test_numeric_column_not_mistaken_for_time():
    df = pd.DataFrame({"value": _values(50), "other": _values(50) * 2})
    rep = detect_time_column(df)
    assert rep["time_col"] is None


def test_epoch_seconds():
    epochs = pd.date_range("2024-01-01", periods=20, freq="D").as_unit("s").astype("int64")
    df = pd.DataFrame({"timestamp": epochs, "y": _values(20)})
    rep = detect_time_column(df)
    assert rep["time_col"] == "timestamp"
    parsed = parse_time_column(df, "timestamp", rep["candidates"][0]["method"])
    assert parsed.iloc[1] == pd.Timestamp("2024-01-02")


def test_two_date_columns_is_ambiguous():
    d = pd.date_range("2024-01-01", periods=20).strftime("%Y-%m-%d")
    df = pd.DataFrame({"order_day": d, "ship_day": d, "y": _values(20)})
    rep = detect_time_column(df)
    assert rep["ambiguous"] is True
    assert len(rep["candidates"]) == 2


def test_series_id_column_detected():
    d = list(pd.date_range("2024-01-01", periods=20).strftime("%Y-%m-%d"))
    df = pd.DataFrame({"ds": d * 3, "unique_id": ["a"] * 20 + ["b"] * 20 + ["c"] * 20, "y": _values(60)})
    rep = detect_time_column(df)
    assert rep["time_col"] == "ds"
    assert rep["id_col"] == "unique_id"


def test_day_first_dates():
    d = pd.date_range("2024-01-03", periods=40, freq="D").strftime("%d/%m/%Y")
    df = pd.DataFrame({"Date": d, "y": _values(40)})
    rep = detect_time_column(df)
    assert rep["candidates"][0]["method"] == "string_dayfirst"
    parsed = parse_time_column(df, "Date", "string_dayfirst")
    assert parsed.iloc[0] == pd.Timestamp("2024-01-03")
    assert parsed.is_monotonic_increasing
