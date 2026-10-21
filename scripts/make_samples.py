"""Generate demo CSVs in data/samples/ covering the cases the first agents must handle."""

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "data" / "samples"
rng = np.random.default_rng(42)


def seasonal(n, period, amp=10.0, trend=0.05, noise=1.0, level=100.0):
    t = np.arange(n)
    return level + trend * t + amp * np.sin(2 * np.pi * t / period) + rng.normal(0, noise, n)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    samples = {}

    idx = pd.date_range("2024-01-01", periods=24 * 42, freq="h")
    samples["hourly_energy.csv"] = pd.DataFrame(
        {"timestamp": idx, "load_mw": seasonal(len(idx), 24, amp=30, level=500)})

    idx = pd.date_range("2023-01-01", periods=365, freq="D")
    df = pd.DataFrame({"date": idx.strftime("%Y-%m-%d"), "sales": seasonal(len(idx), 7, amp=15)})
    samples["daily_sales_with_gaps.csv"] = df.drop(index=[30, 31, 32, 100, 101, 250])

    idx = pd.date_range("2022-01-03", periods=500, freq="B")
    samples["business_day_stock.csv"] = pd.DataFrame(
        {"Date": idx.strftime("%d/%m/%Y"), "close": 100 + np.cumsum(rng.normal(0, 1, len(idx)))})

    idx = pd.date_range("2012-01-01", periods=144, freq="MS")
    samples["monthly_passengers.csv"] = pd.DataFrame(
        {"month": idx.strftime("%Y-%m"), "passengers": seasonal(len(idx), 12, amp=40, trend=2, level=300)})

    secs = np.sort(rng.uniform(0, 14 * 86400, 600)).astype(int)
    t = pd.Timestamp("2024-03-01") + pd.to_timedelta(secs, unit="s")
    samples["irregular_sensor.csv"] = pd.DataFrame(
        {"event_time": t.strftime("%Y-%m-%dT%H:%M:%S"), "reading": rng.normal(20, 2, len(t))})

    idx = pd.date_range("2024-01-01", periods=96 * 7, freq="15min")
    df = pd.DataFrame({"ts": idx, "value": seasonal(len(idx), 96)})
    dups = df.sample(15, random_state=1)
    samples["15min_with_duplicates.csv"] = pd.concat([df, dups]).sample(frac=1, random_state=2)

    idx = pd.date_range("2024-01-01", periods=120, freq="D")
    samples["multi_store_long.csv"] = pd.concat(
        [pd.DataFrame({"ds": idx.strftime("%Y-%m-%d"), "unique_id": sid, "y": seasonal(len(idx), 7, level=lvl)})
         for sid, lvl in [("store_1", 50), ("store_2", 80), ("store_3", 20)]])

    for name, df in samples.items():
        df.to_csv(OUT / name, index=False)
        print(f"wrote {name} ({len(df)} rows)")


if __name__ == "__main__":
    main()
