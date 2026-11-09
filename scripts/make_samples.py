"""Generate demo CSVs in data/samples/ covering the cases the first agents must handle."""

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "data" / "samples"
rng = np.random.default_rng(42)


def seasonal(n, period, amp=10.0, trend=0.05, noise=1.0, level=100.0):
    t = np.arange(n)
    return level + trend * t + amp * np.sin(2 * np.pi * t / period) + rng.normal(0, noise, n)


def holiday_sales() -> pd.DataFrame:
    """Two years of US daily store sales: weekly pattern, holiday effects and six data-entry outliers.

    Uses its own random generator so adding it leaves the other samples unchanged.
    """
    r = np.random.default_rng(2024)
    idx = pd.date_range("2023-01-01", "2024-12-31", freq="D")
    t = np.arange(len(idx))
    weekly = np.array([0.95, 0.9, 0.92, 0.97, 1.08, 1.2, 1.0])[idx.dayofweek]
    y = (200 + 0.08 * t) * weekly * r.normal(1, 0.04, len(idx))
    effect = pd.Series(1.0, index=idx)
    for year in (2023, 2024):
        thanksgiving = pd.Timestamp(f"{year}-11-01") + pd.offsets.WeekOfMonth(week=3, weekday=3)
        effect[thanksgiving] = 0.45                        # stores close early
        effect[thanksgiving + pd.Timedelta(days=1)] = 2.1  # Black Friday
        for d in range(1, 10):                             # Christmas shopping run-up
            effect[pd.Timestamp(f"{year}-12-24") - pd.Timedelta(days=d)] = 1 + 0.06 * (10 - d)
        effect[pd.Timestamp(f"{year}-12-24")] = 1.5
        effect[pd.Timestamp(f"{year}-12-25")] = 0.15        # closed
        effect[pd.Timestamp(f"{year}-01-01")] = 0.3
        effect[pd.Timestamp(f"{year}-07-04")] = 1.4
        effect[pd.Timestamp(f"{year}-05-01") + pd.offsets.LastWeekOfMonth(weekday=0)] = 1.35  # Memorial Day
        effect[pd.Timestamp(f"{year}-09-01") + pd.offsets.WeekOfMonth(week=0, weekday=0)] = 1.3  # Labor Day
    y = y * effect.to_numpy()
    for day, factor in [("2023-03-14", 3.0), ("2023-06-08", 0.2), ("2023-09-20", 2.6),
                        ("2024-02-27", 0.25), ("2024-04-17", 3.2), ("2024-08-13", 2.8)]:
        y[idx.get_loc(pd.Timestamp(day))] *= factor          # data-entry errors, nowhere near a holiday
    return pd.DataFrame({"date": idx.strftime("%Y-%m-%d"), "sales": y.round(2)})


def store_demand() -> pd.DataFrame:
    """Three stores, daily, with covariates: temperature (past), promo and price (future, planned and filled in for
    the next 14 days, where sales are still empty), store_type (static) and a free-text note column.

    Uses its own random generator so adding it leaves the other samples unchanged.
    """
    r = np.random.default_rng(7)
    observed, horizon = 180, 14
    idx = pd.date_range("2024-01-01", periods=observed + horizon, freq="D")
    t = np.arange(len(idx))
    frames = []
    for store, kind, level in [("north", "mall", 120), ("south", "street", 80), ("east", "mall", 140)]:
        temperature = 12 + 10 * np.sin(2 * np.pi * (t - 100) / 365) + r.normal(0, 2, len(idx))
        promo = (r.random(len(idx)) < 0.12).astype(int)
        price = np.round(9.99 - 1.5 * promo + r.normal(0, 0.05, len(idx)), 2)
        sales = (level + 12 * np.sin(2 * np.pi * t / 7) + 0.8 * temperature + 10 * promo
                 + r.normal(0, 4, len(idx)))
        sales[observed:] = np.nan  # the future: covariates known, sales not yet
        temp = temperature.round(1)
        temp[observed:] = np.nan  # weather is only measured, not known ahead
        frames.append(pd.DataFrame({
            "date": idx.strftime("%Y-%m-%d"), "store": store, "sales": sales.round(1), "temperature": temp,
            "promo": promo, "price": price, "store_type": kind,
            "note": [f"{store}-{i}" for i in range(len(idx))],
        }))
    return pd.concat(frames, ignore_index=True)


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

    samples["daily_sales_holidays.csv"] = holiday_sales()
    samples["store_demand_covariates.csv"] = store_demand()

    for name, df in samples.items():
        df.to_csv(OUT / name, index=False)
        print(f"wrote {name} ({len(df)} rows)")


if __name__ == "__main__":
    main()
