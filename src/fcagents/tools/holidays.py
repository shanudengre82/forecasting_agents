"""Holiday calendars and a robust test of whether holidays move the series (deterministic, numpy/pandas only)."""

from __future__ import annotations

import re
from functools import lru_cache

import holidays
import numpy as np
import pandas as pd

HOLIDAY_TYPES = {"official": "Official (public holidays)", "unofficial": "Unofficial (observances, other days)"}
DEFAULT_COUNTRY = "US"
DEFAULT_WINDOW = 7
MAX_WINDOW = 14
MIN_OCCURRENCES = 3
BASELINE_WEEKS = 4  # same weekday, up to 4 weeks before and after
BASELINE_SKIP_DAYS = 1  # holidays and the days right next to them are left out of the baseline
MIN_EFFECT_Z = 1.2  # mean |robust z| an offset needs to count as an effect (ordinary days average about 0.8)
EFFECT_Z = 3.0  # per holiday and offset: |z| in every year (with 2+ years of data)
EFFECT_Z_SINGLE = 5.0  # stricter when the data covers the holiday only once (day itself only)
ALPHA = 0.05
N_PERMUTATIONS = 2000


# --------------------------------------------------------------------------- calendars


@lru_cache(maxsize=1)
def countries() -> dict[str, str]:
    """{ISO code: country name}, sorted by name."""
    names = {code: key.replace("_", " ").title() for key, (_, code, *_rest) in holidays.registry.COUNTRIES.items()}
    supported = set(holidays.list_supported_countries())
    return dict(sorted(((c, n) for c, n in names.items() if c in supported), key=lambda kv: kv[1]))


def _cls(country: str):
    return holidays.country_holidays(country).__class__


@lru_cache(maxsize=64)
def subdivisions(country: str) -> dict[str, str]:
    """{subdivision code: name}, e.g. {"CA": "California"} for the US. Empty if the country has none."""
    cls = _cls(country)
    names = {code: name for name, code in getattr(cls, "subdivisions_aliases", {}).items()}
    return {code: names.get(code, code) for code in cls.subdivisions}


@lru_cache(maxsize=64)
def unofficial_categories(country: str) -> tuple[str, ...]:
    return tuple(c for c in _cls(country).supported_categories if c != holidays.PUBLIC)


def holiday_calendar(country: str, subdiv: str | None, types: tuple[str, ...], start, end) -> pd.Series:
    """Holiday names indexed by date (midnight Timestamps) between ``start`` and ``end``."""
    cats: tuple[str, ...] = ()
    if "official" in types:
        cats += (holidays.PUBLIC,)
    if "unofficial" in types:
        cats += unofficial_categories(country)
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    if not cats or end < start:
        return pd.Series(dtype=object, name="holiday")
    h = holidays.country_holidays(country, subdiv=subdiv or None, years=range(start.year, end.year + 1),
                                  categories=cats)
    cal = pd.Series({pd.Timestamp(d): "; ".join(h.get_list(d)) for d in h}, dtype=object, name="holiday")
    return cal.sort_index().loc[start:end]


# --------------------------------------------------------------------------- impact


def daily_view(series: pd.Series, freq: str, agg: str | None) -> pd.Series | None:
    """One value per calendar day, or None if the frequency is coarser than a day."""
    if freq in ("B", "C", "D"):
        return series.groupby(series.index.normalize()).mean().asfreq("D")
    try:
        step = pd.Timedelta(pd.tseries.frequencies.to_offset(freq))
    except (ValueError, TypeError):
        return None  # weeks, months, quarters…
    if step > pd.Timedelta(days=1):
        return None
    days = series.resample("D")
    return days.sum(min_count=1) if agg in ("sum", "count") else days.mean()


def _shift_days(dates: pd.DatetimeIndex, offset: int) -> pd.DatetimeIndex:
    return dates + pd.Timedelta(days=offset)


def holiday_impact(series: pd.Series, freq: str, calendar: pd.Series, window: int = DEFAULT_WINDOW,
                   agg: str | None = None, seed: int = 0) -> tuple[dict, dict]:
    """(facts, extras). Robust z-scores against a same-weekday baseline, then for each offset around the holidays
    the mean |z| (holidays can push the series up or down, so signed effects would cancel) with a permutation
    p-value against random ordinary days."""
    window = int(max(0, min(window, MAX_WINDOW)))
    daily = daily_view(series, freq, agg)
    base = {"window": window, "n_holidays_in_calendar": int(len(calendar))}
    if daily is None:
        return base | {"applicable": False}, {}
    daily = daily.astype(float)
    hol_days = calendar.index[(calendar.index >= daily.index[0]) & (calendar.index <= daily.index[-1])]
    in_window = pd.Series(False, index=daily.index)
    for o in range(-window, window + 1):
        in_window |= daily.index.isin(_shift_days(hol_days, o))
    near = pd.Series(False, index=daily.index)
    for o in range(-BASELINE_SKIP_DAYS, BASELINE_SKIP_DAYS + 1):
        near |= daily.index.isin(_shift_days(hol_days, o))

    # The median of same-weekday neighbours shrugs off the odd run-up day, so only days at a holiday are skipped.
    ordinary = daily.where(~near)
    neighbours = pd.concat([ordinary.shift(7 * k) for k in range(1, BASELINE_WEEKS + 1)]
                           + [ordinary.shift(-7 * k) for k in range(1, BASELINE_WEEKS + 1)], axis=1)
    baseline = neighbours.median(axis=1).where(neighbours.notna().sum(axis=1) >= 2)
    resid = daily - baseline
    normal = resid[~in_window].dropna()
    scale = 1.4826 * float((normal - normal.median()).abs().median()) if len(normal) else 0.0
    scale = scale or float(normal.std()) or 1.0
    z = resid / scale

    occ = [d for d in hol_days if pd.notna(z.get(d))]
    facts = base | {"applicable": True, "n_occurrences": len(occ), "n_days": int(daily.notna().sum())}
    if len(occ) < MIN_OCCURRENCES:
        return facts | {"holidays_with_effect": [], "window_effects": [], "n_effects": 0,
                        "protected_dates": [d.strftime("%Y-%m-%d") for d in occ]}, {}

    occ_idx = pd.DatetimeIndex(occ)
    names = calendar.reindex(occ_idx)

    # 1) Overall: are holidays further from normal than ordinary days? (mean |z|, permutation test)
    rng = np.random.default_rng(seed)
    pool = z[~in_window].dropna().abs().to_numpy()
    day0 = z.reindex(occ_idx).abs()
    size0 = float(day0.mean())
    null = rng.choice(pool, size=(N_PERMUTATIONS, len(occ_idx))).mean(axis=1)
    p0 = float((np.sum(null >= size0) + 1) / (N_PERMUTATIONS + 1))

    # 2) Per holiday and offset: an effect is a large move in the same direction in every year.
    long = []
    for o in range(-window, window + 1):
        shifted = _shift_days(occ_idx, o)
        vals = z.reindex(shifted).to_numpy()
        if o:  # a day that is itself a holiday belongs to that holiday, not to this one's window
            vals = np.where(shifted.isin(hol_days), np.nan, vals)
        long += [{"date": d, "holiday": n, "offset": o, "z": v} for d, n, v in zip(occ_idx, names, vals)]
    long = pd.DataFrame(long).dropna(subset=["z"])
    long = long.assign(holiday=long["holiday"].str.split("; ")).explode("holiday")
    g = long.groupby(["holiday", "offset"])["z"]
    stats = pd.DataFrame({"mean_z": g.mean(), "n": g.size(), "min_abs": g.apply(lambda v: v.abs().min()),
                          "same_sign": g.apply(lambda v: bool((v > 0).all() or (v < 0).all()))}).reset_index()
    need = np.where(stats["n"] >= 2, EFFECT_Z, np.where(stats["offset"] == 0, EFFECT_Z_SINGLE, np.inf))
    stats["effect"] = stats["same_sign"] & (stats["min_abs"] >= need)
    effects = stats[stats["effect"]].sort_values(["holiday", "offset"])

    profile = (long.groupby("offset")["z"].agg(mean_abs_z=lambda v: v.abs().mean(), mean_z="mean", n="size")
               .round(2).reset_index())
    profile["has_effect"] = profile["offset"].isin(effects["offset"])

    occurrences = pd.DataFrame({
        "date": occ_idx, "holiday": names.to_numpy(), "value": daily.reindex(occ_idx).to_numpy(),
        "expected": baseline.reindex(occ_idx).to_numpy(), "z": z.reindex(occ_idx).round(2).to_numpy(),
    })
    occurrences["change_pct"] = ((occurrences["value"] / occurrences["expected"] - 1) * 100).round(1)

    protected = set(occ_idx)  # holidays themselves are never treated as outliers
    for _, e in effects.iterrows():
        dates = calendar.index[calendar.str.contains(e["holiday"], regex=False)]
        protected |= set(_shift_days(dates, int(e["offset"])))
    protected &= set(daily.index)

    day0_effects = effects[effects["offset"] == 0]
    facts |= {
        "effect_day0_abs_z": round(size0, 2),
        "ordinary_abs_z": round(float(pool.mean()), 2),
        "p_day0": round(p0, 4),
        "overall_significant": bool(p0 < ALPHA and size0 >= MIN_EFFECT_Z),
        "holidays_with_effect": sorted(day0_effects["holiday"].unique().tolist()),
        "window_effects": [{"holiday": r["holiday"], "offset": int(r["offset"]), "mean_z": round(float(r["mean_z"]), 2)}
                           for _, r in effects[effects["offset"] != 0].iterrows()],
        "n_effects": int(len(effects)),
        "top_holidays": (occurrences.assign(holiday=occurrences["holiday"].str.split("; ")).explode("holiday")
                         .groupby("holiday")["change_pct"].mean().round(1)
                         .sort_values(key=np.abs, ascending=False).head(5).to_dict()),
        "protected_dates": sorted(d.strftime("%Y-%m-%d") for d in protected),
    }
    extras = {"profile": profile, "occurrences": occurrences, "effects": effects.reset_index(drop=True),
              "daily": daily, "baseline": baseline}
    return facts, extras


# --------------------------------------------------------------------------- covariates


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower().replace("'", "")).strip("_")


def holiday_covariates(index: pd.DatetimeIndex, calendar: pd.Series, effects: pd.DataFrame) -> pd.DataFrame:
    """0/1 columns aligned to ``index``: ``is_holiday``, plus one per detected effect, named
    ``holiday_<name>`` for the day itself and ``holiday_<name>_m3`` / ``_p1`` for days before / after."""
    days = index.normalize()
    cols = {"is_holiday": days.isin(calendar.index)}
    for _, e in effects.iterrows():
        o = int(e["offset"])
        dates = calendar.index[calendar.str.contains(e["holiday"], regex=False)]
        suffix = "" if o == 0 else f"_{'m' if o < 0 else 'p'}{abs(o)}"
        cols[f"holiday_{_slug(e['holiday'])}{suffix}"] = days.isin(_shift_days(dates, o))
    return pd.DataFrame(cols, index=index).astype("int8")


def protected_index(index: pd.DatetimeIndex, protected_dates: list[str]) -> pd.DatetimeIndex:
    """Timestamps of ``index`` that fall on a protected (holiday-affected) day."""
    if not protected_dates:
        return index[:0]
    return index[index.normalize().isin(pd.DatetimeIndex(protected_dates))]
