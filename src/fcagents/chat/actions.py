"""What the chat may change on each page, and the check every proposed change must pass."""

from __future__ import annotations

from dataclasses import dataclass

from fcagents.tools.missing import FILL_METHODS
from fcagents.tools.resample import AGGREGATIONS, FREQ_OPTIONS

PAGE_KEYS = {
    "data": ("time_col", "target_col", "series_id"),
    "resampling": ("resample_freq", "resample_agg"),
    "missing": ("fill_method",),
}
DEFAULTS = {"time_col": None, "target_col": None, "series_id": None,
            "resample_freq": "auto", "resample_agg": "mean", "fill_method": "auto"}
KEY_LABELS = {"time_col": "time column", "target_col": "target column", "series_id": "series",
              "resample_freq": "frequency", "resample_agg": "aggregation", "fill_method": "fill method"}


@dataclass(frozen=True)
class Action:
    key: str
    value: str | None  # None = auto-detect (data page defaults)

    def describe(self) -> str:
        if self.value is None:
            return f"{KEY_LABELS[self.key]} → auto-detect"
        label = {"resample_freq": FREQ_OPTIONS, "fill_method": FILL_METHODS}.get(self.key, {}).get(self.value)
        return f"{KEY_LABELS[self.key]} → `{self.value}`" + (f" ({label})" if label else "")


def allowed_values(key: str, ctx: dict) -> list[str]:
    """``ctx`` carries what the data offers: time_cols, target_cols, series_ids."""
    return {
        "resample_freq": ["auto", *FREQ_OPTIONS],
        "resample_agg": list(AGGREGATIONS),
        "fill_method": ["auto", *FILL_METHODS],
        "time_col": list(ctx.get("time_cols", [])),
        "target_col": list(ctx.get("target_cols", [])),
        "series_id": [str(s) for s in ctx.get("series_ids", [])],
    }[key]


def validate(page: str, action: Action, ctx: dict) -> str | None:
    """Return why ``action`` is not allowed on ``page``, or None if it is."""
    if action.key not in PAGE_KEYS.get(page, ()):
        return f"The {KEY_LABELS.get(action.key, action.key)} can't be changed on this page."
    allowed = allowed_values(action.key, ctx)
    if action.value not in allowed and action.value != DEFAULTS[action.key]:
        return f"`{action.value}` is not a valid {KEY_LABELS[action.key]}. Choose one of: {', '.join(allowed)}."
    return None
