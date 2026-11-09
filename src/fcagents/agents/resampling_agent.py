from __future__ import annotations

from fcagents.core.agent import BaseAgent, Decision
from fcagents.core.state import PipelineState
from fcagents.tools.resample import (
    MIN_POINTS,
    SPARSE_EMPTY_SHARE,
    recommend_freq,
    resample_options,
    resample_series,
)


class ResamplingAgent(BaseAgent):
    name = "resampling"
    per_series = True
    title = "Resampling Agent"
    task = "Convert an irregularly sampled series onto a fixed frequency between 1 hour and 1 week."
    scope = ("Do not invent point counts. Explain the trade-off: finer frequencies give more points but more "
             "empty bins; coarser ones give fewer, denser points.")

    summary = ("Converts irregularly spaced observations into a fixed-frequency series between 1 hour and 1 week, "
               "using your choice of frequency and aggregation. Empty bins are left as missing timestamps for the "
               "Missing Values agent.")
    runs_when = ("When the Regularity agent says the series is irregular, or when you pick a frequency for a "
                 "regular series (e.g. aggregate hourly data to daily).")
    inputs = ("Series from the Data page", "Your resampling preferences (page controls or chat)")
    outputs = ("Chosen frequency and whether it was yours or auto-recommended",
               "Data points, bins and empty-bin share for every candidate frequency",
               "The resampled series, empty bins dropped (replaces the raw one for later agents)")
    decisions = {
        "resampled": "Enough data points and few empty bins; the pipeline continues.",
        "sparse": "Enough points, but over half the bins are empty; a coarser frequency is suggested.",
        "too_few_points": f"Fewer than {MIN_POINTS} data points; finer frequencies are suggested and the pipeline stops.",
    }
    user_controls = ("Target frequency: auto or h, 2h, 3h, 4h, 6h, 12h, D, 2D, 3D, W (controls or chat)",
                     "Aggregation: mean, sum, median, min, max, first, last, count (controls or chat)")
    tools = ("tools.resample.resample_options", "tools.resample.recommend_freq", "tools.resample.resample_series")

    def applies(self, state: PipelineState) -> str | None:
        if state.fact("regularity", "status") != "irregular" and state.resample_freq in (None, "auto"):
            return (f"Skipped: series already has fixed frequency '{state.freq}'. "
                    "Pick a target frequency to aggregate it anyway.")
        return None

    def run_tools(self, state: PipelineState) -> dict:
        raw = state.series
        options = resample_options(raw)
        recommended = recommend_freq(options)
        user_choice = state.resample_freq not in (None, "auto")
        freq = state.resample_freq if user_choice else recommended

        resampled, facts = resample_series(raw, freq, state.resample_agg)
        state.series, state.freq = resampled, freq

        chosen = next(o for o in options if o["freq"] == freq)
        finer = [o for o in options[: options.index(chosen)] if o["enough_points"]]
        return facts | {
            "chosen_freq": freq,
            "freq_source": "user" if user_choice else "auto",
            "recommended_freq": recommended,
            "min_points": MIN_POINTS,
            "options": options,
            "finer_options_with_enough_points": [{"freq": o["freq"], "n_points": o["n_points"]} for o in finer[::-1]],
        }

    def run(self, state, llm):
        raw = state.series
        result = super().run(state, llm)
        result.extras["raw_series"] = raw  # lets the UI overlay the original observations
        return result

    def decide(self, f: dict) -> Decision:
        how = f"{f['agg']} per bin"
        src = "your choice" if f["freq_source"] == "user" else "auto-selected"
        if f["n_points"] < MIN_POINTS:
            finer = f["finer_options_with_enough_points"]
            if finer:
                opts = ", ".join(f"{o['freq']} ({o['n_points']} points)" for o in finer[:3])
                step = f"Select a finer frequency to get at least {MIN_POINTS} points: {opts}."
            else:
                step = (f"Even hourly bins give fewer than {MIN_POINTS} points: the time span is too short "
                        "to model. Collect a longer history.")
            return Decision("warning", "too_few_points", f["n_points"] / MIN_POINTS,
                            f"Resampling to {f['chosen_freq']} ({f['label']}, {src}) yields only "
                            f"{f['n_points']} data points (< {MIN_POINTS}).", step)
        summary = (f"Resampled {f['n_input']} observations to {f['chosen_freq']} ({f['label']}, {src}; "
                   f"{how}): {f['n_points']} data points, {f['n_empty']} of {f['n_bins']} bins empty "
                   f"({f['empty_share']:.0%}).")
        if f["empty_share"] > SPARSE_EMPTY_SHARE:
            return Decision("warning", "sparse", 1 - f["empty_share"], summary +
                            f" More than {SPARSE_EMPTY_SHARE:.0%} of bins are empty.",
                            f"Consider a coarser frequency (recommended: {f['recommended_freq']}); otherwise "
                            "the Missing values page has many empty bins to fill.")
        step = (f"Fill the {f['n_empty']} empty bins on the Missing values page." if f["n_empty"]
                else "Name the new frequency and derive seasonal periods.")
        return Decision("ok", "resampled", 1 - f["empty_share"], summary, step)

    def prompt_facts(self, f: dict) -> dict:
        keys = ["chosen_freq", "label", "freq_source", "recommended_freq", "agg", "n_input",
                "n_bins", "n_points", "empty_share", "min_points", "finer_options_with_enough_points"]
        return {k: f.get(k) for k in keys}
