"""Data Exploration Agent: profiles imported data, highlights anomalies, distributions and column relationships."""

from __future__ import annotations

from fcagents.core.agent import BaseAgent, Decision
from fcagents.core.state import PipelineState
from fcagents.tools.explore import CORR_METHODS, explore_dataset


class DataExplorationAgent(BaseAgent):
    name = "exploration"
    title = "Data Exploration Agent"
    task = ("Summarize the imported dataset's structure, data quality anomalies, column distributions, "
            "and pairwise relationships between features.")
    scope = ("Only discuss dataset shape, column types, missingness, duplicates, distribution skew/IQR anomalies, "
             "and pairwise correlations from the facts. Do NOT discuss resampling frequencies, holidays, or imputation.")

    summary = ("Profiles the imported dataset (CSV, TSV, JSON, Excel or Parquet) before forecasting: audits "
               "column types, missingness, duplicate rows, distribution skewness, IQR anomalies, and pairwise "
               "relationships between columns.")
    runs_when = "Always (first step after importing a dataset)."
    inputs = ("Imported dataset (CSV, TSV, JSON, Excel or Parquet)", "Correlation method (Pearson or Spearman)")
    outputs = ("Per-column statistical & quality profile", "Anomaly summary (duplicates, missing, skew, IQR outliers)",
               "Pairwise correlation matrix & strongest feature relationships")
    decisions = {
        "clean": "Dataset imported cleanly with no major structural anomalies, missingness, or extreme skew.",
        "findings": "Dataset imported; notable anomalies, missing values, skewed distributions, or strong correlations were found.",
        "empty": "Dataset has no rows or no columns; import a non-empty file to proceed.",
    }
    user_controls = ("Distribution column picker", "Correlation method (Pearson / Spearman)",
                     "Relationship X and Y column pickers")
    tools = ("tools.explore.load_dataset", "tools.explore.profile_columns",
             "tools.explore.detect_anomalies", "tools.explore.compute_relationships")

    def run_tools(self, state: PipelineState) -> dict:
        facts, extras = explore_dataset(state.raw, corr_method=state.corr_method)
        self._last_extras = extras
        return facts

    def decide(self, f: dict) -> Decision:
        if f.get("empty"):
            return Decision("error", "empty", 1.0,
                            f"The imported dataset is empty ({f['n_rows']} rows × {f['n_cols']} columns).",
                            "Upload a non-empty CSV, TSV, JSON, Excel, or Parquet file, or pick a sample dataset.")

        kinds = ", ".join(f"{n} {k}" for k, n in f["column_kinds"].items())
        parts = [f"**{f['n_rows']:,} rows × {f['n_cols']} columns** ({kinds})."]

        notes: list[str] = []
        if f["n_duplicate_rows"]:
            notes.append(f"{f['n_duplicate_rows']} duplicate rows ({f['duplicate_pct']}%)")
        if f["any_missing_cols"]:
            cols = ", ".join(f"`{c}`" for c in f["any_missing_cols"][:4])
            more = f" (+{len(f['any_missing_cols']) - 4} more)" if len(f["any_missing_cols"]) > 4 else ""
            notes.append(f"missing values in {cols}{more} ({f['missing_cell_pct']}% of cells)")
        if f["constant_cols"]:
            notes.append("constant column(s): " + ", ".join(f"`{c}`" for c in f["constant_cols"][:3]))
        if f["total_iqr_anomalies"]:
            top_col, top_n = max(f["iqr_anomaly_cols"].items(), key=lambda kv: kv[1])
            notes.append(f"{f['total_iqr_anomalies']} IQR extreme values (most in `{top_col}`: {top_n})")
        if f["skewed_cols"]:
            notes.append("strongly skewed: " + ", ".join(f"`{c}`" for c in f["skewed_cols"][:3]))

        if notes:
            parts.append("Anomalies & quality checks: " + "; ".join(notes) + ".")
        else:
            parts.append("No duplicate rows, missing cells, constant columns, or extreme IQR anomalies detected.")

        if f["top_pairs"]:
            p0 = f["top_pairs"][0]
            method_label = CORR_METHODS.get(f["corr_method"], f["corr_method"])
            parts.append(
                f"Strongest {method_label} relationship: `{p0['col_a']}` vs `{p0['col_b']}` (r = {p0['r']:+.3f})."
            )

        has_warnings = bool(f["n_duplicate_rows"] or f["high_missing_cols"] or f["constant_cols"])
        has_findings = bool(notes or f["top_pairs"])
        if has_warnings:
            return Decision("warning", "findings", 1.0, " ".join(parts),
                            "Inspect the distribution and relationship charts below, then continue to Data.")
        if has_findings:
            return Decision("ok", "findings", 1.0, " ".join(parts),
                            "Explore column distributions and correlations below, then proceed to Data.")
        return Decision("ok", "clean", 1.0, " ".join(parts),
                        "Data looks structurally clean — proceed to Data to set up timestamps and targets.")

    def prompt_facts(self, f: dict) -> dict:
        if f.get("empty"):
            return {"n_rows": f["n_rows"], "n_cols": f["n_cols"], "empty": True}
        return {
            "n_rows": f["n_rows"],
            "n_cols": f["n_cols"],
            "column_kinds": f["column_kinds"],
            "missing_cell_pct": f["missing_cell_pct"],
            "n_duplicate_rows": f["n_duplicate_rows"],
            "constant_cols": f["constant_cols"][:5],
            "skewed_cols": f["skewed_cols"][:5],
            "iqr_anomaly_cols": dict(list(f["iqr_anomaly_cols"].items())[:5]),
            "corr_method": f["corr_method"],
            "top_pairs": f["top_pairs"][:5],
        }

    def run(self, state: PipelineState, llm):
        result = super().run(state, llm)
        if hasattr(self, "_last_extras"):
            result.extras |= self._last_extras
        return result
