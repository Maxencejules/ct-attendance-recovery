"""Markdown write-ups for the ``analyze`` and ``forecast`` commands."""

from __future__ import annotations

import pandas as pd

from .forecasting import MODEL_LABELS, MODEL_NAMES, ForecastResult, ModelFrame, comparison_sentences
from .loader import AttendanceData
from .markdown import markdown_table


def _group_label(frame: pd.DataFrame) -> pd.Series:
    same = frame["category"] == frame["student_group"]
    return frame["student_group"].where(same, frame["category"] + " / " + frame["student_group"])


def data_handling_lines(data: AttendanceData) -> list:
    rep = data.report
    lines = [
        f"- Rows read: {rep.rows_read} ({rep.district_rows} district rows, "
        f"{rep.statewide_rows} statewide rows; duplicates dropped: {rep.duplicate_rows_dropped}).",
        "- Statewide rows (district code 00000CT) are not used in any average or model; "
        "they appear only as the `state_reported_pct` cross-check column.",
        f"- Non-numeric cells treated as suppressed/missing: {rep.token_summary()}.",
        f"- Out-of-range values cleared: {rep.out_of_range_cells}; "
        f"count/rate pairs cleared because one half was missing: {rep.unpaired_values_cleared}.",
    ]
    per_year = ", ".join(f"{y}: {n}" for y, n in rep.rows_missing_year.items())
    lines.append(f"- Rows with no usable rate, by year (all rows incl. statewide): {per_year}.")
    scales = ", ".join(f"{y}: {s}" for y, s in rep.rate_scale.items())
    lines.append(f"- Rate scale detected in the file: {scales}.")
    if rep.reporting_periods:
        lines.append(
            f"- Reporting period of the latest year: {', '.join(rep.reporting_periods)} "
            "(year-to-date figures)."
        )
    return lines


def analysis_markdown(data: AttendanceData, trajectories: pd.DataFrame, gaps: pd.DataFrame) -> str:
    baseline = gaps["baseline_year"].iloc[0] if len(gaps) else data.default_baseline()
    latest = data.latest_year

    traj = trajectories.copy()
    traj["group"] = _group_label(traj)
    rate_wide = traj.pivot_table(index="group", columns="year", values="weighted_rate_pct", sort=False)
    rate_wide = rate_wide.reindex(columns=data.years)

    gap_wide = pd.DataFrame()
    detail = pd.DataFrame()
    if len(gaps):
        g = gaps.copy()
        g["group"] = _group_label(g)
        gap_wide = g.pivot_table(index="group", columns="year", values="gap_pp", sort=False)
        latest_rows = g[g["year"] == latest].sort_values("gap_pp")
        detail = latest_rows[
            [
                "group",
                "matched_rows",
                "baseline_rate_pct",
                "rate_pct",
                "gap_pp",
                "median_district_gap_pp",
                "rows_at_or_above_baseline",
            ]
        ].rename(
            columns={
                "matched_rows": "matched district rows",
                "baseline_rate_pct": f"{baseline} %",
                "rate_pct": f"{latest} %",
                "gap_pp": "gap (pp)",
                "median_district_gap_pp": "median district gap (pp)",
                "rows_at_or_above_baseline": "rows at/above baseline",
            }
        )

    parts = [
        "# Connecticut attendance recovery by student group",
        "",
        f"Source file: `{data.report.source}`  ",
        f"School years: {', '.join(data.years)}; baseline: {baseline}; latest: {latest}.",
        "",
        "## Data handling",
        "",
        *data_handling_lines(data),
        "",
        "## Student-weighted attendance rate by group (%)",
        "",
        "Weighted across district rows by each row's student count for that year. "
        "Every district row that reports a year is included, so the set of districts "
        "can differ slightly between years.",
        "",
        markdown_table(rate_wide, digits=2, index=True),
        "",
    ]
    if len(gaps):
        parts += [
            f"## Change versus {baseline} (percentage points)",
            "",
            f"Computed on matched district rows (rows reporting both {baseline} and the "
            "year shown). Negative values mean attendance is still below the baseline.",
            "",
            markdown_table(gap_wide, digits=2, index=True),
            "",
            f"## {latest} compared with {baseline}, sorted by gap",
            "",
            markdown_table(detail, digits=2),
            "",
        ]
    parts += [
        "## Notes",
        "",
        "- Category-level totals are not computed because groups inside a category overlap "
        "(for example, the combined free/reduced-price meal group contains the two separate groups).",
        "- District-level aggregates cannot show changes for individual students or schools.",
        "",
    ]
    return "\n".join(parts)


def forecast_markdown(data: AttendanceData, frame: ModelFrame, result: ForecastResult) -> str:
    metrics = result.metrics.copy()
    metrics.insert(0, "forecaster", metrics["model"].map(MODEL_LABELS))
    metrics = metrics.drop(columns=["model"])

    folds = result.fold_metrics.pivot_table(index="fold", columns="model", values="mae_pp")
    folds = folds.reindex(columns=list(MODEL_NAMES))
    fold_sizes = result.fold_metrics.drop_duplicates("fold").set_index("fold")[["test_rows", "test_districts"]]
    folds = fold_sizes.join(folds)

    by_group = result.by_group.copy()
    by_group.insert(0, "group", _group_label(by_group))
    by_group = by_group.drop(columns=["category", "student_group"])

    reasons = frame.exclusions["reason"].value_counts().rename_axis("reason").reset_index(name="rows")
    n_districts = frame.rows["district_code"].nunique()

    parts = [
        f"# Retrospective district validation for {frame.target_year} attendance rates",
        "",
        f"Source file: `{data.report.source}`",
        "",
        "## Setup",
        "",
        f"- Target: each district x student-group row's {frame.target_year} attendance rate "
        "(percentage points).",
        f"- Inputs: rates and log student counts for {', '.join(frame.prior_years)}, "
        "their recent changes, and one-hot category and student group.",
        "- Baseline: persistence, i.e. the row's previous-year rate. "
        "Reference: persistence plus the mean change seen in the fold's training rows.",
        "- Ridge and gradient boosting are trained on the change from the previous year's rate; "
        "their forecast is the previous year's rate plus the predicted change.",
        f"- Validation: GroupKFold with {result.n_splits} folds grouped by district code, "
        f"so no district appears in both the training and test part of a fold "
        f"({n_districts} districts, {len(frame.rows)} rows).",
        "- This is same-year retrospective validation: training uses other districts' "
        "target-year outcomes. It does not measure future-year prediction. Use the "
        "`backtest` command for a time-held-out evaluation.",
        "",
        "## Rows excluded",
        "",
        markdown_table(reasons) if len(reasons) else "None.",
        "",
        "## Out-of-fold error (percentage points, all folds pooled)",
        "",
        markdown_table(metrics, digits=3),
        "",
        "## Comparison with persistence",
        "",
        *[f"- {line}" for line in comparison_sentences(result.metrics, result.n_splits)],
        "",
        "## MAE by fold (pp)",
        "",
        markdown_table(folds, digits=3, index=True),
        "",
        "## MAE by student group (pp)",
        "",
        markdown_table(by_group, digits=3),
        "",
    ]
    return "\n".join(parts)


def backtest_markdown(data, result) -> str:
    folds = result.fold_metrics
    return "\n".join([
        "# Time-held-out attendance backtest", "",
        f"Source file: `{data.report.source}`", "",
        f"- Fixed feature window: {result.history_years} prior school years; {result.n_origins} evaluated origins.",
        "- Each origin fits preprocessing and all models only on strictly earlier target years. "
        "The current target's outcomes and student counts never enter its predictions.",
        "- Districts may recur across years. This measures transfer across time for observed "
        "districts, separately from the district holdout used by `forecast`.",
        "- Hyperparameters are fixed. Complete prior-window counts/rates and an observed target "
        "rate are required for scoring; the exclusion file audits omitted rows per origin and role.",
        "- Errors are unweighted district/group percentage-point errors, pooled across origins. "
        "Groups overlap and repeated yearly observations are dependent; no confidence or causal claim follows.",
        "", "## Pooled error", "", markdown_table(result.metrics, digits=3),
        "", "## Origin and model results", "", markdown_table(folds, digits=3),
        "", "## MAE by student group", "", markdown_table(result.by_group, digits=3),
        "", "## Limits", "",
        "The included four-year snapshot supplies only one two-history-year time holdout. "
        "Earlier years span COVID disruptions and the latest year is year to date. "
        "A time holdout cannot establish performance in unobserved future years or explain "
        "changes for individual students. No model is promoted automatically.", "",
    ])
