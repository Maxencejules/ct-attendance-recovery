"""Command-line entry point: ``python -m attendance_recovery {analyze,forecast}``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .loader import load_attendance


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _describe_load(data) -> list:
    rep = data.report
    missing = ", ".join(f"{y}: {n}" for y, n in rep.rows_missing_year.items())
    lines = [
        f"Read {rep.rows_read} rows from {rep.source} "
        f"({rep.district_rows} district rows, {rep.statewide_rows} statewide rows excluded from averages/models).",
        f"School years: {', '.join(data.years)}"
        + (f" (latest reporting period: {', '.join(rep.reporting_periods)})" if rep.reporting_periods else ""),
        f"Rows with a suppressed or missing rate, by year: {missing}",
    ]
    return lines


def run_analyze(args: argparse.Namespace) -> int:
    from .reports import analysis_markdown
    from .trends import group_trajectories, recovery_gaps

    data = load_attendance(args.data, rate_unit=getattr(args, "rate_unit", "auto"))
    trajectories = group_trajectories(data)
    gaps = recovery_gaps(data, args.baseline)
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    trajectories.to_csv(out / "trajectories.csv", index=False, float_format="%.4f")
    gaps.to_csv(out / "recovery_gaps.csv", index=False, float_format="%.4f")
    _write(out / "attendance_summary.md", analysis_markdown(data, trajectories, gaps))

    for line in _describe_load(data):
        print(line)
    everyone = trajectories[trajectories["student_group"] == "All Students"]
    if len(everyone):
        series = ", ".join(f"{r.year} {r.weighted_rate_pct:.2f}%" for r in everyone.itertuples())
        print(f"All Students, student-weighted: {series}")
    latest = gaps[(gaps["year"] == data.latest_year) & gaps["gap_pp"].notna()].sort_values("gap_pp")
    if len(latest):
        base = latest["baseline_year"].iloc[0]
        worst, best = latest.iloc[0], latest.iloc[-1]
        below = int((latest["gap_pp"] < 0).sum())
        print(
            f"{data.latest_year} vs {base}: {below} of {len(latest)} groups below baseline; "
            f"largest gap {worst.gap_pp:+.2f} pp ({worst.student_group}), "
            f"smallest {best.gap_pp:+.2f} pp ({best.student_group})"
        )
    print(f"Wrote {out / 'trajectories.csv'}, {out / 'recovery_gaps.csv'}, {out / 'attendance_summary.md'}")
    return 0


def run_forecast(args: argparse.Namespace) -> int:
    from .forecasting import MODEL_LABELS, build_model_frame, comparison_sentences, cross_validate
    from .reports import forecast_markdown

    data = load_attendance(args.data, rate_unit=getattr(args, "rate_unit", "auto"))
    frame = build_model_frame(data)
    result = cross_validate(frame, n_splits=args.folds, random_state=args.seed)
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    result.metrics.to_csv(out / "forecast_metrics.csv", index=False, float_format="%.4f")
    result.fold_metrics.to_csv(out / "forecast_fold_metrics.csv", index=False, float_format="%.4f")
    result.predictions.to_csv(out / "forecast_predictions.csv", index=False, float_format="%.4f")
    result.by_group.to_csv(out / "forecast_by_group.csv", index=False, float_format="%.4f")
    frame.exclusions.to_csv(out / "forecast_excluded_rows.csv", index=False)
    _write(out / "forecast_summary.md", forecast_markdown(data, frame, result))

    for line in _describe_load(data)[:2]:
        print(line)
    print(f"Target: {frame.target_year} rate; inputs from {', '.join(frame.prior_years)}")
    n_districts = frame.rows["district_code"].nunique()
    print(f"Rows used: {len(frame.rows)} from {n_districts} districts; excluded: {len(frame.exclusions)}")
    for reason, n in frame.exclusions["reason"].value_counts().items():
        print(f"  {n:5d}  {reason}")
    print(f"GroupKFold by district, {args.folds} folds (no district in both train and test)")
    print("Retrospective same-year district validation; training uses other districts' target-year outcomes.")
    print(f"{'forecaster':38s} {'MAE pp':>8s} {'RMSE pp':>8s}")
    for row in result.metrics.itertuples():
        print(f"{MODEL_LABELS[row.model]:38s} {row.mae_pp:8.3f} {row.rmse_pp:8.3f}")
    for line in comparison_sentences(result.metrics, result.n_splits):
        print(line)
    print(f"Wrote forecast_*.csv and forecast_summary.md to {out}")
    return 0


def run_backtest(args: argparse.Namespace) -> int:
    from .backtesting import expanding_origin_backtest
    from .reports import backtest_markdown

    data = load_attendance(args.data, rate_unit=args.rate_unit, require_paired=False)
    result = expanding_origin_backtest(data, history_years=args.history_years, random_state=args.seed)
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    tables = {
        "backtest_metrics.csv": result.metrics,
        "backtest_fold_metrics.csv": result.fold_metrics,
        "backtest_predictions.csv": result.predictions,
        "backtest_by_group.csv": result.by_group,
        "backtest_excluded_rows.csv": result.exclusions,
    }
    for name, table in tables.items():
        table.to_csv(out / name, index=False, float_format="%.4f")
    _write(out / "backtest_summary.md", backtest_markdown(data, result))
    print(f"Expanding-origin backtest: {result.n_origins} time-held-out year(s), {result.history_years} prior years per row")
    for row in result.metrics.itertuples():
        print(f"{row.model}: MAE {row.mae_pp:.3f} pp; RMSE {row.rmse_pp:.3f} pp")
    print(f"Wrote backtest_*.csv and backtest_summary.md to {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="attendance_recovery",
        description="Attendance recovery analysis and forecasting for Connecticut district attendance data.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze", help="weighted trajectories and gaps versus a baseline year")
    analyze.add_argument("--data", type=Path, required=True, help="attendance CSV downloaded from data.ct.gov")
    analyze.add_argument("--out", type=Path, required=True, help="directory for CSV and Markdown outputs")
    analyze.add_argument("--baseline", default=None, help="baseline school year (default 2019-2020 if present)")
    analyze.set_defaults(func=run_analyze)

    forecast = sub.add_parser("forecast", help="cross-validated forecast of the latest year's rates")
    forecast.add_argument("--data", type=Path, required=True, help="attendance CSV downloaded from data.ct.gov")
    forecast.add_argument("--out", type=Path, required=True, help="directory for CSV and Markdown outputs")
    forecast.add_argument("--folds", type=int, default=5, help="number of GroupKFold folds (default 5)")
    forecast.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random_state passed to gradient boosting (default 0); with the fixed settings "
        "used here the model has no random step, so this does not change the results",
    )
    forecast.set_defaults(func=run_forecast)
    backtest = sub.add_parser("backtest", help="time-held-out expanding-origin evaluation")
    backtest.add_argument("--data", type=Path, required=True, help="attendance CSV downloaded from data.ct.gov")
    backtest.add_argument("--out", type=Path, required=True, help="directory for CSV and Markdown outputs")
    backtest.add_argument("--history-years", type=int, default=2, help="fixed prior-year feature window (default 2)")
    backtest.add_argument("--seed", type=int, default=0, help="gradient boosting random state")
    backtest.set_defaults(func=run_backtest)
    for command in (analyze, forecast, backtest):
        command.add_argument("--rate-unit", choices=("auto", "percent", "fraction"), default="auto",
                             help="unit of unmarked rate cells; explicit %% cells always use percent")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
