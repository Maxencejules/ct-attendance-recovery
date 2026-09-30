"""Expanding-origin backtests of a fixed lookback, using earlier target years only.

Districts may occur in both historical training data and a later test year. This
evaluates time-held-out performance, not generalisation to unseen districts.
Only lagged rates/counts are features; a target year's count is never used to
select or predict that target. Its rate is required solely as an observed label.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from numbers import Integral

import numpy as np
import pandas as pd

from .forecasting import (
    CATEGORICAL_FEATURES,
    DRIFT,
    MODEL_NAMES,
    PERSISTENCE,
    STATEWIDE_REASON,
    errors_by_group,
    mae,
    make_models,
    rmse,
    summarise,
)
from .loader import ID_COLUMNS, AttendanceData, count_col, rate_col


@dataclass
class BacktestResult:
    predictions: pd.DataFrame
    fold_metrics: pd.DataFrame
    metrics: pd.DataFrame
    by_group: pd.DataFrame
    exclusions: pd.DataFrame
    n_origins: int
    history_years: int


def _validate(data: AttendanceData, history_years: int) -> None:
    if (
        isinstance(history_years, bool)
        or not isinstance(history_years, Integral)
        or history_years < 1
    ):
        raise ValueError("history_years must be a positive integer")
    starts = []
    for year in data.years:
        match = re.fullmatch(r"(\d{4})-(\d{4})", str(year))
        if not match or int(match[2]) != int(match[1]) + 1:
            raise ValueError(
                "backtesting requires consecutive YYYY-YYYY school-year labels"
            )
        starts.append(int(match[1]))
    if any(right != left + 1 for left, right in zip(starts, starts[1:], strict=False)):
        raise ValueError(
            "backtesting school years must be unique, chronological and consecutive"
        )
    if len(starts) < history_years + 2:
        raise ValueError(
            f"backtesting needs at least {history_years + 2} school years "
            f"for history_years={history_years}"
        )
    required = ID_COLUMNS + ["is_statewide"] + [rate_col(year) for year in data.years]
    # The latest count is not needed: that year is never an earlier feature window.
    required += [count_col(year) for year in data.years[:-1]]
    missing = [column for column in required if column not in data.table.columns]
    if missing:
        raise ValueError("backtesting missing column(s): " + ", ".join(missing))


def _window(
    data: AttendanceData, target_index: int, history_years: int, numeric: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target = data.years[target_index]
    # Lag 1 always means the immediately preceding school year, at every origin.
    lag_years = [data.years[target_index - lag] for lag in range(1, history_years + 1)]
    records, exclusions = [], []
    for _, observation in data.table.iterrows():
        identity = {column: observation[column] for column in ID_COLUMNS}
        reason = ""
        if observation["is_statewide"]:
            reason = STATEWIDE_REASON
        elif not np.isfinite(observation[rate_col(target)]):
            reason = f"target year {target} rate suppressed or missing"
        else:
            missing = [
                year
                for year in lag_years
                if not np.isfinite(observation[rate_col(year)])
                or not np.isfinite(observation[count_col(year)])
                or observation[count_col(year)] < 0
            ]
            if missing:
                reason = "history year(s) suppressed or missing: " + ", ".join(missing)
        if reason:
            exclusions.append({**identity, "target_year": target, "reason": reason})
            continue
        record = {
            **identity,
            "target_year": target,
            "actual": 100.0 * observation[rate_col(target)],
        }
        for lag, year in enumerate(lag_years, start=1):
            record[f"rate_pct_lag{lag}"] = 100.0 * observation[rate_col(year)]
            record[f"log_students_lag{lag}"] = np.log1p(observation[count_col(year)])
        if history_years >= 2:
            record["change_pp_last"] = record["rate_pct_lag1"] - record["rate_pct_lag2"]
            record["change_pp_since_first"] = (
                record["rate_pct_lag1"] - record[f"rate_pct_lag{history_years}"]
            )
        record[PERSISTENCE] = record["rate_pct_lag1"]
        records.append(record)
    rows = pd.DataFrame.from_records(
        records, columns=ID_COLUMNS + ["target_year", "actual", PERSISTENCE] + numeric
    )
    excluded = pd.DataFrame.from_records(
        exclusions, columns=ID_COLUMNS + ["target_year", "reason"]
    )
    return rows, excluded


def expanding_origin_backtest(
    data: AttendanceData, history_years: int = 2, random_state: int = 0
) -> BacktestResult:
    """Test each eligible later target once, fitting only strictly earlier targets.

    The first training target follows ``history_years`` preceding years; the
    first test target is one year later. All available earlier target windows
    accumulate in training. Preprocessing is fitted afresh on those rows only.
    A missing target count does not exclude a row, but a missing lagged count
    does. An empty training/test origin fails explicitly rather than silently
    disappearing from the evaluation. Exclusions distinguish origin, role and
    the excluded observation's own target year.
    """
    _validate(data, history_years)
    numeric = [f"rate_pct_lag{lag}" for lag in range(1, history_years + 1)]
    numeric += [f"log_students_lag{lag}" for lag in range(1, history_years + 1)]
    if history_years >= 2:
        numeric += ["change_pp_last", "change_pp_since_first"]
    features = numeric + list(CATEGORICAL_FEATURES)
    predictions, fold_records, audited_exclusions = [], [], []
    windows = {}
    for index in range(history_years, len(data.years)):
        windows[index] = _window(data, index, history_years, numeric)
    for fold, index in enumerate(range(history_years + 1, len(data.years)), start=1):
        target = data.years[index]
        training = pd.concat(
            [windows[earlier][0] for earlier in range(history_years, index)],
            ignore_index=True,
        )
        test = windows[index][0]
        if training.empty or test.empty:
            role = "training" if training.empty else "test"
            raise ValueError(f"no complete {role} rows for backtest origin {target}")
        train_latest = training["target_year"].max()
        if not train_latest < target:
            raise RuntimeError(
                "backtest training target must be strictly earlier than test target"
            )
        for earlier in range(history_years, index + 1):
            excluded = windows[earlier][1].copy()
            excluded["fold"] = fold
            excluded["origin_year"] = target
            excluded["role"] = "test" if earlier == index else "train"
            excluded["train_latest_year"] = train_latest
            audited_exclusions.append(excluded)
        result = test[ID_COLUMNS + ["actual", PERSISTENCE]].copy()
        result["fold"] = fold
        result["target_year"] = target
        result["train_latest_year"] = train_latest
        change = (training["actual"] - training[PERSISTENCE]).to_numpy(dtype=float)
        result[DRIFT] = result[PERSISTENCE] + float(np.mean(change))
        models = make_models(numeric, list(CATEGORICAL_FEATURES), random_state)
        for name, model in models.items():
            model.fit(training[features], change)
            result[name] = result[PERSISTENCE] + model.predict(test[features])
        predictions.append(result)
        for name in MODEL_NAMES:
            fold_records.append(
                {
                    "fold": fold,
                    "target_year": target,
                    "train_latest_year": train_latest,
                    "train_target_years": ", ".join(
                        sorted(training["target_year"].unique())
                    ),
                    "train_rows": len(training),
                    "test_rows": len(test),
                    "train_districts": training["district_code"].nunique(),
                    "test_districts": test["district_code"].nunique(),
                    "model": name,
                    "mae_pp": mae(result["actual"], result[name]),
                    "rmse_pp": rmse(result["actual"], result[name]),
                }
            )
    predicted = pd.concat(predictions, ignore_index=True)
    folds = pd.DataFrame.from_records(fold_records)
    excluded = pd.concat(audited_exclusions, ignore_index=True)
    return BacktestResult(
        predicted,
        folds,
        summarise(predicted, folds),
        errors_by_group(predicted),
        excluded,
        len(predictions),
        int(history_years),
    )
