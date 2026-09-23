"""Forecast the latest school year's attendance rate from earlier years.

Each modelling row is one district x student group.  The target is that row's
attendance rate in the latest year in the file; the inputs are the row's
rates and student counts in the earlier years plus its category and student
group.  Everything is expressed in percentage points so errors read directly
as "points of attendance".

Four forecasters are compared with district-grouped cross-validation
(``GroupKFold``), so a district's rows are never split between training and
test folds:

* ``persistence`` - the row's rate in the most recent earlier year (the
  baseline every other forecaster is judged against);
* ``persistence_plus_mean_change`` - persistence plus the average
  year-over-year change seen in the training rows (a reference point showing
  how much of any gain comes from a uniform statewide shift);
* ``ridge`` - ridge regression on standardised numeric inputs plus one-hot
  category and student group;
* ``hist_gradient_boosting`` - scikit-learn's HistGradientBoostingRegressor on
  the same inputs (one-hot category and student group).

Both fitted models learn the change from the persistence value; their
forecast is persistence plus that predicted change.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .loader import ID_COLUMNS, AttendanceData, count_col, rate_col

PERSISTENCE = "persistence"
DRIFT = "persistence_plus_mean_change"
LEARNED_MODELS = ("ridge", "hist_gradient_boosting")
MODEL_NAMES = (PERSISTENCE, DRIFT) + LEARNED_MODELS
MODEL_LABELS = {
    PERSISTENCE: "Persistence (previous year's rate)",
    DRIFT: "Persistence + mean training change",
    "ridge": "Ridge regression",
    "hist_gradient_boosting": "Histogram gradient boosting",
}
CATEGORICAL_FEATURES = ["category", "student_group"]
STATEWIDE_REASON = "statewide aggregate row"


@dataclass
class ModelFrame:
    """Rows ready for modelling plus a record of every row left out."""

    rows: pd.DataFrame
    numeric_features: list
    categorical_features: list
    target_year: str
    prior_years: list
    exclusions: pd.DataFrame

    @property
    def groups(self) -> np.ndarray:
        return self.rows["district_code"].to_numpy()

    @property
    def features(self) -> pd.DataFrame:
        return self.rows[self.numeric_features + self.categorical_features]


@dataclass
class ForecastResult:
    predictions: pd.DataFrame
    fold_metrics: pd.DataFrame
    metrics: pd.DataFrame
    by_group: pd.DataFrame
    n_splits: int


def _exclusion_reason(row: pd.Series, target_year: str, prior_years: list) -> str:
    if row["is_statewide"]:
        return STATEWIDE_REASON

    def missing(year: str) -> bool:
        return pd.isna(row[rate_col(year)]) or pd.isna(row[count_col(year)])

    if missing(target_year):
        return f"target year {target_year} suppressed or missing"
    gaps = [y for y in prior_years if missing(y)]
    if gaps:
        return "earlier year(s) suppressed or missing: " + ", ".join(gaps)
    return ""


def build_model_frame(data: AttendanceData) -> ModelFrame:
    """Select complete district rows and derive model inputs.

    Rows are excluded (and listed in ``exclusions`` with a reason) when they
    are the statewide aggregate or when any year's rate or count is missing.
    """
    if len(data.years) < 2:
        raise ValueError("forecasting needs at least two school years in the file")
    target_year = data.years[-1]
    prior_years = data.years[:-1]
    table = data.table

    reasons = table.apply(_exclusion_reason, axis=1, args=(target_year, prior_years))
    keep = reasons == ""
    exclusions = table.loc[~keep, ID_COLUMNS].copy()
    exclusions["reason"] = reasons[~keep]
    exclusions = exclusions.reset_index(drop=True)

    kept = table.loc[keep].reset_index(drop=True)
    rows = kept[ID_COLUMNS].copy()
    numeric: list = []
    for year in prior_years:
        name = f"rate_pct_{year}"
        rows[name] = 100.0 * kept[rate_col(year)]
        numeric.append(name)
    for year in prior_years:
        name = f"log_students_{year}"
        rows[name] = np.log1p(kept[count_col(year)])
        numeric.append(name)
    last, first = prior_years[-1], prior_years[0]
    if len(prior_years) >= 2:
        before_last = prior_years[-2]
        rows["change_pp_last"] = rows[f"rate_pct_{last}"] - rows[f"rate_pct_{before_last}"]
        rows["change_pp_since_first"] = rows[f"rate_pct_{last}"] - rows[f"rate_pct_{first}"]
        numeric += ["change_pp_last", "change_pp_since_first"]

    rows[PERSISTENCE] = rows[f"rate_pct_{last}"]
    rows["actual"] = 100.0 * kept[rate_col(target_year)]
    rows["students_target_year"] = kept[count_col(target_year)]

    return ModelFrame(
        rows=rows,
        numeric_features=numeric,
        categorical_features=list(CATEGORICAL_FEATURES),
        target_year=target_year,
        prior_years=list(prior_years),
        exclusions=exclusions,
    )


def make_models(numeric: list, categorical: list, random_state: int = 0) -> dict:
    """Unfitted pipelines for the two learned forecasters."""

    def prep(scale: bool) -> ColumnTransformer:
        return ColumnTransformer(
            [
                ("num", StandardScaler() if scale else "passthrough", numeric),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
            ]
        )

    return {
        "ridge": Pipeline([("prep", prep(True)), ("model", Ridge(alpha=1.0))]),
        "hist_gradient_boosting": Pipeline(
            [
                ("prep", prep(False)),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        learning_rate=0.05,
                        max_iter=200,
                        max_leaf_nodes=15,
                        min_samples_leaf=20,
                        l2_regularization=1.0,
                        early_stopping=False,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def district_folds(groups, n_splits: int = 5) -> list:
    """GroupKFold splits keyed by district; raises if any district leaks."""
    groups = np.asarray(groups)
    n_districts = len(np.unique(groups))
    if n_splits < 2:
        raise ValueError("need at least 2 folds")
    if n_splits > n_districts:
        raise ValueError(f"{n_splits} folds requested but only {n_districts} districts available")
    folds = []
    for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(np.zeros(len(groups)), groups=groups):
        shared = set(groups[train_idx]) & set(groups[test_idx])
        if shared:
            raise RuntimeError(f"district(s) in both train and test: {sorted(shared)[:5]}")
        folds.append((train_idx, test_idx))
    return folds


def mae(actual, predicted) -> float:
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    return float(np.mean(np.abs(a - p)))


def rmse(actual, predicted) -> float:
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    return float(np.sqrt(np.mean((a - p) ** 2)))


def persistence_forecast(frame: ModelFrame) -> np.ndarray:
    """Previous year's rate (percentage points) for every modelling row."""
    return frame.rows[f"rate_pct_{frame.prior_years[-1]}"].to_numpy(dtype=float)


def cross_validate(frame: ModelFrame, n_splits: int = 5, random_state: int = 0) -> ForecastResult:
    """Out-of-fold forecasts and error summaries for all three forecasters."""
    if len(frame.rows) == 0:
        raise ValueError("no complete rows available for forecasting")
    X = frame.features
    baseline = persistence_forecast(frame)
    actual = frame.rows["actual"].to_numpy(dtype=float)
    change = actual - baseline

    predictions = frame.rows[ID_COLUMNS].copy()
    predictions["fold"] = 0
    predictions["actual"] = actual
    predictions[PERSISTENCE] = baseline
    for name in MODEL_NAMES[1:]:
        predictions[name] = np.nan

    fold_records = []
    folds = district_folds(frame.groups, n_splits)
    for fold_no, (train_idx, test_idx) in enumerate(folds, start=1):
        predictions.loc[test_idx, "fold"] = fold_no
        predictions.loc[test_idx, DRIFT] = baseline[test_idx] + float(np.mean(change[train_idx]))
        models = make_models(frame.numeric_features, frame.categorical_features, random_state)
        for name, model in models.items():
            model.fit(X.iloc[train_idx], change[train_idx])
            predictions.loc[test_idx, name] = baseline[test_idx] + model.predict(X.iloc[test_idx])
        for name in MODEL_NAMES:
            fold_records.append(
                {
                    "fold": fold_no,
                    "model": name,
                    "test_rows": len(test_idx),
                    "test_districts": len(np.unique(frame.groups[test_idx])),
                    "train_districts": len(np.unique(frame.groups[train_idx])),
                    "mae_pp": mae(actual[test_idx], predictions.loc[test_idx, name]),
                    "rmse_pp": rmse(actual[test_idx], predictions.loc[test_idx, name]),
                }
            )
    fold_metrics = pd.DataFrame.from_records(fold_records)
    metrics = summarise(predictions, fold_metrics)
    by_group = errors_by_group(predictions)
    return ForecastResult(predictions, fold_metrics, metrics, by_group, n_splits)


def summarise(predictions: pd.DataFrame, fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Pooled out-of-fold MAE/RMSE per forecaster, relative to persistence."""
    actual = predictions["actual"]
    base_mae = mae(actual, predictions[PERSISTENCE])
    base_rmse = rmse(actual, predictions[PERSISTENCE])
    base_folds = fold_metrics[fold_metrics["model"] == PERSISTENCE].set_index("fold")["mae_pp"]
    records = []
    for name in MODEL_NAMES:
        per_fold = fold_metrics[fold_metrics["model"] == name].set_index("fold")["mae_pp"]
        m = mae(actual, predictions[name])
        r = rmse(actual, predictions[name])
        records.append(
            {
                "model": name,
                "rows": len(predictions),
                "mae_pp": m,
                "rmse_pp": r,
                "mae_fold_sd_pp": float(per_fold.std(ddof=1)) if len(per_fold) > 1 else np.nan,
                "mae_minus_persistence_pp": m - base_mae,
                "rmse_minus_persistence_pp": r - base_rmse,
                "folds_with_lower_mae_than_persistence": int((per_fold < base_folds).sum()),
            }
        )
    return pd.DataFrame.from_records(records)


def errors_by_group(predictions: pd.DataFrame) -> pd.DataFrame:
    """MAE per student group for each forecaster."""
    records = []
    for (category, group), part in predictions.groupby(["category", "student_group"], sort=False):
        record = {"category": category, "student_group": group, "rows": len(part)}
        for name in MODEL_NAMES:
            record[f"mae_pp_{name}"] = mae(part["actual"], part[name])
        records.append(record)
    return pd.DataFrame.from_records(records)


def comparison_sentences(metrics: pd.DataFrame, n_splits: int) -> list:
    """Plain statements of how each forecaster did against persistence."""
    table = metrics.set_index("model")
    base = table.loc[PERSISTENCE]
    lines = []
    for name in MODEL_NAMES[1:]:
        row = table.loc[name]
        label = MODEL_LABELS[name]
        numbers = (
            f"MAE {row['mae_pp']:.3f} vs {base['mae_pp']:.3f} pp, "
            f"RMSE {row['rmse_pp']:.3f} vs {base['rmse_pp']:.3f} pp"
        )
        better_mae = row["mae_pp"] < base["mae_pp"]
        better_rmse = row["rmse_pp"] < base["rmse_pp"]
        if better_mae and better_rmse:
            verdict = "beat the persistence baseline on both MAE and RMSE"
        elif not better_mae and not better_rmse:
            verdict = "did NOT beat the persistence baseline"
        elif better_mae:
            verdict = "had lower MAE but higher RMSE than the persistence baseline (mixed result)"
        else:
            verdict = "had lower RMSE but higher MAE than the persistence baseline (mixed result)"
        folds = int(row["folds_with_lower_mae_than_persistence"])
        lines.append(f"{label} {verdict} ({numbers}; lower MAE in {folds} of {n_splits} folds).")
    drift = table.loc[DRIFT]
    for name in LEARNED_MODELS:
        diff = table.loc[name, "mae_pp"] - drift["mae_pp"]
        direction = "lower" if diff < 0 else "higher"
        lines.append(
            f"{MODEL_LABELS[name]} MAE is {abs(diff):.3f} pp {direction} than "
            f"persistence + mean training change ({drift['mae_pp']:.3f} pp)."
        )
    return lines
