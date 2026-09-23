"""Student-weighted attendance trajectories and gaps against a baseline year.

All averages here combine district rows (never the statewide row) and weight
each district's rate by that district's student count for the same group and
year:  ``sum(rate * count) / sum(count)``.  The statewide row published in the
file is kept only as a cross-check column.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .loader import AttendanceData, count_col, rate_col


def weighted_mean(values, weights) -> float:
    """Weighted mean that skips pairs with a missing value or a non-positive weight.

    Returns NaN when no usable pair remains.
    """
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    usable = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not usable.any():
        return float("nan")
    return float(np.sum(v[usable] * w[usable]) / np.sum(w[usable]))


def _group_order(table: pd.DataFrame) -> list:
    """(category, student_group) pairs in the order they first appear."""
    pairs = table[["category", "student_group"]].drop_duplicates()
    return list(pairs.itertuples(index=False, name=None))


def group_trajectories(data: AttendanceData) -> pd.DataFrame:
    """Weighted attendance rate for every student group and school year.

    Columns: category, student_group, year, districts_reporting, students,
    weighted_rate_pct, state_reported_pct.
    """
    districts = data.district_table()
    statewide = data.table[data.table["is_statewide"]]
    records = []
    for category, group in _group_order(data.table):
        rows = districts[(districts["category"] == category) & (districts["student_group"] == group)]
        state = statewide[(statewide["category"] == category) & (statewide["student_group"] == group)]
        for year in data.years:
            rates = rows[rate_col(year)]
            counts = rows[count_col(year)]
            usable = rates.notna() & counts.notna() & (counts > 0)
            state_rate = state[rate_col(year)].iloc[0] if len(state) else np.nan
            records.append(
                {
                    "category": category,
                    "student_group": group,
                    "year": year,
                    "districts_reporting": int(usable.sum()),
                    "students": float(counts[usable].sum()),
                    "weighted_rate_pct": 100.0 * weighted_mean(rates, counts),
                    "state_reported_pct": 100.0 * state_rate if pd.notna(state_rate) else np.nan,
                }
            )
    return pd.DataFrame.from_records(records)


def recovery_gaps(data: AttendanceData, baseline: str | None = None) -> pd.DataFrame:
    """Change in weighted attendance versus ``baseline`` for each later year.

    For a given group and year only district rows that report both the
    baseline year and that year are used ("matched" rows), so the gap is not
    driven by districts entering or leaving the average.

    Columns: category, student_group, baseline_year, year, matched_rows,
    baseline_rate_pct, rate_pct, gap_pp, median_district_gap_pp,
    rows_at_or_above_baseline.

    Raises ValueError if ``baseline`` is not in the data or is the latest year
    (there would be no later year to compare it with).
    """
    baseline = baseline or data.default_baseline()
    if baseline not in data.years:
        raise ValueError(f"baseline year {baseline!r} not in data (years: {', '.join(data.years)})")
    later = data.years[data.years.index(baseline) + 1 :]
    if not later:
        raise ValueError(
            f"baseline year {baseline!r} is the latest year in the data; it must be earlier than "
            f"the latest year (years: {', '.join(data.years)})"
        )
    districts = data.district_table()
    records = []
    for category, group in _group_order(data.table):
        rows = districts[(districts["category"] == category) & (districts["student_group"] == group)]
        base_rate = rows[rate_col(baseline)].to_numpy(dtype=float)
        base_count = rows[count_col(baseline)].to_numpy(dtype=float)
        for year in later:
            rate = rows[rate_col(year)].to_numpy(dtype=float)
            count = rows[count_col(year)].to_numpy(dtype=float)
            matched = (
                np.isfinite(base_rate) & np.isfinite(base_count) & (base_count > 0)
                & np.isfinite(rate) & np.isfinite(count) & (count > 0)
            )
            base_pct = 100.0 * weighted_mean(base_rate[matched], base_count[matched])
            year_pct = 100.0 * weighted_mean(rate[matched], count[matched])
            district_gaps = 100.0 * (rate[matched] - base_rate[matched])
            records.append(
                {
                    "category": category,
                    "student_group": group,
                    "baseline_year": baseline,
                    "year": year,
                    "matched_rows": int(matched.sum()),
                    "baseline_rate_pct": base_pct,
                    "rate_pct": year_pct,
                    "gap_pp": year_pct - base_pct,
                    "median_district_gap_pp": float(np.median(district_gaps)) if matched.any() else np.nan,
                    "rows_at_or_above_baseline": int(np.sum(rate[matched] >= base_rate[matched])),
                }
            )
    return pd.DataFrame.from_records(records)
