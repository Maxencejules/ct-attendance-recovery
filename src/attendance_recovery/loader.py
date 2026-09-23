"""Read a Connecticut "School Attendance by Student Group and District" CSV.

The published files have one row per district and student group, and one
pair of columns (student count, attendance rate) per school year, for
example ``2022-2023 attendance rate - year to date``.  This module turns such
a file into a table with a predictable column layout:

* ``district_code``, ``district_name``, ``category``, ``student_group``
* ``is_statewide`` (True for the state aggregate row, code ``00000CT``)
* ``count_<year>`` and ``rate_<year>`` for every school year found

Rates are stored as fractions between 0 and 1.  Cells that are blank, contain
a suppression marker such as ``*``, or otherwise do not parse as numbers are
stored as NaN, and every such cell is counted in a :class:`LoadReport`.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Union

import numpy as np
import pandas as pd

STATE_CODE = "00000CT"
ALL_STUDENTS_LABEL = "All Students"
ID_COLUMNS = ["district_code", "district_name", "category", "student_group"]

_HEADER_ALIASES = {
    "district code": "district_code",
    "district name": "district_name",
    "category": "category",
    "student group": "student_group",
    "reporting period": "reporting_period",
}
_YEAR_HEADER = re.compile(
    r"^\s*(\d{4})\s*-\s*(\d{4}|\d{2})\s+(student count|attendance rate)\b",
    re.IGNORECASE,
)
# A rate column whose largest value exceeds this is read as a percentage.
_PERCENT_THRESHOLD = 1.5


def count_col(year: str) -> str:
    """Name of the student-count column for a school year."""
    return f"count_{year}"


def rate_col(year: str) -> str:
    """Name of the attendance-rate column for a school year."""
    return f"rate_{year}"


@dataclass
class LoadReport:
    """What happened while reading the file."""

    source: str
    rows_read: int = 0
    statewide_rows: int = 0
    district_rows: int = 0
    duplicate_rows_dropped: int = 0
    rate_scale: dict = field(default_factory=dict)
    missing_cells: dict = field(default_factory=dict)
    missing_tokens: Counter = field(default_factory=Counter)
    out_of_range_cells: int = 0
    unpaired_values_cleared: int = 0
    rows_missing_year: dict = field(default_factory=dict)
    reporting_periods: list = field(default_factory=list)

    def token_summary(self) -> str:
        if not self.missing_tokens:
            return "none"
        parts = [f"{tok!r} x{n}" for tok, n in sorted(self.missing_tokens.items())]
        return ", ".join(parts)


@dataclass
class AttendanceData:
    """Parsed attendance table plus the list of school years it covers."""

    table: pd.DataFrame
    years: list
    report: LoadReport

    @property
    def latest_year(self) -> str:
        return self.years[-1]

    def default_baseline(self) -> str:
        """2019-2020 when present (last pre-pandemic year), else the oldest year."""
        return "2019-2020" if "2019-2020" in self.years else self.years[0]

    def district_table(self) -> pd.DataFrame:
        return self.table[~self.table["is_statewide"]]

    def long(self) -> pd.DataFrame:
        """One row per district x student group x school year."""
        frames = []
        base = self.table[ID_COLUMNS + ["is_statewide"]]
        for year in self.years:
            part = base.copy()
            part["year"] = year
            part["student_count"] = self.table[count_col(year)].to_numpy()
            part["attendance_rate"] = self.table[rate_col(year)].to_numpy()
            frames.append(part)
        out = pd.concat(frames, ignore_index=True)
        out["year"] = pd.Categorical(out["year"], categories=self.years, ordered=True)
        return out


def _normalise_year(start: str, end: str) -> str:
    if len(end) == 2:
        end = start[:2] + end
    return f"{start}-{end}"


def _find_columns(columns) -> tuple[dict, dict]:
    """Map raw headers to id fields and to (year, kind) pairs."""
    id_map: dict = {}
    year_map: dict = {}
    for raw in columns:
        key = " ".join(str(raw).strip().lower().split())
        if key in _HEADER_ALIASES:
            id_map[_HEADER_ALIASES[key]] = raw
            continue
        match = _YEAR_HEADER.match(str(raw))
        if match:
            year = _normalise_year(match.group(1), match.group(2))
            kind = "count" if match.group(3).lower() == "student count" else "rate"
            year_map[(year, kind)] = raw
    return id_map, year_map


def _to_number(raw: pd.Series, label: str, report: LoadReport) -> pd.Series:
    text = raw.fillna("").astype(str).str.strip()
    cleaned = text.str.replace(",", "", regex=False).str.rstrip("%").str.strip()
    values = pd.to_numeric(cleaned, errors="coerce").astype(float)
    values[~np.isfinite(values)] = np.nan
    bad = values.isna()
    report.missing_cells[label] = int(bad.sum())
    report.missing_tokens.update(text[bad].replace("", "<blank>").tolist())
    return values


def load_attendance(source: Union[str, Path, IO[str]]) -> AttendanceData:
    """Read and clean an attendance CSV (path or open text stream)."""
    name = str(source) if isinstance(source, (str, Path)) else getattr(source, "name", "<stream>")
    if isinstance(source, (str, Path)):
        raw = pd.read_csv(source, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    else:
        raw = pd.read_csv(source, dtype=str, keep_default_na=False)
    report = LoadReport(source=name, rows_read=len(raw))

    id_map, year_map = _find_columns(raw.columns)
    missing_ids = [c for c in ("district_code", "student_group") if c not in id_map]
    if missing_ids:
        raise ValueError(f"missing required column(s): {', '.join(missing_ids)}")
    years = sorted({y for (y, _k) in year_map if (y, "count") in year_map and (y, "rate") in year_map})
    if not years:
        raise ValueError("no '<yyyy>-<yyyy> student count' / 'attendance rate' column pairs found")

    table = pd.DataFrame(index=raw.index)
    for field_name in ID_COLUMNS:
        if field_name in id_map:
            table[field_name] = raw[id_map[field_name]].astype(str).str.strip()
        else:
            table[field_name] = ""
    table["district_code"] = table["district_code"].str.upper()
    table.loc[table["district_name"] == "", "district_name"] = table["district_code"]
    # The published files leave Category blank on the "All Students" row.
    blank_cat = table["category"] == ""
    table.loc[blank_cat, "category"] = table.loc[blank_cat, "student_group"]
    table.loc[table["category"] == "", "category"] = ALL_STUDENTS_LABEL
    table["is_statewide"] = table["district_code"] == STATE_CODE

    for year in years:
        counts = _to_number(raw[year_map[(year, "count")]], f"{year} student count", report)
        rates = _to_number(raw[year_map[(year, "rate")]], f"{year} attendance rate", report)

        finite = rates.dropna()
        if len(finite) and finite.max() > _PERCENT_THRESHOLD:
            rates = rates / 100.0
            report.rate_scale[year] = "percent"
        else:
            report.rate_scale[year] = "fraction"

        bad_rate = rates.notna() & ((rates < 0) | (rates > 1))
        bad_count = counts.notna() & (counts < 0)
        report.out_of_range_cells += int(bad_rate.sum() + bad_count.sum())
        rates[bad_rate] = np.nan
        counts[bad_count] = np.nan

        unpaired = counts.isna() ^ rates.isna()
        report.unpaired_values_cleared += int(unpaired.sum())
        counts[unpaired] = np.nan
        rates[unpaired] = np.nan

        table[count_col(year)] = counts
        table[rate_col(year)] = rates

    before = len(table)
    table = table.drop_duplicates(subset=["district_code", "category", "student_group"], keep="first")
    report.duplicate_rows_dropped = before - len(table)
    table = table.reset_index(drop=True)
    for year in years:
        report.rows_missing_year[year] = int(table[rate_col(year)].isna().sum())

    report.statewide_rows = int(table["is_statewide"].sum())
    report.district_rows = int((~table["is_statewide"]).sum())
    if "reporting_period" in id_map:
        periods = raw[id_map["reporting_period"]].astype(str).str.strip()
        report.reporting_periods = sorted(p for p in periods.unique() if p)

    return AttendanceData(table=table, years=years, report=report)
