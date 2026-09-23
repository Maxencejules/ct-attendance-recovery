"""Shared test helpers: import path setup and small CSV fixtures (no network)."""

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Percent-scale rates (as in older published files), with '*' and blank cells.
# Hand-checked values used by the tests:
#   All Students 2021-2022, districts only: (100*.90 + 300*.94 + 100*.92) / 500 = 0.928
#   All Students 2019-2020: Gamma is blank -> (100*.95 + 300*.97) / 400 = 0.965
#   Gamma 2020-2021 has a count but a '*' rate, so the pair is cleared.
TINY_CSV = """\
District code,District name,Category,Student group,2021-2022 student count - year to date,2021-2022 attendance rate - year to date,2020-2021 student count,2020-2021 attendance rate,2019-2020 student count,2019-2020 attendance rate,Reporting period,Date update
00000CT,Connecticut,,All Students,500,93.10,480,92.00,510,96.20,Jun-22,7/22/2022
0010011,Alpha District,,All Students,100,90.00,90,88.00,100,95.00,Jun-22,7/22/2022
0020011,Beta District,,All Students,300,94.00,290,93.00,300,97.00,Jun-22,7/22/2022
0030011,Gamma District,,All Students,100,92.00,100,*,,,Jun-22,7/22/2022
0010011,Alpha District,English Learners,English Learners,*,*,10,85.00,12,90.00,Jun-22,7/22/2022
0020011,Beta District,English Learners,English Learners,40,89.00,,,38,92.50,Jun-22,7/22/2022
"""

SYNTH_GROUPS = [
    ("", "All Students", 0.0, 1.0),
    ("Students With Disabilities", "Students With Disabilities", -2.5, 0.16),
    ("English Learners", "English Learners", -1.5, 0.10),
    ("Race/Ethnicity", "White", 0.8, 0.45),
    ("Race/Ethnicity", "Hispanic/Latino of any race", -1.2, 0.30),
    ("High Needs", "Students With High Needs", -1.8, 0.50),
]
SYNTH_YEARS = ["2019-2020", "2020-2021", "2021-2022", "2022-2023"]
_YEAR_SHIFT = {"2019-2020": 0.0, "2020-2021": -2.0, "2021-2022": -2.8, "2022-2023": -2.3}


def synthetic_rows(n_districts: int = 15, seed: int = 7) -> list:
    """District x group rows shaped like the published file (fraction rates)."""
    rng = random.Random(seed)
    rows = []
    for d in range(n_districts):
        code = f"{d + 1:03d}0011"
        name = f"Synthetic District {d + 1}"
        level = rng.uniform(92.5, 96.0)
        trend = rng.uniform(-0.8, 0.8)
        size = rng.randint(400, 9000)
        for category, group, offset, share in SYNTH_GROUPS:
            record = {"code": code, "name": name, "category": category, "group": group}
            for k, year in enumerate(SYNTH_YEARS):
                pct = level + offset + _YEAR_SHIFT[year] + trend * k + rng.gauss(0, 0.5)
                pct = min(max(pct, 70.0), 99.5)
                count = max(5, int(size * share * rng.uniform(0.95, 1.05)))
                record[year] = [str(count), f"{pct / 100:.4f}"]
            if rng.random() < 0.10:  # blank an earlier year, like suppressed cells
                record[rng.choice(SYNTH_YEARS[:-1])] = ["", ""]
            rows.append(record)
    rows[3]["2020-2021"] = ["*", "*"]
    rows[8]["2022-2023"] = ["*", "*"]
    return rows


def write_synthetic_csv(path: Path, n_districts: int = 15, seed: int = 7) -> Path:
    rows = synthetic_rows(n_districts, seed)
    header = ["District code", "District name", "Category", "Student group"]
    for year in reversed(SYNTH_YEARS):
        suffix = " - year to date" if year == SYNTH_YEARS[-1] else ""
        header += [f"{year} student count{suffix}", f"{year} attendance rate{suffix}"]
    header += ["Reporting period", "Update date"]

    state = []
    for category, group, _o, _s in SYNTH_GROUPS:
        members = [r for r in rows if r["group"] == group]
        record = {"code": "00000CT", "name": "Connecticut", "category": category, "group": group}
        for year in SYNTH_YEARS:
            pairs = [(int(r[year][0]), float(r[year][1])) for r in members if r[year][1] not in ("", "*")]
            total = sum(c for c, _ in pairs)
            rate = sum(c * v for c, v in pairs) / total
            record[year] = [str(total), f"{rate:.4f}"]
        state.append(record)

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for record in state + rows:
            line = [record["code"], record["name"], record["category"], record["group"]]
            for year in reversed(SYNTH_YEARS):
                line += record[year]
            line += ["May 2023", "06/16/2023"]
            writer.writerow(line)
    return path
