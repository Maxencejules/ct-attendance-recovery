# ct-attendance-recovery

A small Python command-line tool for the Connecticut State Department of
Education's public "School Attendance by Student Group and District" data.
It provides three distinct evaluations:

1. **`analyze`**: computes a student-weighted attendance rate for each
   student group in each school year, combining district rows. It then
   measures how far each group's latest rate is from its 2019-2020 rate.
2. **`forecast`**: predicts each district x student-group row's attendance
   rate in the latest year from the same row's earlier years. It compares
   ridge regression and histogram gradient boosting with a persistence
   baseline (last year's rate) using cross-validation grouped by district.
   This is retrospective same-year validation: the fold's training districts
   already have observed outcomes in the target year.
3. **`backtest`**: holds out each eligible later year, fitting only strictly
   earlier target years with a fixed history window. It evaluates transfer
   across time for observed districts; it does not use the held-out year's
   rates or student counts to produce its predictions.

The data file, `data/school_attendance_by_student_group_and_district_2022-2023.csv`
(dataset `he4h-bgqh`, Public Domain), is included. See [Data](#data).

## Requirements

- Python 3.12. Tested with 3.12.3.
- numpy, pandas and scikit-learn. `requirements.txt` sets each minimum to the
  version the code was tested with: numpy 1.26.4, pandas 2.2.3 and
  scikit-learn 1.5.2.

Install the dependencies in whatever environment you use, for example:

```
python -m pip install -r requirements-lock.txt
```

The complete Python 3.12 verification environment is pinned in
`requirements-lock.txt`; `requirements.txt` retains library minimums.
The package is not installed itself. Run it from the
repository root with `src` on `PYTHONPATH`.

## Usage

```
PYTHONPATH=src python -m attendance_recovery analyze \
    --data data/school_attendance_by_student_group_and_district_2022-2023.csv --out outputs/

PYTHONPATH=src python -m attendance_recovery forecast \
    --data data/school_attendance_by_student_group_and_district_2022-2023.csv --out outputs/

PYTHONPATH=src python -m attendance_recovery backtest \
    --data data/school_attendance_by_student_group_and_district_2022-2023.csv \
    --history-years 2 --rate-unit fraction --out outputs/backtest/
```

Options:

- `analyze --baseline YYYY-YYYY` sets the baseline year. The default is
  2019-2020 when the file has it, otherwise the oldest year. The baseline
  must be earlier than the latest year in the file; otherwise the command
  prints an `error:` line and exits with status 2 without writing outputs.
- `--rate-unit auto|percent|fraction` applies to all commands and controls
  unmarked rate cells. Explicit `%` cells always use percent. Auto retains
  column inference using valid candidates; outliers above 100 cannot rescale
  other rows. Mixed unmarked values above and below 1 fail until a unit is
  selected. An unmarked `0.5` is ambiguous: use `percent` for 0.5% or
  `fraction` for 50%. Use `fraction` for the included CSDE snapshot.
- `backtest --history-years N` uses the same N-year lag feature layout at
  every origin. At least N+2 consecutive school years are required. The
  default N=2 gives one evaluated origin in the included four-year file;
  N=1 gives two. Empty origins fail explicitly before output creation.
- `forecast --folds N` sets the number of GroupKFold folds (default 5).
  `forecast --seed N` is passed as `random_state` to the gradient-boosting
  model (default 0). With the fixed settings used here (no early stopping,
  no feature subsampling, and fewer rows than the 200,000 at which
  scikit-learn subsamples for binning) that model has no random step, so
  its results do not depend on `--seed`. On the included data, runs with
  `--seed 0` and `--seed 7` wrote identical `forecast_metrics.csv` files.

Both commands print a short summary. They write these files to `--out`
(`outputs/` is git-ignored):

| File | Contents |
| --- | --- |
| `trajectories.csv` | For each group and year: districts reporting, students, student-weighted rate (%), and the rate on the file's statewide row, for comparison |
| `recovery_gaps.csv` | For each group and each year after the baseline: matched district rows, weighted baseline and year rates, gap (percentage points), median district gap, rows at or above the baseline |
| `attendance_summary.md` | How the data was handled, the rate table, the gap table, and the latest year's gaps sorted |
| `forecast_metrics.csv` | Pooled out-of-fold MAE and RMSE for each forecaster, and the difference from persistence |
| `forecast_fold_metrics.csv` | MAE and RMSE per fold, with fold sizes |
| `forecast_predictions.csv` | Out-of-fold prediction from every forecaster for every modelled row, with its fold number |
| `forecast_by_group.csv` | MAE for each student group and forecaster |
| `forecast_excluded_rows.csv` | Every row left out of the forecast and the reason |
| `forecast_summary.md` | The forecast setup, exclusions, the metrics tables and a plain comparison with persistence |
| `backtest_metrics.csv`, `backtest_fold_metrics.csv` | Pooled errors and each origin/model's training years, row counts and errors |
| `backtest_predictions.csv`, `backtest_by_group.csv` | Time-held-out predictions with training cutoffs and group errors |
| `backtest_excluded_rows.csv`, `backtest_summary.md` | Exclusions by origin and train/test role, evaluation method and limits |

## Method

**Parsing** (`loader.py`)
- The loader finds the school-year columns by header (`<yyyy>-<yyyy> student
  count` / `attendance rate`), so the files for other years parse too.
- District codes are read as text, which keeps their leading zeros.
- Rates given as percentages are converted to fractions, preserving explicit
  percent markers even when a column contains only small percentages.
- Duplicate canonical headers, missing count/rate column pairs, malformed or
  nonconsecutive years, blank district/group IDs and conflicting observations
  fail explicitly. Only identical normalized observations are deduplicated;
  differing reporting periods are conflicts.
- Blank cells, `*` and any other non-numeric text count as suppressed. Each
  one is tallied by token.
- If only one of a year's count and rate is present, both are cleared.
  Out-of-range values are cleared too.
- Negative or fractional counts are invalid. For backtest scoring, an observed
  target rate is retained without its target count; lagged counts remain required.
- The statewide row (district code `00000CT`) is flagged.

**Descriptive analysis** (`trends.py`)
- A group's rate for a year is `sum(rate x students) / sum(students)` over
  the district rows that report that year. The statewide row is left out of
  this sum. It is kept only as a cross-check column.
- A recovery gap is computed on *matched* rows only: district rows that
  report both the baseline year and the year being compared. That way,
  districts moving in and out of suppression do not shift the gap.
- The median of the district-level gaps is reported alongside it.
- Category-level totals are not computed, because groups within a category
  overlap.

**Forecast** (`forecasting.py`)
- Target: the latest year's rate, in percentage points.
- Inputs: each earlier year's rate and log student count, the last one-year
  change, the change since the first year, and one-hot category and student
  group. The latest year's own student count is not used.
- Excluded rows: the statewide row, and any row with a suppressed or missing
  rate or count in any year. There is no imputation.
- Forecasters:
  - `persistence`: the previous year's rate.
  - `persistence_plus_mean_change`: persistence plus the mean change in the
    fold's training rows. This is a reference that shows how much a uniform
    shift explains.
  - `ridge`: `Ridge(alpha=1.0)` on standardised numeric inputs.
  - `hist_gradient_boosting`: `HistGradientBoostingRegressor` with fixed
    settings (learning rate 0.05, 200 iterations, 15 leaves, at least 20
    samples per leaf, L2 1.0, no early stopping).
- Both learned models are fitted to the change from the previous year. Their
  forecast is persistence plus the predicted change. Hyperparameters are
  fixed, not tuned.
- Validation: `GroupKFold` keyed on district code. The code checks every fold
  and raises an error if any district appears in both train and test. MAE
  and RMSE are pooled over all out-of-fold predictions.

**Time-held-out evaluation** (`backtesting.py`)
- Each test target is scored once. Training accumulates only windows whose
  target year is strictly earlier, and preprocessing is refitted on those rows.
- Generic lag features keep the same meaning across origins; the four existing
  forecasters and fixed hyperparameters are reused. The drift reference is
  learned from past changes, so a shock in the held-out year cannot be learned
  from that year's labels.
- Same districts may recur across years. This is a different question from
  same-year validation on unseen districts. Neither mode establishes causal
  explanations or performance in an unobserved future year.
- Errors weight district/group rows equally. Student groups overlap and
  repeated yearly observations are dependent; no confidence interval is implied.

## Results on the included data

The output below is copied from a single run of the two commands in
[Usage](#usage) on 2026-09-23, with Python 3.12.3, numpy 1.26.4,
pandas 2.2.3 and scikit-learn 1.5.2 (that interpreter was used as
`python`).

`analyze`:

```
Read 2031 rows from data/school_attendance_by_student_group_and_district_2022-2023.csv (2018 district rows, 13 statewide rows excluded from averages/models).
School years: 2019-2020, 2020-2021, 2021-2022, 2022-2023 (latest reporting period: May 2023)
Rows with a suppressed or missing rate, by year: 2019-2020: 76, 2020-2021: 61, 2021-2022: 35, 2022-2023: 0
All Students, student-weighted: 2019-2020 94.78%, 2020-2021 93.12%, 2021-2022 92.08%, 2022-2023 92.32%
2022-2023 vs 2019-2020: 13 of 13 groups below baseline; largest gap -4.69 pp (Students Experiencing Homelessness), smallest -1.88 pp (White)
Wrote outputs/trajectories.csv, outputs/recovery_gaps.csv, outputs/attendance_summary.md
```

This table is from `outputs/attendance_summary.md` of the same run. It
covers matched district rows, 2022-2023 against 2019-2020:

| group | matched district rows | 2019-2020 % | 2022-2023 % | gap (pp) | median district gap (pp) | rows at/above baseline |
| --- | --- | --- | --- | --- | --- | --- |
| Homelessness / Students Experiencing Homelessness | 19 | 88.46 | 83.77 | -4.69 | -5.71 | 1 |
| Free/Reduced Lunch / Free Meal Eligible | 149 | 93.11 | 89.68 | -3.42 | -2.70 | 4 |
| Free/Reduced Lunch / Free/Reduced Price Meal Eligible | 191 | 93.39 | 90.16 | -3.24 | -2.54 | 3 |
| Race/Ethnicity / Black or African American | 120 | 94.00 | 90.89 | -3.11 | -2.26 | 2 |
| Race/Ethnicity / Hispanic/Latino of any race | 152 | 93.61 | 90.57 | -3.04 | -2.47 | 2 |
| High Needs / Students With High Needs | 194 | 93.61 | 90.59 | -3.02 | -2.31 | 4 |
| Students With Disabilities | 182 | 92.75 | 89.92 | -2.83 | -2.28 | 9 |
| English Learners | 111 | 93.89 | 91.15 | -2.74 | -2.12 | 3 |
| Free/Reduced Lunch / Reduced Price Meal Eligible | 142 | 95.18 | 92.64 | -2.55 | -2.11 | 7 |
| All Students | 199 | 94.78 | 92.32 | -2.46 | -2.11 | 2 |
| Race/Ethnicity / All other races | 111 | 95.62 | 93.43 | -2.20 | -1.91 | 5 |
| High Needs / Students Without High Needs | 193 | 96.07 | 94.14 | -1.93 | -1.78 | 2 |
| Race/Ethnicity / White | 179 | 95.43 | 93.55 | -1.88 | -1.77 | 3 |

Cross-check from `outputs/trajectories.csv`: the district-weighted
All Students rate for 2019-2020 is 94.7830%. The file's statewide row gives
94.7900%.

`forecast`:

```
Read 2031 rows from data/school_attendance_by_student_group_and_district_2022-2023.csv (2018 district rows, 13 statewide rows excluded from averages/models).
School years: 2019-2020, 2020-2021, 2021-2022, 2022-2023 (latest reporting period: May 2023)
Target: 2022-2023 rate; inputs from 2019-2020, 2020-2021, 2021-2022
Rows used: 1917 from 199 districts; excluded: 114
     29  earlier year(s) suppressed or missing: 2019-2020, 2020-2021
     26  earlier year(s) suppressed or missing: 2019-2020
     17  earlier year(s) suppressed or missing: 2019-2020, 2020-2021, 2021-2022
     13  statewide aggregate row
     11  earlier year(s) suppressed or missing: 2020-2021
     10  earlier year(s) suppressed or missing: 2021-2022
      4  earlier year(s) suppressed or missing: 2020-2021, 2021-2022
      4  earlier year(s) suppressed or missing: 2019-2020, 2021-2022
GroupKFold by district, 5 folds (no district in both train and test)
forecaster                               MAE pp  RMSE pp
Persistence (previous year's rate)        1.003    1.406
Persistence + mean training change        1.039    1.410
Ridge regression                          0.765    1.048
Histogram gradient boosting               0.813    1.154
Persistence + mean training change did NOT beat the persistence baseline (MAE 1.039 vs 1.003 pp, RMSE 1.410 vs 1.406 pp; lower MAE in 1 of 5 folds).
Ridge regression beat the persistence baseline on both MAE and RMSE (MAE 0.765 vs 1.003 pp, RMSE 1.048 vs 1.406 pp; lower MAE in 5 of 5 folds).
Histogram gradient boosting beat the persistence baseline on both MAE and RMSE (MAE 0.813 vs 1.003 pp, RMSE 1.154 vs 1.406 pp; lower MAE in 4 of 5 folds).
Ridge regression MAE is 0.274 pp lower than persistence + mean training change (1.039 pp).
Histogram gradient boosting MAE is 0.226 pp lower than persistence + mean training change (1.039 pp).
Wrote forecast_*.csv and forecast_summary.md to outputs
```

Per-group errors are in `outputs/forecast_summary.md` from the same run.
Both learned models did worse than persistence on one group: Students
Experiencing Homelessness, 19 rows. Its MAE was 2.062 pp for persistence,
2.133 pp for ridge and 2.314 pp for gradient boosting.

## Data

| Field | Value |
| --- | --- |
| Title | School Attendance by Student Group and District, 2022-2023 |
| Publisher | Connecticut State Department of Education, via Connecticut Open Data (data.ct.gov) |
| Dataset ID | `he4h-bgqh` |
| Source URL | https://data.ct.gov/api/views/he4h-bgqh/rows.csv (dataset page https://data.ct.gov/d/he4h-bgqh) |
| Retrieved | 2026-09-23 |
| License | Public Domain, per the portal metadata (https://api.us.socrata.com/api/catalog/v1?ids=he4h-bgqh) |
| SHA256 | `d6858fe016fb8100f197f8a7c56ed7d4e7d232a96e63707665e709378708c10a` |

Citation: Connecticut State Department of Education. *School Attendance by
Student Group and District, 2022-2023* (dataset he4h-bgqh). Connecticut Open
Data, https://data.ct.gov/d/he4h-bgqh. Retrieved 2026-09-23. Public Domain.

On the retrieval date, no newer year of this dataset was listed on the
portal. `data/README.md` has the catalog query, a description of the columns
and the suppression rules.

## Data notes and limitations

- **Aggregate data.** The data is district-level only. It says nothing about
  individual students or schools, and the tool does not try to explain *why*
  attendance changed.
- **Suppression.** CSDE suppresses small or sensitive cells, and in this file
  they are blank. Small groups are therefore thin:
  - Students Experiencing Homelessness has 19 matched district rows in the
    latest-year gap.
  - The forecast drops 114 of the 2031 rows: 101 district rows with a
    suppressed year, plus the 13 statewide rows.
  - Rows that survive suppression may not be typical of the rows that were
    suppressed.
- **Reporting periods.** Only the 2022-2023 columns are labelled "year to
  date", with a reporting period of May 2023. How each year's rate was
  defined is up to the publisher. The 2019-2020 and 2020-2021 school years
  were disrupted by COVID-19, so comparisons against 2019-2020 inherit
  whatever that year's measurement captured.
- **Differences from the statewide row.** District-weighted averages differ
  slightly from the file's statewide row (see the cross-check above). Both
  are shown in `trajectories.csv`.
- **One target year.** The forecast was evaluated on a single target year,
  2022-2023, with district-grouped folds. The result shows how well the
  models generalise to unseen districts in that year. It does not show how
  they would do in a different year.

## Tests

```
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests
```

`tests/helpers.py` puts `src/` on `sys.path`, so the tests do not need
`PYTHONPATH`. The tests use a small inline CSV and a synthetic CSV built at
run time. They do not use the network. They cover:

- parsing, and the handling of blank and `*` cells;
- weighted averages and gaps, checked against hand-computed values;
- that GroupKFold never puts a district in both train and test;
- the persistence baseline and the exclusion reasons;
- end-to-end runs of `analyze` and `forecast`, both in-process and through
  `python -m attendance_recovery`.

Result on 2026-09-23, with the same interpreter and library versions as
above: `Ran 43 tests ... OK`.

The expanded suite also tests canonical-header and duplicate integrity,
explicit rate units, future/current-label perturbations, a held-out 20-point
shock, hand-calculated temporal errors and end-to-end target-count independence.
Native CI runs the complete suite on Ubuntu and Windows, followed by all three
commands on the unchanged public snapshot. Reproduce the evidence locally:

```
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python ci/prove_attendance.py
```

`outputs/ci-proof/proof.json` records dataset/source/output SHA256 values,
runtime versions, measured metrics, explicit training cutoffs and an identical
CSV rerun check. CI uploads the reports and proof. Historical results above
remain same-year district results; compare them separately with time-held-out
metrics, which can be worse than persistence.

On the unchanged snapshot with the locked Python 3.12 environment, the
two-year-history backtest trains on target 2021-2022 and evaluates target
2022-2023: 1,943 scored rows, with all training cutoffs strictly earlier.
The latest count is not required for scoring. A second run produces identical
CSV files. Measured percentage-point errors are:

| Forecaster | Time-held-out MAE | Time-held-out RMSE |
| --- | ---: | ---: |
| Persistence | 1.0118 | 1.4214 |
| Persistence + mean historical change | 1.5135 | 1.9570 |
| Ridge | 1.6593 | 2.2411 |
| Histogram gradient boosting | 1.7015 | 2.1524 |

Persistence wins this time holdout. The same-year ridge advantage above
cannot establish a future-year advantage. One origin supplies no meaningful
cross-origin standard deviation; the metrics CSV leaves it empty. Native CI
recomputes the evidence rather than trusting this table.

## Layout

```
data/                          source CSV and its README
src/attendance_recovery/
    __main__.py, cli.py        command-line interface
    loader.py                  CSV parsing and suppression handling
    trends.py                  weighted trajectories and recovery gaps
    forecasting.py             model frame, GroupKFold evaluation, metrics
    reports.py, markdown.py    Markdown summaries
tests/                         unittest suite and fixtures
```
