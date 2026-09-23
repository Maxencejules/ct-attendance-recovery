# Data

## `school_attendance_by_student_group_and_district_2022-2023.csv`

| Field | Value |
| --- | --- |
| Title | School Attendance by Student Group and District, 2022-2023 |
| Publisher | Connecticut State Department of Education (CSDE) |
| Portal | Connecticut Open Data, data.ct.gov |
| Dataset ID | `he4h-bgqh` |
| Dataset page | https://data.ct.gov/d/he4h-bgqh |
| Download URL | https://data.ct.gov/api/views/he4h-bgqh/rows.csv |
| Retrieved | 2026-09-23 |
| License | Public Domain (the `license` field in the portal metadata returned by https://api.us.socrata.com/api/catalog/v1?ids=he4h-bgqh) |
| Portal "data updated" timestamp | 2023-06-16T17:18:43Z |
| SHA256 | `d6858fe016fb8100f197f8a7c56ed7d4e7d232a96e63707665e709378708c10a` |
| Size | 276,489 bytes; 2,031 data rows plus a header row |

The file is committed exactly as downloaded (the SHA256 above is of the
committed file).

**Suggested citation:** Connecticut State Department of Education. *School
Attendance by Student Group and District, 2022-2023* (dataset he4h-bgqh).
Connecticut Open Data, https://data.ct.gov/d/he4h-bgqh. Retrieved 2026-09-23.
Public Domain.

### Why this year

On 2026-09-23 the catalog query
`https://api.us.socrata.com/api/catalog/v1?domains=data.ct.gov&q=School%20Attendance%20by%20Student%20Group`
listed three datasets with this title: 2020-2021 (`226r-dnit`), 2021-2022
(`t4hx-jd4c`) and 2022-2023 (`he4h-bgqh`). No newer year was listed, so the
2022-2023 file is used. It carries rates for 2019-2020, 2020-2021, 2021-2022
and 2022-2023.

### Contents

One row per district and student group. Columns:

- `District code`, `District name` - `00000CT` / `Connecticut` is the
  statewide aggregate row.
- `Category`, `Student group` - 13 student groups in 6 named categories;
  the "All Students" row has a blank category. Groups inside a category can overlap,
  e.g. "Free/Reduced Price Meal Eligible" contains "Free Meal Eligible" and
  "Reduced Price Meal Eligible".
- `<year> student count` and `<year> attendance rate` for each school year.
  Rates are fractions (0.9232 = 92.32%). The 2022-2023 columns are labelled
  "year to date"; the `Reporting period` column is "May 2023" on every row.
- `Reporting period`, `Update date`.

### Suppression

The portal description says that an empty cell means the value was
suppressed to protect student confidentiality or because the group is too
small to report reliably. In this file every suppressed value is a blank
cell (344 blank numeric cells; the 2022-2023 columns have none). The loader
also treats `*` and any other non-numeric text as suppressed.
