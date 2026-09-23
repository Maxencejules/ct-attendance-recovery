import io
import math
import unittest

import helpers  # noqa: F401  (puts src/ on sys.path)
from helpers import TINY_CSV

from attendance_recovery.loader import count_col, load_attendance, rate_col


def _row(data, code, group):
    t = data.table
    match = t[(t["district_code"] == code) & (t["student_group"] == group)]
    assert len(match) == 1, (code, group)
    return match.iloc[0]


class ParsingTests(unittest.TestCase):
    def setUp(self):
        self.data = load_attendance(io.StringIO(TINY_CSV))

    def test_years_found_and_sorted(self):
        self.assertEqual(self.data.years, ["2019-2020", "2020-2021", "2021-2022"])
        self.assertEqual(self.data.latest_year, "2021-2022")
        self.assertEqual(self.data.default_baseline(), "2019-2020")

    def test_district_codes_keep_leading_zeros(self):
        self.assertIn("0010011", set(self.data.table["district_code"]))

    def test_statewide_row_is_flagged(self):
        flagged = self.data.table[self.data.table["is_statewide"]]
        self.assertEqual(list(flagged["district_code"]), ["00000CT"])
        self.assertEqual(self.data.report.statewide_rows, 1)
        self.assertEqual(self.data.report.district_rows, 5)
        self.assertEqual(len(self.data.district_table()), 5)

    def test_percent_rates_are_converted_to_fractions(self):
        self.assertEqual(set(self.data.report.rate_scale.values()), {"percent"})
        self.assertAlmostEqual(_row(self.data, "0010011", "All Students")[rate_col("2021-2022")], 0.90)

    def test_blank_category_becomes_all_students(self):
        self.assertEqual(_row(self.data, "0020011", "All Students")["category"], "All Students")

    def test_suppressed_and_blank_cells_become_nan_and_are_counted(self):
        alpha_el = _row(self.data, "0010011", "English Learners")
        self.assertTrue(math.isnan(alpha_el[rate_col("2021-2022")]))
        self.assertTrue(math.isnan(alpha_el[count_col("2021-2022")]))
        beta_el = _row(self.data, "0020011", "English Learners")
        self.assertTrue(math.isnan(beta_el[rate_col("2020-2021")]))
        report = self.data.report
        self.assertEqual(report.missing_tokens["*"], 3)
        self.assertEqual(report.missing_tokens["<blank>"], 4)
        self.assertEqual(report.missing_cells["2020-2021 attendance rate"], 2)

    def test_count_without_rate_is_cleared(self):
        gamma = _row(self.data, "0030011", "All Students")
        self.assertTrue(math.isnan(gamma[count_col("2020-2021")]))
        self.assertEqual(self.data.report.unpaired_values_cleared, 1)

    def test_rows_missing_per_year(self):
        self.assertEqual(
            self.data.report.rows_missing_year,
            {"2019-2020": 1, "2020-2021": 2, "2021-2022": 1},
        )

    def test_long_format(self):
        long = self.data.long()
        self.assertEqual(len(long), 6 * 3)
        self.assertEqual(list(long["year"].cat.categories), self.data.years)


class FormatVariantTests(unittest.TestCase):
    def test_fraction_scale_short_years_and_thousands_separator(self):
        text = (
            "District Code,District Name,Category,Student Group,"
            "2022-23 student count - year to date,2022-23 attendance rate - year to date,"
            "2021-22 student count,2021-22 attendance rate,Reporting period\n"
            '0010011,Alpha,,All Students,"1,200",0.9312,1150,0.9105,May 2023\n'
            "0010011,Alpha,English Learners,English Learners,N/A,0.88,40,-0.5,May 2023\n"
        )
        data = load_attendance(io.StringIO(text))
        self.assertEqual(data.years, ["2021-2022", "2022-2023"])
        self.assertEqual(set(data.report.rate_scale.values()), {"fraction"})
        alpha = data.table.iloc[0]
        self.assertEqual(alpha[count_col("2022-2023")], 1200.0)
        self.assertAlmostEqual(alpha[rate_col("2022-2023")], 0.9312)
        # 'N/A' count clears its rate; a negative rate is out of range.
        el = data.table.iloc[1]
        self.assertTrue(math.isnan(el[rate_col("2022-2023")]))
        self.assertTrue(math.isnan(el[rate_col("2021-2022")]))
        self.assertEqual(data.report.out_of_range_cells, 1)
        self.assertEqual(data.report.missing_tokens["N/A"], 1)
        self.assertEqual(data.report.reporting_periods, ["May 2023"])

    def test_missing_required_column_raises(self):
        with self.assertRaises(ValueError):
            load_attendance(io.StringIO("District name,Category\nA,B\n"))

    def test_no_year_columns_raises(self):
        with self.assertRaises(ValueError):
            load_attendance(io.StringIO("District code,Student group\n0010011,All Students\n"))

    def test_duplicate_rows_are_dropped(self):
        lines = TINY_CSV.splitlines()
        text = "\n".join(lines + [lines[2]]) + "\n"
        data = load_attendance(io.StringIO(text))
        self.assertEqual(data.report.duplicate_rows_dropped, 1)
        self.assertEqual(len(data.table), 6)


if __name__ == "__main__":
    unittest.main()
