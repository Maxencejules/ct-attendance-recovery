import io
import unittest

import helpers  # noqa: F401
from attendance_recovery.loader import load_attendance, rate_col


HEADER = 'District code,Student group,2022-2023 student count,2022-2023 attendance rate\n'


class ObservationIntegrityTests(unittest.TestCase):
    def test_small_explicit_percent_is_not_a_fraction(self):
        data = load_attendance(io.StringIO(HEADER + '001,All Students,100,1%\n'))
        self.assertAlmostEqual(data.table.iloc[0][rate_col('2022-2023')], .01)

    def test_invalid_outlier_does_not_change_valid_fraction(self):
        data = load_attendance(io.StringIO(HEADER + '001,All Students,100,.94\n002,All Students,100,940\n'))
        self.assertAlmostEqual(data.table.iloc[0][rate_col('2022-2023')], .94)
        self.assertTrue(data.table.iloc[1].isna()[rate_col('2022-2023')])

    def test_explicit_scale_resolves_ambiguous_unmarked_values(self):
        text = HEADER + '001,All Students,100,1\n'
        fraction = load_attendance(io.StringIO(text), rate_unit='fraction')
        percent = load_attendance(io.StringIO(text), rate_unit='percent')
        self.assertEqual(fraction.table.iloc[0][rate_col('2022-2023')], 1)
        self.assertEqual(percent.table.iloc[0][rate_col('2022-2023')], .01)

    def test_explicit_percent_markers_are_honoured_with_fraction_option(self):
        data = load_attendance(io.StringIO(HEADER + '001,All Students,100,1%\n'), rate_unit='fraction')
        self.assertEqual(data.table.iloc[0][rate_col('2022-2023')], .01)

    def test_fraction_option_does_not_rescale_other_rows(self):
        data = load_attendance(io.StringIO(HEADER + '001,All Students,100,.94\n002,All Students,100,94\n'), rate_unit='fraction')
        self.assertEqual(data.table.iloc[0][rate_col('2022-2023')], .94)
        self.assertTrue(data.table.iloc[1].isna()[rate_col('2022-2023')])

    def test_conflicting_duplicates_rejected_in_either_order(self):
        rows = ['001,All Students,100,.9\n', '001,All Students,100,.5\n']
        for ordered in (rows, rows[::-1]):
            with self.subTest(order=ordered), self.assertRaisesRegex(ValueError, 'conflicting'):
                load_attendance(io.StringIO(HEADER + ''.join(ordered)))

    def test_identical_normalised_observations_are_deduplicated(self):
        data = load_attendance(io.StringIO(HEADER + '001,All Students,100,.90\n001,All Students,100.0,0.9\n'))
        self.assertEqual(len(data.table), 1)
        self.assertEqual(data.report.duplicate_rows_dropped, 1)

    def test_conflicting_reporting_periods_rejected(self):
        text = HEADER.rstrip('\n') + ',Reporting period\n001,All Students,100,.9,May\n001,All Students,100,.9,June\n'
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            load_attendance(io.StringIO(text))

    def test_canonical_and_exact_duplicate_headers_rejected(self):
        for column in ('2022-23 attendance rate', '2022-2023 attendance rate'):
            text = HEADER.rstrip('\n') + ',' + column + '\n001,All Students,100,.9,.5\n'
            with self.subTest(column=column), self.assertRaisesRegex(ValueError, 'duplicate'):
                load_attendance(io.StringIO(text))

    def test_duplicate_identifier_headers_rejected(self):
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            load_attendance(io.StringIO(HEADER.rstrip('\n') + ',District Code\n001,All Students,100,.9,002\n'))

    def test_incomplete_year_pair_rejected(self):
        with self.assertRaisesRegex(ValueError, 'pair'):
            load_attendance(io.StringIO(HEADER.rstrip('\n') + ',2021-2022 student count\n001,All Students,100,.9,100\n'))

    def test_invalid_school_year_rejected(self):
        with self.assertRaisesRegex(ValueError, 'school year'):
            load_attendance(io.StringIO(HEADER.replace('2022-2023', '2022-2025') + '001,All Students,100,.9\n'))

    def test_blank_required_identity_rejected(self):
        for row in (',All Students,100,.9\n', '001,,100,.9\n'):
            with self.subTest(row=row), self.assertRaisesRegex(ValueError, 'blank'):
                load_attendance(io.StringIO(HEADER + row))

    def test_fractional_student_count_cleared(self):
        data = load_attendance(io.StringIO(HEADER + '001,All Students,2.5,.9\n'))
        self.assertTrue(data.table.iloc[0].isna()[rate_col('2022-2023')])

    def test_invalid_rate_unit_rejected(self):
        with self.assertRaisesRegex(ValueError, 'rate_unit'):
            load_attendance(io.StringIO(HEADER + '001,All Students,100,.9\n'), rate_unit='guess')

    def test_auto_rejects_ambiguous_mixed_ranges(self):
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            load_attendance(io.StringIO(HEADER + '001,All Students,100,.94\n002,All Students,100,94\n'))

    def test_low_unmarked_percentage_above_one_is_converted(self):
        data = load_attendance(io.StringIO(HEADER + '001,All Students,100,1.2\n'))
        self.assertEqual(data.table.iloc[0][rate_col('2022-2023')], .012)

    def test_extra_malformed_year_columns_are_not_silently_ignored(self):
        for year in ('2023-XX', '20233-2024'):
            text = HEADER.rstrip('\n') + f',{year} student count,{year} attendance rate\n001,All Students,100,.9,100,.8\n'
            with self.subTest(year=year), self.assertRaisesRegex(ValueError, 'school year'):
                load_attendance(io.StringIO(text))


if __name__ == '__main__':
    unittest.main()
