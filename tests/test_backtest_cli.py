import contextlib
import csv
import io
import tempfile
import unittest
from pathlib import Path

import helpers  # noqa: F401
import pandas as pd
from helpers import write_synthetic_csv
from attendance_recovery.cli import main
from attendance_recovery.loader import load_attendance, rate_col


class BacktestCommandTests(unittest.TestCase):
    def test_command_reports_strict_time_boundary_and_keeps_missing_target_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = write_synthetic_csv(root / 'data.csv')
            with source.open(newline='', encoding='utf-8') as handle:
                rows = list(csv.reader(handle))
            count_index = rows[0].index('2022-2023 student count - year to date')
            original = root / 'original'
            altered = root / 'altered'
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(['backtest', '--data', str(source), '--out', str(original)]), 0)
            for row in rows[1:]:
                row[count_index] = ''
            with source.open('w', newline='', encoding='utf-8') as handle:
                csv.writer(handle).writerows(rows)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(['backtest', '--data', str(source), '--out', str(altered)]), 0)
            pd.testing.assert_frame_equal(pd.read_csv(original / 'backtest_predictions.csv'),
                                          pd.read_csv(altered / 'backtest_predictions.csv'))
            predictions = pd.read_csv(altered / 'backtest_predictions.csv')
            self.assertTrue((predictions.train_latest_year < predictions.target_year).all())
            self.assertFalse((predictions.district_code == '00000CT').any())
            self.assertIn('strictly earlier target years', (altered / 'backtest_summary.md').read_text())
            self.assertEqual(len(list(altered.glob('backtest_*.csv'))), 5)

    def test_insufficient_years_writes_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'data.csv'
            source.write_text(helpers.TINY_CSV)
            out = root / 'out'
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(['backtest', '--data', str(source), '--out', str(out)]), 2)
            self.assertFalse(out.exists())

    def test_conflicts_fail_before_output_directory_is_created(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'data.csv'
            lines = helpers.TINY_CSV.splitlines()
            conflict = lines[2].replace('90.00', '50.00')
            source.write_text('\n'.join(lines + [conflict]))
            out = root / 'out'
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(['analyze', '--data', str(source), '--out', str(out)]), 2)
            self.assertFalse(out.exists())

    def test_unpaired_scoring_mode_retains_observed_rate(self):
        text = 'District code,Student group,2022-2023 student count,2022-2023 attendance rate\n001,All Students,,90\n'
        paired = load_attendance(io.StringIO(text))
        scoring = load_attendance(io.StringIO(text), require_paired=False)
        self.assertTrue(pd.isna(paired.table.iloc[0][rate_col('2022-2023')]))
        self.assertEqual(scoring.table.iloc[0][rate_col('2022-2023')], .9)


if __name__ == '__main__':
    unittest.main()
