import io
import math
import unittest

import helpers  # noqa: F401  (puts src/ on sys.path)
from helpers import TINY_CSV

from attendance_recovery.loader import load_attendance
from attendance_recovery.trends import group_trajectories, recovery_gaps, weighted_mean


class WeightedMeanTests(unittest.TestCase):
    def test_hand_computed(self):
        # (0.90*100 + 0.94*300) / 400 = 0.93
        self.assertAlmostEqual(weighted_mean([0.90, 0.94], [100, 300]), 0.93)

    def test_skips_missing_and_zero_weights(self):
        self.assertAlmostEqual(weighted_mean([0.8, float("nan"), 0.6, 0.1], [1, 5, 3, 0]), 0.65)

    def test_nothing_usable_is_nan(self):
        self.assertTrue(math.isnan(weighted_mean([float("nan")], [10])))
        self.assertTrue(math.isnan(weighted_mean([], [])))


class TrajectoryTests(unittest.TestCase):
    def setUp(self):
        self.data = load_attendance(io.StringIO(TINY_CSV))
        self.traj = group_trajectories(self.data)

    def _get(self, group, year):
        t = self.traj
        return t[(t["student_group"] == group) & (t["year"] == year)].iloc[0]

    def test_all_students_weighted_rates(self):
        latest = self._get("All Students", "2021-2022")
        # districts only: (90 + 282 + 92) / 500
        self.assertAlmostEqual(latest["weighted_rate_pct"], 92.8)
        self.assertEqual(latest["districts_reporting"], 3)
        self.assertEqual(latest["students"], 500)
        base = self._get("All Students", "2019-2020")
        self.assertAlmostEqual(base["weighted_rate_pct"], 96.5)
        self.assertEqual(base["districts_reporting"], 2)
        mid = self._get("All Students", "2020-2021")
        # Gamma cleared: (90*0.88 + 290*0.93) / 380
        self.assertAlmostEqual(mid["weighted_rate_pct"], 100 * (79.2 + 269.7) / 380)

    def test_statewide_row_is_only_a_cross_check(self):
        latest = self._get("All Students", "2021-2022")
        self.assertAlmostEqual(latest["state_reported_pct"], 93.10)
        self.assertNotAlmostEqual(latest["weighted_rate_pct"], latest["state_reported_pct"])

    def test_group_with_no_state_row(self):
        el = self._get("English Learners", "2019-2020")
        # (12*0.90 + 38*0.925) / 50
        self.assertAlmostEqual(el["weighted_rate_pct"], 100 * (10.8 + 35.15) / 50)
        self.assertTrue(math.isnan(el["state_reported_pct"]))

    def test_one_row_per_group_and_year(self):
        self.assertEqual(len(self.traj), 2 * 3)


class RecoveryGapTests(unittest.TestCase):
    def setUp(self):
        self.data = load_attendance(io.StringIO(TINY_CSV))
        self.gaps = recovery_gaps(self.data)

    def test_gap_uses_matched_rows_only(self):
        g = self.gaps
        row = g[(g["student_group"] == "All Students") & (g["year"] == "2021-2022")].iloc[0]
        self.assertEqual(row["matched_rows"], 2)  # Gamma has no baseline
        self.assertAlmostEqual(row["baseline_rate_pct"], 96.5)
        self.assertAlmostEqual(row["rate_pct"], 93.0)  # (90 + 282) / 400
        self.assertAlmostEqual(row["gap_pp"], -3.5)
        self.assertAlmostEqual(row["median_district_gap_pp"], -4.0)  # Alpha -5, Beta -3
        self.assertEqual(row["rows_at_or_above_baseline"], 0)

    def test_gap_for_middle_year(self):
        g = self.gaps
        row = g[(g["student_group"] == "All Students") & (g["year"] == "2020-2021")].iloc[0]
        self.assertAlmostEqual(row["gap_pp"], 100 * (79.2 + 269.7) / 380 - 96.5)

    def test_baseline_is_not_listed_as_a_later_year(self):
        self.assertEqual(set(self.gaps["year"]), {"2020-2021", "2021-2022"})
        self.assertEqual(set(self.gaps["baseline_year"]), {"2019-2020"})

    def test_custom_and_unknown_baseline(self):
        later = recovery_gaps(self.data, "2020-2021")
        self.assertEqual(set(later["year"]), {"2021-2022"})
        with self.assertRaises(ValueError):
            recovery_gaps(self.data, "2015-2016")

    def test_latest_year_as_baseline_is_rejected(self):
        # 2021-2022 is the newest year in the tiny fixture: nothing to compare it with.
        with self.assertRaisesRegex(ValueError, "must be earlier than the latest year"):
            recovery_gaps(self.data, "2021-2022")


if __name__ == "__main__":
    unittest.main()
