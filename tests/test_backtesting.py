import copy
import unittest

import helpers  # noqa: F401 (adds src to sys.path)
import numpy as np
import pandas as pd

from attendance_recovery.backtesting import expanding_origin_backtest
from attendance_recovery.forecasting import DRIFT, MODEL_NAMES, PERSISTENCE
from attendance_recovery.loader import AttendanceData, LoadReport, count_col, rate_col


def fixture(rates=None, *, statewide=False):
    rates = rates if rates is not None else [0.95, 0.94, 0.90, 0.92, 0.93, 0.94]
    years = [f"{2017 + index}-{2018 + index}" for index in range(len(rates))]
    records = []
    for index, code in enumerate(["A", "B", "C"] + (["00000CT"] if statewide else [])):
        row = dict(
            district_code=code,
            district_name=code,
            category="All Students",
            student_group="All Students",
            is_statewide=code == "00000CT",
        )
        for year, rate in zip(years, rates, strict=True):
            row[rate_col(year)] = rate
            row[count_col(year)] = 100.0 + index * 10
        records.append(row)
    return AttendanceData(
        pd.DataFrame.from_records(records), years, LoadReport(source="synthetic")
    )


class TemporalBacktestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = fixture()
        # Vary training changes across districts so the learned model can react
        # to leakage; a constant target would make even a leaky feature harmless.
        for district, row_index in enumerate(cls.data.table.index):
            for year_index, year in enumerate(cls.data.years):
                cls.data.table.loc[row_index, rate_col(year)] += (
                    0.002 * district * year_index
                )
        cls.result = expanding_origin_backtest(cls.data)

    def test_each_test_target_is_scored_once_with_strictly_past_training(self):
        result = self.result
        self.assertEqual(result.n_origins, 3)
        self.assertEqual(
            list(result.predictions.target_year.unique()), self.data.years[3:]
        )
        self.assertTrue(
            (
                result.predictions.train_latest_year < result.predictions.target_year
            ).all()
        )
        self.assertFalse(
            result.predictions.duplicated(
                ["target_year", "district_code", "student_group"]
            ).any()
        )
        self.assertEqual(len(result.predictions), 9)
        folds = result.fold_metrics.drop_duplicates("fold")
        self.assertEqual(list(folds.train_rows), [3, 6, 9])
        for origin in folds.itertuples():
            self.assertTrue(
                all(
                    year < origin.target_year
                    for year in origin.train_target_years.split(", ")
                )
            )
        self.assertEqual(list(result.metrics.model), list(MODEL_NAMES))
        self.assertFalse(result.predictions[list(MODEL_NAMES)].isna().any().any())
        self.assertEqual(len(result.by_group), 1)

    def test_hand_checked_persistence_and_drift_errors(self):
        result = expanding_origin_backtest(fixture([0.95, 0.94, 0.90, 0.92, 0.93]))
        metrics = result.metrics.set_index("model")
        np.testing.assert_allclose(result.predictions[PERSISTENCE], [90] * 3 + [92] * 3)
        np.testing.assert_allclose(result.predictions[DRIFT], [86] * 3 + [91] * 3)
        self.assertAlmostEqual(metrics.loc[PERSISTENCE, "mae_pp"], 1.5)
        self.assertAlmostEqual(metrics.loc[PERSISTENCE, "rmse_pp"], np.sqrt(2.5))
        self.assertAlmostEqual(metrics.loc[DRIFT, "mae_pp"], 4)
        self.assertAlmostEqual(metrics.loc[DRIFT, "rmse_pp"], np.sqrt(20))

    def test_current_target_shock_cannot_be_learned_from_that_target(self):
        result = expanding_origin_backtest(fixture([0.90, 0.90, 0.90, 0.70]))
        np.testing.assert_allclose(result.predictions[PERSISTENCE], 90)
        np.testing.assert_allclose(result.predictions[DRIFT], 90)
        np.testing.assert_allclose(result.predictions["actual"], 70)
        self.assertAlmostEqual(
            result.metrics.set_index("model").loc[DRIFT, "mae_pp"], 20
        )

    def test_current_labels_do_not_change_current_predictions(self):
        changed = copy.deepcopy(self.data)
        current = changed.years[4]
        changed.table[rate_col(current)] = [0.50, 0.60, 0.70]
        result = expanding_origin_backtest(changed)
        before = self.result.predictions.query("target_year == @current")
        after = result.predictions.query("target_year == @current")
        np.testing.assert_allclose(before[list(MODEL_NAMES)], after[list(MODEL_NAMES)])
        self.assertFalse(np.allclose(before.actual, after.actual))

    def test_future_labels_and_counts_do_not_change_earlier_origins(self):
        changed = copy.deepcopy(self.data)
        future = changed.years[4]
        changed.table[rate_col(future)] = [0.10, 0.20, 0.30]
        changed.table[count_col(future)] = [1000000, 2000000, 3000000]
        result = expanding_origin_backtest(changed)
        before = self.result.predictions.query("target_year < @future")
        after = result.predictions.query("target_year < @future")
        pd.testing.assert_frame_equal(
            before.reset_index(drop=True), after.reset_index(drop=True)
        )

    def test_current_target_count_is_not_a_feature_or_eligibility_rule(self):
        data = fixture([0.95, 0.94, 0.90, 0.92])
        before = expanding_origin_backtest(data)
        data.table[count_col(data.latest_year)] = np.nan
        after = expanding_origin_backtest(data)
        pd.testing.assert_frame_equal(before.predictions, after.predictions)
        data.table = data.table.drop(columns=[count_col(data.latest_year)])
        pd.testing.assert_frame_equal(
            before.predictions, expanding_origin_backtest(data).predictions
        )

    def test_exclusion_audit_uses_local_window_and_identifies_train_or_test(self):
        data = fixture(statewide=True)
        data.table.loc[data.table.district_code == "A", count_col(data.years[0])] = (
            np.nan
        )
        result = expanding_origin_backtest(data)
        audit = result.exclusions
        self.assertEqual(set(audit.role), {"train", "test"})
        self.assertTrue((audit.origin_year >= audit.target_year).all())
        missing = audit[audit.district_code == "A"]
        self.assertEqual(set(missing.target_year), {data.years[2]})
        self.assertEqual(set(missing.role), {"train"})
        self.assertTrue(missing.reason.str.contains(data.years[0], regex=False).all())
        # An old suppressed year outside a later window does not disqualify its test row.
        self.assertEqual(sum(result.predictions.district_code == "A"), 3)
        self.assertNotIn("00000CT", set(result.predictions.district_code))
        self.assertTrue(
            audit[audit.district_code == "00000CT"]
            .reason.str.contains("statewide")
            .all()
        )

    def test_single_year_lookback_is_supported_without_current_features(self):
        result = expanding_origin_backtest(fixture([0.9, 0.91, 0.92]), history_years=1)
        self.assertEqual(result.n_origins, 1)
        np.testing.assert_allclose(result.predictions[PERSISTENCE], 91)

    def test_empty_training_and_empty_test_origins_fail_explicitly(self):
        data = fixture([0.95, 0.94, 0.90, 0.92])
        data.table[rate_col(data.years[0])] = np.nan
        with self.assertRaisesRegex(ValueError, "no complete training rows"):
            expanding_origin_backtest(data)
        data = fixture([0.95, 0.94, 0.90, 0.92])
        data.table[rate_col(data.latest_year)] = np.nan
        with self.assertRaisesRegex(ValueError, "no complete test rows"):
            expanding_origin_backtest(data)

    def test_invalid_history_and_insufficient_years(self):
        for history in [0, -1, 2.5, True]:
            with self.subTest(history=history), self.assertRaises(ValueError):
                expanding_origin_backtest(self.data, history_years=history)
        with self.assertRaisesRegex(ValueError, "at least 4"):
            expanding_origin_backtest(fixture([0.9, 0.9, 0.9]))

    def test_invalid_year_labels_order_duplicates_and_gaps(self):
        for years in [
            ["2017-2017"],
            ["2017-18"],
            list(reversed(self.data.years)),
            self.data.years[:2] + self.data.years[3:],
            [self.data.years[0]] + self.data.years,
        ]:
            data = copy.deepcopy(self.data)
            data.years = years
            with self.subTest(years=years), self.assertRaises(ValueError):
                expanding_origin_backtest(data)


if __name__ == "__main__":
    unittest.main()
