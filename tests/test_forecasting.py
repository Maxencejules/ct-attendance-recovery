import io
import tempfile
import unittest
from pathlib import Path

import numpy as np

import helpers  # noqa: F401  (puts src/ on sys.path)
from helpers import TINY_CSV, write_synthetic_csv

from attendance_recovery.forecasting import (
    DRIFT,
    MODEL_NAMES,
    PERSISTENCE,
    STATEWIDE_REASON,
    build_model_frame,
    comparison_sentences,
    cross_validate,
    district_folds,
    mae,
    persistence_forecast,
    rmse,
)
from attendance_recovery.loader import load_attendance, rate_col


class ModelFrameTests(unittest.TestCase):
    def setUp(self):
        self.data = load_attendance(io.StringIO(TINY_CSV))
        self.frame = build_model_frame(self.data)

    def test_only_complete_district_rows_are_kept(self):
        kept = list(zip(self.frame.rows["district_code"], self.frame.rows["student_group"]))
        self.assertEqual(kept, [("0010011", "All Students"), ("0020011", "All Students")])
        self.assertEqual(self.frame.target_year, "2021-2022")
        self.assertEqual(self.frame.prior_years, ["2019-2020", "2020-2021"])

    def test_every_exclusion_has_a_reason(self):
        reasons = dict(
            zip(
                zip(self.frame.exclusions["district_code"], self.frame.exclusions["student_group"]),
                self.frame.exclusions["reason"],
            )
        )
        self.assertEqual(reasons[("00000CT", "All Students")], STATEWIDE_REASON)
        self.assertEqual(reasons[("0010011", "English Learners")], "target year 2021-2022 suppressed or missing")
        self.assertEqual(
            reasons[("0030011", "All Students")],
            "earlier year(s) suppressed or missing: 2019-2020, 2020-2021",
        )
        self.assertEqual(reasons[("0020011", "English Learners")], "earlier year(s) suppressed or missing: 2020-2021")
        self.assertEqual(len(self.frame.rows) + len(self.frame.exclusions), len(self.data.table))

    def test_statewide_row_never_reaches_the_model(self):
        self.assertNotIn("00000CT", set(self.frame.rows["district_code"]))

    def test_persistence_is_previous_year_rate(self):
        np.testing.assert_allclose(persistence_forecast(self.frame), [88.0, 93.0])
        np.testing.assert_allclose(self.frame.rows["actual"], [90.0, 94.0])

    def test_needs_two_years(self):
        text = "District code,Student group,2022-2023 student count,2022-2023 attendance rate\n0010011,All Students,10,0.9\n"
        with self.assertRaises(ValueError):
            build_model_frame(load_attendance(io.StringIO(text)))


class MetricTests(unittest.TestCase):
    def test_mae_and_rmse(self):
        self.assertAlmostEqual(mae([1, 2, 3], [2, 2, 5]), 1.0)
        self.assertAlmostEqual(rmse([1, 2, 3], [2, 2, 5]), np.sqrt(5 / 3))


class DistrictFoldTests(unittest.TestCase):
    def test_no_district_in_both_train_and_test(self):
        groups = np.repeat([f"D{i:02d}" for i in range(17)], [1 + i % 4 for i in range(17)])
        folds = district_folds(groups, n_splits=5)
        self.assertEqual(len(folds), 5)
        seen_test = []
        for train_idx, test_idx in folds:
            self.assertEqual(set(groups[train_idx]) & set(groups[test_idx]), set())
            self.assertEqual(len(train_idx) + len(test_idx), len(groups))
            seen_test.extend(set(groups[test_idx]))
        # every district is tested exactly once
        self.assertEqual(sorted(seen_test), sorted(set(groups)))

    def test_too_many_folds_rejected(self):
        with self.assertRaises(ValueError):
            district_folds(np.array(["A", "A", "B"]), n_splits=3)
        with self.assertRaises(ValueError):
            district_folds(np.array(["A", "B"]), n_splits=1)


class CrossValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        path = write_synthetic_csv(Path(cls.tmp.name) / "synthetic.csv", n_districts=15)
        cls.data = load_attendance(path)
        cls.frame = build_model_frame(cls.data)
        cls.result = cross_validate(cls.frame, n_splits=5)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_each_district_sits_in_one_fold(self):
        preds = self.result.predictions
        folds_per_district = preds.groupby("district_code")["fold"].nunique()
        self.assertTrue((folds_per_district == 1).all())
        self.assertEqual(set(preds["fold"]), {1, 2, 3, 4, 5})
        for fold in range(1, 6):
            test_districts = set(preds.loc[preds["fold"] == fold, "district_code"])
            train_districts = set(preds.loc[preds["fold"] != fold, "district_code"])
            self.assertEqual(test_districts & train_districts, set())

    def test_persistence_predictions_match_previous_year(self):
        preds = self.result.predictions
        table = self.data.table.set_index(["district_code", "student_group"])
        prev = self.frame.prior_years[-1]
        for row in preds.head(25).itertuples():
            expected = 100 * table.loc[(row.district_code, row.student_group), rate_col(prev)]
            self.assertAlmostEqual(getattr(row, PERSISTENCE), expected)

    def test_all_forecasters_scored(self):
        metrics = self.result.metrics.set_index("model")
        self.assertEqual(list(metrics.index), list(MODEL_NAMES))
        self.assertTrue(np.isfinite(metrics[["mae_pp", "rmse_pp"]].to_numpy()).all())
        self.assertEqual(metrics.loc[PERSISTENCE, "mae_minus_persistence_pp"], 0.0)
        self.assertFalse(self.result.predictions[list(MODEL_NAMES)].isna().any().any())
        self.assertIn(DRIFT, set(self.result.fold_metrics["model"]))

    def test_synthetic_exclusions(self):
        reasons = set(self.frame.exclusions["reason"])
        self.assertIn(STATEWIDE_REASON, reasons)
        self.assertIn("target year 2022-2023 suppressed or missing", reasons)

    def test_comparison_sentences_are_plain(self):
        lines = comparison_sentences(self.result.metrics, self.result.n_splits)
        self.assertEqual(len(lines), 5)
        for line in lines[:3]:
            self.assertTrue(
                "beat the persistence baseline" in line
                or "did NOT beat the persistence baseline" in line
                or "mixed result" in line,
                line,
            )


if __name__ == "__main__":
    unittest.main()
