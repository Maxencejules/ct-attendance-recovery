import contextlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import helpers
from helpers import SYNTH_GROUPS, SYNTH_YEARS, write_synthetic_csv

from attendance_recovery.cli import main


def _run(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(argv)
    return code, out.getvalue(), err.getvalue()


class EndToEndTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.csv = write_synthetic_csv(self.root / "synthetic.csv", n_districts=15)
        self.out = self.root / "outputs"

    def tearDown(self):
        self.tmp.cleanup()

    def test_analyze_writes_tables_and_summary(self):
        code, stdout, _ = _run(["analyze", "--data", str(self.csv), "--out", str(self.out)])
        self.assertEqual(code, 0)
        for name in ("trajectories.csv", "recovery_gaps.csv", "attendance_summary.md"):
            self.assertTrue((self.out / name).is_file(), name)
        traj = pd.read_csv(self.out / "trajectories.csv")
        self.assertEqual(len(traj), len(SYNTH_GROUPS) * len(SYNTH_YEARS))
        gaps = pd.read_csv(self.out / "recovery_gaps.csv")
        self.assertEqual(len(gaps), len(SYNTH_GROUPS) * (len(SYNTH_YEARS) - 1))
        summary = (self.out / "attendance_summary.md").read_text(encoding="utf-8")
        self.assertIn("## Data handling", summary)
        self.assertIn("All Students, student-weighted:", stdout)
        self.assertIn("groups below baseline", stdout)

    def test_forecast_writes_metrics_and_predictions(self):
        code, stdout, _ = _run(["forecast", "--data", str(self.csv), "--out", str(self.out), "--folds", "5"])
        self.assertEqual(code, 0)
        metrics = pd.read_csv(self.out / "forecast_metrics.csv")
        self.assertEqual(
            list(metrics["model"]),
            ["persistence", "persistence_plus_mean_change", "ridge", "hist_gradient_boosting"],
        )
        preds = pd.read_csv(self.out / "forecast_predictions.csv", dtype={"district_code": str})
        self.assertTrue(preds["fold"].between(1, 5).all())
        excluded = pd.read_csv(self.out / "forecast_excluded_rows.csv", dtype={"district_code": str})
        self.assertIn("00000CT", set(excluded["district_code"]))
        self.assertNotIn("00000CT", set(preds["district_code"]))
        self.assertTrue((self.out / "forecast_summary.md").is_file())
        self.assertIn("GroupKFold by district", stdout)
        self.assertIn("persistence baseline", stdout)

    def test_missing_file_reports_error(self):
        code, _, stderr = _run(["analyze", "--data", str(self.root / "nope.csv"), "--out", str(self.out)])
        self.assertEqual(code, 2)
        self.assertIn("error:", stderr)

    def test_latest_year_baseline_is_a_clean_error(self):
        latest = SYNTH_YEARS[-1]
        code, _, stderr = _run(["analyze", "--data", str(self.csv), "--out", str(self.out), "--baseline", latest])
        self.assertEqual(code, 2)
        self.assertIn("error:", stderr)
        self.assertIn("must be earlier than the latest year", stderr)
        self.assertNotIn("Traceback", stderr)
        self.assertFalse(self.out.exists())  # the error is raised before anything is written

    def test_module_entry_point_runs_both_commands(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = str(helpers.SRC_DIR) + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        for command in ("analyze", "forecast"):
            proc = subprocess.run(
                [sys.executable, "-m", "attendance_recovery", command, "--data", str(self.csv), "--out", str(self.out)],
                capture_output=True,
                text=True,
                env=env,
                timeout=300,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("Wrote", proc.stdout)


if __name__ == "__main__":
    unittest.main()
