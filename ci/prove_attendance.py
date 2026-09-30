"""Run the real public snapshot through all commands and preserve derived evidence."""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import os
import platform
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from attendance_recovery.cli import main  # noqa: E402

SNAPSHOT_SHA256 = 'd6858fe016fb8100f197f8a7c56ed7d4e7d232a96e63707665e709378708c10a'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def output_hashes(out):
    # Do not include a prior proof.json or unrelated files at the output root.
    return {str(path.relative_to(out)): sha(path)
            for command in ('analyze', 'forecast', 'backtest')
            for path in sorted((out / command).iterdir()) if path.is_file()}


def run(command, source, out):
    transcript = io.StringIO()
    with contextlib.redirect_stdout(transcript):
        status = main([command, '--data', str(source), '--out', str(out), '--rate-unit', 'fraction'])
    if status:
        raise RuntimeError(f'{command} exited {status}')
    (out / f'{command}-console.txt').write_text(transcript.getvalue(), encoding='utf-8')


def prove(out):
    source = ROOT / 'data/school_attendance_by_student_group_and_district_2022-2023.csv'
    assert sha(source) == SNAPSHOT_SHA256, 'Public snapshot changed; review its provenance before replacing the hash'
    for command in ('analyze', 'forecast', 'backtest'):
        run(command, source, out / command)
    # Independently check the known snapshot's units with CSV arithmetic.
    with source.open(encoding='utf-8', newline='') as handle:
        all_students = [row for row in csv.DictReader(handle)
                        if row['Student group'] == 'All Students' and row['District code'] != '00000CT']
    count_key = '2022-2023 student count - year to date'
    rate_key = '2022-2023 attendance rate - year to date'
    expected_rate = 100 * sum(float(row[count_key]) * float(row[rate_key]) for row in all_students) / sum(float(row[count_key]) for row in all_students)
    trajectories = pd.read_csv(out / 'analyze/trajectories.csv')
    observed_rate = trajectories.loc[(trajectories.student_group == 'All Students') & (trajectories.year == '2022-2023'), 'weighted_rate_pct'].item()
    assert np.isclose(expected_rate, observed_rate, atol=1e-4, rtol=0), 'Snapshot unit/weighted-rate mismatch'
    predictions = pd.read_csv(out / 'backtest/backtest_predictions.csv', dtype={'district_code': str})
    assert (predictions.train_latest_year < predictions.target_year).all()
    assert not (predictions.district_code == '00000CT').any()
    assert not predictions.duplicated(['target_year', 'district_code', 'category', 'student_group']).any()
    numeric = predictions[['actual', 'persistence', 'persistence_plus_mean_change', 'ridge', 'hist_gradient_boosting']]
    assert np.isfinite(numeric.to_numpy()).all()
    with tempfile.TemporaryDirectory() as directory:
        repeat = Path(directory) / 'repeat'
        run('backtest', source, repeat)
        assert all(sha(path) == sha(repeat / path.name) for path in (out / 'backtest').glob('*.csv'))
    proof = {
        'dataset_sha256': sha(source),
        'rate_unit': 'fraction',
        'independent_latest_weighted_rate_pct': expected_rate,
        'github_sha': os.environ.get('GITHUB_SHA'),
        'source_sha256': {str(path.relative_to(ROOT)): sha(path) for path in sorted((ROOT / 'src').rglob('*.py'))},
        'requirements_lock_sha256': sha(ROOT / 'requirements-lock.txt'),
        'python': platform.python_version(), 'platform': platform.platform(),
        'numpy': np.__version__, 'pandas': pd.__version__, 'scikit_learn': sklearn.__version__,
        'time_holdout': {'origins': int(predictions.target_year.nunique()), 'rows': len(predictions),
                         'train_latest_years': sorted(predictions.train_latest_year.unique().tolist()),
                         'target_years': sorted(predictions.target_year.unique().tolist()),
                         'history_years': 2, 'csv_repeated_identically': True},
        'backtest_metrics': json.loads(pd.read_csv(out / 'backtest/backtest_metrics.csv').to_json(orient='records')),
        'retrospective_district_metrics': json.loads(pd.read_csv(out / 'forecast/forecast_metrics.csv').to_json(orient='records')),
        'output_sha256': output_hashes(out),
    }
    # One origin has no cross-origin standard deviation; dataframe JSON converts it to null.
    text = json.dumps(proof, indent=2, allow_nan=False)
    (out / 'proof.json').write_text(text + '\n', encoding='utf-8')
    print(json.dumps(proof['time_holdout']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=ROOT / 'outputs/ci-proof')
    prove(parser.parse_args().out)
