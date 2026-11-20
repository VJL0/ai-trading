"""Bounded paired development tournament. Frozen holdout remains inaccessible to workers."""
from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from ..autonomous.engine import code_hash, economic_metrics
from ..kronos.benchmark import common_rows, paired_evidence
from ..models.strategy import StrategyStats, evaluate_score
from ..provenance import dumps
from ..splits import plan_walk_forward
from .targets import probability_metrics


def benchmark(cache: Path, output: Path, families: list[str], train_rows=256, test_rows=32):
    if not 32 <= train_rows <= 2048 or not 1 <= test_rows <= 256:
        raise ValueError('diagnostic row budget must be 32..2048 train and 1..256 test')
    manifest = json.loads((cache/'manifest.json').read_text())
    data = cache/'development.npz'
    if hashlib.sha256(data.read_bytes()).hexdigest() != manifest['cache_sha256']:
        raise ValueError('development cache integrity mismatch')
    with np.load(data, allow_pickle=False) as archive:
        a = dict(archive)
    plan = plan_walk_forward(a['decision'], a['end'], embargo_steps=60, min_unique_times=200)
    valid = a.get('calendar_aligned', np.ones(len(a['X']), bool)).copy()
    for symbol in np.unique(a['symbol_id']):
        rows = np.flatnonzero(a['symbol_id'] == symbol)
        valid[rows] &= np.searchsorted(a[f'avail_{symbol}'], a['decision'][rows], side='right') >= 128
    folds = [{'train': common_rows(f.fit_idx[valid[f.fit_idx]], train_rows).tolist(),
              'cal': common_rows(f.calib_idx[valid[f.calib_idx]], max(32, test_rows)).tolist(),
              'test': common_rows(f.test_idx[valid[f.test_idx]], test_rows).tolist()} for f in plan.folds]
    output.mkdir(parents=True, exist_ok=False)
    report = {'source_sha256': code_hash(), 'dataset': manifest['dataset_key'], 'is_simulated': manifest['is_simulated'],
              'holdout': 'UNTOUCHED', 'folds': folds, 'models': {}, 'promotion_eligible': False,
              'scope': 'bounded development diagnostic; cannot authorize deployment'}
    days = np.unique(a['decision'][np.concatenate([f['test'] for f in folds])]//86_400_000_000_000)
    series = {}
    for family in families:
        directory = output/family
        directory.mkdir()
        request = {'cache': str(data.resolve()), 'family': family, 'folds': folds}
        (directory/'request.json').write_text(json.dumps(request))
        env = {k: v for k, v in os.environ.items() if k in ('PATH', 'HOME', 'TMPDIR', 'LANG')}
        env.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', VECLIB_MAXIMUM_THREADS='2')
        started = time.monotonic()
        with (directory/'worker.log').open('w') as log:
            process = subprocess.Popen([sys.executable, '-m', 'services.research.advanced.benchmark_worker', str(directory)],
                                       env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                returncode = process.wait(timeout=1800)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                returncode = -9
        if returncode:
            report['models'][family] = {'status': 'FAILED', 'log': str(directory/'worker.log')}
        else:
            predictions = json.loads((directory/'output.json').read_text())
            stats, stress = StrategyStats(), StrategyStats()
            daily = {int(day): 0. for day in days}
            metrics = []
            for fold in predictions:
                rows = np.asarray(fold['rows'], int)
                pred = np.asarray(fold['predictions'])
                before_span = stats.span_minutes
                taken = evaluate_score(a['decision'][rows], pred[:, 0]-fold['allowance'], a['exit'][rows],
                                       a['y'][rows, 0], manifest['bar_minutes'], stats)
                stress.add(a['y'][rows[taken], 0]-a['cost'][rows[taken]], stats.span_minutes-before_span, len(rows))
                for row in rows[taken]:
                    daily[int(a['decision'][row]//86_400_000_000_000)] += float(a['y'][row, 0])
                metrics.append({'mae': np.mean(abs(pred-a['y'][rows]), axis=0).tolist(),
                                'calibration': probability_metrics(pred, a['y'][rows]),
                                'head_latency_ms_per_row': fold['head_latency_ms_per_row']})
            series[family] = np.array([daily[int(d)] for d in days])
            report['models'][family] = {'status': 'COMPLETED', 'metrics': stats.metrics(60),
                'risk': economic_metrics(np.array(stats.net_bps)), 'stress_2x': stress.metrics(60),
                'fold_metrics': metrics, 'total_seconds': time.monotonic()-started,
                'promotion_blockers': ['requires full-budget independent confirmation',
                    *(['foundation pretraining overlap not independently excluded'] if family.startswith('foundation_') else [])]}
        (output/'report.json').write_text(dumps(report))
    if 'ridge' in series:
        for family, values in series.items():
            report['models'][family]['paired_vs_ridge'] = paired_evidence(values, series['ridge'], len(families))
    (output/'report.json').write_text(dumps(report))
    from .tracking import record_tournament
    record_tournament(output)
    return report
