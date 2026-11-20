"""Paired Kronos benchmarks reuse suite labels, purged folds and execution-cost accounting."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from ..autonomous.engine import Journal, economic_metrics
from ..autonomous.evidence import block_p_value
from ..autonomous.learners import Hypothesis
from ..models.registry import load_registry, register_research_challenger
from ..models.strategy import StrategyStats, evaluate_score
from ..models.vol import vol_metrics
from ..provenance import dumps
from ..splits import plan_walk_forward
from .adapter import DEFAULT_ROOT, ForecastConfig, KronosAdapter, sha
from .data import context, future_calendar
from .tuning import tune_heads


def common_rows(indices: np.ndarray, maximum: int) -> np.ndarray:
    """Deterministic time-spread sampling before any predictions/outcomes are inspected."""
    if maximum < 1:
        raise ValueError('row budget must be positive')
    return indices[np.unique(np.linspace(0, len(indices)-1, min(maximum, len(indices)), dtype=int))]


def paired_evidence(candidate: np.ndarray, baseline: np.ndarray, trials: int, seed: int = 42) -> dict[str, Any]:
    """Session-block inference for incremental net mean and risk-adjusted value."""
    delta = candidate-baseline
    p = block_p_value(delta, seed=seed)
    def sharpe(a: np.ndarray) -> float:
        return float(np.mean(a)/max(np.std(a, ddof=1), 1e-8) * np.sqrt(252)) if len(a) > 1 else 0.
    alpha = .05 / max(1, 2*trials)
    lower = None
    if len(delta) >= 60:
        rng = np.random.default_rng(seed)
        ids = (rng.integers(0, len(delta), (4000, int(np.ceil(len(delta)/5)), 1)) + np.arange(5)) % len(delta)
        ids = ids.reshape(4000, -1)[:, :len(delta)]
        differences = [sharpe(candidate[i])-sharpe(baseline[i]) for i in ids]
        lower = float(np.quantile(differences, alpha))
    return {'p_incremental_net': p, 'delta_session_sharpe': sharpe(candidate)-sharpe(baseline),
            'sharpe_difference_lower_bound': lower, 'alpha': alpha,
            'credible': bool(p <= alpha and lower is not None and lower > 0),
            'method': 'paired five-session circular block bootstrap; all-trial correction'}


def cpu_comparators(a: dict[str, np.ndarray], target: np.ndarray, train: np.ndarray, rows: np.ndarray,
                    test: np.ndarray, manifest: dict[str, Any], registry: Any, seed: int) -> dict[str, np.ndarray]:
    """Fresh interpreter avoids mixing libomp from torch and LightGBM in one macOS process."""
    with tempfile.TemporaryDirectory(prefix='lat-kronos-cpu-') as temporary:
        directory = Path(temporary)
        np.savez_compressed(directory / 'input.npz', X_train=a['X'][train], y_train=target[train],
                            X_predict=a['X'][rows], vol_train=a['vol_X'][train], rv_train=a['y'][train, 1]**2,
                            vol_test=a['vol_X'][test])
        (directory / 'input.json').write_text(json.dumps({'seed': seed, 'vol_spec': manifest['vol_spec'],
                                                         'registry': registry.meta if registry else None}))
        env = {k: v for k, v in os.environ.items() if k in ('PATH', 'HOME', 'TMPDIR', 'LANG', 'SYSTEMROOT')}
        env.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', VECLIB_MAXIMUM_THREADS='2')
        completed = subprocess.run([sys.executable, '-m', 'services.research.kronos.cpu_worker', str(directory)],
                       env=env, check=False, timeout=600, capture_output=True, text=True)
        if completed.returncode:
            raise RuntimeError(f'CPU comparator failed: {completed.stderr[-4000:]}')
        with np.load(directory / 'output.npz', allow_pickle=False) as archive:
            return dict(archive)


def evaluate_challenger(root: Path, jid: str, spec: Hypothesis, a: dict[str, np.ndarray],
                        manifest: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    journal = Journal(root)
    if spec.kronos_artifact_hash != sha(DEFAULT_ROOT / 'manifest.json'):
        raise ValueError('Kronos weight manifest changed after hypothesis registration')
    if spec.kronos_tune_steps:
        justified = any(r['result'] and r['spec']['family'] == spec.family and not r['spec'].get('kronos_tune_steps')
                        and r['result'].get('objective', 0) > 0 and r['result']['metrics']['net_return'] > 0
                        and not r['result']['is_simulated'] for r in journal.rows(manifest['dataset_key']))
        if not justified:
            raise ValueError('fine-tuning requires a profitable real-data development parent')
    cfg = ForecastConfig(variant=spec.family.removeprefix('kronos_'), context=spec.kronos_context,
                         samples=spec.kronos_samples, seed=spec.seed)
    adapter = KronosAdapter(cfg)
    plan = plan_walk_forward(a['decision'], a['end'], embargo_steps=60, min_unique_times=200)
    horizons = (5, 15, 60)
    if manifest['bar_minutes'] != 1:
        raise ValueError('autonomous Kronos benchmark currently requires one-minute bars')
    registry = load_registry(root.parent)
    names = [spec.family, 'lightgbm', 'catboost', 'ridge', 'selected_registry']
    predictions, fold_reports, paths, path_rows, tuning = [], [], [], [], []
    pooled = {name: StrategyStats() for name in names}
    stress = {name: StrategyStats() for name in names}
    daily: dict[str, dict[int, float]] = {name: {} for name in names}
    vol_actual, vol_kronos, vol_har = [], [], []
    artifacts = root / 'runs' / jid
    artifacts.mkdir(parents=True, exist_ok=True)
    sampling_metadata: dict[str, Any] = {}
    missing_selected = registry is None
    for fold in plan.folds:
        journal.event('TRAIN', f'{spec.family} paired benchmark fold {fold.fold+1}/{len(plan.folds)}', jid)
        aligned = a['calendar_aligned']
        train = fold.fit_idx[aligned[fold.fit_idx]]
        cal = common_rows(fold.calib_idx[aligned[fold.calib_idx]], max(20, spec.kronos_rows_per_fold))
        test = common_rows(fold.test_idx[aligned[fold.test_idx]], spec.kronos_rows_per_fold)
        if len(cal) < 20 or not len(test):
            raise ValueError('insufficient common calibration/test rows')
        if spec.kronos_tune_steps:
            adapter = KronosAdapter(cfg)  # reset to pretrained weights on every fold
            tuning.append(tune_heads(adapter, a, train, spec.kronos_tune_steps, int(a['decision'][cal].min())))
            from safetensors.torch import save_file
            save_file({k: v.detach().cpu().contiguous() for k, v in adapter.model.head.state_dict().items()},
                      str(artifacts / f'fold-{fold.fold}-head.safetensors'))
        rows = np.r_[cal, test]
        kp, lower, vols, summaries = [], [], [], []
        for row in rows:
            forecast = adapter.forecast(context(a, int(row), cfg.context),
                                        future_calendar(int(a['decision'][row]), 61, 1),
                                        decision=cast(pd.Timestamp, pd.Timestamp(int(a['decision'][row]), tz='UTC')),
                                        seed=(spec.seed+int(row)) % (2**31))
            sampling_metadata = forecast.metadata
            summary = forecast.features(horizons, float(a['cost'][row]))
            kp.append([summary[f'h{h}_mean_bps'] for h in horizons])
            lower.append([summary[f'h{h}_q10_bps'] for h in horizons])
            vols.append(summary['h60_variance'])
            summaries.append(summary)
            if row in test:
                paths.append(forecast.paths)
                path_rows.append(int(row))
        kp, lower = np.asarray(kp), np.asarray(lower)
        target = np.column_stack([a[f'net_{h}'] for h in horizons])
        all_pred = {spec.family: kp}
        cpu = cpu_comparators(a, target, train, rows, test, manifest, registry, spec.seed)
        har_pred = cpu.pop('har_rv')
        all_pred.update(cpu)
        vol_actual.extend(a['y'][test, 1]**2)
        vol_kronos.extend(vols[len(cal):])
        vol_har.extend(har_pred)
        report: dict[str, Any] = {'fold': fold.fold, 'fit': len(train), 'calibration_rows': cal.tolist(),
                                  'test_rows': test.tolist(), 'models': {}}
        for name, pred in all_pred.items():
            allowance = np.maximum(0, np.quantile(pred[:len(cal)]-target[cal], .9, axis=0))
            p = pred[len(cal):]
            score = p-allowance
            if name == spec.family:
                score = np.minimum(score, lower[len(cal):])
            # Economics use the same 60-minute target, costs, row set and one-position simulator.
            stats = StrategyStats()
            taken = evaluate_score(a['decision'][test], score[:, 2], a['exit'][test], a['y'][test, 0], 1, stats)
            pooled[name].add(a['y'][test[taken], 0], stats.span_minutes, len(test))
            stress[name].add(a['y'][test[taken], 0]-a['cost'][test[taken]], stats.span_minutes, len(test))
            days = a['decision'][test] // 86_400_000_000_000
            for day in np.unique(days):
                daily[name].setdefault(int(day), 0.)
            for t in taken:
                daily[name][int(days[t])] += float(a['y'][test[t], 0])
            report['models'][name] = stats.metrics(60) | {'mae_bps_by_horizon': np.abs(p-target[test]).mean(axis=0).tolist(),
                                                        'allowance_bps': allowance.tolist(), 'trade_rows': test[taken].tolist()}
            if name == spec.family:
                report.update(stats.metrics(60))
                report['trade_rows'] = test[taken].tolist()
            for i, row in enumerate(test):
                predictions.append({'row': int(row), 'fold': fold.fold, 'model': name,
                                    'symbol': manifest['symbols'][int(a['symbol_id'][row])],
                                    'decision_ns': int(a['decision'][row]), 'exit_ns': int(a['exit'][row]),
                                    'prediction': p[i].tolist(), 'score': float(score[i, 2]),
                                    'features': summaries[len(cal)+i] if name == spec.family else None})
        fold_reports.append(report)
    trials = len(journal.rows()) * len(names) * len(horizons)
    days = sorted(daily[spec.family])
    candidate_daily = np.array([daily[spec.family][d] for d in days])
    comparisons = {name: paired_evidence(candidate_daily, np.array([daily[name][d] for d in days]), trials)
                   for name in names[1:] if daily[name]}
    metrics = pooled[spec.family].metrics(60)
    risk = economic_metrics(np.array(pooled[spec.family].net_bps))
    blockers = ['pretraining data overlap unknown; require independently timestamped post-checkpoint evidence',
                'sequence challenger has not passed independent confirmation and LEAN parity']
    if missing_selected:
        blockers.append('current selected registry unavailable or invalid')
    if not comparisons or not all(c['credible'] for c in comparisons.values()):
        blockers.append('incremental net risk-adjusted value not established against every comparator')
    result = {'metrics': metrics, 'stress_2x_costs': stress[spec.family].metrics(60), 'risk': risk,
              'folds': fold_reports, 'sessions': len(days), 'p_positive': block_p_value(candidate_daily),
              'p_incremental': max((v['p_incremental_net'] for v in comparisons.values()), default=1.),
              'objective': float((metrics.get('net_sharpe') or 0)-10*(risk['max_drawdown'] or 0))
              if metrics['n_trades'] >= 20 else -1e9,
              'benchmarks': {name: {'metrics': pooled[name].metrics(60), 'stress_2x_costs': stress[name].metrics(60)}
                             for name in names if daily[name]}, 'paired_comparisons': comparisons,
              'volatility': {'kronos': vol_metrics(np.array(vol_actual), np.array(vol_kronos)),
                             'har_rv': vol_metrics(np.array(vol_actual), np.array(vol_har)),
                             'har_base_reason': manifest['har_base_reason']},
              'is_simulated': manifest['is_simulated'], 'validation': 'DEVELOPMENT_ONLY',
              'promotion_eligible': False, 'promotion_blockers': blockers,
              'selected_registry': ((registry.meta.get('registry_sha256') or sha(registry.path / 'registry.json'))
                                    if registry else None),
              'comparison_policy': 'selected return architectures refitted inside each fold; HAR-RV is a volatility comparator',
              'horizons_minutes': horizons, 'calendar_aligned_rows': int(a['calendar_aligned'].sum()),
              'calendar_excluded_rows': int((~a['calendar_aligned']).sum()), 'row_budget_per_fold': spec.kronos_rows_per_fold,
              'sampling': sampling_metadata, 'tuning': tuning, 'elapsed_seconds': time.monotonic()-started,
              'code_hash': manifest['code_hash'], 'dataset_key': manifest['dataset_key'],
              'no_edge_note': 'Development research only; sampling quantiles are not calibrated probabilities.'}
    (artifacts / 'predictions.json').write_text(dumps(predictions))
    result['prediction_sha256'] = sha(artifacts / 'predictions.json')
    np.savez_compressed(artifacts / 'forecast_paths.npz', rows=np.array(path_rows), paths=np.stack(paths))
    result['paths_sha256'] = sha(artifacts / 'forecast_paths.npz')
    (artifacts / 'result.json').write_text(dumps(result))
    register_research_challenger(root.parent, jid, spec.describe(), result, artifacts)
    return result
