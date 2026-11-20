"""Durable autonomous search with development-only workers and honest economic accounting.

SQLite checkpoints jobs, not opaque Python objects. NPZ caches contain numeric arrays only.
No module in this package imports execution, risk-policy mutation, broker or paid AI clients.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
from contextlib import closing
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..models.strategy import StrategyStats, evaluate_score
from ..provenance import NO_EDGE_NOTE, dumps
from ..splits import plan_walk_forward
from .evidence import block_p_value, screen
from .learners import Hypothesis, available, fit


def identity(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def code_hash() -> str:
    root = Path(__file__).resolve().parents[1]
    files = []
    for folder, dirs, names in os.walk(root):
        dirs[:] = sorted(set(dirs)-{'qlib_env', 'foundation_env', 'rd_env', '.venv', '__pycache__'})
        files.extend(Path(folder)/name for name in names if name.endswith('.py'))
    files.extend((root.parent / 'memory').rglob('*.py'))
    # Bind numerical/runtime versions as well as source; never traverse installed environments.
    files.extend(root/env/'uv.lock' for env in ('qlib_env', 'foundation_env', 'rd_env'))
    files.append(root.parents[1]/'uv.lock')
    return identity([(str(p.relative_to(root)) if p.is_relative_to(root) else str(p),
                      hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(files) if p.is_file()])


def now() -> str:
    return datetime.now(UTC).isoformat()


class Journal:
    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / 'research.sqlite3'
        with closing(self.connect()) as db, db:
            db.executescript('''
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, dataset TEXT NOT NULL, spec TEXT NOT NULL,
                    state TEXT NOT NULL, created TEXT NOT NULL, updated TEXT NOT NULL,
                    result TEXT, error TEXT);
                CREATE TABLE IF NOT EXISTS events (
                    seq INTEGER PRIMARY KEY, at TEXT NOT NULL, job TEXT, stage TEXT NOT NULL, detail TEXT NOT NULL);
            ''')

    def connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        return db

    def event(self, stage: str, detail: str, job: str | None = None) -> None:
        with closing(self.connect()) as db, db:
            db.execute('INSERT INTO events(at,job,stage,detail) VALUES(?,?,?,?)', (now(), job, stage, detail))

    def enqueue(self, dataset: str, spec: Hypothesis) -> str:
        if spec.family.startswith('kronos_') and not spec.kronos_artifact_hash:
            from ..kronos.adapter import DEFAULT_ROOT, sha
            spec = replace(spec, kronos_artifact_hash=sha(DEFAULT_ROOT / 'manifest.json'))
        if (spec.family.startswith('foundation_') or spec.family in ('tabiclv2', 'kumo', 'mitra')):
            from ..advanced.foundation import WEIGHTS
            path = WEIGHTS/spec.family.removeprefix('foundation_')/'manifest.json'
            fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
            if spec.foundation_artifact_hash and spec.foundation_artifact_hash != fingerprint:
                raise ValueError('foundation manifest changed')
            spec = replace(spec, foundation_artifact_hash=fingerprint)
        jid = identity([dataset, spec.describe(), code_hash()])
        with closing(self.connect()) as db, db:
            db.execute('INSERT OR IGNORE INTO jobs VALUES(?,?,?,?,?,?,?,?)',
                       (jid, dataset, json.dumps(spec.describe()), 'QUEUED', now(), now(), None, None))
        return jid

    def claim(self, jid: str) -> bool:
        with closing(self.connect()) as db, db:
            return db.execute("UPDATE jobs SET state='RUNNING',updated=? WHERE id=? AND state='QUEUED'",
                              (now(), jid)).rowcount == 1

    def finish(self, jid: str, result: dict[str, Any] | None = None, error: str | None = None) -> None:
        state = 'FAILED' if error else 'COMPLETED'
        with closing(self.connect()) as db, db:
            db.execute('UPDATE jobs SET state=?,updated=?,result=?,error=? WHERE id=?',
                       (state, now(), dumps(result) if result is not None else None, error, jid))
        self.event(state, error or 'Development walk-forward evaluation recorded', jid)

    def rows(self, dataset: str | None = None) -> list[dict[str, Any]]:
        with closing(self.connect()) as db:
            query = ('SELECT * FROM jobs WHERE dataset=? ORDER BY created,id'
                     if dataset else 'SELECT * FROM jobs ORDER BY created,id')
            rows = db.execute(query, (dataset,) if dataset else ()).fetchall()
        return [{**dict(r), 'spec': json.loads(r['spec']), 'result': json.loads(r['result']) if r['result'] else None}
                for r in rows]

    def snapshot(self) -> dict[str, Any]:
        rows = self.rows()
        with closing(self.connect()) as db:
            events = [dict(r) for r in db.execute('SELECT * FROM events ORDER BY seq DESC LIMIT 40')]
        for row in rows:
            if row['result']:
                row['screen'] = screen(row['result'], len(rows))
        return {'jobs': rows[-100:], 'counts': {s: sum(r['state'] == s for r in rows)
                                               for s in ('QUEUED', 'RUNNING', 'COMPLETED', 'FAILED')},
                'events': events, 'learners': available(), 'no_edge_note': NO_EDGE_NOTE,
                'execution_authority': False, 'promotion_status': 'RESEARCH_ONLY'}


def propose(journal: Journal, dataset: str, budget: int = 24) -> list[str]:
    """Explore all available families, then mutate development winners. Never consult a holdout."""
    prior = journal.rows(dataset)
    specs = [Hypothesis(k) for k, v in available().items() if v['available']
             and not k.startswith('foundation_') and k not in ('tabiclv2', 'kumo', 'mitra')]
    # Fixed matched challengers precede capacity search; every attempt counts in the same journal.
    from services.memory.ablation import MODES
    specs = [Hypothesis('ridge'), *[Hypothesis('ridge', memory_mode=m) for m in MODES],
             *[s for s in specs if s.family != 'ridge']]
    # Expensive pretrained challengers enter after the cheap baseline families.
    specs += [Hypothesis(k) for k, v in available().items() if v['available']
              and (k.startswith('foundation_') or k in ('tabiclv2', 'kumo', 'mitra'))]
    ranked = sorted((r for r in prior if r['result'] and not r['result'].get('is_simulated', True)
                     and r['result'].get('objective', -1e9) > -1e9),
                    key=lambda r: r['result'].get('objective', -1e9), reverse=True)
    for row in ranked[:3]:
        parent = Hypothesis(**row['spec'])
        if parent.family.startswith('kronos_'):
            if row['result'].get('objective', 0) > 0 and row['result'].get('metrics', {}).get('net_return', 0) > 0:
                specs.append(replace(parent, kronos_context=min(512, parent.kronos_context * 2)))
                specs.append(replace(parent, kronos_tune_steps=16))
            continue
        if parent.family.startswith('temporal_'):
            specs.append(replace(parent, pretrain_epochs=2))
            specs.append(replace(parent, meta_steps=4))
            specs.append(replace(parent, epochs=min(16, parent.epochs*2)))
            continue
        for feature in ('signed_log', 'interactions'):
            specs.append(Hypothesis(parent.family, parent.seed, min(parent.capacity * 2, 512),
                                    parent.alpha * 2, feature))
    from ..kronos.adapter import DEFAULT_ROOT, sha
    specs = [replace(s, kronos_artifact_hash=sha(DEFAULT_ROOT / 'manifest.json'))
             if s.family.startswith('kronos_') and not s.kronos_artifact_hash else s for s in specs]
    from ..advanced.foundation import WEIGHTS
    specs = [replace(s, foundation_artifact_hash=sha(WEIGHTS/s.family.removeprefix('foundation_')/'manifest.json'))
             if s.family.startswith('foundation_') or s.family in ('tabiclv2', 'kumo', 'mitra') else s for s in specs]
    seen = {json.dumps(r['spec'], sort_keys=True) for r in prior}
    result = []
    for spec in specs:
        if len(prior) + len(result) >= budget:
            break
        key = json.dumps(spec.describe(), sort_keys=True)
        if key not in seen:
            result.append(journal.enqueue(dataset, spec))
            seen.add(key)
    if result:
        journal.event('HYPOTHESIZE', f'Queued {len(result)} reproducible hypotheses; campaign budget {budget}')
    return result


def prepare_cache(journal: Journal, data: Any, quotes: Any = None, *, book: Any = None) -> str:
    from ..costs import CostModel
    from ..frames import last_clean_segment, prepare_bars
    from ..models.config import SuiteConfig
    from ..models.targets import build_suite_dataset
    cfg = SuiteConfig()
    # Hash actual source content, not just its user-visible dataset name.
    digest = hashlib.sha256(data.bars.astype(str).to_csv(index=False).encode()).hexdigest()
    quote_digest = identity({s: [t.tolist(), v.tolist()] for s, (t, v) in quotes.by_symbol.items()}) if quotes else None
    key = identity([data.dataset_version, digest, quote_digest, data.adjustments, cfg.describe(), code_hash()])
    directory = journal.root / 'cache' / key
    if (directory / 'manifest.json').exists():
        manifest = json.loads((directory / 'manifest.json').read_text())
        if book is not None:
            book.expose(key, manifest['symbols'], manifest['development_start'], manifest['development_end'],
                        is_simulated=manifest.get('is_simulated') is True)
        return key
    journal.event('OBSERVE', f'Preparing {data.dataset_version}; point-in-time features and costs')
    prepared = prepare_bars(data.bars)
    bars, quality = last_clean_segment(prepared)
    if len(bars) != len(prepared):
        raise ValueError('research requires contiguous data; refusing future-dependent segment selection')
    ds = build_suite_dataset(bars, CostModel(), cfg, quotes=quotes)
    plan = plan_walk_forward(ds.decision_ns, ds.label_end_ns, keys=ds.keys, embargo_steps=60, min_unique_times=200)
    idx = plan.dev_idx
    # No frozen-test feature, target or prediction is written to the worker cache.
    from ..advanced.targets import NAMES, multitask_targets
    y = multitask_targets(ds, bars)
    good = np.isfinite(y[idx]).all(axis=1) & np.isfinite(ds.X[idx]).all(axis=1)
    idx = idx[good]
    if len(idx) < 200:
        raise ValueError('insufficient finite development samples')
    directory.mkdir(parents=True, exist_ok=True)
    from ..kronos.data import sequence_arrays
    sequences = sequence_arrays(bars, ds, idx)
    target = directory / 'development.npz'
    with (directory / 'development.tmp').open('wb') as f:
        np.savez_compressed(f, X=ds.X[idx], y=y[idx], decision=ds.decision_ns[idx], end=ds.label_end_ns[idx],
                            exit=ds.exit_ns[60][idx], cost=ds.cost_rt_bps[idx], **sequences)  # pyright: ignore[reportArgumentType]
    os.replace(directory / 'development.tmp', target)
    manifest = {'dataset_version': ds.dataset_version, 'dataset_key': key, 'source_hash': digest,
                'is_simulated': ds.is_simulated, 'bar_minutes': ds.bar_minutes, 'horizon': 60,
                'target_names': NAMES, 'feature_columns': ds.feature_columns, 'symbols': sorted(ds.symbols),
                'vol_spec': vars(ds.vol_spec), 'har_base_reason': ds.har_base_reason,
                'development_start': int(ds.decision_ns[idx].min()), 'development_end': int(ds.label_end_ns[idx].max()),
                'data_quality': quality, 'adjustments': data.adjustments,
                'split': plan.describe(), 'cache_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                'code_hash': code_hash(), 'no_edge_note': NO_EDGE_NOTE,
                'holdout_status': 'UNTOUCHED_NOT_IN_WORKER_CACHE'}
    tmp = directory / 'manifest.tmp'
    tmp.write_text(dumps(manifest))
    os.replace(tmp, directory / 'manifest.json')
    if book is not None:
        book.expose(key, ds.symbols, manifest['development_start'], manifest['development_end'],
                    is_simulated=ds.is_simulated)
    journal.event('ANALYZE', f'Cached {len(idx)} development rows; final test excluded')
    return key


def economic_metrics(returns: np.ndarray) -> dict[str, float | None]:
    r = np.asarray(returns, dtype=float) / 10000
    if not len(r):
        return {'max_drawdown': 0.0, 'tail_mean_5pct': 0.0, 'sortino_per_trade': None}
    equity = np.r_[1.0, np.cumprod(1 + r)]
    drawdown = 1 - equity / np.maximum.accumulate(equity)
    downside = np.sqrt(np.mean(np.minimum(r, 0) ** 2))
    return {'max_drawdown': float(drawdown.max()),
            'tail_mean_5pct': float(np.sort(r)[:max(1, math.ceil(len(r) * .05))].mean()),
            'sortino_per_trade': float(r.mean() / downside) if downside else None}


def evaluate(root: Path, jid: str) -> dict[str, Any]:
    journal = Journal(root)
    rows = [r for r in journal.rows() if r['id'] == jid]
    if not rows:
        raise ValueError('unknown job')
    row = rows[0]
    spec = Hypothesis(**row['spec'])
    directory = root / 'cache' / row['dataset']
    manifest = json.loads((directory / 'manifest.json').read_text())
    path = directory / 'development.npz'
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['cache_sha256'] or code_hash() != manifest['code_hash']:
        raise ValueError('cache/source integrity mismatch')
    with np.load(path, allow_pickle=False) as archive:
        a = {k: archive[k] for k in archive.files}
    if spec.family.startswith('foundation_') or spec.family in ('tabiclv2', 'kumo', 'mitra'):
        from ..advanced.integration import evaluate_foundation
        return evaluate_foundation(root, jid, spec, directory, manifest)
    if spec.family.startswith('kronos_'):
        from ..kronos.benchmark import evaluate_challenger
        return evaluate_challenger(root, jid, spec, a, manifest)
    # An inner reserved block is still development data; never interpreted as independent confirmation.
    plan = plan_walk_forward(a['decision'], a['end'], embargo_steps=60, min_unique_times=200)
    from ..advanced.inputs import model_inputs
    def inputs(indices):
        return model_inputs(spec, a, indices)
    pooled, stressed, baseline = StrategyStats(), StrategyStats(), StrategyStats()
    folds, predictions = [], []
    daily: dict[int, float] = {}
    base_daily: dict[int, float] = {}
    for fold in plan.folds:
        journal.event('TRAIN', f"{spec.family}: fold {fold.fold + 1}/{len(plan.folds)}", jid)
        model = fit(spec, inputs(fold.fit_idx), a['y'][fold.fit_idx])
        cal = fold.calib_idx
        if len(cal) < 20:
            raise ValueError('insufficient purged uncertainty calibration block')
        # One-sided conservative residual allowance, estimated without test labels.
        from ..advanced.calibration import CalibratedTargets
        model = CalibratedTargets(model).calibrate(inputs(cal), a['y'][cal])
        residual = model.predict(inputs(cal))[:, 0] - a['y'][cal, 0]
        allowance = max(0.0, float(np.quantile(residual, .9)))
        t = fold.test_idx
        pred = np.asarray(model.predict(inputs(t)))
        if pred.shape != (len(t), a['y'].shape[1]) or not np.isfinite(pred).all():
            raise ValueError('invalid predictions')
        score = pred[:, 0] - allowance
        if spec.memory_mode:
            from services.memory.ablation import EpisodicAdjustment
            if spec.family != 'ridge':
                raise ValueError('memory ablations require the matched ridge baseline')
            adjustment = EpisodicAdjustment(spec.memory_mode).fit(
                a['X'][fold.fit_idx], a['X'][cal], -residual, a['end'][cal], a['decision'][t].min())
            score = score + adjustment.adjustment(a['X'][t])
        stats = StrategyStats()
        args = (a['decision'][t], score, a['exit'][t], a['y'][t, 0], manifest['bar_minutes'])
        taken = evaluate_score(*args, stats)
        pooled.add(a['y'][t[taken], 0], stats.span_minutes, len(t))
        stressed.add(a['y'][t[taken], 0] - a['cost'][t[taken]], stats.span_minutes, len(t))
        # Same execution convention for a simple momentum comparator.
        momentum = a['X'][t, manifest['feature_columns'].index('ret_60')]
        base_taken = evaluate_score(a['decision'][t], momentum, a['exit'][t], a['y'][t, 0],
                                    manifest['bar_minutes'], baseline)
        days = a['decision'][t] // 86_400_000_000_000
        for day in np.unique(days):
            daily.setdefault(int(day), 0.0)
            base_daily.setdefault(int(day), 0.0)
        for collection, trades in ((daily, taken), (base_daily, base_taken)):
            for trade in trades:
                collection[int(days[trade])] += float(a['y'][t[trade], 0])
        from ..advanced.targets import probability_metrics
        folds.append({'probability_calibration': probability_metrics(pred, a['y'][t]), 'fold': fold.fold,
                      **stats.metrics(60), 'uncertainty_allowance_bps': allowance,
                      'trade_rows': t[taken].tolist(),
                      'target_mae': np.abs(pred - a['y'][t]).mean(axis=0).tolist()})
        predictions.extend({'row': int(i), 'prediction': p.tolist(), 'score': float(s)}
                           for i, p, s in zip(t, pred, score, strict=True))
    metrics = pooled.metrics(60)
    sharpe = metrics.get('net_sharpe') or 0.0
    risk = economic_metrics(np.asarray(pooled.net_bps))
    result = {'metrics': metrics, 'stress_2x_costs': stressed.metrics(60), 'momentum_baseline': baseline.metrics(60),
              'risk': risk, 'folds': folds,
              'objective': float(sharpe - 10 * (risk['max_drawdown'] or 0)) if metrics['n_trades'] >= 20 else -1e9,
              'sessions': len(daily), 'p_positive': block_p_value(np.array(list(daily.values()))),
              'p_incremental': block_p_value(np.array([daily[d] - base_daily[d] for d in sorted(daily)])),
              'is_simulated': manifest['is_simulated'], 'promotion_eligible': False,
              'validation': 'DEVELOPMENT_ONLY', 'code_hash': manifest['code_hash'],
              'no_edge_note': NO_EDGE_NOTE}
    if spec.memory_mode:
        result['memory_ablation'] = spec.memory_mode
        result['promotion_blockers'] = ['Memory challenger requires paired baseline confirmation; research-only']
    result['daily_net_bps'] = {str(day): value for day, value in sorted(daily.items())}
    artifacts = root / 'runs' / jid
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / 'predictions.json').write_text(dumps(predictions))
    result['prediction_sha256'] = hashlib.sha256((artifacts / 'predictions.json').read_bytes()).hexdigest()
    (artifacts / 'result.json').write_text(dumps(result))
    return result


def snapshot(root: Path) -> dict[str, Any]:
    if not (root / 'research.sqlite3').exists():
        return {'jobs': [], 'events': [], 'counts': {}, 'promotion_status': 'NOT_STARTED', 'execution_authority': False}
    result = Journal(root).snapshot()
    if (root.parent / 'autonomous-validation.sqlite3').exists():
        from .validation import ValidationBook
        result['paper'] = ValidationBook(root.parent).status()
        result['promotion_status'] = 'PAPER_ACTIVE' if result['paper']['active'] else 'NO_PROMOTION'
    return result
