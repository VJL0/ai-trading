"""Independent confirmation and reversible PAPER-only promotion bookkeeping.

The application-wide book is outside individual campaign directories. Interval claims are
symbol/time based, independent of dataset aliases, model families and output directories.
Claims commit BEFORE holdout evaluation; crashes consume an interval too.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import ExitStack, closing
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from ..costs import CostModel
from ..frames import last_clean_segment, prepare_bars
from ..ml import _registry_lock
from ..models.config import SuiteConfig
from ..models.strategy import StrategyStats, evaluate_score
from ..models.targets import build_suite_dataset
from ..provenance import dumps
from ..splits import holdout_split
from .artifacts import PortableModel, export_model
from .engine import Journal, code_hash, economic_metrics, evaluate, identity, now, prepare_cache
from .evidence import block_p_value, screen
from .learners import Hypothesis, fit, transform


class ValidationBook:
    def __init__(self, experiments: Path) -> None:
        self.experiments = experiments.resolve()
        self.path = self.experiments / 'autonomous-validation.sqlite3'
        self.experiments.mkdir(parents=True, exist_ok=True)
        with closing(self.connect()) as db, db:
            db.executescript('''
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS exposures (
                    owner TEXT NOT NULL, symbol TEXT NOT NULL, lo INTEGER NOT NULL, hi INTEGER NOT NULL,
                    kind TEXT NOT NULL, PRIMARY KEY(owner,symbol,lo,hi,kind));
                CREATE TABLE IF NOT EXISTS confirmations (
                    id INTEGER PRIMARY KEY, job TEXT UNIQUE NOT NULL, created TEXT NOT NULL,
                    state TEXT NOT NULL, candidate TEXT NOT NULL, report TEXT);
                CREATE TABLE IF NOT EXISTS paper_state (singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                    active INTEGER, previous INTEGER, updated TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS transitions (id INTEGER PRIMARY KEY, at TEXT NOT NULL,
                    previous INTEGER, active INTEGER, reason TEXT NOT NULL);
            ''')

    def connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        return db

    def expose(self, owner: str, symbols: list[str], lo: int, hi: int, *, is_simulated: bool = False) -> None:
        if not symbols or lo > hi:
            raise ValueError('invalid development exposure')
        kind = 'DEVELOPMENT_SIMULATED' if is_simulated else 'DEVELOPMENT'
        with closing(self.connect()) as db, db:
            db.executemany('INSERT OR IGNORE INTO exposures VALUES(?,?,?,?,?)',
                           [(owner, s, lo, hi, kind) for s in sorted(set(symbols))])

    def _legacy_check(self, symbols: list[str], lo: int, hi: int) -> None:
        for family in ('models', 'ml'):
            path = self.experiments / family / 'frozen_test_registry.json'
            if not path.exists():
                continue
            entries = json.loads(path.read_text())
            if not isinstance(entries, dict):
                raise ValueError('unreadable legacy holdout registry')
            for entry in entries.values():
                interval = entry.get('interval_ns')
                if not isinstance(interval, list) or len(interval) != 2:
                    raise ValueError('legacy evaluation has unknown interval; cannot establish untouched data')
                if interval[0] <= hi and interval[1] >= lo and set(symbols).intersection(entry.get('symbols', symbols)):
                    raise ValueError('holdout overlaps an existing model-suite/ML evaluation')

    def import_development_history(self, symbols: list[str]) -> None:
        for path in (self.experiments / 'autonomous' / 'cache').glob('*/manifest.json'):
            metadata = json.loads(path.read_text())
            if metadata.get('is_simulated') is True:
                # Repair early journal entries whose known synthetic provenance was not distinguished.
                with closing(self.connect()) as db, db:
                    db.execute("UPDATE OR IGNORE exposures SET kind='DEVELOPMENT_SIMULATED'"
                               " WHERE owner=? AND kind='DEVELOPMENT'", (path.parent.name,))
                continue
            with np.load(path.parent / 'development.npz', allow_pickle=False) as cache:
                lo, hi = int(cache['decision'].min()), int(cache['end'].max())
            self.expose(path.parent.name, metadata.get('symbols') or symbols, lo, hi)

    def reserve(self, job: str, candidate: dict[str, Any], symbols: list[str], lo: int, hi: int) -> int:
        if not symbols or lo > hi:
            raise ValueError('invalid holdout interval')
        with ExitStack() as locks:
            for family in ('ml', 'models'):
                locks.enter_context(_registry_lock(self.experiments / family / 'frozen_test_registry.json'))
            self.import_development_history(symbols)
            self._legacy_check(symbols, lo, hi)
            with closing(self.connect()) as db, db:
                db.execute('BEGIN IMMEDIATE')
                for symbol in symbols:
                    if db.execute("SELECT 1 FROM exposures WHERE symbol=? AND lo<=? AND hi>=?"
                                  " AND kind!='DEVELOPMENT_SIMULATED' LIMIT 1",
                                  (symbol, hi, lo)).fetchone():
                        raise ValueError('holdout overlaps prior research exposure or confirmation')
                cursor = db.execute('INSERT INTO confirmations(job,created,state,candidate) VALUES(?,?,?,?)',
                                    (job, now(), 'RESERVED', dumps(candidate)))
                cid = int(cursor.lastrowid or 0)
                db.executemany('INSERT INTO exposures VALUES(?,?,?,?,?)',
                               [(job, s, lo, hi, 'CONFIRMATION') for s in sorted(set(symbols))])
                return cid

    def complete(self, cid: int, report: dict[str, Any]) -> None:
        state = 'CONFIRMED' if report.get('passed') is True else 'REJECTED'
        with closing(self.connect()) as db, db:
            if db.execute("UPDATE confirmations SET state=?,report=? WHERE id=? AND state='RESERVED'",
                          (state, dumps(report), cid)).rowcount != 1:
                raise ValueError('confirmation already completed or missing')

    def promote(self, cid: int) -> None:
        with closing(self.connect()) as db, db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT * FROM confirmations WHERE id=?', (cid,)).fetchone()
            if row is None or row['state'] != 'CONFIRMED':
                raise ValueError('only independently confirmed candidates can enter PAPER')
            report = json.loads(row['report'])
            if report.get('is_simulated') is not False or report.get('code_hash') != code_hash():
                raise ValueError('synthetic/unknown provenance or changed source')
            if datetime.fromisoformat(report['expires_at']) <= datetime.now(UTC):
                raise ValueError('expired confirmation')
            PortableModel(Path(report['artifact_dir']), report['files'])
            current = db.execute('SELECT active FROM paper_state WHERE singleton=1').fetchone()
            previous = current['active'] if current else None
            if previous == cid:
                return
            db.execute('INSERT OR REPLACE INTO paper_state VALUES(1,?,?,?)', (cid, previous, now()))
            db.execute('INSERT INTO transitions(at,previous,active,reason) VALUES(?,?,?,?)',
                       (now(), previous, cid, 'independent validation passed; PAPER only'))

    def rollback(self, reason: str, expected_id: int | None = None) -> None:
        """Rollback to a previously confirmed candidate or NO_TRADE. Never expand authorization."""
        with closing(self.connect()) as db, db:
            db.execute('BEGIN IMMEDIATE')
            current = db.execute('SELECT * FROM paper_state WHERE singleton=1').fetchone()
            if not current or (expected_id is not None and current['active'] != expected_id):
                return
            db.execute('UPDATE paper_state SET active=?,previous=NULL,updated=? WHERE singleton=1',
                       (current['previous'], now()))
            db.execute('INSERT INTO transitions(at,previous,active,reason) VALUES(?,?,?,?)',
                       (now(), current['active'], current['previous'], reason))

    def active(self) -> dict[str, Any] | None:
        with closing(self.connect()) as db:
            row = db.execute('SELECT c.* FROM confirmations c JOIN paper_state p ON p.active=c.id WHERE p.singleton=1').fetchone()
        if row is None:
            return None
        return {**dict(row), 'candidate': json.loads(row['candidate']), 'report': json.loads(row['report'])}

    def status(self) -> dict[str, Any]:
        with closing(self.connect()) as db:
            states = [dict(r) for r in db.execute('SELECT id,job,created,state FROM confirmations ORDER BY id DESC LIMIT 30')]
            transitions = [dict(r) for r in db.execute('SELECT * FROM transitions ORDER BY id DESC LIMIT 20')]
        active = self.active()
        return {'active': active['id'] if active else None, 'confirmations': states, 'transitions': transitions,
                'execution_authority': False, 'mode': 'PAPER_ONLY'}


def confirm(root: Path, jid: str, data: Any, quotes: Any, book: ValidationBook) -> dict[str, Any]:
    journal = Journal(root)
    rows = journal.rows()
    row = next(r for r in rows if r['id'] == jid)
    result = row['result']
    eligibility = screen(result or {}, len(rows))
    if row['state'] != 'COMPLETED' or not eligibility['eligible_for_independent_validation']:
        return {'passed': False, 'stage': 'SCREEN', **eligibility}
    if prepare_cache(journal, data, quotes) != row['dataset']:
        raise ValueError('candidate data/source lineage changed')
    manifest = json.loads((root / 'cache' / row['dataset'] / 'manifest.json').read_text())
    original = json.loads((root / 'runs' / jid / 'predictions.json').read_text())
    reproduced = evaluate(root, jid)
    repeated = json.loads((root / 'runs' / jid / 'predictions.json').read_text())
    # Independent process refit must reproduce trades as well as scores. Neural numeric tolerance is explicit.
    if [p['row'] for p in original] != [p['row'] for p in repeated]:
        raise ValueError('reproduction row mismatch')
    a, b = np.array([p['score'] for p in original]), np.array([p['score'] for p in repeated])
    if not np.allclose(a, b, rtol=1e-5, atol=1e-4) or not np.array_equal(a > 0, b > 0):
        raise ValueError('independent reproduction failed')
    if [f.get('trade_rows') for f in result['folds']] != [f.get('trade_rows') for f in reproduced['folds']]:
        raise ValueError('independent reproduction changed trade selection')
    if not screen(reproduced, len(rows))['eligible_for_independent_validation']:
        raise ValueError('reproduced candidate failed development screen')
    boundary = manifest['split']['final_test_start_ns']
    symbols = sorted(set(data.bars['symbol'].astype(str)))
    hi = int(data.bars['available_at_ts'].max().value)
    spec = Hypothesis(**row['spec'])
    candidate = {'job': jid, 'spec': row['spec'], 'dataset_key': row['dataset'], 'code_hash': code_hash(),
                 'holdout_start': boundary, 'holdout_end': hi, 'symbols': symbols, 'screen': eligibility}
    age = datetime.now(UTC).timestamp() - hi / 1e9
    if age < 0 or age > 4 * 86400:
        raise ValueError('confirmation requires recent completed market data')
    cid = book.reserve(jid, candidate, symbols, boundary, hi)
    journal.event('VALIDATE', f'Confirmation {cid} reserved before opening final interval', jid)
    completed = False
    try:
        prepared = prepare_bars(data.bars)
        bars, _ = last_clean_segment(prepared)
        if len(bars) != len(prepared):
            raise ValueError('confirmation data requires contiguous history')
        ds = build_suite_dataset(bars, CostModel(), SuiteConfig(), quotes=quotes)
        if ds.is_simulated:
            raise ValueError('synthetic data cannot confirm a model')
        with np.load(root / 'cache' / row['dataset'] / 'development.npz', allow_pickle=False) as cache:
            dev = {k: cache[k] for k in cache.files}
        train, calib, _, _ = holdout_split(np.arange(len(dev['X'])), dev['decision'], dev['end'],
                                           np.unique(dev['decision']), .25, 60)
        from ..advanced.inputs import model_inputs
        model = fit(spec, model_inputs(spec, dev, train), dev['y'][train])
        from ..advanced.calibration import CalibratedTargets
        model = CalibratedTargets(model).calibrate(model_inputs(spec, dev, calib), dev['y'][calib])
        residual = model.predict(model_inputs(spec, dev, calib))[:, 0] - dev['y'][calib, 0]
        allowance = max(0.0, float(np.quantile(residual, .9)))
        artifact_dir = book.experiments / 'autonomous-artifacts' / str(cid)
        files = export_model(model, artifact_dir)
        portable = PortableModel(artifact_dir, files)
        probe = model_inputs(spec, dev, calib[:100])
        if not np.allclose(model.predict(probe), portable.predict(probe), rtol=1e-4, atol=1e-4):
            raise ValueError('serialized inference differs from training model')
        test = np.flatnonzero(ds.decision_ns >= boundary)
        if spec.family.startswith('temporal_'):
            from ..kronos.data import sequence_arrays
            holdout = sequence_arrays(bars, ds, test)
            holdout.update(X=ds.X[test], decision=ds.decision_ns[test])
            inputs = model_inputs(spec, holdout, np.arange(len(test)))
        else:
            inputs = transform(ds.X[test], spec.transform)
        pred = portable.predict(inputs)[:, 0] - allowance
        stats, baseline = StrategyStats(), StrategyStats()
        taken = evaluate_score(ds.decision_ns[test], pred, ds.exit_ns[60][test], ds.net_bps[60][test], ds.bar_minutes, stats)
        base_taken = evaluate_score(ds.decision_ns[test], ds.X[test, ds.feature_columns.index('ret_60')],
                                    ds.exit_ns[60][test], ds.net_bps[60][test], ds.bar_minutes, baseline)
        days = ds.decision_ns[test] // 86_400_000_000_000
        unique = np.unique(days)
        daily = np.array([ds.net_bps[60][test[taken]][days[taken] == day].sum() for day in unique])
        base_daily = np.array([ds.net_bps[60][test[base_taken]][days[base_taken] == day].sum() for day in unique])
        p_positive, p_incremental = block_p_value(daily), block_p_value(daily - base_daily)
        # Summable alpha spending across every confirmation, including crashes and rejected attempts.
        alpha = .05 / (2 * cid * (cid + 1))
        risk = economic_metrics(np.array(stats.net_bps))
        stressed = float((ds.net_bps[60][test[taken]] - ds.cost_rt_bps[test[taken]]).sum())
        passed = (len(unique) >= 60 and len(taken) >= 50 and p_positive <= alpha and p_incremental <= alpha
                  and stressed > 0 and (risk['max_drawdown'] or 0) <= .15)
        report = {'passed': passed, 'is_simulated': False, 'metrics': stats.metrics(60), 'risk': risk,
                  'sessions': len(unique), 'p_positive': p_positive, 'p_incremental': p_incremental, 'alpha': alpha,
                  'artifact_dir': str(artifact_dir), 'files': files, 'code_hash': code_hash(),
                  'allowance_bps': allowance, 'feature_columns': list(ds.feature_columns),
                  'training_mean': dev['X'][train].mean(axis=0).tolist(),
                  'training_std': np.maximum(dev['X'][train].std(axis=0), 1e-8).tolist(),
                  'expires_at': (datetime.now(UTC) + timedelta(days=7)).isoformat(),
                  'provenance_digest': identity(candidate), 'execution_authority': False}
        book.complete(cid, report)
        completed = True
        if passed:
            book.promote(cid)
            journal.event('PROMOTE', f'Confirmation {cid} activated for PAPER signals only', jid)
        else:
            journal.event('REJECT', f'Confirmation {cid} did not meet preregistered evidence requirements', jid)
        return report
    except Exception as exc:
        if not completed:
            book.complete(cid, {'passed': False, 'error': f'{type(exc).__name__}: {exc}'})
        raise
