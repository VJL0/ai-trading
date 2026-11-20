from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from services.research.autonomous.engine import Journal, code_hash, economic_metrics, evaluate, prepare_cache, propose
from services.research.autonomous.learners import Hypothesis, available, fit, transform
from services.research.synthetic import make_synthetic_bars


def test_atomic_claim_and_restart_deduplication(tmp_path):
    journal = Journal(tmp_path)
    jid = journal.enqueue('data', Hypothesis('ridge'))
    assert journal.enqueue('data', Hypothesis('ridge')) == jid
    with ThreadPoolExecutor(8) as pool:
        claimed = list(pool.map(lambda _: Journal(tmp_path).claim(jid), range(16)))
    assert sum(claimed) == 1
    journal.finish(jid, error='interrupted')
    assert Journal(tmp_path).rows()[0]['state'] == 'FAILED'
    assert not Journal(tmp_path).claim(jid)


def test_budget_and_missing_dependencies(tmp_path):
    journal = Journal(tmp_path)
    assert len(propose(journal, 'dataset', budget=2)) == 2
    assert propose(journal, 'dataset', budget=2) == []
    with pytest.raises(ValueError, match='unregistered'):
        fit(Hypothesis('arbitrary_python'), np.ones((100, 3)), np.ones((100, 3)))


@pytest.mark.parametrize('family', ['ridge', 'bayesian', 'extra_trees', 'lightgbm', 'catboost', 'online', 'regime', 'pca'])
def test_learners_finite_multi_task_predictions(family):
    if not available()[family]['available']:
        pytest.skip('optional dependency absent')
    rng = np.random.default_rng(91)
    X = rng.normal(size=(120, 5))
    y = np.column_stack((X[:, 0] * 2, X[:, 1] * .01, X[:, 2] + 3))
    model = fit(Hypothesis(family, capacity=8), X[:100], y[:100])
    prediction = model.predict(X[100:])
    assert prediction.shape == (20, 3)
    assert np.isfinite(prediction).all()


def test_feature_grammar_has_no_future_or_mutation():
    x = np.arange(30, dtype=float).reshape(10, 3)
    for name in ('identity', 'signed_log', 'interactions'):
        full = transform(x, name)
        np.testing.assert_array_equal(full[:5], transform(x[:5], name))
    with pytest.raises(ValueError):
        transform(x, 'eval')


def test_cache_excludes_entire_outer_holdout(tmp_path):
    bars = make_synthetic_bars(('AAA', 'BBB'), timeframe='1Min', n_sessions=12)
    data = SimpleNamespace(bars=bars, dataset_version='test', adjustments={})
    key = prepare_cache(Journal(tmp_path), data)
    cache = tmp_path / 'cache' / key
    manifest = json.loads((cache / 'manifest.json').read_text())
    with np.load(cache / 'development.npz', allow_pickle=False) as a:
        boundary = manifest['split']['final_test_start_ns']
        assert a['decision'].max() < boundary
        assert a['end'].max() < boundary
    assert manifest['is_simulated'] is True
    assert not list(tmp_path.rglob('*frozen_test_registry*'))


def make_cache(root: Path) -> str:
    rng = np.random.default_rng(45)
    n = 2400
    x = rng.normal(size=(n, 5))
    decision = np.arange(n, dtype=np.int64) * 60_000_000_000
    y = rng.normal(size=(n, 3))
    cache = root / 'cache' / 'numeric'
    cache.mkdir(parents=True)
    path = cache / 'development.npz'
    np.savez(path, X=x, y=y, decision=decision, end=decision + 60_000_000_000,
             exit=decision + 60_000_000_000, cost=np.ones(n))
    manifest = {'cache_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'code_hash': code_hash(),
                'bar_minutes': 1, 'feature_columns': ['ret_60', 'b', 'c', 'd', 'e'], 'is_simulated': True}
    (cache / 'manifest.json').write_text(json.dumps(manifest))
    return 'numeric'


def test_real_walkforward_reproducible_and_never_promotes(tmp_path):
    journal = Journal(tmp_path)
    key = make_cache(tmp_path)
    jid = journal.enqueue(key, Hypothesis('ridge'))
    first = evaluate(tmp_path, jid)
    second = evaluate(tmp_path, jid)
    assert first == second
    assert first['promotion_eligible'] is False
    assert first['validation'] == 'DEVELOPMENT_ONLY'
    assert len(first['folds']) == 4
    with (tmp_path / 'cache' / key / 'development.npz').open('ab') as file:
        file.write(b'corrupt')
    with pytest.raises(ValueError, match='integrity'):
        evaluate(tmp_path, jid)


def test_drawdown_includes_initial_capital():
    result = economic_metrics(np.array([-1000., 100.]))
    assert result['max_drawdown'] == pytest.approx(.1)
    assert result['tail_mean_5pct'] == pytest.approx(-.1)


def test_multiple_testing_gate_never_accepts_synthetic_or_sparse_results():
    from services.research.autonomous.evidence import block_p_value, screen
    strong = {'is_simulated': False, 'metrics': {'n_trades': 200}, 'sessions': 150,
              'p_positive': .0005, 'p_incremental': .0005, 'stress_2x_costs': {'net_return': .1},
              'folds': [{'net_return': .1}] * 4, 'risk': {'max_drawdown': .05}}
    assert screen(strong, 10)['eligible_for_independent_validation']
    assert not screen(strong, 100)['eligible_for_independent_validation']
    assert not screen(strong | {'is_simulated': True}, 1)['eligible_for_independent_validation']
    assert not screen(strong | {'sessions': 5}, 1)['eligible_for_independent_validation']
    assert block_p_value(np.ones(10)) == 1.0
    assert block_p_value(np.full(100, np.nan)) == 1.0
    assert block_p_value(-np.ones(100)) == 1.0


def test_worker_failure_is_recorded_and_secrets_are_not_forwarded(tmp_path, monkeypatch):
    import subprocess

    from services.research.autonomous.__main__ import run_batch
    journal = Journal(tmp_path)
    jid = journal.enqueue('missing-cache', Hypothesis('ridge'))
    monkeypatch.setenv('ALPACA_SECRET_KEY', 'must-not-reach-worker')
    real_popen = subprocess.Popen
    captured = []

    def popen(*args, **kwargs):
        if 'env' in kwargs:
            captured.append(kwargs['env'])
        return real_popen(*args, **kwargs)

    monkeypatch.setattr(subprocess, 'Popen', popen)
    with (tmp_path / 'test.lock').open('w') as lock:
        run_batch(journal, 'missing-cache', 1, 30, 6, lock.fileno())
    assert journal.rows()[0]['state'] == 'FAILED'
    assert journal.rows()[0]['error']
    assert all('ALPACA_SECRET_KEY' not in env for env in captured)
    assert (tmp_path / 'logs' / f'{jid}.log').exists()


def test_timeout_is_a_persisted_failure(tmp_path):
    from services.research.autonomous.__main__ import run_batch
    journal = Journal(tmp_path)
    journal.enqueue('missing-cache', Hypothesis('ridge'))
    with (tmp_path / 'test.lock').open('w') as lock:
        run_batch(journal, 'missing-cache', 1, .000001, 6, lock.fileno())
    assert journal.rows()[0]['state'] == 'FAILED'
    assert 'wall-clock budget' in journal.rows()[0]['error']


def test_synthetic_success_does_not_trigger_further_alpha_search(tmp_path):
    journal = Journal(tmp_path)
    propose(journal, 'synthetic', budget=100)
    for row in journal.rows():
        journal.finish(row['id'], {'objective': 100.0, 'is_simulated': True})
    assert propose(journal, 'synthetic', budget=100) == []


def test_history_selection_resists_short_batches_and_synthetic_replacement():
    from services.research.autonomous.datasets import pick_history

    def manifest(name, start, end, **extra):
        return dict(dataset_version=name, start=start + 'T00:00:00Z', end=end + 'T00:00:00Z',
                    timeframes=['1Min'], is_synthetic=False, **extra)

    history = manifest('history', '2024-01-01', '2026-01-01')
    fragment = manifest('batch', '2026-01-01', '2026-01-02')
    synthetic = {**history, 'dataset_version': 'synthetic', 'is_synthetic': True, 'end': fragment['end']}
    rows = [history, fragment, synthetic, {**fragment, 'start': None}]
    store = SimpleNamespace(list_manifests=lambda **kwargs: rows)
    assert pick_history(store)['dataset_version'] == 'history'
    refreshed = manifest('refreshed', '2024-02-01', '2026-02-01')
    rows.append(refreshed)
    assert pick_history(store)['dataset_version'] == 'refreshed'
    rows[:] = [fragment, synthetic]
    assert pick_history(store)['dataset_version'] == 'batch'
    rows[:] = [synthetic]
    assert pick_history(store)['dataset_version'] == 'synthetic'
    rows[:] = [{**history, 'kind': 'lean_sample_import'}, {**history, 'defects_injected': ['bad']}]
    assert pick_history(store) is None
