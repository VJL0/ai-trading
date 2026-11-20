import json
import sqlite3
from datetime import UTC, datetime

import numpy as np
import pytest

from packages.core.db import Database
from services.memory.ablation import MODES, EpisodicAdjustment
from services.memory.store import Memory, encode
from services.memory.worker import import_research


def test_transactional_capture_and_recovery(tmp_path):
    with Database(tmp_path / 'operations.sqlite3') as db:
        def insert(c):
            c.execute("INSERT INTO proposals VALUES('p','c','2026-01-01T00:00:00Z','NO_TRADE','test','SPY',?)",
                      (json.dumps({'reason': 'negative net edge', 'expected_return_bps': -2}),))
        def rollback(c):
            insert(c)
            raise ValueError('rollback')
        with pytest.raises(ValueError, match='rollback'):
            db.write(rollback)
        assert db.query('SELECT * FROM memory_outbox') == []
        db.write(insert)
        memory = Memory(tmp_path / 'memory.sqlite3')
        assert memory.sync(db) == 1
        with memory.connect() as c:
            c.execute('DELETE FROM cursors')  # simulate crash before checkpoint
        assert memory.sync(db) == 1
        result = memory.search('negative edge', asof='2099-01-01T00:00:00Z')
        assert len(result['hits']) == 1
        assert 'NO_TRADE' in result['hits'][0]['text']
        assert memory.search('negative', asof='2025-01-01T00:00:00Z')['hits'] == []
        with pytest.raises(sqlite3.IntegrityError):
            db.write(lambda c: c.execute("DELETE FROM memory_outbox"))


def test_point_in_time_hybrid_and_immutable(tmp_path):
    memory = Memory(tmp_path / 'memory.sqlite3')
    for source, symbol, at, state in [('a','SPY','2026-01-01T00:00:00Z', 0.),
                                     ('b','SPY','2027-01-01T00:00:00Z', 0.),
                                     ('c','QQQ','2026-01-01T00:00:00Z', 4.)]:
        memory.append(source='test', source_id=source, body={'reason': 'volatility reversal'},
                      available_at=at, symbol=symbol, state={'vol': state})
    hits = memory.search('volatility " OR *', asof='2026-06-01T00:00:00Z', symbol='SPY', state={'vol': 0})['hits']
    assert len(hits) == 1 and hits[0]['source_id'] == 'a'
    assert hits[0]['untrusted_data']
    with memory.connect() as c, pytest.raises(sqlite3.IntegrityError):
        c.execute("UPDATE memories SET status='VALIDATED'")
    with pytest.raises(ValueError, match='validation'):
        memory.append(source='x', source_id='x', body={}, available_at='2026-01-01T00:00:00Z', status='VALIDATED')
    with pytest.raises(ValueError, match='hypothesis'):
        memory.append(source='x', source_id='x', body={}, available_at='2026-01-01T00:00:00Z', kind='reflection')
    assert np.array_equal(encode('reversal volatility'), encode('reversal volatility'))


def test_timezone_and_backfill_availability(tmp_path):
    memory = Memory(tmp_path / 'memory.sqlite3')
    memory.append(source='x', source_id='1', body={}, available_at='2026-01-01T09:00:00-05:00')
    assert not memory.search('', asof='2026-01-01T13:59:00Z')['hits']
    assert memory.search('', asof='2026-01-01T14:00:00Z')['hits']
    with pytest.raises(ValueError, match='timezone'):
        memory.search('', asof='2026-01-01')


@pytest.mark.parametrize('mode', MODES)
def test_ablation_training_only_and_repeatable(mode):
    rng = np.random.default_rng(5)
    x = rng.normal(size=(100, 4))
    cal = rng.normal(size=(32, 4))
    residual = np.full(32, -2.)
    fit = EpisodicAdjustment(mode).fit(x, cal, residual, np.arange(32), 33)
    assert np.allclose(fit.adjustment(x[:5]), -1.)
    with pytest.raises(ValueError, match='unavailable'):
        EpisodicAdjustment(mode).fit(x, cal, residual, np.arange(32), 31)
    assert np.allclose(fit.adjustment(x[:5]), -1.)


def test_research_failure_reflection_is_not_fact(tmp_path):
    from services.research.autonomous.engine import Journal
    from services.research.autonomous.learners import Hypothesis
    journal = Journal(tmp_path / 'research')
    jid = journal.enqueue('dev', Hypothesis('ridge'))
    journal.finish(jid, error='no usable training data')
    memory = Memory(tmp_path / 'memory.sqlite3')
    import_research(memory, journal.root)
    import_research(memory, journal.root)
    hits = memory.search('failure', asof=datetime.now(UTC).isoformat(), kind='reflection')['hits']
    assert len(hits) == 1 and hits[0]['status'] == 'HYPOTHESIS'
    assert json.loads(hits[0]['text'])['established_fact'] is False


async def test_shared_read_only_tool(tmp_path):
    from services.tools.handlers import dispatch
    from services.tools.registry import TOOL_REGISTRY
    from tests.unit.tools.fake_runtime import FakeToolRuntime
    rt = FakeToolRuntime()
    with Database(tmp_path / 'db.sqlite3') as db:
        rt.db = db
        result = await dispatch(rt, 'search_memory', {'query': 'edge'}, actor='tester')
        assert result['memory']['execution_authority'] is False
        assert TOOL_REGISTRY['search_memory'].read_only


def test_observed_outcome_keeps_counterfactual_and_execution_separate(tmp_path):
    from datetime import timedelta

    from services.memory.outcomes import observe_outcomes
    from tests.unit.ai.conftest import T0, make_signal, make_snapshot
    sig = make_signal()
    first = make_snapshot(snapshot_id='start')
    last = make_snapshot(at=T0 + timedelta(minutes=30), snapshot_id='end')
    memory = Memory(tmp_path / 'memory.sqlite3')
    with Database(tmp_path / 'ledger.sqlite3') as db:
        db.write(lambda c: c.execute('INSERT INTO quant_signals VALUES(?,?,?,?,?,?,?,?,?)',
                 (sig.signal_id, 'c', sig.strategy_id, sig.symbol, sig.generated_at.isoformat(),
                  1, str(sig.net_edge_bps), sig.dataset_version, sig.model_dump_json())))
        for snap in (first, last):
            db.write(lambda c, snap=snap: c.execute('INSERT INTO market_snapshots VALUES(?,?,?,?,?,?)',
                     (snap.snapshot_id, snap.symbol, snap.data_asof.isoformat(), snap.feed.value,
                      int(snap.is_simulated), snap.model_dump_json())))
        assert observe_outcomes(db, memory) == 1
        assert observe_outcomes(db, memory) == 0
        hit = memory.search('', asof='2099-01-01T00:00:00Z')['hits'][0]
        outcome = json.loads(hit['text'])
        assert outcome['measurement'] == 'MID_MARK_COUNTERFACTUAL_NOT_EXECUTED_PNL'
        assert float(outcome['prediction_error_bps']) == -18.
        assert float(outcome['counterfactual_net_bps']) == -2.1
        assert memory.search('', asof=T0.isoformat())['hits'] == []


def test_comparison_rejects_unpaired_and_synthetic_results():
    from services.memory.comparison import compare
    from services.research.autonomous.learners import Hypothesis
    result = {'daily_net_bps': {str(i): 0. for i in range(120)}, 'code_hash': 'a',
              'metrics': {'net_sharpe': 1, 'n_trades': 100}, 'risk': {'max_drawdown': .01},
              'is_simulated': True, 'p_positive': .0001, 'stress_2x_costs': {'net_return': .01}}
    base = {'id': 'base', 'dataset': 'dev', 'spec': Hypothesis('ridge').describe(), 'result': result}
    other = {'id': 'other', 'dataset': 'dev', 'spec': Hypothesis('ridge', memory_mode='numeric').describe(),
             'result': {**result, 'daily_net_bps': {str(i): 1. for i in range(120)},
                        'metrics': {'net_sharpe': 2, 'n_trades': 150}}}
    assert compare([base, other], 'dev')['comparisons'][0]['decision'] == 'REJECTED'
    other['result']['code_hash'] = 'changed'
    assert compare([base, other], 'dev')['comparisons'][0]['decision'] == 'INSUFFICIENT_EVIDENCE'


def test_semantic_confirmation_requires_real_verified_artifact(tmp_path):
    from services.research.autonomous.artifacts import export_model
    from services.research.autonomous.learners import Hypothesis, fit
    from services.research.autonomous.validation import ValidationBook
    book = ValidationBook(tmp_path / 'experiments')
    cid = book.reserve('job', {'spec': 'ridge'}, ['SPY'], 1, 2)
    memory = Memory(tmp_path / 'memory.sqlite3')
    with pytest.raises(ValueError, match='confirmation'):
        memory.remember_confirmation(book, cid)
    rng = np.random.default_rng(12)
    model = fit(Hypothesis('ridge'), rng.normal(size=(30, 3)), rng.normal(size=(30, 3)))
    artifact = tmp_path / 'artifact'
    files = export_model(model, artifact)
    book.complete(cid, {'passed': True, 'is_simulated': False, 'artifact_dir': str(artifact), 'files': files,
                        'code_hash': 'test', 'expires_at': '2026-01-01T00:00:00+00:00'})
    memory.remember_confirmation(book, cid)
    memory.remember_confirmation(book, cid)
    hits = memory.search('candidate', asof='2099-01-01T00:00:00Z', kind='semantic')['hits']
    assert len(hits) == 1 and hits[0]['status'] == 'INDEPENDENTLY_CONFIRMED'
    assert 'expires_at' in hits[0]['text']
    (artifact / next(iter(files))).write_text('corrupt')
    with pytest.raises((ValueError, FileNotFoundError)):
        memory.remember_confirmation(book, cid)


def test_future_corpus_cannot_change_past_rankings(tmp_path):
    memory = Memory(tmp_path / 'memory.sqlite3')
    for i, text in enumerate(('momentum momentum costs', 'momentum costs costs costs', 'volatility costs')):
        memory.append(source='past', source_id=str(i), body={'text': text}, available_at='2026-01-01T00:00:00Z')
    before = memory.search('momentum costs', asof='2026-02-01T00:00:00Z')
    for i in range(20):
        memory.append(source='future', source_id=str(i), body={'text': 'momentum ' * 1000},
                      available_at='2027-01-01T00:00:00Z')
    assert memory.search('momentum costs', asof='2026-02-01T00:00:00Z') == before


def test_malformed_model_numbers_do_not_poison_capture(tmp_path):
    with Database(tmp_path / 'db.sqlite3') as db:
        db.write(lambda c: c.execute(
            "INSERT INTO memory_outbox(source,source_id,event_at,body) VALUES('predictions','bad',?,?)",
            ('2026-01-01T00:00:00Z', json.dumps({'symbol': 'SPY', 'expected_return_bps': 'unknown'}))))
        memory = Memory(tmp_path / 'memory.sqlite3')
        assert memory.sync(db) == 1
        assert 'unknown' in memory.search('unknown', asof='2099-01-01T00:00:00Z')['hits'][0]['text']
