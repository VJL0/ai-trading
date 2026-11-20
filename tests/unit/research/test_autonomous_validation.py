from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from services.research.autonomous.artifacts import PortableModel, export_model
from services.research.autonomous.engine import code_hash
from services.research.autonomous.learners import Hypothesis, available, fit
from services.research.autonomous.paper import NoTradeModel, PaperModel, selected_paper_model
from services.research.autonomous.validation import ValidationBook
from services.research.ml import FrozenTest, FrozenTestReuseError


@pytest.mark.parametrize('family', ['ridge', 'bayesian', 'online', 'pca', 'regime', 'extra_trees',
                                   'lightgbm', 'catboost', 'xgboost'])
def test_portable_models_match_without_pickle(tmp_path, family):
    if not available()[family]['available']:
        pytest.skip('optional learner dependency absent')
    rng = np.random.default_rng(18)
    x, y = rng.normal(size=(140, 8)), rng.normal(size=(140, 3))
    model = fit(Hypothesis(family, capacity=8), x[:100], y[:100])
    directory = tmp_path / 'model'
    files = export_model(model, directory)
    portable = PortableModel(directory, files)
    np.testing.assert_allclose(portable.predict(x[100:]), model.predict(x[100:]), rtol=1e-5, atol=1e-5)
    assert not any(p.suffix in ('.pkl', '.pickle', '.joblib') for p in directory.iterdir())
    (directory / 'model.json').write_text('{}')
    with pytest.raises(ValueError, match='integrity'):
        PortableModel(directory, files)


def test_reservation_is_atomic_alias_independent_and_crash_consumes_interval(tmp_path):
    def reserve(i):
        try:
            return ValidationBook(tmp_path).reserve(str(i), {'dataset_alias': str(i)}, ['SPY'], 100, 200)
        except ValueError:
            return None
    ValidationBook(tmp_path)
    with ThreadPoolExecutor(4) as pool:
        ids = list(pool.map(reserve, range(4)))
    assert sum(x is not None for x in ids) == 1
    with pytest.raises(ValueError, match='overlap'):
        ValidationBook(tmp_path).reserve('retry-after-crash', {}, ['SPY'], 200, 300)
    assert ValidationBook(tmp_path).reserve('later', {}, ['SPY'], 201, 300) > 0


def test_development_exposure_prevents_confirmation(tmp_path):
    book = ValidationBook(tmp_path)
    book.expose('dataset-A', ['SPY'], 1, 100)
    with pytest.raises(ValueError, match='exposure'):
        book.reserve('different-dataset-alias', {}, ['SPY'], 50, 200)


def test_legacy_and_new_holdout_registries_protect_each_other(tmp_path):
    book = ValidationBook(tmp_path)
    cid = book.reserve('new', {}, ['SPY'], 100, 200)
    assert cid == 1
    frozen = FrozenTest(np.array([0]), 'legacy', registry=tmp_path / 'models' / 'frozen_test_registry.json',
                        interval_ns=(150, 250), symbols=['SPY'])
    with pytest.raises(FrozenTestReuseError):
        frozen.evaluate_once(lambda _: {}, run_id='legacy', dataset_version='alias')
    other = FrozenTest(np.array([0]), 'other', registry=tmp_path / 'ml' / 'frozen_test_registry.json',
                       interval_ns=(300, 400), symbols=['SPY'])
    other.evaluate_once(lambda _: {}, run_id='old', dataset_version='original-name')
    with pytest.raises(ValueError, match='existing'):
        book.reserve('alias', {}, ['SPY'], 350, 450)


def confirmation(book, tmp_path, job, lo):
    rng = np.random.default_rng(7)
    x, y = rng.normal(size=(100, 8)), rng.normal(size=(100, 3))
    model = fit(Hypothesis('ridge'), x, y)
    directory = tmp_path / job
    files = export_model(model, directory)
    cid = book.reserve(job, {'spec': Hypothesis('ridge').describe(), 'symbols': ['SPY']}, ['SPY'], lo, lo + 50)
    report = {'passed': True, 'is_simulated': False, 'code_hash': code_hash(), 'files': files,
              'artifact_dir': str(directory), 'expires_at': (datetime.now(UTC) + timedelta(days=1)).isoformat(),
              'training_mean': x.mean(axis=0).tolist(), 'training_std': x.std(axis=0).tolist()}
    book.complete(cid, report)
    return cid


def test_paper_promotion_reversible_cas_and_never_live(tmp_path):
    book = ValidationBook(tmp_path)
    first = confirmation(book, tmp_path, 'a', 100)
    second = confirmation(book, tmp_path, 'b', 200)
    book.promote(first)
    book.promote(second)
    assert book.active()['id'] == second
    book.rollback('stale inference instance', expected_id=first)
    assert book.active()['id'] == second
    assert selected_paper_model(tmp_path, 'LIVE', 'ALPACA_PAPER') is None
    assert selected_paper_model(tmp_path, 'PAPER', 'FAKE') is None
    assert isinstance(selected_paper_model(tmp_path, 'PAPER', 'ALPACA_PAPER'), PaperModel)
    book.rollback('performance degradation', expected_id=second)
    assert book.active()['id'] == first
    book.rollback('no valid predecessor', expected_id=first)
    assert book.active() is None
    assert isinstance(selected_paper_model(tmp_path, 'PAPER', 'ALPACA_PAPER'), NoTradeModel)


def test_rejected_and_synthetic_confirmations_cannot_promote(tmp_path):
    book = ValidationBook(tmp_path)
    rejected = book.reserve('bad', {}, ['SPY'], 1, 5)
    book.complete(rejected, {'passed': False})
    with pytest.raises(ValueError, match='confirmed'):
        book.promote(rejected)
    simulated = book.reserve('simulated', {}, ['SPY'], 10, 15)
    book.complete(simulated, {'passed': True, 'is_simulated': True})
    with pytest.raises(ValueError, match='provenance'):
        book.promote(simulated)


def test_corrupt_legacy_book_is_not_a_fresh_holdout(tmp_path):
    root = tmp_path / 'models'
    root.mkdir()
    (root / 'frozen_test_registry.json').write_text(json.dumps([]))
    with pytest.raises(ValueError, match='unreadable'):
        ValidationBook(tmp_path).reserve('a', {}, ['SPY'], 1, 5)


def test_full_confirmation_rejects_short_holdout_and_keeps_it_consumed(tmp_path, monkeypatch):
    """Fabricated input exercises plumbing only; no test result can authorize real paper deployment."""
    from types import SimpleNamespace

    from services.research.autonomous import validation
    from services.research.autonomous.engine import Journal, evaluate, prepare_cache
    from services.research.synthetic import make_synthetic_bars

    bars = make_synthetic_bars(('SPY', 'QQQ'), timeframe='1Min', n_sessions=12, start='2026-10-20')
    # Test-only provenance override to reach the independent holdout rejection branch.
    bars['is_synthetic'] = False
    data = SimpleNamespace(bars=bars, dataset_version='fabricated-validation-plumbing-test', adjustments={})
    journal = Journal(tmp_path / 'autonomous')
    book = ValidationBook(tmp_path)
    key = prepare_cache(journal, data, book=book)
    jid = journal.enqueue(key, Hypothesis('ridge'))
    journal.claim(jid)
    journal.finish(jid, evaluate(journal.root, jid))
    monkeypatch.setattr(validation, 'screen', lambda *_: {'eligible_for_independent_validation': True})
    # Freeze only the recency clock at the end of this explicit synthetic fixture.
    class TestDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 11, 5, 22, tzinfo=UTC)
    monkeypatch.setattr(validation, 'datetime', TestDateTime)
    result = validation.confirm(journal.root, jid, data, None, book)
    assert result['passed'] is False
    assert result['sessions'] < 60
    assert book.status()['confirmations'][0]['state'] == 'REJECTED'
    assert book.active() is None
    with pytest.raises(ValueError, match='overlap'):
        validation.confirm(journal.root, jid, data, None, book)


def test_synthetic_calendar_overlap_does_not_expose_real_returns(tmp_path):
    book = ValidationBook(tmp_path)
    book.expose('synthetic-fixture', ['SPY'], 100, 200, is_simulated=True)
    assert book.reserve('real-market', {}, ['SPY'], 100, 200) == 1
    with pytest.raises(ValueError, match='overlap'):
        book.reserve('second-real-look', {}, ['SPY'], 100, 200)
