from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from services.research.autonomous.engine import Journal, prepare_cache
from services.research.autonomous.evidence import screen
from services.research.kronos.adapter import Forecast, ForecastConfig, KronosAdapter, sha
from services.research.kronos.benchmark import common_rows, paired_evidence
from services.research.kronos.data import context, future_calendar
from services.research.kronos.lean import export_replay
from services.research.synthetic import make_synthetic_bars


def history(n=12):
    ix = pd.date_range('2026-11-17 14:30', periods=n, freq='min', tz='UTC')
    return pd.DataFrame({'open': 100., 'high': 102., 'low': 99., 'close': 101., 'volume': 1000.,
                         'available_at_ts': ix+pd.Timedelta(minutes=1)}, index=ix)


def fake_adapter():
    pytest.importorskip('torch')
    adapter = KronosAdapter.__new__(KronosAdapter)
    adapter.config = ForecastConfig(samples=2)
    adapter.device = 'cpu'
    adapter.fallback_reason = None
    adapter.parameter_count = 1
    adapter.manifest = {}
    def generate(x, h, f, seed):
        return np.tile([100., 102., 99., 101., 1000., 100000.], (2, len(f), 1))
    adapter._generate = generate
    return adapter


def test_adapter_rejects_unpublished_context_and_bad_dates():
    a = fake_adapter()
    frame = history()
    decision = frame.available_at_ts.iloc[-1]
    future = pd.date_range(decision, periods=16, freq='min')
    assert a.forecast(frame, future, decision=decision).paths.shape == (2, 16, 6)
    frame.iloc[-1, frame.columns.get_loc('available_at_ts')] = decision+pd.Timedelta(minutes=1)
    with pytest.raises(ValueError, match='unpublished'):
        a.forecast(frame, future, decision=decision)
    with pytest.raises(ValueError, match='future calendar'):
        a.forecast(history(), future[::-1], decision=decision)


def test_forecast_uncertainty_retains_path_dispersion():
    p = np.ones((4, 61, 6))*100
    p[:, 60, 0] = [98, 99, 101, 102]
    f = Forecast(p, pd.date_range('2026-01-01', periods=61, tz='UTC'), {})
    features = f.features((5, 15, 60), 10)
    assert features['h60_mean_bps'] == pytest.approx(-10)
    assert features['h60_std_bps'] > 0
    assert features['h60_q10_bps'] < features['h60_q90_bps']
    assert features['h60_p_positive'] == .5
    with pytest.raises(ValueError):
        f.features((61,))


def test_variable_lengths_do_not_pad_normalization():
    a = fake_adapter()
    frames = [history(8), history(12)]
    requests = [(f, pd.date_range(f.available_at_ts.iloc[-1], periods=5, freq='min'), f.available_at_ts.iloc[-1])
                for f in frames]
    assert len(a.forecast_many(requests)) == 2


def test_sequence_cache_excludes_holdout_and_future_mutation(tmp_path):
    bars = make_synthetic_bars(('AAA', 'BBB'), timeframe='1Min', n_sessions=12)
    key = prepare_cache(Journal(tmp_path), SimpleNamespace(bars=bars, dataset_version='test', adjustments={}))
    directory = tmp_path / 'cache' / key
    manifest = json.loads((directory / 'manifest.json').read_text())
    with np.load(directory / 'development.npz', allow_pickle=False) as archive:
        a = dict(archive)
    for sym in range(2):
        assert a[f'avail_{sym}'].max() < manifest['split']['final_test_start_ns']
    row = len(a['decision'])//2
    original = context(a, row, 128)
    for sym in range(2):
        a[f'bars_{sym}'][a[f'avail_{sym}'] > a['decision'][row]] *= 100
    pd.testing.assert_frame_equal(original, context(a, row, 128))
    assert (original.available_at_ts.astype('int64') <= a['decision'][row]).all()


def test_masks_match_unpadded_and_causality():
    torch = pytest.importorskip('torch')
    pytest.importorskip('einops')
    from services.research.kronos.vendor.module import MultiHeadAttentionWithRoPE, MultiHeadCrossAttentionWithRoPE
    torch.manual_seed(2)
    layer = MultiHeadAttentionWithRoPE(16, 4).eval()
    x = torch.randn(2, 7, 16)
    padding = torch.tensor([[False]*4+[True]*3, [False]*7])
    actual = layer(x, padding)
    assert torch.allclose(actual[0, :4], layer(x[:1, :4])[0], atol=1e-6)
    changed = x.clone()
    changed[:, 3:] += 100
    assert torch.allclose(layer(x)[:, :3], layer(changed)[:, :3], atol=1e-6)
    cross = MultiHeadCrossAttentionWithRoPE(16, 4).train()
    assert torch.isfinite(cross(x, x, x, padding)).all()
    assert torch.allclose(cross(x, x, x, padding)[0, :4], cross(x[:1, :4], x[:1, :4], x[:1, :4])[0], atol=1e-6)


def test_sampling_in_eval_mode_and_no_path_average():
    pytest.importorskip('torch')
    pytest.importorskip('huggingface_hub')
    from services.research.kronos.vendor.kronos import Kronos, KronosTokenizer
    tokenizer = KronosTokenizer(6, 16, 4, 32, 1, 1, .1, .1, .1, 5, 5, .1, 1., 1., 1., 5)
    model = Kronos(5, 5, 1, 16, 4, 32, .1, .1, .1, .1, True)
    a = fake_adapter()
    a.tokenizer, a.model = tokenizer, model
    del a._generate
    f = history()
    d = f.available_at_ts.iloc[-1]
    future = pd.date_range(d, periods=6, freq='min')
    first = a.forecast(f, future, decision=d)
    second = a.forecast(f, future, decision=d)
    np.testing.assert_array_equal(first.paths, second.paths)
    assert not model.training and not tokenizer.training
    assert first.paths.shape == (2, 6, 6)


def test_common_rows_and_sparse_promotion_fail_closed():
    np.testing.assert_array_equal(common_rows(np.arange(100), 4), [0, 33, 66, 99])
    assert not paired_evidence(np.ones(30), np.zeros(30), 1)['credible']
    strong = {'is_simulated': False, 'metrics': {'n_trades': 200}, 'sessions': 150,
              'p_positive': 0., 'p_incremental': 0., 'stress_2x_costs': {'net_return': .1},
              'folds': [{'net_return': .1}]*4, 'risk': {'max_drawdown': .05},
              'promotion_blockers': ['pretraining overlap unknown']}
    assert not screen(strong, 1)['eligible_for_independent_validation']


def test_lean_export_verifies_evidence(tmp_path):
    p = tmp_path / 'predictions.json'
    p.write_text(json.dumps([{'model': 'kronos_base', 'decision_ns': 1, 'exit_ns': 2, 'symbol': 'SPY', 'score': 4}]))
    (tmp_path / 'result.json').write_text(json.dumps({'prediction_sha256': sha(p), 'dataset_key': 'x', 'is_simulated': True}))
    output = tmp_path / 'scores.csv'
    assert export_replay(tmp_path, 'kronos_base', output)['rows'] == 1
    p.write_text('[]')
    with pytest.raises(ValueError, match='integrity'):
        export_replay(tmp_path, 'kronos_base', output)


def test_future_calendar_does_not_read_market_outcomes():
    d = pd.Timestamp('2026-11-17 15:31:00.001', tz='UTC').value
    assert future_calendar(d, 61, 1)[0] == pd.Timestamp('2026-11-17 15:32', tz='UTC')
    with pytest.raises(ValueError, match='session'):
        future_calendar(pd.Timestamp('2026-11-17 20:55', tz='UTC').value, 61, 1)


def test_tuning_changes_only_heads_and_rejects_boundary_overlap():
    torch = pytest.importorskip('torch')
    pytest.importorskip('huggingface_hub')
    from services.research.kronos.tuning import tune_heads
    from services.research.kronos.vendor.kronos import Kronos, KronosTokenizer
    a = fake_adapter()
    a.tokenizer = KronosTokenizer(6, 16, 4, 32, 1, 1, 0., 0., 0., 5, 5, .1, 1., 1., 1., 5)
    a.model = Kronos(5, 5, 1, 16, 4, 32, 0., 0., 0., 0., True)
    f = history()
    f['amount'] = 100000.
    values = f[['open', 'high', 'low', 'close', 'volume', 'amount']].to_numpy(dtype=np.float32)
    data = {'decision': np.array([f.available_at_ts.iloc[-1].value]), 'end': np.array([10]),
            'symbol_id': np.array([0]), 'bars_0': values, 'start_0': f.index.asi8,
            'avail_0': f.available_at_ts.astype('int64').to_numpy()}
    before = {k: v.clone() for k, v in a.model.state_dict().items()}
    assert tune_heads(a, data, np.array([0]), 2, 11)['steps'] == 2
    changed = [k for k, v in a.model.state_dict().items() if not torch.equal(v, before[k])]
    assert changed and all(k.startswith('head.') for k in changed)
    assert not a.model.training and not a.tokenizer.training
    with pytest.raises(ValueError, match='overlap'):
        tune_heads(a, data, np.array([0]), 2, 10)


def test_mps_failure_falls_back_only_for_backend_errors(monkeypatch):
    a = fake_adapter()
    a.device = 'mps'
    original = a._generate
    calls = []
    def generate(*args):
        calls.append(a.device)
        if a.device == 'mps':
            raise RuntimeError('MPS operation not implemented')
        return original(*args)
    def cpu(reason):
        a.device, a.fallback_reason = 'cpu', reason
    monkeypatch.setattr(a, '_generate', generate)
    monkeypatch.setattr(a, '_cpu', cpu)
    f = history()
    d = f.available_at_ts.iloc[-1]
    r = a.forecast(f, pd.date_range(d, periods=6, freq='min'), decision=d)
    assert calls == ['mps', 'cpu'] and r.metadata['cpu_fallback']


def test_paired_benchmark_uses_identical_oos_rows_and_is_research_only(tmp_path, monkeypatch):
    import services.research.kronos.benchmark as benchmark
    from services.research.autonomous.learners import Hypothesis
    bars = make_synthetic_bars(('AAA', 'BBB'), timeframe='1Min', n_sessions=12)
    root = tmp_path / 'autonomous'
    journal = Journal(root)
    key = prepare_cache(journal, SimpleNamespace(bars=bars, dataset_version='test', adjustments={}))
    directory = root / 'cache' / key
    manifest = json.loads((directory / 'manifest.json').read_text())
    with np.load(directory / 'development.npz', allow_pickle=False) as archive:
        arrays = dict(archive)
    # This test targets paired row plumbing; HAR itself is tested against real lag inputs elsewhere.
    arrays['vol_X'] = np.full_like(arrays['vol_X'], 1e-6)
    weights = tmp_path / 'weights'
    weights.mkdir()
    (weights / 'manifest.json').write_text('{}')
    monkeypatch.setattr(benchmark, 'DEFAULT_ROOT', weights)
    monkeypatch.setattr(benchmark, 'KronosAdapter', lambda cfg: fake_adapter())
    registry = SimpleNamespace(meta={'selected': {f'ret_{h}m': 'zero_baseline' for h in (5, 15, 60)},
                                    'config': {}, 'registry_sha256': 'test'})
    monkeypatch.setattr(benchmark, 'load_registry', lambda _: registry)
    spec = Hypothesis('kronos_base', kronos_rows_per_fold=3, kronos_artifact_hash=sha(weights / 'manifest.json'))
    jid = journal.enqueue(key, spec)
    result = benchmark.evaluate_challenger(root, jid, spec, arrays, manifest)
    assert result['promotion_eligible'] is False and result['promotion_blockers']
    predictions = json.loads((root / 'runs' / jid / 'predictions.json').read_text())
    models = {p['model'] for p in predictions}
    assert models == {'kronos_base', 'ridge', 'lightgbm', 'catboost', 'selected_registry'}
    rowsets = [{p['row'] for p in predictions if p['model'] == m} for m in models]
    assert all(rows == rowsets[0] for rows in rowsets)
    assert max(p['decision_ns'] for p in predictions) < manifest['split']['final_test_start_ns']
    assert result['volatility']['har_rv']['n_test'] == 12
    entry = tmp_path / 'models' / 'challengers' / f'{jid}.json'
    assert json.loads(entry.read_text())['status'] == 'RESEARCH_ONLY'
