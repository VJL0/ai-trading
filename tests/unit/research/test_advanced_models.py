"""Real train/predict/portable artifact checks; synthetic data is not ROI evidence."""
import numpy as np
import pytest

from services.research.advanced.inputs import model_inputs
from services.research.autonomous.artifacts import PortableModel, export_model
from services.research.autonomous.learners import Hypothesis, fit


@pytest.mark.parametrize('family', ['elastic_net', 'tabm', 'temporal_patchtst', 'temporal_itransformer',
                                  'temporal_moderntcn', 'temporal_multitask_transformer'])
def test_actual_training_and_portable_parity(family, tmp_path):
    rng = np.random.default_rng(15)
    x = rng.normal(size=(48, 16, 6) if family.startswith('temporal_') else (48, 6)).astype('float32')
    y = rng.normal(size=(48, 3)).astype('float32')
    spec = Hypothesis(family, capacity=16, epochs=1, pretrain_epochs=1, meta_steps=1, alpha=.01)
    model = fit(spec, x, y)
    p = model.predict(x[:5])
    assert np.isfinite(p).all() and p.shape == (5, 3)
    files = export_model(model, tmp_path/'model')
    np.testing.assert_allclose(PortableModel(tmp_path/'model', files).predict(x[:5]), p, atol=2e-4, rtol=2e-4)
    if hasattr(model, 'model'):
        assert not model.model.training


def test_temporal_inputs_exclude_future():
    bars = np.ones((32, 6), dtype='float32')
    a = {'symbol_id': np.array([0]), 'decision': np.array([16]), 'avail_0': np.arange(32), 'bars_0': bars}
    spec = Hypothesis('temporal_patchtst', sequence_length=16)
    before = model_inputs(spec, a, [0])
    bars[17:] = 999
    np.testing.assert_array_equal(model_inputs(spec, a, [0]), before)


def test_hmm_prefix_invariance():
    from services.research.advanced.statistical import FilteredHMM
    x = np.random.default_rng(3).normal(size=(200, 2))
    model = FilteredHMM().fit(x[:100])
    np.testing.assert_allclose(model.filtered(x[100:150]), model.filtered(x[100:])[:50])
    np.testing.assert_allclose(model.filtered(x).sum(axis=1), 1)
