import numpy as np
import pytest

from services.research.autonomous.artifacts import PortableModel, export_model
from services.research.autonomous.learners import Hypothesis, available, fit


@pytest.mark.parametrize('family', ['double_ensemble', 'symbolic_gp'])
def test_cpu_train_export(family, tmp_path):
    if not available()[family]["available"]:
        pytest.skip(f"optional runtime for {family} not installed (double_ensemble needs the isolated qlib env)")
    x = np.random.default_rng(4).normal(size=(200, 8))
    y = x[:, :3]*2
    m = fit(Hypothesis(family, capacity=8), x, y)
    files = export_model(m, tmp_path/'model')
    np.testing.assert_allclose(m.predict(x[:10]), PortableModel(tmp_path/'model', files).predict(x[:10]))


def test_logged_policies_require_support():
    from services.research.advanced.policy import ContextualBandit, OfflineFittedQ
    rng = np.random.default_rng(8)
    x = rng.normal(size=(300, 4))
    action = np.arange(300) % 3
    reward = (action == 1)*x[:, 0]-.01
    probability = np.full(300, 1/3)
    model = ContextualBandit().fit(x[:210], action[:210], reward[:210], probability[:210])
    with pytest.raises(ValueError):
        model.evaluate(x[210:], action[210:], reward[210:], probability[210:])
    model = OfflineFittedQ().fit_transitions(x, action, reward, probability, x, np.ones(300), iterations=2)
    assert model.predict(x).shape == (300,)
    with pytest.raises(ValueError):
        model.fit(x, action, reward, np.zeros(300))


def test_gjr_and_markov_are_finite_and_causal():
    from services.research.advanced.statistical import GJRGarch, MarkovSwitching
    rng = np.random.default_rng(9)
    returns = np.r_[rng.normal(-.002, .001, 200), rng.normal(.002, .005, 200)]
    g = GJRGarch().fit(returns)
    v = g.forecast(returns)
    assert 0 < v[5] < v[15] < v[60]
    m = MarkovSwitching().fit(returns)
    np.testing.assert_allclose(m.filtered(returns[:200]), m.filtered(returns)[:200])
