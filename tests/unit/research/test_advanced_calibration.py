import numpy as np

from services.research.advanced.calibration import CalibratedTargets
from services.research.autonomous.artifacts import PortableModel, export_model
from services.research.autonomous.learners import Hypothesis, fit


def test_calibration_artifact_parity_and_probability_bounds(tmp_path):
    rng = np.random.default_rng(21)
    X = rng.normal(size=(200, 5))
    y = rng.normal(size=(200, 10))
    y[:, 5:8] = y[:, 5:8] > 0
    base = fit(Hypothesis('ridge'), X[:100], y[:100])
    model = CalibratedTargets(base).calibrate(X[100:150], y[100:150])
    pred = model.predict(X[150:])
    assert ((pred[:, 5:8] >= 0) & (pred[:, 5:8] <= 1)).all()
    np.testing.assert_allclose(pred[:, :5], base.predict(X[150:])[:, :5])
    files = export_model(model, tmp_path/'model')
    np.testing.assert_allclose(pred, PortableModel(tmp_path/'model', files).predict(X[150:]))
