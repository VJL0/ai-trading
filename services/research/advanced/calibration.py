"""Success probabilities calibrated only on the purged calibration block."""
import numpy as np
from sklearn.isotonic import IsotonicRegression


class CalibratedTargets:
    def __init__(self, model):
        self.model = model
        self.calibrators = {}

    def calibrate(self, X, y):
        pred = self.model.predict(X)
        if y.shape[1] >= 8:
            for column in (5, 6, 7):
                if len(y) >= 20 and len(np.unique(y[:, column])) == 2:
                    self.calibrators[column] = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1).fit(
                        pred[:, column], y[:, column])
        return self

    def predict(self, X):
        pred = np.asarray(self.model.predict(X)).copy()
        for column, calibrator in self.calibrators.items():
            pred[:, column] = calibrator.predict(pred[:, column])
        if pred.shape[1] >= 8:
            pred[:, 5:8] = np.clip(pred[:, 5:8], 0, 1)
        return pred
