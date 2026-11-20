"""Research-only logged-policy learners. No order, broker or promotion authority.

Propensities must come from recorded behavior, not retrospectively fabricated labels.
Offline policy value remains observational and does not establish deployable trading edge.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def validate_log(X, action, reward, propensity, actions):
    if len(X) < 100 or not (len(X) == len(action) == len(reward) == len(propensity)):
        raise ValueError('at least 100 aligned logged transitions required')
    if not all(np.isfinite(v).all() for v in (X, reward, propensity)):
        raise ValueError('nonfinite policy log')
    if np.any(propensity <= 0) or np.any(propensity > 1) or np.any(action != action.astype(int)):
        raise ValueError('valid recorded behavior propensities and discrete actions required')
    if set(np.unique(action)) != set(range(actions)):
        raise ValueError('logged behavior must cover every action')


class ContextualBandit:
    def __init__(self, actions=3, alpha=10.):
        self.actions, self.alpha = actions, alpha

    def fit(self, X, action, reward, propensity):
        validate_log(X, action, reward, propensity, self.actions)
        self.scaler = StandardScaler().fit(X)
        z = self.scaler.transform(X)
        self.models = []
        for a in range(self.actions):
            rows = action == a
            if rows.sum() < 20:
                raise ValueError('insufficient per-action support')
            self.models.append(Ridge(alpha=self.alpha).fit(z[rows], reward[rows],
                                                          sample_weight=np.minimum(1/propensity[rows], 20)))
        return self

    def values(self, X):
        return np.column_stack([m.predict(self.scaler.transform(X)) for m in self.models])

    def predict(self, X):
        return self.values(X).argmax(axis=1)

    def evaluate(self, X, action, reward, propensity):
        validate_log(X, action, reward, propensity, self.actions)
        values = self.values(X)
        selected = values.argmax(axis=1)
        weights = (selected == action)/propensity
        correction = weights*(reward-values[np.arange(len(X)), action.astype(int)])
        effective = weights.sum()**2/max(float(np.sum(weights**2)), 1e-12)
        return {'doubly_robust_value': float(np.mean(values.max(axis=1)+correction)),
                'effective_sample_size': float(effective), 'research_only': True}


class OfflineFittedQ(ContextualBandit):
    """Behavior-supported fitted-Q iteration with a conservative value penalty."""
    def fit_transitions(self, X, action, reward, propensity, next_X, terminal, iterations=10, discount=.95):
        if next_X.shape != X.shape or len(terminal) != len(X) or not 0 <= discount < 1:
            raise ValueError('invalid transition contract')
        super().fit(X, action, reward, propensity)
        for _ in range(iterations):
            q = self.values(next_X)
            # Pessimistic continuation penalizes disagreement between supported actions.
            continuation = q.max(axis=1)-q.std(axis=1)
            target = reward+discount*(1-np.asarray(terminal, bool))*continuation
            super().fit(X, action, target, propensity)
        return self
