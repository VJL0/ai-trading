"""Fitted GJR-GARCH and Gaussian HMM with strictly filtered state probabilities."""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import logsumexp


class GJRGarch:
    def fit(self, returns: np.ndarray):
        from arch import arch_model
        r = np.asarray(returns, float)
        if len(r) < 100 or not np.isfinite(r).all():
            raise ValueError('GJR-GARCH requires at least 100 finite returns')
        self.unit = 1/max(float(np.std(r)), 1e-8)
        fitted = arch_model(r*self.unit, mean='zero', vol='garch', p=1, o=1, q=1, dist='t').fit(disp='off', show_warning=False)
        self.params = fitted.params.to_dict()
        self.initial_variance = float(np.var(r*self.unit))
        self.converged = fitted.convergence_flag == 0
        if not self.converged:
            raise ValueError('GJR-GARCH did not converge')
        return self

    def forecast(self, observed: np.ndarray, horizons=(5, 15, 60)):
        p = self.params
        variance = self.initial_variance
        for r in np.asarray(observed)*self.unit:
            variance = p['omega']+(p['alpha[1]']+p['gamma[1]']*(r < 0))*r*r+p['beta[1]']*variance
        path = []
        persistence = p['alpha[1]']+.5*p['gamma[1]']+p['beta[1]']
        for _ in range(max(horizons)):
            path.append(max(variance, 1e-12)/(self.unit**2))
            variance = p['omega']+persistence*variance
        return {h: float(np.sum(path[:h])) for h in horizons}


class FilteredHMM:
    def __init__(self, states=3, seed=42):
        self.states, self.seed = states, seed

    def fit(self, X: np.ndarray):
        from hmmlearn.hmm import GaussianHMM
        self.mean, self.scale = X.mean(axis=0), np.maximum(X.std(axis=0), 1e-8)
        model = GaussianHMM(self.states, covariance_type='diag', n_iter=100, random_state=self.seed,
                            min_covar=1e-4).fit((X-self.mean)/self.scale)
        self.start, self.transition, self.means = model.startprob_, model.transmat_, model.means_
        self.variance = np.diagonal(np.asarray(model.covars_), axis1=1, axis2=2)
        return self

    def filtered(self, X: np.ndarray) -> np.ndarray:
        z = (X-self.mean)/self.scale
        emission = -.5*(np.log(2*np.pi*self.variance).sum(axis=1)[None]
                        + ((z[:, None]-self.means)**2/self.variance).sum(axis=2))
        prior = self.start.copy()
        result = []
        for row in emission:
            logits = np.log(np.maximum(prior, 1e-300))+row
            posterior = np.exp(logits-logsumexp(logits))
            result.append(posterior)
            prior = posterior@self.transition
        return np.asarray(result)


class MarkovSwitching:
    """Fixed-parameter switching mean/variance; outputs filtered, never smoothed states."""
    def __init__(self, states=2):
        self.states = states

    def fit(self, returns):
        from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
        self.scale = max(float(np.std(returns)), 1e-8)
        result: Any = MarkovRegression(np.asarray(returns)/self.scale, k_regimes=self.states,
                                  trend='c', switching_variance=True).fit(disp=False, maxiter=200)
        if not result.mle_retvals.get('converged', False):
            raise ValueError('Markov-switching optimizer did not converge')
        self.parameters = result.params
        return self

    def filtered(self, returns):
        from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
        model = MarkovRegression(np.asarray(returns)/self.scale, k_regimes=self.states,
                                 trend='c', switching_variance=True)
        return np.asarray(model.filter(self.parameters).filtered_marginal_probabilities)
