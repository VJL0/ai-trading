"""Bounded genetic symbolic factor discovery using training-only inner chronological validation."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def expression(node, X):
    if isinstance(node, int):
        return np.clip(X[:, node], -20, 20)
    op, left, right = node
    a, b = expression(left, X), expression(right, X)
    if op == 'add':
        value = a+b
    elif op == 'subtract':
        value = a-b
    elif op == 'multiply':
        value = a*b
    elif op == 'divide':
        value = a/(1+abs(b))
    else:
        raise ValueError('unregistered symbolic operator')
    return np.clip(value, -100, 100)


class GeneticFactors:
    def __init__(self, generations=6, population=32, seed=42):
        self.generations, self.population, self.seed = generations, population, seed

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        # Reserve inner tail before normalizing; no outer calibration/test observations enter search.
        split = int(len(X)*.7)
        self.scaler = StandardScaler().fit(X[:split])
        z = self.scaler.transform(X)
        if split < 30 or len(X)-split < 20:
            raise ValueError('symbolic discovery requires at least 100 chronological training rows')
        operators = ('add', 'subtract', 'multiply', 'divide')

        def random_tree(depth=0):
            if depth >= 2 or rng.random() < .4:
                return int(rng.integers(X.shape[1]))
            return [str(rng.choice(operators)), random_tree(depth+1), random_tree(depth+1)]

        if self.generations < 1 or self.population < 4:
            raise ValueError("positive generations and population >= 4 required")
        survivors = []
        population = [random_tree() for _ in range(self.population)]
        for _ in range(self.generations):
            scored = []
            for tree in population:
                values = expression(tree, z)[:, None]
                head = Ridge(alpha=10).fit(values[:split], y[:split, 0])
                error = float(np.mean((head.predict(values[split:])-y[split:, 0])**2))
                scored.append((error, tree))
            survivors = [tree for _, tree in sorted(scored, key=lambda item: item[0])[:max(2, self.population//4)]]
            population = survivors + [[str(rng.choice(operators)),
                                      survivors[int(rng.integers(len(survivors)))], random_tree()]
                                     if rng.random() < .5 else random_tree()
                                     for _ in range(self.population-len(survivors))]
        self.trees = survivors
        self.head = Ridge(alpha=10).fit(self.features(X), y)
        return self

    def features(self, X):
        z = self.scaler.transform(X)
        return np.column_stack([z, *[expression(tree, z) for tree in self.trees]])

    def predict(self, X):
        return self.head.predict(self.features(X))
