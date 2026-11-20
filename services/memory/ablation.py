"""Frozen calibration-episode retrieval challengers. No test labels enter memory.

All modes share a fixed ridge baseline, folds, calibration episodes, costs and position
simulator. Text modes encode coarse market-state descriptions, not LLM-generated facts.
"""
from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler

from .store import encode

MODES = ('numeric', 'keyword', 'vector', 'hybrid', 'reflection')


def descriptions(x: np.ndarray) -> list[str]:
    return [' '.join(f'feature{j}_{"low" if v < -0.5 else "high" if v > 0.5 else "normal"}'
                     for j, v in enumerate(row)) for row in x]


class EpisodicAdjustment:
    def __init__(self, mode: str, *, max_episodes: int = 256, neighbors: int = 16):
        if mode not in MODES:
            raise ValueError('unknown memory ablation')
        self.mode, self.max_episodes, self.neighbors = mode, max_episodes, neighbors

    def fit(self, training_x, calibration_x, residual, label_end, test_start):
        if len(calibration_x) != len(residual) or np.any(np.asarray(label_end) >= test_start):
            raise ValueError('memory outcomes unavailable before test interval')
        if len(calibration_x) < self.neighbors:
            raise ValueError('insufficient calibration episodes')
        if not all(np.isfinite(a).all() for a in (training_x, calibration_x, residual)):
            raise ValueError('nonfinite memory data')
        self.scaler = StandardScaler().fit(training_x)
        # Fixed chronological subsampling, independent of labels and performance.
        idx = np.linspace(0, len(calibration_x) - 1, min(self.max_episodes, len(calibration_x)), dtype=int)
        self.x = np.asarray(self.scaler.transform(calibration_x[idx]), dtype=float)
        self.residual = np.asarray(residual)[idx]
        self.text = descriptions(self.x)
        self.vector = np.stack([encode(t) for t in self.text])
        self.tokens = [set(t.split()) for t in self.text]
        return self

    def adjustment(self, test_x) -> np.ndarray:
        if not np.isfinite(test_x).all():
            raise ValueError('nonfinite query')
        x = np.asarray(self.scaler.transform(test_x), dtype=float)
        values = []
        for row, text in zip(x, descriptions(x), strict=True):
            rankings = []
            if self.mode in ('numeric', 'hybrid', 'reflection'):
                rankings.append(np.argsort(np.mean((self.x - row) ** 2, axis=1), kind='stable'))
            if self.mode in ('vector', 'hybrid'):
                rankings.append(np.argsort(-(self.vector @ encode(text)), kind='stable'))
            if self.mode in ('keyword', 'hybrid'):
                tokens = set(text.split())
                overlap = [len(tokens & t) / max(1, len(tokens | t)) for t in self.tokens]
                rankings.append(np.argsort(-np.asarray(overlap), kind='stable'))
            scores = np.zeros(len(self.x))
            for ranking in rankings:
                scores[ranking] += 1 / (61 + np.arange(len(ranking)))
            selected = np.argsort(-scores, kind='stable')[:self.neighbors]
            correction = float(np.mean(self.residual[selected])) * .5
            # Reflection is a conservative local overprediction veto, not a discovered fact.
            values.append(min(0., correction) if self.mode == 'reflection' else correction)
        return np.asarray(values)
