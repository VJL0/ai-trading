"""Validated inference adapter. Emits signals only; deterministic risk remains downstream.

A missing promotion leaves the existing suite in charge. An invalid/expired promotion
fails closed to NO_TRADE. Explicit rollback can restore the prior validated version.
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..models.config import SuiteConfig
from ..models.features import symbol_model_features
from ..signals import ModelPrediction
from .artifacts import PortableModel
from .engine import code_hash, identity
from .learners import transform
from .validation import ValidationBook


class NoTradeModel:
    strategy_id = 'autonomous_validated_paper'
    model_version = 'autonomous:NO_TRADE'
    horizon_bars = 60
    calibration_evaluated = False

    def predict_latest(self, sym_bars: Any) -> ModelPrediction | None:
        return None


class PaperModel(NoTradeModel):
    def __init__(self, book: ValidationBook, record: dict[str, Any]) -> None:
        self.book, self.record = book, record
        self.report = record['report']
        self.model_version = f"autonomous:confirmation:{record['id']}"
        self.model = PortableModel(Path(self.report['artifact_dir']), self.report['files'])
        self.recent_drift: list[bool] = []
        self.last_decision: Any = None
        self.retired = False

    def predict_latest(self, sym_bars: Any) -> ModelPrediction | None:
        if (self.retired or sym_bars.empty or str(sym_bars['timeframe'].iloc[0]) != '1Min'
                or bool(sym_bars['is_synthetic'].any())):
            return None
        if datetime.fromisoformat(self.report['expires_at']) <= datetime.now(UTC):
            self.book.rollback('confirmation expired; revalidation required', self.record['id'])
            self.retired = True
            return None
        if str(sym_bars['symbol'].iloc[-1]) not in self.record['candidate']['symbols']:
            return None
        spec = SuiteConfig().feature_spec
        features = symbol_model_features(sym_bars, spec)
        x = features[self.report['feature_columns']].iloc[-1].to_numpy(dtype=float)
        if not np.isfinite(x).all():
            return None
        z = (x - np.asarray(self.report['training_mean'])) / np.asarray(self.report['training_std'])
        drift = bool(np.max(np.abs(z)) > 8)
        decision = (str(sym_bars['symbol'].iloc[-1]), str(sym_bars['available_at_ts'].iloc[-1]))
        if decision != self.last_decision:
            self.recent_drift = [*self.recent_drift, drift][-20:]
            self.last_decision = decision
        if len(self.recent_drift) == 20 and sum(self.recent_drift) >= 10:
            self.book.rollback('persistent feature-distribution shift; revalidation required', self.record['id'])
            self.retired = True
            return None
        if drift:
            return None
        try:
            candidate = self.record['candidate']['spec']
            if candidate['family'].startswith('temporal_'):
                from ..advanced.inputs import normalize_history
                values = sym_bars[[c + '_f' for c in ('open', 'high', 'low', 'close', 'volume')]].to_numpy()
                values = np.column_stack([values, values[:, 4]*values[:, :4].mean(axis=1)])
                inputs = normalize_history(values, candidate['sequence_length'])[None]
            else:
                inputs = transform(x[None], candidate['transform'])
            pred = self.model.predict(inputs)
            net = float(pred[0, 0])
            allowance = float(self.report['allowance_bps'])
        except Exception:
            self.book.rollback('inference failure; reverted paper candidate', self.record['id'])
            self.retired = True
            return None
        if not np.isfinite(net) or not np.isfinite(allowance) or allowance < 0:
            return None
        # The target is NET of the training cost convention (round trip at the suite reference notional). The engine
        # re-prices that convention with live inputs and charges the live round trip exactly once; the validation
        # allowance is kept as an explicit uncertainty buffer subtracted before the threshold (never hidden).
        return ModelPrediction(net, net - allowance, None, False, basis='NET',
                               reference_notional=float(SuiteConfig().reference_notional), uncertainty_bps=allowance)


_CACHE: dict[str, PaperModel] = {}


def selected_paper_model(experiments: Path, mode: str, gateway: str) -> NoTradeModel | None:
    if mode != 'PAPER' or gateway != 'ALPACA_PAPER':
        return None
    if not (experiments / 'autonomous-validation.sqlite3').exists():
        return None
    try:
        book = ValidationBook(experiments)
        record = book.active()
        if record is None:
            # Before the first promotion, preserve the existing suite. After rollback to
            # no candidate, block instead of silently reverting to a heuristic strategy.
            return NoTradeModel() if book.status()['transitions'] else None
        report = record['report']
        if (record['state'] != 'CONFIRMED' or report.get('passed') is not True
                or report.get('is_simulated') is not False or report.get('code_hash') != code_hash()
                or datetime.fromisoformat(report['expires_at']) <= datetime.now(UTC)):
            book.rollback('invalid, expired or source-incompatible paper promotion', record['id'])
            return NoTradeModel()
        key = identity([str(experiments.resolve()), record['id'], report])
        if key not in _CACHE:
            _CACHE.clear()
            _CACHE[key] = PaperModel(book, record)
        return _CACHE[key]
    except Exception:
        return NoTradeModel()
