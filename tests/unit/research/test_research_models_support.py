"""Shared helpers for the model-suite tests (explicitly SYNTHETIC data; no test functions here)."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from services.research.costs import CostModel
from services.research.models.config import SuiteConfig
from services.research.models.registry import ModelRegistry, save_registry
from services.research.models.walkforward import TrainResult, train_suite
from services.research.synthetic import make_synthetic_bars, xnys_sessions


def fast_config(**overrides: Any) -> SuiteConfig:
    """Small, fast, deterministic model parameters for unit tests."""
    base: dict[str, Any] = {
        "lgbm_params": {"num_leaves": 7, "learning_rate": 0.1, "num_boost_round": 40, "min_data_in_leaf": 40},
        "rank_params": {"num_leaves": 7, "learning_rate": 0.1, "num_boost_round": 20, "min_data_in_leaf": 40},
        "cost_params": {"num_leaves": 7, "learning_rate": 0.1, "num_boost_round": 20, "min_data_in_leaf": 100},
        "catboost_params": {"iterations": 60, "depth": 3, "learning_rate": 0.1},
        "n_folds": 3,
    }
    base.update(overrides)
    return SuiteConfig(**base)


def noise_bars(symbols: tuple[str, ...] = ("AAA", "BBB", "CCC"), *, sessions: int = 8, seed: int = 11,
               latency_s: float = 1.0) -> pd.DataFrame:
    """Seeded random walk (no edge) on the real XNYS calendar, 1-minute bars."""
    return make_synthetic_bars(symbols, timeframe="1Min", n_sessions=sessions, seed=seed, drift=0.0, vol=0.012,
                               latency_s=latency_s, dataset_version="synthetic-noise-test")


def _minute_spans(sessions: int, start: str = "2024-03-04") -> tuple[pd.DatetimeIndex, pd.DatetimeIndex, np.ndarray]:
    """(starts, ends, minute-of-session) of every regular-session 1-minute bar of ``sessions`` XNYS sessions."""
    starts_l: list[pd.DatetimeIndex] = []
    minute_of_session: list[np.ndarray] = []
    for o, c in xnys_sessions(start, sessions):
        n = (int(c.value) - int(o.value)) // 60_000_000_000
        starts_l.append(pd.date_range(o, periods=n, freq="1min"))
        minute_of_session.append(np.arange(n))
    ns = np.concatenate([s.tz_convert("UTC").as_unit("ns").asi8 for s in starts_l])
    starts = pd.DatetimeIndex(pd.to_datetime(ns, unit="ns", utc=True)).as_unit("ns")
    ends = starts + pd.Timedelta(minutes=1)
    return starts, ends, np.concatenate(minute_of_session)


def bars_from_log_returns(bodies: dict[str, np.ndarray], starts: pd.DatetimeIndex, ends: pd.DatetimeIndex, *,
                          dataset_version: str, seed: int, latency_s: dict[str, float] | None = None) -> pd.DataFrame:
    """Decimal OHLCV 1-minute bars from per-symbol one-bar log returns (open = previous close level)."""
    frames: list[pd.DataFrame] = []
    lat = latency_s or {}
    for k, (sym, body) in enumerate(sorted(bodies.items())):
        rng = np.random.RandomState(seed + 97 * k)
        logc = np.log(100.0 + 10 * k) + np.cumsum(body)
        close = np.exp(logc)
        open_ = np.exp(logc - body)
        wick = np.abs(rng.normal(0.0, 1.5 / 10_000.0, size=len(body)))
        high = np.maximum(open_, close) * (1 + wick)
        low = np.minimum(open_, close) * (1 - wick)
        o, h, lo, c = (np.round(x, 2) for x in (open_, high, low, close))
        h = np.maximum.reduce([o, h, lo, c])
        lo = np.minimum.reduce([o, h, lo, c])
        vol = (5000 * (1 + np.abs(rng.normal(0, 0.3, size=len(body))))).astype(np.int64)
        frames.append(pd.DataFrame({
            "symbol": sym, "timeframe": "1Min", "start_ts": starts, "end_ts": ends,
            "open": [Decimal(f"{x:.2f}") for x in o], "high": [Decimal(f"{x:.2f}") for x in h],
            "low": [Decimal(f"{x:.2f}") for x in lo], "close": [Decimal(f"{x:.2f}") for x in c],
            "volume": [Decimal(int(v)) for v in vol],
            "available_at_ts": ends + pd.Timedelta(seconds=lat.get(sym, 1.0)),
            "dataset_version": dataset_version, "is_synthetic": True,
        }))
    return pd.concat(frames, ignore_index=True)


def planted_bars(symbols: tuple[str, ...] = ("AAA", "BBB", "CCC"), *, sessions: int = 9, seed: int = 5,
                 up_bps: float = 1.5, down_bps: float = -0.5, vol_bps: float = 5.0) -> pd.DataFrame:
    """1-minute bars with a PLANTED, learnable time-of-day drift (positive control for selection).

    Minutes [90, 270) after the open drift ``up_bps`` per bar, all other minutes ``down_bps``; noise ``vol_bps``.
    ``tod`` is a model feature, so a fitted model can learn the drift; the zero baseline cannot.
    """
    starts, ends, mos = _minute_spans(sessions)
    mu = np.where((mos >= 90) & (mos < 270), up_bps, down_bps) / 10_000.0
    bodies = {sym: mu + np.random.RandomState(seed + 97 * k).normal(0.0, vol_bps / 10_000.0, size=len(mos))
              for k, sym in enumerate(symbols)}
    return bars_from_log_returns(bodies, starts, ends, dataset_version="synthetic-planted-test", seed=seed)


def regime_bars(symbols: tuple[str, ...] = ("AAA", "BBB", "CCC"), *, sessions: int = 10, seed: int = 13,
                calm_bps: float = 2.0, turbulent_bps: float = 12.0, p_stay: float = 0.97, block: int = 5
                ) -> tuple[pd.DataFrame, np.ndarray]:
    """1-minute bars whose COMMON market factor switches volatility by a persistent hidden 2-state Markov chain.

    The state changes only between ``block``-minute blocks (staying with probability ``p_stay``). Returns the bars and
    the true state per minute (1 = turbulent) so tests can check the filtered regime.
    """
    starts, ends, mos = _minute_spans(sessions)
    n = len(mos)
    rng = np.random.RandomState(seed)
    n_blocks = n // block + 1
    state = np.zeros(n_blocks, dtype=np.int64)
    for k in range(1, n_blocks):
        state[k] = state[k - 1] if rng.uniform() < p_stay else 1 - state[k - 1]
    s_min = state[np.arange(n) // block]
    market = rng.normal(0.0, 1.0, n) * np.where(s_min == 1, turbulent_bps, calm_bps) / 10_000.0
    bodies = {sym: market + np.random.RandomState(seed + 31 * (k + 1)).normal(0.0, 1.0 / 10_000.0, size=n)
              for k, sym in enumerate(symbols)}
    return bars_from_log_returns(bodies, starts, ends, dataset_version="synthetic-regime-test", seed=seed), s_min


def train_and_save(bars: pd.DataFrame, base: Path, cfg: SuiteConfig | None = None, **kw: Any
                   ) -> tuple[TrainResult, ModelRegistry]:
    res = train_suite(bars, CostModel(), base, config=cfg or fast_config(), **kw)
    return res, save_registry(res, base)


def strip_ids(d: dict[str, Any]) -> dict[str, Any]:
    out = dict(d)
    out.pop("prediction_id", None)
    return out
