"""The long-only, one-position strategy implied by a score, and its net performance statistics.

Rule (applied identically to every candidate and to the zero baseline): walk the decision times of an evaluation
window chronologically; when flat, take the symbol with the highest score at that decision time (ties -> first
symbol alphabetically); enter iff ``score > threshold``; hold for the horizon; the position blocks new entries until
its exit bar's data is published (``exit_ns``). Each trade earns its realised NET return (gross open-to-open return
minus the cost model's round trip). The zero-forecast baseline never trades (net return 0, net Sharpe 0).

Annualisation (stated in every report): ``net_sharpe = mean / stdev(ddof=1) of per-trade net returns *
sqrt(trades per year)``, trades per year = trades / covered years, covered years = (evaluation decision times x bar
minutes) / (252 x 390). Flat periods earn 0 (no cash yield); this equals the annual Sharpe of the trade P&L when trades
are independent. ``p_value`` is the one-sided t-test of H0: mean per-trade net return <= 0.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats as _st

from ..frames import REGULAR_SESSION_MINUTES, TRADING_DAYS_PER_YEAR

ANNUALISATION_NOTE = (
    "net_sharpe = mean/stdev of per-trade net returns x sqrt(trades per year); trades per year = trades / covered "
    "years; covered years = evaluation decision times x bar minutes / (252 x 390); flat time earns 0"
)
MINUTES_PER_YEAR = TRADING_DAYS_PER_YEAR * REGULAR_SESSION_MINUTES


def simulate_one_position(
    decision_ns: np.ndarray,
    score: np.ndarray,
    exit_ns: np.ndarray,
    *,
    threshold: float = 0.0,
) -> np.ndarray:
    """Indices (into the given arrays) of the trades taken. Arrays must be sorted by decision time."""
    d = np.asarray(decision_ns, dtype=np.int64)
    s = np.asarray(score, dtype=np.float64)
    ex = np.asarray(exit_ns, dtype=np.int64)
    if len(d) == 0:
        return np.empty(0, dtype=np.int64)
    if (np.diff(d) < 0).any():
        raise ValueError("decision times must be sorted")
    _, starts = np.unique(d, return_index=True)
    bounds = [*starts.tolist(), len(d)]
    busy_until = np.iinfo(np.int64).min
    taken: list[int] = []
    for a, b in itertools.pairwise(bounds):
        if d[a] < busy_until:
            continue
        block = s[a:b]
        finite = np.isfinite(block)
        if not finite.any():
            continue
        j = int(np.argmax(np.where(finite, block, -np.inf)))
        if block[j] > threshold:
            taken.append(a + j)
            busy_until = int(ex[a + j])
    return np.asarray(taken, dtype=np.int64)


@dataclass
class StrategyStats:
    """Pooled trade results over one or more evaluation windows."""

    net_bps: list[float] = field(default_factory=lambda: [])
    span_minutes: float = 0.0
    n_rows: int = 0

    def add(self, trade_net_bps: np.ndarray, span_minutes: float, n_rows: int) -> None:
        self.net_bps.extend(float(x) for x in np.asarray(trade_net_bps, dtype=np.float64))
        self.span_minutes += float(span_minutes)
        self.n_rows += int(n_rows)

    def metrics(self, horizon_minutes: int) -> dict[str, Any]:
        r = np.asarray(self.net_bps, dtype=np.float64)
        n = len(r)
        years = self.span_minutes / MINUTES_PER_YEAR if self.span_minutes > 0 else float("nan")
        sessions = self.span_minutes / REGULAR_SESSION_MINUTES if self.span_minutes > 0 else float("nan")
        out: dict[str, Any] = {
            "n_test": self.n_rows,
            "n_trades": n,
            "net_return": float(r.sum() / 10_000.0) if n else 0.0,
            "mean_net_bps": float(r.mean()) if n else None,
            "hit_rate": float((r > 0).mean()) if n else None,
            "turnover": float(n / sessions) if n and math.isfinite(sessions) and sessions > 0 else 0.0,
            "exposure": float(min(1.0, n * horizon_minutes / self.span_minutes)) if self.span_minutes > 0 else 0.0,
            "covered_sessions": sessions if math.isfinite(sessions) else None,
        }
        if n == 0:
            out.update({"net_sharpe": 0.0, "p_value": 1.0, "t_stat": None})
            return out
        sd = float(r.std(ddof=1)) if n > 1 else float("nan")
        if n < 2 or not math.isfinite(sd) or sd == 0:
            out.update({"net_sharpe": None, "p_value": 1.0, "t_stat": None})
            return out
        mean = float(r.mean())
        trades_per_year = n / years if math.isfinite(years) and years > 0 else float("nan")
        sharpe = mean / sd * math.sqrt(trades_per_year) if math.isfinite(trades_per_year) else float("nan")
        t = mean / (sd / math.sqrt(n))
        out.update(
            {
                "net_sharpe": sharpe if math.isfinite(sharpe) else None,
                "t_stat": t,
                "p_value": float(_st.t.sf(t, n - 1)),
                "trades_per_year": trades_per_year,
            }
        )
        return out


def window_span_minutes(decision_ns: np.ndarray, bar_minutes: int) -> float:
    """Trading minutes covered by an evaluation window: distinct decision times x bar minutes."""
    return float(len(np.unique(np.asarray(decision_ns, dtype=np.int64))) * bar_minutes)


def evaluate_score(
    decision_ns: np.ndarray,
    score: np.ndarray,
    exit_ns: np.ndarray,
    net_bps: np.ndarray,
    bar_minutes: int,
    stats: StrategyStats,
    *,
    threshold: float = 0.0,
) -> np.ndarray:
    """Simulate one evaluation window and add its trades to ``stats``. Returns the trade indices."""
    idx = simulate_one_position(decision_ns, score, exit_ns, threshold=threshold)
    stats.add(np.asarray(net_bps, dtype=np.float64)[idx], window_span_minutes(decision_ns, bar_minutes), len(decision_ns))
    return idx
