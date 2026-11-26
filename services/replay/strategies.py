"""Replay strategies: a ``SignalModel`` for the runtime's ``QuantSignalEngine`` plus, where one exists, the same rule
expressed as QuantLab ``Leg``s for the research simulator (parity).

A strategy never builds orders. It only supplies the forecast the runtime's signal engine turns into an
``EdgeEstimate`` through ``decide_edge`` (costs charged once, threshold); the runtime's ``DecisionCycle`` then
builds the proposal (sizing, limit at the ask, stop at the policy default) and the risk engine decides.

Ids:

* ``runtime-baseline`` - the runtime's default engine with no trained registry (momentum drift baseline,
  60-minute horizon, a decision cycle on every bar). No research counterpart (parity unavailable).
* ``tod-<SYM>-<HHMM>-<M>m`` - buy ``SYM`` on the decision at New York ``HH:MM`` (the bar ending then), hold
  ``M`` minutes (multiple of 5), intraday. Deterministic: a fixed gross forecast at that bar, nothing otherwise.
* ``tod-<SYM>-<HHMM>-<N>d`` - the same entry, held for ``N`` sessions (``holding_sessions=N``), exiting at the
  last session's flatten time. Needs a policy that allows overnight positions and lists the proposal's strategy id
  in ``multi_day_strategy_allowlist``; otherwise the risk engine rejects (as it would in paper).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import pandas as pd

from packages.contracts.policy import RiskPolicy
from services.order_ledger.timeutil import NY_TZ
from services.research.signals import BASELINE_STRATEGY_ID, DEFAULT_MIN_NET_EDGE_BPS, ModelPrediction, SignalModel

TOD_RE = re.compile(r"^tod-(?P<sym>[A-Z][A-Z0-9.]{0,9})-(?P<hhmm>\d{4})-(?P<n>\d+)(?P<unit>[md])$")
TOD_EDGE_BPS = 60.0


class TimeOfDayModel:
    """Deterministic forecast: ``edge_bps`` GROSS on the bar ending at ``clock`` (New York) for ``symbol``."""

    calibration_evaluated = False

    def __init__(self, symbol: str, clock: str, horizon_bars: int, strategy_id: str, edge_bps: float) -> None:
        self.symbol = symbol
        self.clock = clock
        self.horizon_bars = horizon_bars
        self.strategy_id = strategy_id
        self.model_version = f"replay:{strategy_id}:edge{edge_bps:g}"
        self.edge_bps = edge_bps

    def predict_latest(self, sym_bars: pd.DataFrame) -> ModelPrediction | None:
        last = sym_bars.iloc[-1]
        if str(last["symbol"]) != self.symbol:
            return None
        end = pd.Timestamp(last["end_ts"]).tz_convert(NY_TZ)
        if f"{end.hour:02d}:{end.minute:02d}" != self.clock:
            return None
        return ModelPrediction(expected_return_bps=self.edge_bps, score=self.edge_bps, probability_up=None,
                               probability_calibrated=False)


@dataclass(frozen=True)
class ReplayStrategy:
    strategy_id: str
    description: str
    horizon_minutes: int
    model_factory: Callable[[], SignalModel | None]
    holding_sessions: int = 1
    decision_clocks: tuple[str, ...] | None = None
    """New York bar-END times ("HH:MM") at which a decision cycle runs; None = after every bar."""
    proposal_strategy_id: str | None = None
    """Strategy id on the runtime's proposal; None keeps the runtime's own (quant-baseline-v1)."""
    min_net_edge_bps: Decimal = DEFAULT_MIN_NET_EDGE_BPS
    symbols: tuple[str, ...] | None = None
    research: Callable[[Any, Sequence[str], RiskPolicy], list[Any]] | None = None
    """``(IntradayPanel, symbols, policy) -> list[Leg]``; None = no research counterpart (parity unavailable)."""

    @property
    def horizon_within_session(self) -> bool:
        return self.holding_sessions == 1

    def to_json(self) -> dict[str, Any]:
        return {"strategy_id": self.strategy_id, "description": self.description,
                "horizon_minutes": self.horizon_minutes, "holding_sessions": self.holding_sessions,
                "decision_clocks": list(self.decision_clocks) if self.decision_clocks else None,
                "proposal_strategy_id": self.proposal_strategy_id, "min_net_edge_bps": str(self.min_net_edge_bps),
                "symbols": list(self.symbols) if self.symbols else None, "has_research_counterpart":
                self.research is not None}


def _tod(strategy_id: str) -> ReplayStrategy:
    m = TOD_RE.match(strategy_id)
    if m is None:
        raise KeyError(strategy_id)
    sym, hhmm, n, unit = m["sym"], m["hhmm"], int(m["n"]), m["unit"]
    clock = f"{hhmm[:2]}:{hhmm[2:]}"
    if unit == "m":
        if n <= 0 or n % 5 or n > 390:
            raise ValueError("tod horizon minutes must be a positive multiple of 5 up to 390")
        horizon, sessions = n, 1
    else:
        if not 2 <= n <= 20:
            raise ValueError("tod holding sessions must be 2..20")
        horizon, sessions = 60, n
    hb = horizon // 5

    def factory() -> SignalModel:
        return TimeOfDayModel(sym, clock, hb, strategy_id, TOD_EDGE_BPS)

    def legs(panel: Any, symbols: Sequence[str], policy: RiskPolicy) -> list[Any]:
        from services.quantlab.intraday import Leg, decision_slot, flatten_slot

        if sym not in symbols or sym not in set(panel.symbols.tolist()):
            return []
        j = panel.col(sym)
        ds = decision_slot(clock)
        out: list[Leg] = []
        for d in range(len(panel.days)):
            if sessions == 1:
                out.append(Leg(d, j, ds, d, ds + 1 + hb, score=TOD_EDGE_BPS,
                               stop_bps=float(policy.default_stop_loss_bps)))
            elif d + sessions - 1 < len(panel.days):
                last = d + sessions - 1
                out.append(Leg(d, j, ds, last, flatten_slot(panel.last_slot[last]), score=TOD_EDGE_BPS,
                               stop_bps=float(policy.default_stop_loss_bps)))
        return out

    return ReplayStrategy(
        strategy_id=strategy_id, description=(f"buy {sym} on the {clock} New York decision, hold "
                                              + (f"{n} min" if unit == "m" else f"{n} sessions")),
        horizon_minutes=horizon, model_factory=factory, holding_sessions=sessions, decision_clocks=(clock,),
        proposal_strategy_id=strategy_id, symbols=(sym,), research=legs)


def _baseline() -> ReplayStrategy:
    return ReplayStrategy(strategy_id="runtime-baseline",
                          description=f"the runtime's default engine without a trained registry ({BASELINE_STRATEGY_ID})",
                          horizon_minutes=60, model_factory=lambda: None)


def get_strategy(strategy_id: str) -> ReplayStrategy:
    if strategy_id == "runtime-baseline":
        return _baseline()
    if TOD_RE.match(strategy_id):
        return _tod(strategy_id)
    raise KeyError(f"unknown replay strategy {strategy_id!r}; known: runtime-baseline, tod-<SYM>-<HHMM>-<M>m, "
                   "tod-<SYM>-<HHMM>-<N>d")


def list_strategies() -> list[dict[str, Any]]:
    return [_baseline().to_json(), _tod("tod-SPY-1000-60m").to_json(), _tod("tod-SPY-1000-2d").to_json()]
