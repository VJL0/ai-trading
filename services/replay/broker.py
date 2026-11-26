"""Replay fill simulator: the real ``FakeBroker`` with an entry fill model (narrow subclass, documented adapter).

Everything except entry BUY matching is the unchanged FakeBroker: intent validation (signature, ledger claim check,
mode/gateway, tick, DAY/regular session, cash cover, long-only), Alpaca-like order events with unique execution
ids, DAY expiry at the close, STOP_LIMIT triggering (bid at or below the stop, then a resting limit that does not
fill on a gap below it), resting SELL fills on later quotes, failure injection (timeouts after accept, lost
lookups, fill/cancel races, dropped events ...).

Entry model (marketable LIMIT BUY against the NEXT bar):

* an entry never fills at acceptance (the decision quote is the bar the decision was made on); it is matched once,
  against the next bar's opening quote (+1 s after the decision); later quotes never stand in for a missing open;
* not marketable there (``limit < ask``: the market gapped past the limit) -> no fill, ever (like QuantLab's
  ``NON_FILL``; the position manager cancels it after ``entry_fill_timeout_s``);
* marketable -> with probability ``entry_nonfill_prob`` no fill (queue/latency), else with probability
  ``entry_partial_prob`` a partial fill of ``U(partial_fraction)`` of the quantity (at least one share, at most
  quantity - 1), else a full fill; at ``min(limit, ask x (1 + slippage))``. The remainder never fills on later
  quotes; only a cancel/fill race (failure injection) can complete it.

Randomness is a per-order seeded generator derived from ``(seed, symbol, submitted_at, qty, limit)``, so a run, a
re-run and a resumed run draw identical outcomes. The matched-entry set is persisted atomically with FakeBroker state
so a restarted replay never gives an entry a second chance.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import timedelta
from decimal import ROUND_CEILING, Decimal
from typing import Any

import numpy as np

from packages.contracts.enums import OrderType, Side
from packages.contracts.models import Quote
from packages.core.money import BPS, ONE, price_tick
from services.order_ledger.fake_broker import FakeBroker, _Order
from services.order_ledger.timeutil import iso


@dataclass(frozen=True)
class FillModel:
    seed: int = 0
    entry_nonfill_prob: float = 0.0
    entry_partial_prob: float = 0.0
    partial_fraction: tuple[float, float] = (0.3, 0.7)
    slippage_bps: Decimal = Decimal(0)
    quote_step_s: int = 5
    stop_crossing: bool = True

    def __post_init__(self) -> None:
        for p in (self.entry_nonfill_prob, self.entry_partial_prob):
            if not 0.0 <= p <= 1.0:
                raise ValueError("fill probabilities must be in [0, 1]")
        lo, hi = self.partial_fraction
        if not 0.0 < lo <= hi < 1.0:
            raise ValueError("partial_fraction must satisfy 0 < lo <= hi < 1")
        if not isinstance(self.slippage_bps, Decimal):
            raise TypeError("slippage_bps must be a Decimal")

    def to_json(self) -> dict[str, Any]:
        return {"seed": self.seed, "entry_nonfill_prob": self.entry_nonfill_prob,
                "entry_partial_prob": self.entry_partial_prob, "partial_fraction": list(self.partial_fraction),
                "slippage_bps": str(self.slippage_bps), "quote_step_s": self.quote_step_s,
                "stop_crossing": self.stop_crossing}


class ReplayBroker(FakeBroker):
    def __init__(self, *args: Any, fill_model: FillModel, **kwargs: Any) -> None:
        self.fill_model = fill_model
        self.entry_outcomes: dict[str, str] = {}
        super().__init__(*args, slippage_bps=fill_model.slippage_bps, **kwargs)

    # ------------------------------------------------------------------ persistence of the matched-entry set
    def _load(self) -> None:
        super()._load()
        if self.state_path is not None and self.state_path.is_file():
            saved = json.loads(self.state_path.read_text())
            previous = self.state_path.with_name(self.state_path.stem + ".entries.json")
            self.entry_outcomes = saved.get("entry_outcomes", {})
            if "entry_outcomes" not in saved and previous.is_file():
                self.entry_outcomes = json.loads(previous.read_text())

    def advance(self, quote: Quote | None = None) -> None:
        """FakeBroker.advance; the state file is rewritten only when the quote changed broker state (a fill, an
        expiry, a stop trigger or an entry match outcome), not on every quote (thousands per session)."""
        before = self._state_token()
        self._quiet = True
        try:
            super().advance(quote)
        finally:
            self._quiet = False
        if self._state_token() != before:
            self._save()

    def _state_token(self) -> tuple[int, int, int]:
        return self._event_seq, sum(1 for o in self.orders.values() if o.triggered), len(self.entry_outcomes)

    _quiet = False

    def _save(self) -> None:
        path = self.state_path
        if self._quiet or path is None:
            return
        # FakeBroker's snapshot shape, with the one-shot match decision in the same atomic write.
        state = {
            "cash": str(self.cash), "starting_cash": str(self.starting_cash),
            "positions": {s: {"qty": str(p.qty), "cost": str(p.cost)} for s, p in self.positions.items()},
            "orders": [o.to_json() for o in self.orders.values()],
            "order_seq": self._order_seq, "event_seq": self._event_seq, "exec_seq": self._exec_seq,
            "entry_outcomes": self.entry_outcomes,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, sort_keys=True))
        tmp.replace(path)

    # ------------------------------------------------------------------ matching
    def _rng(self, o: _Order) -> np.random.Generator:
        key = f"{self.fill_model.seed}|{o.symbol}|{iso(o.submitted_at)}|{o.qty}|{o.limit}"
        return np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest()[:16], 16))

    def _try_fill(self, o: _Order, quote: Quote | None = None) -> None:
        if not (o.side == Side.BUY and o.order_type == OrderType.LIMIT):
            super()._try_fill(o, quote)
            return
        if not o.is_open or o.status == "pending_cancel" or o.coid in self.entry_outcomes:
            return
        if quote is None or quote.symbol != o.symbol or quote.exchange_ts <= o.submitted_at:
            return  # never at acceptance: only the next market event for this symbol
        q = self._usable(quote)
        if q is None:
            return
        if q.exchange_ts != o.submitted_at + timedelta(seconds=1):
            self.entry_outcomes[o.coid] = "NON_FILL_NO_OPEN"
            return
        fm = self.fill_model
        if o.limit < q.ask:
            self.entry_outcomes[o.coid] = "NON_FILL_LIMIT"
            return
        rng = self._rng(o)
        if rng.random() < fm.entry_nonfill_prob:
            self.entry_outcomes[o.coid] = "NON_FILL_RANDOM"
            return
        qty = o.remaining
        outcome = "FILLED"
        if qty > 1 and rng.random() < fm.entry_partial_prob:
            frac = rng.uniform(*fm.partial_fraction)
            part = Decimal(max(1, min(int(qty) - 1, math.floor(float(qty) * frac))))
            qty, outcome = part, "PARTIAL"
        raw = q.ask * (ONE + fm.slippage_bps / BPS)
        px = min(o.limit, raw.quantize(price_tick(raw), rounding=ROUND_CEILING))
        self.entry_outcomes[o.coid] = outcome
        self._book_fill(o, qty, px)

    def resting_stops(self, symbol: str) -> list[Decimal]:
        """Untriggered protective stop prices resting for ``symbol`` (the feed prints them on a crossing)."""
        with self._lock:
            return [o.stop for o in self.orders.values()
                    if o.symbol == symbol and o.is_open and o.order_type == OrderType.STOP_LIMIT and not o.triggered
                    and o.stop is not None]
