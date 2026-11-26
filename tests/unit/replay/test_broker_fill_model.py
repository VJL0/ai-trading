"""ReplayBroker entry fill model on top of the real FakeBroker (signed intents, offline)."""

from __future__ import annotations

import json
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from packages.contracts.enums import OrderType, Side, TradingMode
from packages.core.clock import ManualClock
from packages.core.db import Database
from packages.core.signing import IntentSigner
from services.order_ledger.fake_broker import QuoteBook
from services.order_ledger.ledger import OrderLedger
from services.replay.broker import FillModel, ReplayBroker
from tests.unit.order_ledger.ol_support import T0, IntentFactory, always_open, make_quote


def _rig(tmp: Path, fm: FillModel, *, state: bool = False) -> tuple[ReplayBroker, IntentFactory, ManualClock, QuoteBook]:
    clock = ManualClock(T0)
    signer = IntentSigner.load_or_create(tmp / "keys")
    ledger = OrderLedger(Database(tmp / "ledger.sqlite"), clock, TradingMode.REPLAY, verifier=signer.verifier)
    book = QuoteBook()
    broker = ReplayBroker(clock, book, starting_cash=Decimal("1000000"), verifier=signer.verifier,
                          session_bounds_fn=always_open, fill_model=fm,
                          state_path=(tmp / "broker.json") if state else None)
    return broker, IntentFactory(ledger, signer, clock, mode=TradingMode.REPLAY), clock, book


async def _submit(broker: ReplayBroker, f: IntentFactory, book: QuoteBook, clock: ManualClock, qty: str = "10",
                  limit: str = "100.02") -> str:
    book.set(make_quote("SPY", "100.00", "100.02", clock.now()))
    intent = f.make(qty=qty, limit=limit, record=False)
    await broker.submit(intent)
    return intent.client_order_id


async def test_entry_never_fills_at_acceptance_only_on_next_quote(tmp_path: Path) -> None:
    broker, f, clock, book = _rig(tmp_path, FillModel())
    coid = await _submit(broker, f, book, clock)
    assert broker.orders[coid].filled == 0  # the decision quote was marketable: still no fill
    clock.advance(1)
    broker.advance(make_quote("SPY", "100.00", "100.02", clock.now()))
    o = broker.orders[coid]
    assert o.status == "filled" and o.notional == Decimal("1000.20") and broker.entry_outcomes[coid] == "FILLED"


async def test_gap_past_limit_is_a_permanent_non_fill(tmp_path: Path) -> None:
    broker, f, clock, book = _rig(tmp_path, FillModel())
    coid = await _submit(broker, f, book, clock)
    clock.advance(1)
    broker.advance(make_quote("SPY", "100.10", "100.12", clock.now()))  # opened above the limit
    clock.advance(5)
    broker.advance(make_quote("SPY", "99.90", "99.92", clock.now()))  # later marketable again: no second chance
    assert broker.orders[coid].filled == 0 and broker.entry_outcomes[coid] == "NON_FILL_LIMIT"


async def test_missing_next_open_does_not_match_a_later_quote(tmp_path: Path) -> None:
    broker, f, clock, book = _rig(tmp_path, FillModel())
    coid = await _submit(broker, f, book, clock)
    clock.advance(6)
    broker.advance(make_quote("SPY", "100.00", "100.02", clock.now()))
    assert broker.orders[coid].filled == 0


async def test_random_non_fill_and_partial_fill(tmp_path: Path) -> None:
    broker, f, clock, book = _rig(tmp_path, FillModel(entry_nonfill_prob=1.0))
    coid = await _submit(broker, f, book, clock)
    clock.advance(1)
    broker.advance(make_quote("SPY", "100.00", "100.02", clock.now()))
    assert broker.orders[coid].filled == 0 and broker.entry_outcomes[coid] == "NON_FILL_RANDOM"

    broker, f, clock, book = _rig(tmp_path / "p", FillModel(entry_partial_prob=1.0, partial_fraction=(0.3, 0.6)))
    coid = await _submit(broker, f, book, clock)
    clock.advance(1)
    broker.advance(make_quote("SPY", "100.00", "100.02", clock.now()))
    o = broker.orders[coid]
    assert o.status == "partially_filled" and Decimal(3) <= o.filled <= Decimal(6)
    clock.advance(5)
    broker.advance(make_quote("SPY", "99.90", "99.92", clock.now()))
    assert broker.orders[coid].filled == o.filled  # the remainder never fills on later quotes
    broker.injector.fill_on_cancel = 1
    await broker.cancel(coid)  # cancel/fill race completes it
    assert broker.orders[coid].status == "filled"


async def test_fill_draws_are_deterministic_and_survive_a_restart(tmp_path: Path) -> None:
    fm = FillModel(seed=11, entry_nonfill_prob=0.5, entry_partial_prob=0.5)
    outcomes = []
    for k in range(2):
        broker, f, clock, book = _rig(tmp_path / str(k), fm)
        coids = []
        for i in range(8):
            clock.set(T0 + timedelta(minutes=i))
            coids.append(await _submit(broker, f, book, clock, qty=str(10 + i)))
            clock.advance(1)
            broker.advance(make_quote("SPY", "100.00", "100.02", clock.now()))
        outcomes.append([(broker.entry_outcomes[c], broker.orders[c].filled) for c in coids])
    assert outcomes[0] == outcomes[1]
    assert {o for o, _ in outcomes[0]} >= {"NON_FILL_RANDOM"}

    broker, f, clock, book = _rig(tmp_path / "s", FillModel(entry_nonfill_prob=1.0), state=True)
    coid = await _submit(broker, f, book, clock)
    clock.advance(1)
    broker.advance(make_quote("SPY", "100.00", "100.02", clock.now()))
    # The open order and its non-fill must be durable in the same atomic replacement.
    assert json.loads((tmp_path / "s" / "broker.json").read_text())["entry_outcomes"][coid] == "NON_FILL_RANDOM"
    again = ReplayBroker(clock, book, starting_cash=Decimal("10000"), session_bounds_fn=always_open,
                         fill_model=FillModel(), state_path=tmp_path / "s" / "broker.json")
    clock.advance(1)
    again.advance(make_quote("SPY", "100.00", "100.02", clock.now()))
    assert again.orders[coid].filled == 0 and again.entry_outcomes[coid] == "NON_FILL_RANDOM"


async def test_sells_and_stops_keep_fake_broker_semantics(tmp_path: Path) -> None:
    broker, f, clock, book = _rig(tmp_path, FillModel())
    broker.seed_position("SPY", Decimal("10"), Decimal("100"))
    book.set(make_quote("SPY", "100.00", "100.02", clock.now()))
    stop = f.make(side=Side.SELL, qty="10", limit="98.00", order_type=OrderType.STOP_LIMIT, stop="98.50",
                  record=False)
    await broker.submit(stop)
    assert broker.resting_stops("SPY") == [Decimal("98.50")]
    clock.advance(1)
    broker.advance(make_quote("SPY", "98.50", "98.52", clock.now()))
    assert broker.orders[stop.client_order_id].status == "filled" and broker.resting_stops("SPY") == []


def test_fill_model_validation() -> None:
    with pytest.raises(ValueError, match="probabilities"):
        FillModel(entry_nonfill_prob=1.5)
    with pytest.raises(ValueError, match="partial_fraction"):
        FillModel(partial_fraction=(0.0, 0.5))
    with pytest.raises(TypeError):
        FillModel(slippage_bps=1.0)  # type: ignore[arg-type]
