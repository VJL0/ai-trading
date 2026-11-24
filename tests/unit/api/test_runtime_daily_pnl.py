"""Daily loss accounting across overnight holds, partial exits and restart; no broker/API calls."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from packages.contracts.enums import GatewayKind, OrderEventType, OrderState, Side, TradingMode
from packages.contracts.models import PositionSnapshot
from packages.contracts.policy import RiskPolicy
from packages.core.audit import get_state, set_state
from services.api.runtime import Runtime
from services.order_ledger.timeutil import trading_day
from tests.unit.order_ledger.ol_support import Harness, claim, make_harness, stream_event
from tests.unit.risk.risk_factories import make_account

NEXT_DAY = datetime(2026, 11, 5, 15, tzinfo=UTC)


@pytest.fixture
def h(tmp_path: Path) -> Iterator[Harness]:
    harness = make_harness(tmp_path)
    yield harness
    harness.db.close()


def _fill(h: Harness, side: Side, quantity: str, price: str) -> None:
    intent = h.factory.make(side=side, qty=quantity, limit=price)
    claim(h, intent)
    result = h.ledger.apply_event(stream_event(
        intent.client_order_id, OrderEventType.FILL, OrderState.FILLED, h.clock.now(),
        fill=(quantity, price), cum=quantity, exec_id=intent.intent_id, broker_order_id=intent.intent_id,
    ))
    assert result.applied


def _position(h: Harness, qty: str, price: str, day_pl: str | None) -> PositionSnapshot:
    quantity, mark = Decimal(qty), Decimal(price)
    ledger = next(p for p in h.ledger.positions() if p.symbol == "SPY")
    return PositionSnapshot(
        symbol="SPY", quantity=quantity, avg_entry_price=ledger.avg_entry_price, cost_basis=ledger.cost_basis,
        market_value=quantity * mark, current_price=mark, unrealized_pl=quantity * mark - ledger.cost_basis,
        unrealized_intraday_pl=Decimal(day_pl) if day_pl is not None else None, asof=h.clock.now(), source="BROKER",
    )


def _runtime(h: Harness, positions: list[PositionSnapshot] | None = None) -> Runtime:
    # Assemble only the real risk-context/accounting boundary, with a read-only in-memory broker.
    rt = Runtime.__new__(Runtime)
    rt.db, rt.ledger, rt.clock, rt.entries = h.db, h.ledger, h.clock, h.entries
    rt.entries = SimpleNamespace(is_locked=lambda: (False, []), exits_locked=lambda: (False, []))
    rt.policy = RiskPolicy()
    rt.settings = SimpleNamespace(mode=TradingMode.PAPER, gateway=GatewayKind.FAKE)
    rt.feed_monitor = SimpleNamespace(status=lambda: {"status": "OK"})

    async def account() -> Any:
        return make_account(equity="10400", last_equity="10500", now=h.clock.now())

    async def held() -> list[PositionSnapshot]:
        return positions or []

    rt.gateway = SimpleNamespace(get_account=account, get_positions=held)
    rt.db.write(lambda c: set_state(c, "last_reconciliation", {"status": "CLEAN"}))
    return rt


def test_daily_drawdown_does_not_use_overnight_lifetime_gain(h: Harness) -> None:
    _fill(h, Side.BUY, "10", "100")
    h.clock.set(NEXT_DAY)
    rt = _runtime(h)
    realized, unreal, errors = rt._daily_position_pnl(trading_day(h.clock.now()), [_position(h, "10", "140", "-100")])
    assert not errors
    assert realized == 0 and unreal == Decimal("-100")  # lifetime unrealized is +400
    assert get_state(h.db, "daily_position_marks:2026-11-05")["SPY"]["market_value"] == "1500"


def test_partial_and_full_overnight_exit_keep_previous_close_after_restart(h: Harness) -> None:
    _fill(h, Side.BUY, "10", "100")
    h.clock.set(NEXT_DAY)
    rt = _runtime(h)
    day = trading_day(h.clock.now())
    rt._daily_position_pnl(day, [_position(h, "10", "140", "-100")])
    _fill(h, Side.SELL, "4", "140")
    realized, unreal, errors = rt._daily_position_pnl(day, [_position(h, "6", "130", "-120")])
    assert not errors
    assert (realized, unreal) == (Decimal("-40"), Decimal("-120"))
    assert h.ledger.daily_realized_pl(day) == Decimal("160")  # lifetime gain cannot mask the daily loss
    restarted = h.reopen()
    try:
        rt = _runtime(restarted)
        _fill(restarted, Side.SELL, "6", "130")
        realized, unreal, errors = rt._daily_position_pnl(day, [])
        assert not errors
        assert realized == Decimal("-160") and unreal == 0
    finally:
        restarted.db.close()


def test_partial_exit_before_first_observation_can_recover_opening_mark(h: Harness) -> None:
    _fill(h, Side.BUY, "10", "100")
    h.clock.set(NEXT_DAY)
    _fill(h, Side.SELL, "4", "140")
    result = _runtime(h)._daily_position_pnl(trading_day(h.clock.now()), [_position(h, "6", "130", "-120")])
    assert result == (Decimal("-40"), Decimal("-120"), [])


async def test_missing_overnight_daily_pnl_blocks_entries_but_not_exits(h: Harness) -> None:
    _fill(h, Side.BUY, "10", "100")
    h.clock.set(NEXT_DAY)
    rt = _runtime(h, [_position(h, "10", "140", None)])
    context = await rt.risk_context(None)
    assert context.health.entries_locked
    assert any("no broker intraday P&L" in r for r in context.health.lock_reasons)
    assert context.health.exits_locked is False
    assert get_state(h.db, "daily_pnl")["available"] is False


async def test_missing_opening_mark_after_full_overnight_exit_blocks_entries(h: Harness) -> None:
    _fill(h, Side.BUY, "10", "100")
    h.clock.set(NEXT_DAY)
    _fill(h, Side.SELL, "10", "140")
    context = await _runtime(h).risk_context(None)
    assert context.health.entries_locked and context.health.exits_locked is False
    assert any("full overnight exit" in r for r in context.health.lock_reasons)


async def test_daily_keys_and_fills_use_new_york_date_after_utc_midnight(h: Harness) -> None:
    _fill(h, Side.BUY, "10", "100")
    _fill(h, Side.SELL, "10", "99")
    h.clock.set(datetime(2026, 11, 5, 1, tzinfo=UTC))  # still November 4 in New York
    context = await _runtime(h).risk_context(None)
    assert context.daily_realized_pl == Decimal("-10")
    assert context.start_of_day_equity == Decimal("10500")
    assert get_state(h.db, "daily_pnl")["trading_day"] == "2026-11-04"
    assert get_state(h.db, "sod_equity:2026-11-05") is None


def test_same_day_positions_need_no_overnight_pnl_field(h: Harness) -> None:
    _fill(h, Side.BUY, "10", "100")
    result = _runtime(h)._daily_position_pnl(trading_day(h.clock.now()), [_position(h, "10", "99", None)])
    assert result == (Decimal(0), Decimal("-10"), [])
