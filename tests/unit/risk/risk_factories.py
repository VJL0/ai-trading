"""Shared builders for risk tests (imported by unit, integration and failure-injection risk tests).

Everything is deterministic: a fixed decision time inside an XNYS regular session, a ManualClock and a
baseline proposal/context that the default ``RiskPolicy()`` approves. Each test then breaks exactly one
input. Money values are strings/Decimals only.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any, cast

from packages.contracts.enums import (
    DataFeed,
    DecisionAction,
    FeedMode,
    GatewayKind,
    MarketSession,
    OrderState,
    OrderType,
    Proposer,
    Side,
    TimeInForce,
    TradingMode,
)
from packages.contracts.models import (
    AccountSnapshot,
    BrokerOrder,
    MarketSnapshot,
    PositionSnapshot,
    Quote,
    RiskContext,
    SystemHealth,
    TradeProposal,
)
from packages.contracts.policy import RiskPolicy
from packages.core.clock import ManualClock
from packages.core.signing import IntentSigner
from services.risk.engine import RiskEngine

# Thursday 2026-11-05 10:00 America/New_York (EST): inside the XNYS regular session 09:30-16:00.
NOW = datetime(2026, 11, 5, 15, 0, tzinfo=UTC)
PRE_MARKET = datetime(2026, 11, 5, 13, 0, tzinfo=UTC)  # 08:00 ET
AFTER_HOURS = datetime(2026, 11, 5, 22, 0, tzinfo=UTC)  # 17:00 ET
SATURDAY = datetime(2026, 11, 7, 15, 0, tzinfo=UTC)
THANKSGIVING = datetime(2026, 11, 26, 15, 0, tzinfo=UTC)
EARLY_CLOSE_AFTER = datetime(2026, 11, 27, 18, 30, tzinfo=UTC)  # 13:30 ET on the 13:00 early-close day

SIGNER = IntentSigner.generate()
"""One key per test process (EC key generation is cheap, but sharing keeps engines comparable)."""


def d(value: str | int) -> Decimal:
    return Decimal(value)


def _raw(value: object) -> Decimal:
    """Typing-only pass-through: the value reaches the contract *unconverted*, so the contract's own Dec
    validation runs (strings parse, floats are rejected). Never ``Decimal(value)``: that would launder floats."""
    return cast(Decimal, value)


def _raw_opt(value: object) -> Decimal | None:
    return cast(Decimal | None, value)


def make_quote(
    symbol: str = "SPY",
    bid: str = "500.00",
    ask: str = "500.04",
    *,
    now: datetime = NOW,
    exchange_age_s: float = 1.0,
    received_age_s: float | None = None,
    feed: DataFeed = DataFeed.SYNTHETIC,
) -> Quote:
    ex = now - timedelta(seconds=exchange_age_s)
    rx = now - timedelta(seconds=received_age_s if received_age_s is not None else max(exchange_age_s - 0.5, 0))
    return Quote(
        symbol=symbol, bid=_raw(bid), ask=_raw(ask), bid_size=Decimal(100), ask_size=Decimal(100),
        exchange_ts=ex, received_ts=rx,
        available_at_ts=max(ex, rx), feed=feed, source_id=f"test-{feed.value.lower()}",
        is_synthetic=feed == DataFeed.SYNTHETIC,
    )


def make_market(
    symbol: str = "SPY",
    *,
    quote: Quote | None = None,
    no_quote: bool = False,
    session: MarketSession = MarketSession.REGULAR,
    now: datetime = NOW,
    data_asof: datetime | None = None,
    feed: DataFeed = DataFeed.SYNTHETIC,
) -> MarketSnapshot:
    q = None if no_quote else (quote if quote is not None else make_quote(symbol, now=now, feed=feed))
    simulated = feed == DataFeed.SYNTHETIC
    return MarketSnapshot(
        snapshot_id=f"snap_{symbol}", symbol=symbol, feed=feed,
        feed_mode=FeedMode.SIMULATED if simulated else FeedMode.IEX_FREE, session=session, quote=q,
        data_asof=data_asof if data_asof is not None else now, source_id="test", dataset_version="ds-test",
        is_simulated=simulated,
    )


def make_account(
    *,
    equity: str = "10000.00",
    cash: str = "10000.00",
    buying_power: str | None = None,
    non_marginable_buying_power: str | None = None,
    last_equity: str | None = None,
    status: str = "ACTIVE",
    trading_blocked: bool = False,
    account_blocked: bool = False,
    is_paper: bool = True,
    now: datetime = NOW,
    age_s: float = 1.0,
) -> AccountSnapshot:
    return AccountSnapshot(
        snapshot_id="acct_test", account_id_masked="PA****1234", is_paper=is_paper, status=status,
        equity=_raw(equity), last_equity=_raw_opt(last_equity), cash=_raw(cash),
        buying_power=_raw(buying_power if buying_power is not None else cash),
        non_marginable_buying_power=_raw_opt(non_marginable_buying_power), trading_blocked=trading_blocked,
        account_blocked=account_blocked, asof=now - timedelta(seconds=age_s), source="FAKE_BROKER",
    )


def make_position(symbol: str = "SPY", qty: str = "2", price: str = "500.00", *, now: datetime = NOW) -> PositionSnapshot:
    q, p = Decimal(qty), Decimal(price)
    return PositionSnapshot(
        symbol=symbol, quantity=q, avg_entry_price=p, cost_basis=q * p, market_value=q * p, current_price=p,
        unrealized_pl=Decimal(0), asof=now, source="FAKE_BROKER",
    )


def make_order(
    symbol: str = "SPY",
    side: Side = Side.BUY,
    qty: str = "1",
    limit: str | None = "500.04",
    *,
    state: OrderState = OrderState.SUBMITTED,
    coid: str = "lat-sim-open1",
    now: datetime = NOW,
) -> BrokerOrder:
    return BrokerOrder(
        client_order_id=coid, broker_order_id=None, intent_id=None, gateway=GatewayKind.FAKE, symbol=symbol,
        side=side, quantity=_raw(qty), order_type=OrderType.LIMIT, limit_price=_raw_opt(limit),
        time_in_force=TimeInForce.DAY,
        broker_status="new", state=state, updated_at=now,
    )


def healthy(**overrides: Any) -> SystemHealth:
    base: dict[str, Any] = {
        "feed_ok": True, "broker_connected": True, "reconciled": True, "engine_ok": True,
        "entries_locked": False, "strategy_state_ok": True, "permissions_ok": True,
    }
    base.update(overrides)
    return SystemHealth(**base)


def make_ctx(**overrides: Any) -> RiskContext:
    now = overrides.pop("now", NOW)
    symbol = overrides.pop("symbol", "SPY")
    base: dict[str, Any] = {
        "now": now,
        "mode": TradingMode.PAPER,
        "gateway": GatewayKind.FAKE,
        "account": make_account(now=now),
        "positions": (),
        "open_orders": (),
        "recent_submission_times": (),
        "recent_intent_fingerprints": (),
        "start_of_day_equity": Decimal("10000.00"),
        "daily_realized_pl": Decimal(0),
        "daily_unrealized_pl": Decimal(0),
        "market": make_market(symbol, now=now),
        "health": healthy(),
    }
    base.update(overrides)
    return RiskContext(**base)


def make_proposal(**overrides: Any) -> TradeProposal:
    """BUY 3 SPY @ 500.04 (the ask), stop 495.00: notional 1500.12, planned loss 15.12. Approved by default."""
    now = overrides.pop("now", NOW)
    base: dict[str, Any] = {
        "proposal_id": "prop_test", "causal_id": "cz_test", "created_at": now, "data_asof": now,
        "action": DecisionAction.TRADE, "proposer": Proposer.TEST, "strategy_id": "test-strategy",
        "symbol": "SPY", "side": Side.BUY, "quantity": Decimal("3"), "order_type": OrderType.LIMIT,
        "limit_price": Decimal("500.04"), "stop_price": Decimal("495.00"),
        "max_planned_loss": Decimal("15.12"), "evidence_ids": ("ev_1",), "dataset_version": "ds-test",
    }
    base.update(overrides)
    return TradeProposal(**base)


def make_no_trade(**overrides: Any) -> TradeProposal:
    base: dict[str, Any] = {
        "proposal_id": "prop_nt", "causal_id": "cz_nt", "created_at": NOW, "data_asof": NOW,
        "action": DecisionAction.NO_TRADE, "no_trade_reason": "no edge", "proposer": Proposer.TEST,
        "strategy_id": "test-strategy", "dataset_version": "ds-test",
    }
    base.update(overrides)
    return TradeProposal(**base)


def make_engine(policy: RiskPolicy | None = None, *, now: datetime = NOW, signer: IntentSigner | None = None) -> RiskEngine:
    return RiskEngine(policy or RiskPolicy(), signer or SIGNER, ManualClock(now))


def failed(decision: Any) -> set[str]:
    return {c.name for c in decision.checks if not c.passed}


def check(decision: Any, name: str) -> Any:
    matches = [c for c in decision.checks if c.name == name]
    assert matches, f"check {name} did not run; ran: {[c.name for c in decision.checks]}"
    return matches[0]
