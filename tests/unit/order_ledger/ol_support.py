"""Shared fixtures for order-ledger tests. Everything here is deterministic and offline (no network)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import requests

from packages.contracts.enums import (
    DataFeed,
    DecisionAction,
    EventSource,
    GatewayKind,
    OrderEventType,
    OrderState,
    OrderType,
    Proposer,
    RiskVerdict,
    Side,
    TimeInForce,
    TradingMode,
)
from packages.contracts.models import (
    ApprovedOrderIntent,
    OrderEvent,
    Quote,
    RiskDecision,
    TradeProposal,
)
from packages.core import modes
from packages.core.clock import ManualClock
from packages.core.db import Database
from packages.core.ids import client_order_id, new_causal_id, new_id
from packages.core.secrets import SecretStore
from packages.core.signing import IntentSigner
from services.order_ledger.entries import EntriesLock
from services.order_ledger.fake_broker import FakeBroker, QuoteBook
from services.order_ledger.ledger import OrderLedger

# Wednesday 2026-11-04 10:00 America/New_York (regular session; DST ended 2026-11-01 => UTC-5).
T0 = datetime(2026, 11, 4, 15, 0, tzinfo=UTC)
CLOSE_T0 = datetime(2026, 11, 4, 21, 0, tzinfo=UTC)
NEXT_OPEN = datetime(2026, 11, 5, 14, 30, tzinfo=UTC)
POLICY_VERSION = "pol_testpolicy000001"


def always_open(now: datetime) -> tuple[datetime, datetime] | None:
    day = now.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    return day + timedelta(hours=14, minutes=30), day + timedelta(hours=21)


def make_quote(symbol: str, bid: str, ask: str, at: datetime, *, size: str = "500") -> Quote:
    return Quote(
        symbol=symbol, bid=Decimal(bid), ask=Decimal(ask), bid_size=Decimal(size), ask_size=Decimal(size),
        exchange_ts=at, received_ts=at, available_at_ts=at, feed=DataFeed.SYNTHETIC, source_id="test-quotes",
        is_synthetic=True,
    )


@dataclass
class IntentFactory:
    """Records proposal + APPROVED decision and signs intents exactly like the risk engine would."""

    ledger: OrderLedger
    signer: IntentSigner
    clock: ManualClock
    mode: TradingMode = TradingMode.PAPER
    gateway: GatewayKind = GatewayKind.FAKE
    ttl_s: int = 60
    issued: list[ApprovedOrderIntent] = field(default_factory=list)

    def sign(self, **fields: object) -> ApprovedOrderIntent:
        unsigned = ApprovedOrderIntent.model_validate({**fields, "signature": "0" * 128})
        signature = self.signer.sign(unsigned.signing_string())
        return ApprovedOrderIntent.model_validate({**unsigned.model_dump(), "signature": signature})

    def make(
        self,
        symbol: str = "SPY",
        side: Side = Side.BUY,
        qty: str = "10",
        limit: str = "100.00",
        *,
        requires_human: bool = False,
        record: bool = True,
        ttl_s: int | None = None,
        gateway: GatewayKind | None = None,
        mode: TradingMode | None = None,
        order_type: OrderType = OrderType.LIMIT,
        stop: str | None = None,
    ) -> ApprovedOrderIntent:
        now = self.clock.now()
        stop_price = Decimal(stop) if stop is not None else None
        mode = mode or self.mode
        gateway = gateway or self.gateway
        causal = new_causal_id()
        proposal = TradeProposal(
            proposal_id=new_id("prop"), causal_id=causal, created_at=now, data_asof=now, action=DecisionAction.TRADE,
            proposer=Proposer.TEST, strategy_id="test-strategy", symbol=symbol, side=side, quantity=Decimal(qty),
            order_type=order_type, limit_price=Decimal(limit), stop_price=stop_price, max_planned_loss=Decimal("10"),
            evidence_ids=("ev_test",), dataset_version="test-ds",
        )
        decision = RiskDecision(
            decision_id=new_id("dec"), proposal_id=proposal.proposal_id, causal_id=causal, verdict=RiskVerdict.APPROVED,
            checks=(), reasons=(), policy_version=POLICY_VERSION, mode=mode, evaluated_at=now,
        )
        intent = self.sign(
            intent_id=new_id("intent"), client_order_id=client_order_id(modes.client_order_tag(mode, gateway)),
            proposal_id=proposal.proposal_id, decision_id=decision.decision_id, causal_id=causal, mode=mode, gateway=gateway,
            symbol=symbol, side=side, quantity=Decimal(qty), order_type=order_type, limit_price=Decimal(limit),
            stop_price=stop_price if order_type == OrderType.STOP_LIMIT else None,
            time_in_force=TimeInForce.DAY, extended_hours=False, policy_version=POLICY_VERSION, created_at=now,
            expires_at=now + timedelta(seconds=ttl_s or self.ttl_s), requires_human_approval=requires_human,
            human_approval_id=None, key_id=self.signer.key_id,
        )
        if record:
            self.ledger.record_proposal(proposal)
            self.ledger.record_decision(decision)
            self.ledger.record_intent(intent)
        self.issued.append(intent)
        return intent


@dataclass
class Harness:
    tmp: Path
    db: Database
    clock: ManualClock
    ledger: OrderLedger
    entries: EntriesLock
    signer: IntentSigner
    factory: IntentFactory
    quotes: QuoteBook
    broker: FakeBroker

    def reopen(self) -> Harness:
        """Simulate a process restart: new Database/ledger objects on the same SQLite file; same broker."""
        self.db.close()
        db = Database(self.tmp / "ledger.sqlite")
        ledger = OrderLedger(db, self.clock, self.ledger.mode)
        entries = EntriesLock(db, self.ledger.mode, clock=self.clock)
        factory = IntentFactory(ledger, self.signer, self.clock, mode=self.factory.mode, gateway=self.factory.gateway)
        return Harness(self.tmp, db, self.clock, ledger, entries, self.signer, factory, self.quotes, self.broker)


def make_harness(tmp: Path, *, start: datetime = T0, mode: TradingMode = TradingMode.PAPER, **broker_kw: object) -> Harness:
    clock = ManualClock(start)
    db = Database(tmp / "ledger.sqlite")
    ledger = OrderLedger(db, clock, mode)
    entries = EntriesLock(db, mode, clock=clock)
    signer = IntentSigner.generate()
    factory = IntentFactory(ledger, signer, clock, mode=mode)
    quotes = QuoteBook()
    kw: dict[str, object] = {"session_bounds_fn": always_open, "verifier": signer.verifier}
    kw.update(broker_kw)
    broker = FakeBroker(clock, quotes, **kw)  # type: ignore[arg-type]
    return Harness(tmp, db, clock, ledger, entries, signer, factory, quotes, broker)


def stream_event(
    coid: str,
    etype: OrderEventType,
    state: OrderState,
    at: datetime,
    *,
    event_id: str | None = None,
    fill: tuple[str, str] | None = None,
    cum: str | None = None,
    exec_id: str | None = None,
    broker_order_id: str | None = "bo-1",
    gateway: GatewayKind = GatewayKind.FAKE,
    source: EventSource = EventSource.STREAM,
) -> OrderEvent:
    return OrderEvent(
        event_id=event_id or new_id("ev"), client_order_id=coid, broker_order_id=broker_order_id, event_type=etype,
        state=state, source=source, gateway=gateway,
        fill_quantity=Decimal(fill[0]) if fill else None, fill_price=Decimal(fill[1]) if fill else None,
        cum_filled_quantity=Decimal(cum) if cum is not None else None, broker_exec_id=exec_id, broker_ts=at, received_ts=at,
    )


def pass_gate(_intent: ApprovedOrderIntent) -> list[str]:
    return []


def claim(h: Harness, intent: ApprovedOrderIntent) -> ApprovedOrderIntent:
    """RISK_APPROVED -> SUBMITTING through the ledger's atomic claim (as the ExecutionService does)."""
    claimed = h.ledger.claim_for_submission(intent.client_order_id, "test")
    assert claimed is not None, "claim refused"
    return claimed


def state_of(h: Harness, coid: str) -> OrderState:
    rec = h.ledger.get_record(coid)
    assert rec is not None
    return rec.state


def flags(h: Harness) -> set[str]:
    return set(h.entries.flags())


# ------------------------------------------------------------------------------------------ Alpaca HTTP fakes
# Obviously fake placeholder credentials (PK-prefixed like paper key ids); never real keys.
FAKE_KEY = "PKTESTONLY0000000000"
FAKE_SECRET = "test-secret-not-real-0000000000000000"

ACCOUNT = {
    "id": "8f1e2a4c-0000-4000-8000-000000000001", "account_number": "PA3TESTACCT42", "status": "ACTIVE",
    "currency": "USD", "cash": "100000.00", "buying_power": "200000.00", "non_marginable_buying_power": "100000.00",
    "equity": "100250.5", "last_equity": "100000", "multiplier": "2", "trading_blocked": False,
    "account_blocked": False, "transfers_blocked": False, "shorting_enabled": True, "pattern_day_trader": False,
}
POSITION = {
    "asset_id": "b0b6dd9d-8b9b-48a9-ba46-b9d54906e415", "symbol": "SPY", "exchange": "ARCA", "asset_class": "us_equity",
    "qty": "10", "avg_entry_price": "500.12", "side": "long", "market_value": "5010.00", "cost_basis": "5001.20",
    "unrealized_pl": "8.80", "current_price": "501.00",
}
ORDER = {
    "id": "61e69015-8549-4bfd-b9c3-01e75843f47d", "client_order_id": "lat-paper-01jtest", "symbol": "SPY",
    "asset_class": "us_equity", "created_at": "2026-11-04T15:00:00.123456789Z",
    "updated_at": "2026-11-04T15:00:01.5Z", "submitted_at": "2026-11-04T15:00:00.2Z", "qty": "10",
    "filled_qty": "4", "filled_avg_price": "500.05", "order_type": "limit", "type": "limit", "side": "buy",
    "time_in_force": "day", "limit_price": "500.10", "status": "partially_filled", "extended_hours": False,
}

def _response(status: int, body: Any, url: str) -> requests.Response:
    r = requests.Response()
    r.status_code = status
    r._content = json.dumps(body).encode()
    r.url = url
    r.reason = "TEST"
    return r

class FakeSession:
    """Routes by path; each route is a list of (status, body) consumed in order (last one repeats)."""

    def __init__(self, routes: dict[str, list[tuple[int, Any]]]) -> None:
        self.routes = routes
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def request(self, method: str, url: str, **kw: Any) -> requests.Response:
        self.calls.append((method, url, kw))
        path = url.split("/v2", 1)[1]
        seq = self.routes[path]
        status, body = seq.pop(0) if len(seq) > 1 else seq[0]
        return _response(status, body, url)



def alpaca_routes() -> dict[str, list[tuple[int, Any]]]:
    return {
        "/account": [(200, dict(ACCOUNT))],
        "/positions": [(200, [dict(POSITION)])],
        "/orders": [(200, [dict(ORDER)])],
        "/orders:by_client_order_id": [(200, dict(ORDER))],
    }


def paper_secret_store(key: str = FAKE_KEY, secret: str = FAKE_SECRET) -> SecretStore:
    env = {"ALPACA_PAPER_KEY_ID": key, "ALPACA_PAPER_SECRET_KEY": secret} if key else {}
    return SecretStore(Path("/nonexistent"), use_keychain=False, env=env, dotenv_path=Path("/nonexistent/.env"))
