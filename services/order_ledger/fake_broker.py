"""Deterministic in-process fake broker (``GatewayKind.FAKE``, label SIMULATED) with failure injection.

It behaves like a cash (no-margin) U.S. equities broker for long-only LIMIT DAY orders (and protective
STOP_LIMIT SELL exits) in the regular session: validates at submit, fills marketable limits against a
pluggable point-in-time quote source at the limit or better (configurable slippage, optional partial fills),
keeps resting orders that fill on later quotes (``advance``), expires DAY orders at the session close and
emits Alpaca-like order updates with a unique execution id per fill through ``order_events()``.

A STOP_LIMIT SELL rests untriggered until a usable quote's bid trades at or below its stop (a conservative proxy
for the last trade), then behaves as a LIMIT SELL at its limit price. A gap below the limit leaves it working
and unfilled, exactly the case the position manager must escalate: a stop is never a guaranteed exit price.

Nothing here talks to a network. Every fill is simulated; reports must label it SIMULATED.

Failure injection (``FailureInjector``) covers: timeout after accept (accepted at the broker but the caller
sees ``SubmitTimeout``), network partition before accept, response hang, reject, HTTP-429 rate limit, crash
right after accept (``SimulatedCrash``), stream disconnect, dropped/duplicated/reordered events, failing
reads/lookups and a fill racing a cancel.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from pathlib import Path
from typing import Any

from packages.contracts.enums import (
    EventSource,
    ExecutionLabel,
    GatewayKind,
    OrderEventType,
    OrderState,
    OrderType,
    Side,
    TimeInForce,
    TradingMode,
)
from packages.contracts.interfaces import GatewayRejected, SubmitTimeout
from packages.contracts.models import (
    AccountSnapshot,
    ApprovedOrderIntent,
    BrokerOrder,
    OrderEvent,
    PositionSnapshot,
    Quote,
)
from packages.core import modes
from packages.core.clock import Clock
from packages.core.ids import new_id
from packages.core.money import BPS, ONE, ZERO, floor_shares, price_tick
from packages.core.signing import IntentVerifier

from .errors import BrokerUnavailable, RateLimited, SimulatedCrash, StreamDisconnected
from .sessions import session_bounds
from .timeutil import iso, parse_iso

QuoteSource = Callable[[str], Quote | None]
SessionBounds = Callable[[datetime], tuple[datetime, datetime] | None]

_TERMINAL_STATUS = frozenset({"filled", "canceled", "expired", "rejected"})
_STATUS_STATE: dict[str, OrderState] = {
    "accepted": OrderState.SUBMITTED,
    "new": OrderState.ACKNOWLEDGED,
    "partially_filled": OrderState.PARTIALLY_FILLED,
    "filled": OrderState.FILLED,
    "pending_cancel": OrderState.CANCEL_PENDING,
    "canceled": OrderState.CANCELED,
    "expired": OrderState.EXPIRED,
    "rejected": OrderState.REJECTED,
}


@dataclass
class FailureInjector:
    """Mutable switches; counters are consumed one per triggering call."""

    timeout_after_accept: int = 0
    timeout_before_accept: int = 0
    partition_before_accept: int = 0
    hang_after_accept_s: float = 0.0
    crash_after_accept: int = 0
    reject_next: int = 0
    reject_reason: str = "rejected by fake broker (injected)"
    rate_limit_next: int = 0
    drop_next_events: int = 0
    drop_event_types: frozenset[OrderEventType] = frozenset()
    duplicate_next_events: int = 0
    reorder_window: int = 0
    stream_down: bool = False
    fail_reads: int = 0
    fail_lookups: int = 0
    fill_on_cancel: int = 0

    @staticmethod
    def take(value: int) -> tuple[bool, int]:
        return (True, value - 1) if value > 0 else (False, value)


@dataclass
class _Order:
    coid: str
    broker_id: str
    intent_id: str
    symbol: str
    side: Side
    qty: Decimal
    limit: Decimal
    tif: TimeInForce
    status: str
    filled: Decimal
    notional: Decimal
    submitted_at: datetime
    updated_at: datetime
    session_close: datetime | None
    order_type: OrderType = OrderType.LIMIT
    stop: Decimal | None = None
    triggered: bool = False

    @property
    def remaining(self) -> Decimal:
        return self.qty - self.filled

    @property
    def is_open(self) -> bool:
        return self.status not in _TERMINAL_STATUS

    @property
    def state(self) -> OrderState:
        return _STATUS_STATE[self.status]

    def to_json(self) -> dict[str, Any]:
        return {
            "coid": self.coid, "broker_id": self.broker_id, "intent_id": self.intent_id, "symbol": self.symbol,
            "side": self.side.value, "qty": str(self.qty), "limit": str(self.limit), "tif": self.tif.value,
            "status": self.status, "filled": str(self.filled), "notional": str(self.notional),
            "submitted_at": iso(self.submitted_at), "updated_at": iso(self.updated_at),
            "session_close": iso(self.session_close) if self.session_close else None,
            "order_type": self.order_type.value, "stop": None if self.stop is None else str(self.stop),
            "triggered": self.triggered,
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> _Order:
        return cls(
            coid=d["coid"], broker_id=d["broker_id"], intent_id=d["intent_id"], symbol=d["symbol"], side=Side(d["side"]),
            qty=Decimal(d["qty"]), limit=Decimal(d["limit"]), tif=TimeInForce(d["tif"]), status=d["status"],
            filled=Decimal(d["filled"]), notional=Decimal(d["notional"]), submitted_at=parse_iso(d["submitted_at"]),
            updated_at=parse_iso(d["updated_at"]),
            session_close=parse_iso(d["session_close"]) if d.get("session_close") else None,
            order_type=OrderType(d.get("order_type", OrderType.LIMIT.value)),
            stop=Decimal(d["stop"]) if d.get("stop") is not None else None, triggered=bool(d.get("triggered", False)),
        )


class _FakeStream:
    """Async iterator over events delivered after subscription (like a real websocket: no replay)."""

    def __init__(self, broker: FakeBroker) -> None:
        self._broker = broker
        self._gen = broker.stream_generation
        self._cursor = len(broker.event_log)

    def __aiter__(self) -> _FakeStream:
        return self

    async def __anext__(self) -> OrderEvent:
        b = self._broker
        while True:
            if b.injector.stream_down or self._gen != b.stream_generation:
                raise StreamDisconnected("fake broker order-update stream disconnected")
            if self._cursor < len(b.event_log):
                ev = b.event_log[self._cursor]
                self._cursor += 1
                return ev
            await asyncio.sleep(b.poll_interval_s)


@dataclass
class _Position:
    qty: Decimal = ZERO
    cost: Decimal = ZERO


class FakeBroker:
    kind: GatewayKind = GatewayKind.FAKE
    is_paper: bool = True
    label: ExecutionLabel = ExecutionLabel.SIMULATED

    def __init__(
        self,
        clock: Clock,
        quote_source: QuoteSource | None = None,
        *,
        starting_cash: Decimal = Decimal("100000"),
        slippage_bps: Decimal = ZERO,
        max_fill_per_match: Decimal | None = None,
        use_quote_size: bool = False,
        max_quote_age_s: float = 60.0,
        verifier: IntentVerifier | None = None,
        session_bounds_fn: SessionBounds | None = None,
        state_path: Path | None = None,
        poll_interval_s: float = 0.005,
        account_suffix: str = "0001",
        claim_check: Callable[[str], bool] | None = None,
    ) -> None:
        amounts: tuple[object, ...] = (starting_cash, slippage_bps, max_fill_per_match)
        if any(isinstance(a, float) for a in amounts):
            raise TypeError("FakeBroker amounts must be Decimal (binary floats rejected)")
        self.clock = clock
        self.quote_source = quote_source
        self.slippage_bps = Decimal(slippage_bps)
        self.max_fill_per_match = max_fill_per_match
        self.use_quote_size = use_quote_size
        self.max_quote_age = timedelta(seconds=max_quote_age_s)
        self.verifier = verifier
        self.session_bounds_fn: SessionBounds = session_bounds_fn or session_bounds
        self.state_path = state_path
        self.poll_interval_s = poll_interval_s
        self.account_suffix = account_suffix
        # Gateway ownership (invariant 3): when wired by ``build_gateway`` only intents the order ledger has claimed
        # for submission (state SUBMITTING) execute, so a signed intent handed to the broker object by any other
        # route (admin endpoint, replay, script) is refused.
        self.claim_check = claim_check
        self.injector = FailureInjector()
        self._lock = threading.RLock()
        # broker state
        self.starting_cash = Decimal(starting_cash)
        self.cash = Decimal(starting_cash)
        self.positions: dict[str, _Position] = {}
        self.orders: dict[str, _Order] = {}
        self._order_seq = 0
        self._event_seq = 0
        self._exec_seq = 0
        self._last_quotes: dict[str, Quote] = {}
        # stream + bookkeeping (ephemeral)
        self.event_log: list[OrderEvent] = []
        self.dropped_events: list[OrderEvent] = []
        self._reorder_buffer: list[OrderEvent] = []
        self.stream_generation = 0
        self._drain_cursor = 0
        self.submit_attempts: list[str] = []
        self.accepted_submissions: list[str] = []
        self.cancel_requests: list[str] = []
        self._load()

    # ================================================================== persistence
    def _load(self) -> None:
        if self.state_path is None or not self.state_path.is_file():
            return
        d = json.loads(self.state_path.read_text())
        self.cash = Decimal(d["cash"])
        self.starting_cash = Decimal(d.get("starting_cash", d["cash"]))
        self.positions = {s: _Position(Decimal(p["qty"]), Decimal(p["cost"])) for s, p in d["positions"].items()}
        self.orders = {o["coid"]: _Order.from_json(o) for o in d["orders"]}
        self._order_seq, self._event_seq, self._exec_seq = int(d["order_seq"]), int(d["event_seq"]), int(d["exec_seq"])

    def _save(self) -> None:
        if self.state_path is None:
            return
        d = {
            "cash": str(self.cash), "starting_cash": str(self.starting_cash),
            "positions": {s: {"qty": str(p.qty), "cost": str(p.cost)} for s, p in self.positions.items()},
            "orders": [o.to_json() for o in self.orders.values()],
            "order_seq": self._order_seq, "event_seq": self._event_seq, "exec_seq": self._exec_seq,
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(d, indent=1, sort_keys=True))
        os.replace(tmp, self.state_path)

    # ================================================================== stream
    def order_events(self) -> _FakeStream:
        return _FakeStream(self)

    def disconnect_stream(self) -> None:
        self.injector.stream_down = True
        self.stream_generation += 1

    def reconnect_stream(self) -> None:
        self.injector.stream_down = False

    def drain_events(self) -> list[OrderEvent]:
        """Synchronous test helper: events delivered since the previous drain."""
        out = self.event_log[self._drain_cursor:]
        self._drain_cursor = len(self.event_log)
        return out

    def flush_reordered(self) -> None:
        with self._lock:
            buf, self._reorder_buffer = self._reorder_buffer, []
            for ev in reversed(buf):
                self._deliver(ev)

    def _deliver(self, ev: OrderEvent) -> None:
        self.event_log.append(ev)
        take, self.injector.duplicate_next_events = FailureInjector.take(self.injector.duplicate_next_events)
        if take:
            self.event_log.append(ev)

    def _emit(self, ev: OrderEvent) -> None:
        inj = self.injector
        if ev.event_type in inj.drop_event_types:
            self.dropped_events.append(ev)
            return
        take, inj.drop_next_events = FailureInjector.take(inj.drop_next_events)
        if take:
            self.dropped_events.append(ev)
            return
        if inj.reorder_window > 0:
            self._reorder_buffer.append(ev)
            if len(self._reorder_buffer) >= inj.reorder_window:
                self.flush_reordered()
            return
        self._deliver(ev)

    def _event(
        self, o: _Order, etype: OrderEventType, *, fill_qty: Decimal | None = None, fill_price: Decimal | None = None,
        reason: str | None = None,
    ) -> OrderEvent:
        self._event_seq += 1
        exec_id = None
        if fill_qty is not None:
            self._exec_seq += 1
            exec_id = f"fbx-{self.account_suffix}-{self._exec_seq:08d}"
        now = self.clock.now()
        return OrderEvent(
            event_id=f"fbev-{self.account_suffix}-{self._event_seq:08d}", client_order_id=o.coid, broker_order_id=o.broker_id,
            event_type=etype, state=o.state, source=EventSource.STREAM, gateway=GatewayKind.FAKE,
            fill_quantity=fill_qty, fill_price=fill_price, cum_filled_quantity=o.filled, broker_exec_id=exec_id,
            broker_ts=now, received_ts=now, sequence=self._event_seq, reason=reason,
            raw={"status": o.status, "simulated": True},
        )

    # ================================================================== helpers
    def _quote(self, symbol: str) -> Quote | None:
        q: Quote | None = None
        if self.quote_source is not None:
            try:
                q = self.quote_source(symbol)
            except Exception:
                q = None
        last = self._last_quotes.get(symbol)
        if q is None or (last is not None and last.exchange_ts > q.exchange_ts):
            q = last
        return self._usable(q)

    def _usable(self, q: Quote | None) -> Quote | None:
        """The quote if it may be used *now*: never a future (look-ahead) or stale quote."""
        if q is None:
            return None
        now = self.clock.now()
        if q.available_at_ts > now or now - q.exchange_ts > self.max_quote_age:
            return None
        return q

    def _mark(self, symbol: str) -> Decimal | None:
        q = self._quote(symbol)
        return None if q is None else q.mid

    def _reserved_cash(self) -> Decimal:
        return sum((o.remaining * o.limit for o in self.orders.values() if o.is_open and o.side == Side.BUY), ZERO)

    def _pending_sells(self, symbol: str) -> Decimal:
        return sum((o.remaining for o in self.orders.values() if o.is_open and o.side == Side.SELL and o.symbol == symbol), ZERO)

    def _snapshot(self, o: _Order) -> BrokerOrder:
        return BrokerOrder(
            client_order_id=o.coid, broker_order_id=o.broker_id, intent_id=o.intent_id, gateway=GatewayKind.FAKE,
            symbol=o.symbol, side=o.side, quantity=o.qty, filled_quantity=o.filled,
            avg_fill_price=(o.notional / o.filled).quantize(Decimal("0.00000001")) if o.filled > 0 else None,
            order_type=o.order_type, limit_price=o.limit, time_in_force=o.tif, broker_status=o.status, state=o.state,
            submitted_at=o.submitted_at, updated_at=o.updated_at, stop_price=o.stop,
        )

    # ================================================================== ExecutionGateway: submit / cancel
    def _validate(self, intent: ApprovedOrderIntent) -> None:
        now = self.clock.now()
        if intent.gateway != GatewayKind.FAKE:
            raise GatewayRejected(f"fake broker only accepts FAKE-gateway intents, got {intent.gateway.value}")
        try:
            modes.assert_gateway_allowed(intent.mode, intent.gateway)
        except modes.ModeViolation as exc:
            raise GatewayRejected(f"mode/gateway refused: {exc}") from exc
        if intent.mode not in (TradingMode.PAPER, TradingMode.REPLAY):
            raise GatewayRejected(f"fake broker refuses mode {intent.mode.value}")
        if self.verifier is not None and not self.verifier.verify(intent.signing_string(), intent.signature):
            raise GatewayRejected("intent signature invalid (not signed by the risk engine)")
        if self.claim_check is not None:
            try:
                claimed = self.claim_check(intent.client_order_id)
            except Exception as exc:  # fail closed
                raise GatewayRejected(f"order-ledger claim check failed: {type(exc).__name__}") from exc
            if not claimed:
                raise GatewayRejected("intent was not claimed for submission through the order ledger")
        if intent.client_order_id in self.orders:
            raise GatewayRejected("client_order_id must be unique (duplicate submission refused)")
        if intent.order_type == OrderType.STOP_LIMIT:
            stop = intent.stop_price
            if intent.side != Side.SELL or stop is None or intent.limit_price > stop:
                raise GatewayRejected("STOP_LIMIT accepted only as a SELL exit with limit at or below the stop")
            if stop != stop.quantize(price_tick(stop)):
                raise GatewayRejected("stop price not on a valid tick")
        elif intent.order_type != OrderType.LIMIT:
            raise GatewayRejected(f"unsupported order type {intent.order_type.value}: LIMIT only")
        if intent.time_in_force != TimeInForce.DAY:
            raise GatewayRejected("unsupported time in force: DAY only")
        if intent.extended_hours:
            raise GatewayRejected("extended-hours orders are not supported")
        if intent.quantity <= 0 or intent.quantity != floor_shares(intent.quantity):
            raise GatewayRejected("whole-share positive quantity required")
        if intent.limit_price <= 0 or intent.limit_price != intent.limit_price.quantize(price_tick(intent.limit_price)):
            raise GatewayRejected("limit price not on a valid tick")
        if now >= intent.expires_at:
            raise GatewayRejected("intent expired")
        bounds = self.session_bounds_fn(now)
        if bounds is None or not (bounds[0] <= now < bounds[1]):
            raise GatewayRejected("market closed: regular session only")
        if intent.side == Side.SELL:
            held = self.positions.get(intent.symbol, _Position()).qty
            if intent.quantity > held - self._pending_sells(intent.symbol):
                raise GatewayRejected("insufficient position: long-only, no shorting")
        else:
            available = self.cash - self._reserved_cash()
            if intent.quantity * intent.limit_price > available:
                raise GatewayRejected("insufficient buying power (cash account, no margin)")

    async def submit(self, intent: ApprovedOrderIntent) -> BrokerOrder:
        with self._lock:
            self.submit_attempts.append(intent.client_order_id)
            inj = self.injector
            take, inj.partition_before_accept = FailureInjector.take(inj.partition_before_accept)
            if take:
                raise ConnectionError("network partition (injected): request never reached the broker")
            take, inj.timeout_before_accept = FailureInjector.take(inj.timeout_before_accept)
            if take:
                raise SubmitTimeout("submit timed out before reaching the broker (injected)")
            take, inj.rate_limit_next = FailureInjector.take(inj.rate_limit_next)
            if take:
                raise RateLimited("HTTP 429 too many requests (injected); order not accepted")
            take, inj.reject_next = FailureInjector.take(inj.reject_next)
            if take:
                raise GatewayRejected(inj.reject_reason)
            self._validate(intent)
            o = self._accept(intent)
            snap = self._snapshot(o)
            self._save()
            take, inj.crash_after_accept = FailureInjector.take(inj.crash_after_accept)
            if take:
                raise SimulatedCrash(f"process crashed after the broker accepted {intent.client_order_id} (injected)")
            take, inj.timeout_after_accept = FailureInjector.take(inj.timeout_after_accept)
            if take:
                raise SubmitTimeout("broker accepted the order but the response was lost (injected)")
            hang = inj.hang_after_accept_s
        if hang > 0:
            await asyncio.sleep(hang)
        return snap

    def _accept(self, intent: ApprovedOrderIntent) -> _Order:
        now = self.clock.now()
        self._order_seq += 1
        bounds = self.session_bounds_fn(now)
        o = _Order(
            coid=intent.client_order_id, broker_id=f"fb-{self.account_suffix}-{self._order_seq:06d}", intent_id=intent.intent_id,
            symbol=intent.symbol, side=intent.side, qty=intent.quantity, limit=intent.limit_price, tif=intent.time_in_force,
            status="accepted", filled=ZERO, notional=ZERO, submitted_at=now, updated_at=now,
            session_close=bounds[1] if bounds else None, order_type=intent.order_type, stop=intent.stop_price,
        )
        self.orders[o.coid] = o
        self.accepted_submissions.append(o.coid)
        self._emit(self._event(o, OrderEventType.ACCEPTED))
        o.status = "new"
        self._emit(self._event(o, OrderEventType.NEW))
        self._try_fill(o)
        return o

    def _try_fill(self, o: _Order, quote: Quote | None = None) -> None:
        if not o.is_open or o.status == "pending_cancel":
            return
        q = self._usable(quote) if quote is not None and quote.symbol == o.symbol else None
        if q is None:
            q = self._quote(o.symbol)
        if q is None:
            return
        if o.order_type == OrderType.STOP_LIMIT and not o.triggered:
            if o.stop is None or q.bid > o.stop:
                return
            o.triggered = True  # from now on a resting LIMIT SELL at o.limit (may not fill on a gap)
            o.updated_at = self.clock.now()
        slip = self.slippage_bps / BPS
        if o.side == Side.BUY:
            if o.limit < q.ask:
                return
            raw = q.ask * (ONE + slip)
            px = min(o.limit, raw.quantize(price_tick(raw), rounding=ROUND_CEILING))
            avail = q.ask_size
        else:
            if o.limit > q.bid:
                return
            raw = q.bid * (ONE - slip)
            px = max(o.limit, raw.quantize(price_tick(raw), rounding=ROUND_FLOOR))
            avail = q.bid_size
        qty = o.remaining
        if self.max_fill_per_match is not None:
            qty = min(qty, self.max_fill_per_match)
        if self.use_quote_size:
            qty = min(qty, floor_shares(avail))
        if qty <= 0:
            return
        self._book_fill(o, qty, px)

    def _book_fill(self, o: _Order, qty: Decimal, px: Decimal) -> None:
        now = self.clock.now()
        pos = self.positions.setdefault(o.symbol, _Position())
        if o.side == Side.BUY:
            self.cash -= qty * px
            pos.qty += qty
            pos.cost += qty * px
        else:
            avg = pos.cost / pos.qty if pos.qty > 0 else px
            self.cash += qty * px
            pos.cost -= qty * avg
            pos.qty -= qty
            if pos.qty == 0:
                pos.cost = ZERO
        o.filled += qty
        o.notional += qty * px
        o.updated_at = now
        o.status = "filled" if o.filled >= o.qty else "partially_filled"
        etype = OrderEventType.FILL if o.status == "filled" else OrderEventType.PARTIAL_FILL
        self._emit(self._event(o, etype, fill_qty=qty, fill_price=px))

    async def cancel(self, client_order_id: str) -> None:
        with self._lock:
            self.cancel_requests.append(client_order_id)
            o = self.orders.get(client_order_id)
            if o is None:
                raise GatewayRejected("order not found")
            if not o.is_open:
                raise GatewayRejected(f"order is {o.status}; not cancelable")
            o.status = "pending_cancel"
            o.updated_at = self.clock.now()
            self._emit(self._event(o, OrderEventType.PENDING_CANCEL))
            take, self.injector.fill_on_cancel = FailureInjector.take(self.injector.fill_on_cancel)
            if take:
                # Cancel/fill race: the order executes before the cancel takes effect; the cancel is moot.
                o.status = "new" if o.filled == 0 else "partially_filled"
                self._book_fill(o, o.remaining, o.limit)
            else:
                o.status = "canceled"
                o.updated_at = self.clock.now()
                self._emit(self._event(o, OrderEventType.CANCELED, reason="canceled by request"))
            self._save()

    # ================================================================== market progression
    def advance(self, quote: Quote | None = None) -> None:
        """Feed a new quote (fills resting orders) and expire DAY orders past their session close."""
        with self._lock:
            if quote is not None:
                prev = self._last_quotes.get(quote.symbol)
                if prev is None or quote.exchange_ts >= prev.exchange_ts:
                    self._last_quotes[quote.symbol] = quote
            self.expire_day_orders()
            for o in list(self.orders.values()):
                if o.is_open and (quote is None or o.symbol == quote.symbol):
                    self._try_fill(o, quote)
            self._save()

    def expire_day_orders(self) -> list[str]:
        now = self.clock.now()
        out: list[str] = []
        with self._lock:
            for o in self.orders.values():
                if o.is_open and o.tif == TimeInForce.DAY and (o.session_close is None or now >= o.session_close):
                    o.status = "expired"
                    o.updated_at = now
                    self._emit(self._event(o, OrderEventType.EXPIRED, reason="DAY order expired at the close"))
                    out.append(o.coid)
            if out:
                self._save()
        return out

    def seed_position(self, symbol: str, qty: Decimal, avg_price: Decimal) -> None:
        """Test helper: a position the ledger knows nothing about (e.g. placed outside this system)."""
        with self._lock:
            pos = self.positions.setdefault(symbol, _Position())
            pos.qty += qty
            pos.cost += qty * avg_price
            self._save()

    # ================================================================== BrokerStateReader
    def _check_read(self) -> None:
        take, self.injector.fail_reads = FailureInjector.take(self.injector.fail_reads)
        if take:
            raise BrokerUnavailable("fake broker read failed (injected)")

    async def get_account(self) -> AccountSnapshot:
        with self._lock:
            self._check_read()
            now = self.clock.now()
            equity = self.cash
            for sym, p in self.positions.items():
                if p.qty != 0:
                    mark = self._mark(sym)
                    equity += p.qty * mark if mark is not None else p.cost
            bp = max(self.cash - self._reserved_cash(), ZERO)
            return AccountSnapshot(
                snapshot_id=new_id("acct"), account_id_masked=f"FAKE-****{self.account_suffix}", is_paper=True,
                status="ACTIVE", equity=equity, last_equity=None, cash=self.cash, buying_power=bp,
                non_marginable_buying_power=bp, multiplier=ONE, trading_blocked=False, account_blocked=False,
                transfers_blocked=True, shorting_enabled=False, asof=now, source="FAKE_BROKER",
            )

    async def get_positions(self) -> list[PositionSnapshot]:
        with self._lock:
            self._check_read()
            now = self.clock.now()
            out: list[PositionSnapshot] = []
            for sym in sorted(self.positions):
                p = self.positions[sym]
                if p.qty == 0:
                    continue
                mark = self._mark(sym)
                avg = (p.cost / p.qty).quantize(Decimal("0.00000001"))
                out.append(PositionSnapshot(
                    symbol=sym, quantity=p.qty, avg_entry_price=avg, cost_basis=p.cost,
                    market_value=None if mark is None else p.qty * mark, current_price=mark,
                    unrealized_pl=None if mark is None else p.qty * mark - p.cost, asof=now, source="FAKE_BROKER",
                ))
            return out

    async def get_order_by_client_id(self, client_order_id: str) -> BrokerOrder | None:
        with self._lock:
            take, self.injector.fail_lookups = FailureInjector.take(self.injector.fail_lookups)
            if take:
                raise BrokerUnavailable("fake broker order lookup failed (injected)")
            self._check_read()
            o = self.orders.get(client_order_id)
            return None if o is None else self._snapshot(o)

    async def list_orders(self, *, status: str = "all", after: datetime | None = None) -> list[BrokerOrder]:
        with self._lock:
            self._check_read()
            out: list[BrokerOrder] = []
            for o in sorted(self.orders.values(), key=lambda x: (x.submitted_at, x.coid)):
                if status == "open" and not o.is_open:
                    continue
                if status == "closed" and o.is_open:
                    continue
                if after is not None and o.submitted_at <= after:
                    continue
                out.append(self._snapshot(o))
            return out

    # ================================================================== misc
    def submission_count(self, client_order_id: str | None = None) -> int:
        if client_order_id is None:
            return len(self.submit_attempts)
        return sum(1 for c in self.submit_attempts if c == client_order_id)

    async def aclose(self) -> None:
        self._save()


@dataclass
class QuoteBook:
    """Simple mutable quote source for tests/replay: ``book.set(quote)``; ``book(symbol)`` returns latest."""

    quotes: dict[str, Quote] = field(default_factory=dict)

    def set(self, quote: Quote) -> None:
        self.quotes[quote.symbol] = quote

    def __call__(self, symbol: str) -> Quote | None:
        return self.quotes.get(symbol)
