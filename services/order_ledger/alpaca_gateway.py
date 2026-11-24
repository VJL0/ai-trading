"""Signed-intent Alpaca PAPER execution through alpaca-py, independent of LEAN.

REST snapshots drive the existing durable ledger (including cumulative partial-fill
catch-up). A disconnected poll raises into ExecutionService's reconnect/lock loop.
Mutations are single-attempt; timeouts never mean that an order was rejected.
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from decimal import Decimal
from typing import Any
from urllib.parse import urlsplit

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide
from alpaca.trading.enums import OrderType as AlpacaOrderType
from alpaca.trading.enums import TimeInForce as AlpacaTIF
from alpaca.trading.requests import LimitOrderRequest, StopLimitOrderRequest

from packages.contracts.enums import EventSource, GatewayKind, OrderState, OrderType, Side, TimeInForce, TradingMode
from packages.contracts.interfaces import GatewayRejected, SubmitTimeout
from packages.contracts.models import ApprovedOrderIntent, BrokerOrder, OrderEvent
from packages.core.clock import Clock
from packages.core.money import price_tick
from packages.core.secrets import SecretName, SecretStore
from packages.core.signing import IntentVerifier
from services.risk.market_hours import is_regular_session_open

from .alpaca_reader import AlpacaPaperStateReader, map_order
from .entries import EntriesLock
from .errors import BrokerPayloadError, BrokerUnavailable, ReadOnlyViolation
from .ledger import OrderLedger
from .state_machine import PRE_SUBMISSION, TERMINAL


class _PaperOrderSession:
    """Exact endpoint/method allowlist; no redirects, retry adapters, or live URLs."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner

    def request(self, method: str, url: str, **kwargs: Any) -> Any:
        parsed = urlsplit(url)
        path = parsed.path
        allowed = (method == "POST" and path == "/v2/orders") or (
            method == "DELETE" and path.startswith("/v2/orders/") and len(path.split("/")) == 4)
        if parsed.scheme != "https" or parsed.netloc != "paper-api.alpaca.markets" or not allowed or parsed.query:
            raise ReadOnlyViolation("Alpaca execution refused endpoint or method")
        kwargs["allow_redirects"] = False
        kwargs["timeout"] = (3.0, 7.0)
        response = self.inner.request(method, url, **kwargs)
        if 300 <= response.status_code < 400:
            raise BrokerUnavailable("Alpaca execution refused redirect")
        return response

    def close(self) -> None:
        self.inner.close()


class _ExactLimitRequest(LimitOrderRequest):
    """Preserve signed decimal text on the wire despite SDK float request fields."""

    exact_quantity: str
    exact_price: str

    def to_request_fields(self) -> dict[str, Any]:
        fields = super().to_request_fields()
        fields["qty"] = fields.pop("exact_quantity")
        fields["limit_price"] = fields.pop("exact_price")
        return fields


class _ExactStopLimitRequest(StopLimitOrderRequest):
    """Protective STOP_LIMIT SELL with the signed decimal text preserved on the wire."""

    exact_quantity: str
    exact_price: str
    exact_stop: str

    def to_request_fields(self) -> dict[str, Any]:
        fields = super().to_request_fields()
        fields["qty"] = fields.pop("exact_quantity")
        fields["limit_price"] = fields.pop("exact_price")
        fields["stop_price"] = fields.pop("exact_stop")
        return fields


class AlpacaPaperGateway(AlpacaPaperStateReader):
    kind = GatewayKind.ALPACA_PAPER
    events_preapplied = True

    def __init__(self, secrets: SecretStore, *, ledger: OrderLedger, verifier: IntentVerifier,
                 clock: Clock, http_session: Any = None, poll_interval_s: float = 2.0) -> None:
        super().__init__(secrets, clock=clock, http_session=http_session, timeout_s=8.0)
        self.ledger = ledger
        self.verifier = verifier
        self.poll_interval_s = poll_interval_s
        self._submit_lock = asyncio.Lock()
        # Separate read-only and mutation clients: do not weaken the existing reader guard.
        key = secrets.get(SecretName.ALPACA_PAPER_KEY_ID)
        secret = secrets.get(SecretName.ALPACA_PAPER_SECRET_KEY)
        assert key is not None and secret is not None  # validated by reader
        self._writer = TradingClient(key.get_secret_value(), secret.get_secret_value(), paper=True, raw_data=True)
        self._writer._retry = 0
        self._writer._session = _PaperOrderSession(  # type: ignore[assignment]
            http_session if http_session is not None else self._writer._session)
        self._attempted: set[str] = set()
        self._observed: dict[str, str] = {}
        self._terminal_seen: set[str] = set()

    def __repr__(self) -> str:
        return "AlpacaPaperGateway(paper=True, signed_intents_only=True)"

    def _validate(self, intent: ApprovedOrderIntent) -> None:
        if intent.mode != TradingMode.PAPER or intent.gateway != self.kind:
            raise GatewayRejected("Alpaca gateway accepts only ALPACA_PAPER/PAPER intents")
        if intent.key_id != self.verifier.key_id or not self.verifier.verify(intent.signing_string(), intent.signature):
            raise GatewayRejected("invalid risk-engine signature")
        rec = self.ledger.get_record(intent.client_order_id)
        if rec is None or rec.state != OrderState.SUBMITTING or rec.intent_json != intent.model_dump_json():
            raise GatewayRejected("intent not identically claimed in durable ledger")
        if self._clock.now() >= intent.expires_at or self._clock.now() < intent.created_at:
            raise GatewayRejected("intent outside validity window")
        if intent.requires_human_approval and not intent.human_approval_id:
            raise GatewayRejected("human approval missing")
        protective = intent.order_type == OrderType.STOP_LIMIT and intent.side == Side.SELL and intent.stop_price is not None
        if (intent.order_type != OrderType.LIMIT and not protective) or intent.time_in_force != TimeInForce.DAY \
                or intent.extended_hours:
            raise GatewayRejected("only regular-session LIMIT/DAY orders (and protective STOP_LIMIT SELLs) supported")
        if protective and intent.stop_price is not None and \
                intent.stop_price != intent.stop_price.quantize(price_tick(intent.stop_price)):
            raise GatewayRejected("invalid stop price tick")
        if intent.quantity != intent.quantity.to_integral_value() or intent.quantity <= 0:
            raise GatewayRejected("positive whole shares required")
        if intent.limit_price != intent.limit_price.quantize(price_tick(intent.limit_price)):
            raise GatewayRejected("invalid limit price tick")
        if not is_regular_session_open(self._clock.now())[0]:
            raise GatewayRejected("market closed: regular session only")
        if len(intent.client_order_id) > 48:
            raise GatewayRejected("Alpaca client order ID exceeds 48 characters")

    @staticmethod
    def _match(order: BrokerOrder, intent: ApprovedOrderIntent) -> BrokerOrder:
        for field in ("client_order_id", "symbol", "side", "quantity", "order_type", "limit_price", "time_in_force",
                      "stop_price"):
            if getattr(order, field) != getattr(intent, field):
                raise BrokerPayloadError(f"broker order differs from signed intent: {field}")
        return order

    async def _sell_preflight(self, intent: ApprovedOrderIntent) -> None:
        """A SELL must be covered by the *broker's* long position net of the broker's other open SELLs.

        Paper accounts can be margin accounts with shorting enabled; a SELL sized from a stale ledger must never
        open a short. A failed read refuses the SELL (it is retried after reconciliation), never sends it.
        """
        positions = await self.get_positions()
        held = sum((p.quantity for p in positions if p.symbol == intent.symbol), Decimal(0))
        open_orders = await self.list_orders(status="open")
        pending = sum((o.quantity - o.filled_quantity for o in open_orders if o.symbol == intent.symbol
                       and o.side == Side.SELL and o.client_order_id != intent.client_order_id), Decimal(0))
        if intent.quantity > held - pending:
            raise GatewayRejected(f"SELL {intent.quantity} {intent.symbol} exceeds the broker long position {held} net of "
                                  f"{pending} pending (no shorting)")

    async def submit(self, intent: ApprovedOrderIntent) -> BrokerOrder:
        async with self._submit_lock:
            self._validate(intent)
            # A failed lookup must never permit a POST. Existing IDs are reconciled, not resubmitted.
            existing = await self.get_order_by_client_id(intent.client_order_id)
            if existing is not None:
                return self._match(existing, intent)
            if intent.client_order_id in self._attempted:
                raise SubmitTimeout("previous submission uncertain; no retry permitted")
            request: LimitOrderRequest | StopLimitOrderRequest
            if intent.order_type == OrderType.STOP_LIMIT and intent.stop_price is not None:
                request = _ExactStopLimitRequest(
                    type=AlpacaOrderType.STOP_LIMIT, symbol=intent.symbol, qty=float(intent.quantity),
                    side=OrderSide.SELL, time_in_force=AlpacaTIF.DAY, limit_price=float(intent.limit_price),
                    stop_price=float(intent.stop_price), extended_hours=False, client_order_id=intent.client_order_id,
                    exact_quantity=str(intent.quantity), exact_price=str(intent.limit_price),
                    exact_stop=str(intent.stop_price))
            else:
                request = _ExactLimitRequest(
                    type=AlpacaOrderType.LIMIT, symbol=intent.symbol, qty=float(intent.quantity),
                    side=OrderSide(intent.side.value.lower()),
                    time_in_force=AlpacaTIF.DAY, limit_price=float(intent.limit_price), extended_hours=False,
                    client_order_id=intent.client_order_id, exact_quantity=str(intent.quantity),
                    exact_price=str(intent.limit_price))
            if intent.side == Side.SELL:
                await self._sell_preflight(intent)
            self._validate(intent)  # reads may have consumed the intent validity window
            lock = EntriesLock(self.ledger.db, self.ledger.mode, clock=self._clock)
            if (lock.exits_locked() if intent.side == Side.SELL else lock.is_locked())[0]:
                raise GatewayRejected("entries/exits locked during broker preflight")
            self._attempted.add(intent.client_order_id)
            try:
                raw = await asyncio.to_thread(self._writer.submit_order, request)
                return self._match(map_order(raw, self._clock.now(), self.kind), intent)
            except Exception:
                # Even HTTP errors can race acceptance or hide a duplicate ID; use the same read path.
                try:
                    existing = await self.get_order_by_client_id(intent.client_order_id)
                    if existing is not None:
                        return self._match(existing, intent)
                except Exception:
                    raise SubmitTimeout("Alpaca submission and reconciliation unavailable") from None
                raise SubmitTimeout("Alpaca submission outcome uncertain; reconcile client order ID") from None

    async def cancel(self, client_order_id: str) -> None:
        rec = self.ledger.get_record(client_order_id)
        if rec is None or rec.gateway != self.kind:
            raise GatewayRejected("cannot cancel an order not owned by this ledger/gateway")
        order = await self.get_order_by_client_id(client_order_id)
        if order is None or not order.broker_order_id:
            raise BrokerUnavailable("cancel target cannot be verified")
        self._match(order, rec.intent)
        if order.state not in TERMINAL:
            try:
                await asyncio.to_thread(self._writer.cancel_order_by_id, order.broker_order_id)
            except Exception:
                raise SubmitTimeout("cancel outcome uncertain; reconcile before further action") from None
        # Cancellation acknowledgement is not proof of cancellation; a fill can win the race.
        found = await self.get_order_by_client_id(client_order_id)
        if found is None:
            raise BrokerUnavailable("cancel outcome cannot be verified")
        self.ledger.apply_broker_order(found, EventSource.RECONCILE)

    async def order_events(self) -> AsyncIterator[OrderEvent]:
        # Poll working orders and every terminal order once per process to recover crash-time fills.
        # Periodic reconciliation continues checking terminal orders for late corrections.
        while True:
            await self.get_account()  # verify connectivity even when there are no orders
            for rec in self.ledger.records():
                if rec.gateway != self.kind or rec.state in PRE_SUBMISSION:
                    continue
                if rec.client_order_id in self._terminal_seen:
                    continue
                if rec.state in TERMINAL and rec.broker_order_id is None:
                    continue
                found = await self.get_order_by_client_id(rec.client_order_id)
                if found is None:
                    raise BrokerUnavailable("previously submitted order cannot be verified")
                snapshot = found.model_dump_json()
                if self._observed.get(rec.client_order_id) == snapshot:
                    continue
                result = self.ledger.apply_broker_order(found, EventSource.RECONCILE)
                self._observed[rec.client_order_id] = snapshot
                if found.state in TERMINAL and not result.anomalies:
                    self._terminal_seen.add(rec.client_order_id)
                if not result.duplicate:
                    yield self.ledger.event_log(rec.client_order_id)[-1]
            await asyncio.sleep(self.poll_interval_s)

    async def aclose(self) -> None:
        self._guard.close()
        self._writer._session.close()
