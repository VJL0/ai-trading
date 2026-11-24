"""Alpaca PAPER broker-state reader (alpaca-py ``TradingClient(paper=True)``), strictly READ-ONLY.

The execution adapter uses this unchanged read-only boundary for reconciliation and
separate guarded SDK mutations for signed orders. This reader only permits GET
requests to the paper endpoint; it never submits or cancels orders.

* Key ids that do not look like paper keys (Alpaca paper ids start with ``PK``; live ids with ``AK``) are
  refused with ``LiveKeyRefused``. The key value is never logged or included in messages.
* SDK floats/strings are converted to ``Decimal`` via ``str`` (``from_external_float`` for floats).
* HTTP 429 on reads: bounded backoff (the SDK's own blocking retry is disabled), then ``BrokerRateLimited``.
* Unsupported broker objects (non-equity assets, non-DAY orders, plain stop / trailing orders, notional orders,
  stop-limit BUYs) raise ``BrokerPayloadError`` rather than being silently degraded; reconciliation then fails
  closed. The only stop order understood is the protective ``stop_limit`` SELL this system signs.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any, TypeVar

from packages.contracts.enums import GatewayKind, OrderState, OrderType, Side, TimeInForce
from packages.contracts.models import AccountSnapshot, BrokerOrder, PositionSnapshot
from packages.core.clock import Clock, SystemClock
from packages.core.ids import new_id
from packages.core.money import D, from_external_float
from packages.core.secrets import SecretName, SecretStore

from .errors import (
    BrokerAuthError,
    BrokerPayloadError,
    BrokerRateLimited,
    BrokerUnavailable,
    LiveKeyRefused,
    MissingPaperCredentials,
    ReadOnlyViolation,
)
from .timeutil import parse_iso

T = TypeVar("T")

PAPER_BASE_URL = "https://paper-api.alpaca.markets"
_SYMBOL = re.compile(r"^[A-Z][A-Z0-9.]{0,9}$")

ALPACA_STATUS_STATE: dict[str, OrderState] = {
    "accepted": OrderState.SUBMITTED,
    "pending_new": OrderState.SUBMITTED,
    "accepted_for_bidding": OrderState.SUBMITTED,
    "new": OrderState.ACKNOWLEDGED,
    "held": OrderState.ACKNOWLEDGED,
    "suspended": OrderState.ACKNOWLEDGED,
    "stopped": OrderState.ACKNOWLEDGED,
    "pending_replace": OrderState.ACKNOWLEDGED,
    "partially_filled": OrderState.PARTIALLY_FILLED,
    "filled": OrderState.FILLED,
    "pending_cancel": OrderState.CANCEL_PENDING,
    "canceled": OrderState.CANCELED,
    "expired": OrderState.EXPIRED,
    "rejected": OrderState.REJECTED,
}
"""Alpaca order status -> ledger state. ``done_for_day``/``calculated`` depend on fills; ``replaced`` is refused."""


def _base_url_text(value: object) -> str:
    """alpaca-py's ``BaseURL`` is a ``(str, Enum)``: ``str()`` gives ``'BaseURL.TRADING_PAPER'``, so use ``.value``."""
    return str(getattr(value, "value", value))


class _ReadOnlySession:
    """Wraps the SDK's ``requests.Session``: only GET requests to the paper endpoint pass."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.refused: list[str] = []

    def request(self, method: str, url: str, **kwargs: Any) -> Any:
        verb = str(method).upper()
        target = str(url)
        if verb != "GET":
            self.refused.append(f"{verb} {_redact_url(target)}")
            raise ReadOnlyViolation(f"read-only Alpaca reader refused a {verb} request")
        if not target.startswith(PAPER_BASE_URL + "/"):
            self.refused.append(f"{verb} {_redact_url(target)}")
            raise ReadOnlyViolation("read-only Alpaca reader refused a non-paper endpoint")
        kwargs["timeout"] = (3.0, 7.0)
        kwargs["allow_redirects"] = False
        return self._inner.request(method, url, **kwargs)

    def close(self) -> None:
        closer = getattr(self._inner, "close", None)
        if closer is not None:
            closer()


def _redact_url(url: str) -> str:
    return url.split("?", 1)[0][:200]


class _PaperReads:
    """The only four SDK calls this module makes."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def account(self) -> Any:
        return self._client.get_account()

    def positions(self) -> Any:
        return self._client.get_all_positions()

    def by_client_id(self, client_order_id: str) -> Any:
        return self._client.get_order_by_client_id(client_order_id)

    def orders(self, status: str, after: datetime | None, limit: int) -> Any:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        req = GetOrdersRequest(status=QueryOrderStatus(status), after=after, limit=limit, nested=False)
        return self._client.get_orders(filter=req)


# ------------------------------------------------------------------------------------------ mapping


def _dec(raw: dict[str, Any], key: str, *, required: bool = True) -> Decimal | None:
    value = raw.get(key)
    if value is None or value == "":
        if required:
            raise BrokerPayloadError(f"broker payload missing '{key}'")
        return None
    try:
        if isinstance(value, float):
            out = from_external_float(value)
        elif isinstance(value, (int, str, Decimal)) and not isinstance(value, bool):
            out = D(value.strip() if isinstance(value, str) else value)
        else:
            raise BrokerPayloadError(f"broker payload '{key}' has unsupported type {type(value).__name__}")
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise BrokerPayloadError(f"broker payload '{key}' is not a decimal: {value!r}") from exc
    if out is None or not out.is_finite():
        raise BrokerPayloadError(f"broker payload '{key}' is not finite")
    return out


def _req_dec(raw: dict[str, Any], key: str) -> Decimal:
    out = _dec(raw, key)
    assert out is not None
    return out


def _ts(raw: dict[str, Any], key: str) -> datetime | None:
    value = raw.get(key)
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise BrokerPayloadError(f"naive timestamp in '{key}'")
        return value
    try:
        return parse_iso(str(value))
    except ValueError as exc:
        raise BrokerPayloadError(f"bad timestamp in '{key}': {value!r}") from exc


def _str(raw: dict[str, Any], key: str) -> str:
    value = raw.get(key)
    if value is None:
        raise BrokerPayloadError(f"broker payload missing '{key}'")
    return str(getattr(value, "value", value))


def _require_dict(raw: Any, what: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise BrokerPayloadError(f"{what}: expected a JSON object, got {type(raw).__name__}")
    return raw  # pyright: ignore[reportUnknownVariableType]


def mask_account(number: str) -> str:
    tail = number[-4:] if len(number) >= 4 else number
    head = number[:2] if len(number) > 6 else ""
    return f"{head}****{tail}"


def map_account(raw: Any, now: datetime, *, is_paper: bool = True) -> AccountSnapshot:
    d = _require_dict(raw, "account")
    number = str(d.get("account_number") or d.get("id") or "")
    if not number:
        raise BrokerPayloadError("account payload has no account number")
    return AccountSnapshot(
        snapshot_id=new_id("acct"), account_id_masked=mask_account(number), is_paper=is_paper,
        status=_str(d, "status").upper(), currency=str(d.get("currency") or "USD"), equity=_req_dec(d, "equity"),
        last_equity=_dec(d, "last_equity", required=False), cash=_req_dec(d, "cash"),
        buying_power=_req_dec(d, "buying_power"),
        non_marginable_buying_power=_dec(d, "non_marginable_buying_power", required=False),
        multiplier=_dec(d, "multiplier", required=False), trading_blocked=bool(d.get("trading_blocked", True)),
        account_blocked=bool(d.get("account_blocked", True)), transfers_blocked=bool(d.get("transfers_blocked", False)),
        shorting_enabled=bool(d.get("shorting_enabled", False)), asof=now, source="ALPACA_PAPER",
    )


def map_position(raw: Any, now: datetime) -> PositionSnapshot:
    d = _require_dict(raw, "position")
    asset_class = str(getattr(d.get("asset_class"), "value", d.get("asset_class")) or "us_equity")
    symbol = _str(d, "symbol")
    if asset_class != "us_equity" or not _SYMBOL.match(symbol):
        raise BrokerPayloadError(f"unsupported broker position {symbol!r} ({asset_class}); only U.S. equities are supported")
    qty = _req_dec(d, "qty")
    side = str(getattr(d.get("side"), "value", d.get("side")) or "long").lower()
    if side == "short" and qty > 0:
        qty = -qty
    return PositionSnapshot(
        symbol=symbol, quantity=qty, avg_entry_price=_req_dec(d, "avg_entry_price"), cost_basis=_req_dec(d, "cost_basis"),
        market_value=_dec(d, "market_value", required=False), current_price=_dec(d, "current_price", required=False),
        unrealized_pl=_dec(d, "unrealized_pl", required=False),
        unrealized_intraday_pl=_dec(d, "unrealized_intraday_pl", required=False), asof=now, source="BROKER",
    )


def map_order(raw: Any, now: datetime, gateway: GatewayKind = GatewayKind.LEAN_ALPACA_PAPER) -> BrokerOrder:
    d = _require_dict(raw, "order")
    if d.get("extended_hours") is True:
        raise BrokerPayloadError("extended-hours broker order is unsupported")
    asset_class = d.get("asset_class", "us_equity")
    if str(getattr(asset_class, "value", asset_class)) != "us_equity":
        raise BrokerPayloadError("non-equity broker order is unsupported")
    order_class = d.get("order_class")
    if order_class is not None and str(getattr(order_class, "value", order_class)) not in ("", "simple"):
        raise BrokerPayloadError("non-simple broker order is unsupported")
    status = _str(d, "status").lower()
    otype = str(getattr(d.get("order_type") or d.get("type"), "value", d.get("order_type") or d.get("type")) or "").lower()
    if otype not in ("limit", "market", "stop_limit"):
        raise BrokerPayloadError(f"unsupported broker order type {otype!r} (order {d.get('client_order_id')})")
    tif = str(getattr(d.get("time_in_force"), "value", d.get("time_in_force")) or "").lower()
    if tif != "day":
        raise BrokerPayloadError(f"unsupported time in force {tif!r} (order {d.get('client_order_id')})")
    side_raw = _str(d, "side").lower()
    if side_raw not in ("buy", "sell"):
        raise BrokerPayloadError(f"unsupported order side {side_raw!r}")
    symbol = _str(d, "symbol")
    if not _SYMBOL.match(symbol):
        raise BrokerPayloadError(f"unsupported symbol {symbol!r}")
    qty = _dec(d, "qty", required=False)
    if qty is None:
        raise BrokerPayloadError("notional (qty-less) orders are not supported")
    filled = _dec(d, "filled_qty", required=False) or Decimal(0)
    stop_price = _dec(d, "stop_price", required=False)
    if otype == "stop_limit" and (side_raw != "sell" or stop_price is None):
        raise BrokerPayloadError(f"stop_limit order {d.get('client_order_id')} is not a protective SELL with a stop price")
    if status in ("done_for_day", "calculated"):
        state = OrderState.PARTIALLY_FILLED if filled > 0 else OrderState.ACKNOWLEDGED
    elif status in ALPACA_STATUS_STATE:
        state = ALPACA_STATUS_STATE[status]
    else:
        raise BrokerPayloadError(f"unsupported broker order status {status!r}")
    updated = _ts(d, "updated_at") or _ts(d, "submitted_at") or _ts(d, "created_at") or now
    coid = _str(d, "client_order_id")
    return BrokerOrder(
        client_order_id=coid, broker_order_id=str(d.get("id")) if d.get("id") else None, intent_id=None, gateway=gateway,
        symbol=symbol, side=Side.BUY if side_raw == "buy" else Side.SELL, quantity=qty, filled_quantity=filled,
        avg_fill_price=_dec(d, "filled_avg_price", required=False),
        order_type={"limit": OrderType.LIMIT, "stop_limit": OrderType.STOP_LIMIT}.get(otype, OrderType.MARKET),
        limit_price=_dec(d, "limit_price", required=False), time_in_force=TimeInForce.DAY, broker_status=status,
        state=state, submitted_at=_ts(d, "submitted_at"), updated_at=updated,
        stop_price=stop_price if otype == "stop_limit" else None,
    )


# ------------------------------------------------------------------------------------------ reader


def _status_code(exc: BaseException) -> int | None:
    code = getattr(exc, "status_code", None)
    if isinstance(code, int):
        return code
    resp = getattr(exc, "response", None)
    code = getattr(resp, "status_code", None)
    return code if isinstance(code, int) else None


class AlpacaPaperStateReader:
    """``BrokerStateReader`` over the Alpaca PAPER trading API. Read-only by construction."""

    kind: GatewayKind = GatewayKind.LEAN_ALPACA_PAPER
    is_paper: bool = True

    def __init__(
        self,
        secret_store: SecretStore,
        *,
        http_session: Any | None = None,
        clock: Clock | None = None,
        timeout_s: float = 15.0,
        max_attempts: int = 4,
        backoff_s: tuple[float, ...] = (0.5, 1.0, 2.0),
        sleep: Callable[[float], Any] | None = None,
        orders_page_limit: int = 500,
    ) -> None:
        key = secret_store.get(SecretName.ALPACA_PAPER_KEY_ID)
        secret = secret_store.get(SecretName.ALPACA_PAPER_SECRET_KEY)
        if key is None or secret is None:
            raise MissingPaperCredentials(
                "Alpaca PAPER credentials are absent. Store them in the macOS Keychain (service 'local-ai-trader', "
                "accounts ALPACA_PAPER_KEY_ID / ALPACA_PAPER_SECRET_KEY) or the git-ignored .env; never paste them into chat."
            )
        key_id = key.get_secret_value().strip()
        if key_id.startswith("AK"):
            raise LiveKeyRefused(
                "ALPACA_PAPER_KEY_ID looks like a LIVE Alpaca key id (prefix 'AK'); only PAPER keys ('PK') are accepted"
            )
        if not key_id.startswith("PK"):
            raise LiveKeyRefused("ALPACA_PAPER_KEY_ID does not look like an Alpaca PAPER key id (expected prefix 'PK')")
        from alpaca.trading.client import TradingClient

        client = TradingClient(api_key=key_id, secret_key=secret.get_secret_value().strip(), paper=True, raw_data=True)
        if _base_url_text(getattr(client, "_base_url", "")) != PAPER_BASE_URL:
            raise ReadOnlyViolation("alpaca-py TradingClient is not pointed at the paper endpoint")
        if not hasattr(client, "_retry") or not hasattr(client, "_session"):
            raise ReadOnlyViolation("unexpected alpaca-py client internals; refusing to run without the read-only guard")
        client._retry = 0  # bounded, non-blocking backoff is handled here (reads only)
        self._guard = _ReadOnlySession(http_session if http_session is not None else client._session)
        client._session = self._guard  # pyright: ignore[reportAttributeAccessIssue]
        self._api = _PaperReads(client)
        self._clock: Clock = clock or SystemClock()
        self._timeout_s = timeout_s
        self._max_attempts = max(1, max_attempts)
        self._backoff_s = backoff_s or (0.0,)
        self._sleep = sleep or asyncio.sleep
        self._orders_page_limit = max(1, min(orders_page_limit, 500))
        self.rate_limited_retries = 0

    def __repr__(self) -> str:  # never include credentials
        return f"AlpacaPaperStateReader(endpoint={PAPER_BASE_URL!r}, read_only=True)"

    async def _call(self, fn: Callable[[], T], what: str, *, not_found_ok: bool = False) -> T | None:
        from alpaca.common.exceptions import APIError
        from requests.exceptions import RequestException

        for attempt in range(self._max_attempts):
            try:
                return await asyncio.wait_for(asyncio.to_thread(fn), timeout=self._timeout_s)
            except ReadOnlyViolation:
                raise
            except APIError as exc:
                code = _status_code(exc)
                if code == 404 and not_found_ok:
                    return None
                if code == 429:
                    if attempt + 1 >= self._max_attempts:
                        raise BrokerRateLimited(f"Alpaca paper {what}: HTTP 429 after {attempt + 1} attempts") from None
                    self.rate_limited_retries += 1
                    await self._sleep(self._backoff_s[min(attempt, len(self._backoff_s) - 1)])
                    continue
                if code in (401, 403):
                    raise BrokerAuthError(f"Alpaca paper {what}: HTTP {code} (credentials refused)") from None
                raise BrokerUnavailable(f"Alpaca paper {what}: HTTP {code}") from None
            except TimeoutError:
                raise BrokerUnavailable(f"Alpaca paper {what}: timed out after {self._timeout_s}s") from None
            except RequestException as exc:
                raise BrokerUnavailable(f"Alpaca paper {what}: {type(exc).__name__}") from None
            except (ValueError, TypeError) as exc:
                raise BrokerPayloadError(f"Alpaca paper {what}: malformed response ({type(exc).__name__})") from None
        raise BrokerUnavailable(f"Alpaca paper {what}: no attempts made")  # pragma: no cover

    async def get_account(self) -> AccountSnapshot:
        raw = await self._call(self._api.account, "account")
        return map_account(raw, self._clock.now(), is_paper=True)

    async def get_positions(self) -> list[PositionSnapshot]:
        raw = await self._call(self._api.positions, "positions")
        if not isinstance(raw, list):
            raise BrokerPayloadError("positions: expected a JSON array")
        now = self._clock.now()
        return [map_position(p, now) for p in raw]  # pyright: ignore[reportUnknownVariableType]

    async def get_order_by_client_id(self, client_order_id: str) -> BrokerOrder | None:
        raw = await self._call(lambda: self._api.by_client_id(client_order_id), "order lookup", not_found_ok=True)
        if raw is None:
            return None
        return map_order(raw, self._clock.now(), self.kind)

    async def list_orders(self, *, status: str = "all", after: datetime | None = None) -> list[BrokerOrder]:
        if status not in ("open", "closed", "all"):
            raise ValueError(f"bad order status filter {status!r}")
        raw = await self._call(lambda: self._api.orders(status, after, self._orders_page_limit), "orders")
        if not isinstance(raw, list):
            raise BrokerPayloadError("orders: expected a JSON array")
        if len(raw) >= self._orders_page_limit:
            raise BrokerPayloadError("order listing reached page limit; complete broker state cannot be verified")
        now = self._clock.now()
        return [map_order(o, now, self.kind) for o in raw]  # pyright: ignore[reportUnknownVariableType]

    @property
    def refused_requests(self) -> list[str]:
        return list(self._guard.refused)
