"""Helpers to assemble risk inputs from plain values, plus order fingerprint utilities.

Everything here is pure. The integrator gathers broker/ledger/feed state and calls these helpers;
the risk engine never fetches anything itself.

Fail-closed defaults: ``unknown_health()`` reports every gate as failed, and ``build_system_health``
locks entries whenever a halt is active or a lock reason is given.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, NamedTuple, overload

from packages.contracts.enums import GatewayKind, Side, TradingMode
from packages.contracts.models import (
    AccountSnapshot,
    ApprovedOrderIntent,
    BrokerOrder,
    MarketSnapshot,
    PositionSnapshot,
    RiskContext,
    SystemHealth,
    TradeProposal,
)
from packages.core.canonical import sha256_hex
from packages.core.money import D

FINGERPRINT_PREFIX = "fp_"


# ---------------------------------------------------------------------------- fingerprints

ANY_STRATEGY = "*"
"""Strategy placeholder for strategy-agnostic fingerprints (an ``ApprovedOrderIntent`` carries no
strategy id, so a ledger that only has the intent stores this form)."""


def _norm_decimal(value: Decimal) -> str:
    if not isinstance(value, Decimal):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError("fingerprint amounts must be Decimal")
    if not value.is_finite():
        raise ValueError("fingerprint amounts must be finite")
    if value == 0:
        return "0"
    return format(value.normalize(), "f")


def order_fingerprint(
    symbol: str, side: Side | str, quantity: Decimal, limit_price: Decimal, strategy_id: str
) -> str:
    """Stable fingerprint of (symbol, side, qty, limit, strategy) used for duplicate-order rejection.

    Decimal values are normalised, so ``Decimal("10")`` and ``Decimal("10.00")`` match. Pass
    ``ANY_STRATEGY`` for the strategy-agnostic form.
    """
    side_value = side.value if isinstance(side, Side) else Side(side).value
    strategy = strategy_id.strip()
    if not strategy:
        raise ValueError("strategy_id must not be empty (use ANY_STRATEGY for the agnostic form)")
    material = "|".join(
        (
            "lat-fp-v1",
            symbol.strip().upper(),
            side_value,
            _norm_decimal(quantity),
            _norm_decimal(limit_price),
            strategy,
        )
    )
    return FINGERPRINT_PREFIX + sha256_hex(material)[:32]


def fingerprint_variants(
    symbol: str, side: Side | str, quantity: Decimal, limit_price: Decimal, strategy_id: str | None
) -> tuple[str, ...]:
    """Every fingerprint form an identical order may have been recorded under: the strategy-specific
    one (when the strategy is known) and the strategy-agnostic one. Duplicate checks test all forms."""
    agnostic = order_fingerprint(symbol, side, quantity, limit_price, ANY_STRATEGY)
    if strategy_id is None or not strategy_id.strip() or strategy_id.strip() == ANY_STRATEGY:
        return (agnostic,)
    return (order_fingerprint(symbol, side, quantity, limit_price, strategy_id), agnostic)


def proposal_fingerprint(proposal: TradeProposal) -> str | None:
    """Strategy-specific fingerprint of a TRADE proposal (None for NO_TRADE / incomplete proposals)."""
    if proposal.symbol is None or proposal.side is None or proposal.quantity is None:
        return None
    if proposal.limit_price is None:
        return None
    return order_fingerprint(
        proposal.symbol, proposal.side, proposal.quantity, proposal.limit_price, proposal.strategy_id
    )


def intent_fingerprint(intent: ApprovedOrderIntent, strategy_id: str | None = None) -> str:
    """Fingerprint of an intent. With the originating proposal's ``strategy_id`` it equals
    ``proposal_fingerprint`` of that proposal; without it (the intent contract carries no strategy) it
    is the strategy-agnostic form. The risk engine's duplicate checks match either form."""
    return order_fingerprint(
        intent.symbol, intent.side, intent.quantity, intent.limit_price, strategy_id or ANY_STRATEGY
    )


@overload
def fingerprint(item: ApprovedOrderIntent, strategy_id: str | None = None) -> str: ...
@overload
def fingerprint(item: TradeProposal, strategy_id: str | None = None) -> str | None: ...
def fingerprint(item: TradeProposal | ApprovedOrderIntent, strategy_id: str | None = None) -> str | None:
    """Fingerprint of a proposal (its own strategy) or an intent (``strategy_id`` or agnostic)."""
    if isinstance(item, TradeProposal):
        return proposal_fingerprint(item)
    return intent_fingerprint(item, strategy_id)


class FingerprintEntry(NamedTuple):
    """One previously authorized intent, e.g. a row of the ``orders`` table."""

    fingerprint: str
    at: datetime
    client_order_id: str | None = None


def recent_fingerprints(
    entries: Iterable[FingerprintEntry],
    *,
    now: datetime,
    window_seconds: int,
    exclude_client_order_id: str | None = None,
) -> tuple[str, ...]:
    """Fingerprints authorized within ``window_seconds`` before ``now`` (future-dated entries kept,
    conservatively). Pass ``exclude_client_order_id`` at the final gate so an intent is not treated
    as a duplicate of itself."""
    cutoff = now - timedelta(seconds=window_seconds)
    out: list[str] = []
    for e in entries:
        if exclude_client_order_id is not None and e.client_order_id == exclude_client_order_id:
            continue
        if e.at >= cutoff:
            out.append(e.fingerprint)
    return tuple(dict.fromkeys(out))


# ---------------------------------------------------------------------------- health


def unknown_health(reason: str = "system state unknown") -> SystemHealth:
    """Every gate failed and entries locked. Use whenever state cannot be established."""
    return SystemHealth(
        feed_ok=False, feed_detail=reason,
        broker_connected=False, broker_detail=reason,
        reconciled=False, reconcile_detail=reason,
        engine_ok=False, engine_detail=reason,
        entries_locked=True, lock_reasons=(reason,),
        strategy_state_ok=False, permissions_ok=False,
    )


def _halt_reason(halt: Mapping[str, Any] | str) -> str:
    if isinstance(halt, str):
        return f"halt {halt}"
    halt_id = str(halt.get("halt_id", "?"))
    reason = str(halt.get("reason", ""))
    return f"halt {halt_id}: {reason}" if reason else f"halt {halt_id}"


def build_system_health(
    *,
    feed_ok: bool,
    broker_connected: bool,
    reconciled: bool,
    engine_ok: bool,
    entries_locked: bool = False,
    lock_reasons: Sequence[str] = (),
    active_halts: Sequence[Mapping[str, Any] | str] = (),
    strategy_state_ok: bool = True,
    permissions_ok: bool = True,
    feed_detail: str = "",
    broker_detail: str = "",
    reconcile_detail: str = "",
    engine_detail: str = "",
) -> SystemHealth:
    """Build ``SystemHealth``; any active halt or lock reason forces ``entries_locked``."""
    reasons = [*lock_reasons, *(_halt_reason(h) for h in active_halts)]
    locked = bool(entries_locked or reasons)
    if locked and not reasons:
        reasons = ["entries locked"]
    return SystemHealth(
        feed_ok=feed_ok, feed_detail=feed_detail,
        broker_connected=broker_connected, broker_detail=broker_detail,
        reconciled=reconciled, reconcile_detail=reconcile_detail,
        engine_ok=engine_ok, engine_detail=engine_detail,
        entries_locked=locked, lock_reasons=tuple(reasons),
        strategy_state_ok=strategy_state_ok, permissions_ok=permissions_ok,
    )


# ---------------------------------------------------------------------------- context


def build_risk_context(
    *,
    now: datetime,
    mode: TradingMode,
    gateway: GatewayKind,
    account: AccountSnapshot | None,
    positions: Sequence[PositionSnapshot] = (),
    open_orders: Sequence[BrokerOrder] = (),
    market: MarketSnapshot | None,
    health: SystemHealth | None,
    recent_submission_times: Sequence[datetime] = (),
    recent_intent_fingerprints: Sequence[str] = (),
    start_of_day_equity: Decimal | None = None,
    daily_realized_pl: Decimal | str | int = 0,
    daily_unrealized_pl: Decimal | str | int | None = None,
) -> RiskContext:
    """Assemble a ``RiskContext``. ``health=None`` => ``unknown_health()`` (fail closed).

    ``daily_unrealized_pl=None`` sums ``unrealized_pl`` of the supplied positions (a conservative
    proxy; the engine additionally compares account equity against ``start_of_day_equity``).
    """
    if daily_unrealized_pl is None:
        unreal = sum((p.unrealized_pl for p in positions if p.unrealized_pl is not None), Decimal(0))
    else:
        unreal = D(daily_unrealized_pl)
    return RiskContext(
        now=now,
        mode=mode,
        gateway=gateway,
        account=account,
        positions=tuple(positions),
        open_orders=tuple(open_orders),
        recent_submission_times=tuple(recent_submission_times),
        recent_intent_fingerprints=tuple(recent_intent_fingerprints),
        start_of_day_equity=start_of_day_equity,
        daily_realized_pl=D(daily_realized_pl),
        daily_unrealized_pl=unreal,
        market=market,
        health=health if health is not None else unknown_health("health not supplied"),
    )
