"""Build the signed-intent runtime gateway: FAKE or official SDK ALPACA_PAPER.

LEAN execution selections are retired; legacy contracts remain readable for audit/research.
Live activation stays compiled off. Factory decisions are audited.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any

from packages.contracts.enums import GatewayKind, OrderState, TradingMode
from packages.contracts.interfaces import ExecutionGateway, GatewayRejected
from packages.contracts.models import AccountSnapshot, ApprovedOrderIntent, BrokerOrder, OrderEvent, PositionSnapshot, Quote
from packages.core import modes
from packages.core.clock import Clock
from packages.core.config import Settings
from packages.core.secrets import SecretStore
from packages.core.signing import PUBLIC_KEY_FILE, IntentVerifier, load_verifier

from .errors import BrokerUnavailable, MissingPaperCredentials
from .ledger import OrderLedger
from .live_gate import LiveReleaseGate, record_mode_decision

QuoteSource = Callable[[str], Quote | None]


class ResearchOnlyGateway:
    """RESEARCH_ONLY has no execution path: every order is refused and broker state is unavailable."""

    kind: GatewayKind = GatewayKind.FAKE
    is_paper: bool = True

    async def submit(self, intent: ApprovedOrderIntent) -> BrokerOrder:
        raise GatewayRejected("RESEARCH_ONLY mode never places orders")

    async def cancel(self, client_order_id: str) -> None:
        raise GatewayRejected("RESEARCH_ONLY mode has no orders")

    def order_events(self) -> Any:
        return _EmptyStream()

    async def get_account(self) -> AccountSnapshot:
        raise BrokerUnavailable("RESEARCH_ONLY mode has no broker")

    async def get_positions(self) -> list[PositionSnapshot]:
        raise BrokerUnavailable("RESEARCH_ONLY mode has no broker")

    async def get_order_by_client_id(self, client_order_id: str) -> BrokerOrder | None:
        raise BrokerUnavailable("RESEARCH_ONLY mode has no broker")

    async def list_orders(self, *, status: str = "all", after: datetime | None = None) -> list[BrokerOrder]:
        raise BrokerUnavailable("RESEARCH_ONLY mode has no broker")


class _EmptyStream:
    def __aiter__(self) -> _EmptyStream:
        return self

    async def __anext__(self) -> OrderEvent:
        import asyncio

        await asyncio.sleep(3600)
        raise StopAsyncIteration


def _verifier(settings: Settings) -> IntentVerifier | None:
    path = settings.keys_dir / PUBLIC_KEY_FILE
    return load_verifier(settings.keys_dir) if path.is_file() else None


def build_gateway(
    settings: Settings,
    secret_store: SecretStore,
    *,
    ledger: OrderLedger,
    clock: Clock,
    quote_source: QuoteSource | None,
    verifier: IntentVerifier | None = None,
    actor: str = "gateway_factory",
    **fake_broker_kwargs: Any,
) -> ExecutionGateway:
    mode, kind = settings.mode, settings.gateway
    db = ledger.db

    def accept(detail: str) -> None:
        record_mode_decision(db, requested_mode=mode.value, actor=actor, result="ACCEPTED",
                             detail=f"gateway={kind.value} label={modes.execution_label(mode, kind).value}: {detail}")

    # ---- live: always denied, listing every unmet requirement
    if mode in (TradingMode.LIVE_APPROVAL, TradingMode.BOUNDED_LIVE) or kind in modes.LIVE_GATEWAYS:
        gate = LiveReleaseGate(settings, secret_store, db=db)
        failures = gate.evaluate(mode if mode in (TradingMode.LIVE_APPROVAL, TradingMode.BOUNDED_LIVE) else None, kind)
        record_mode_decision(db, requested_mode=mode.value, actor=actor, result="DENIED",
                             detail=f"gateway={kind.value}: " + "; ".join(failures))
        raise modes.LiveActivationDenied(failures)
    if ledger.mode != mode:
        record_mode_decision(db, requested_mode=mode.value, actor=actor, result="DENIED",
                             detail=f"ledger mode {ledger.mode.value} != settings mode {mode.value}")
        raise modes.ModeViolation(f"ledger mode {ledger.mode.value} != settings mode {mode.value}")
    if mode == TradingMode.RESEARCH_ONLY:
        accept("RESEARCH_ONLY: no execution gateway; every order refused")
        return ResearchOnlyGateway()
    try:
        modes.assert_gateway_allowed(mode, kind)
    except modes.ModeViolation as exc:
        record_mode_decision(db, requested_mode=mode.value, actor=actor, result="DENIED", detail=f"gateway={kind.value}: {exc}")
        raise
    verifier = verifier or _verifier(settings)

    if kind == GatewayKind.FAKE:
        from .fake_broker import FakeBroker

        if verifier is None:  # fail closed: the fake broker must be able to verify the risk engine's signature
            detail = (f"gateway={kind.value}: no intent verification key at {settings.keys_dir / PUBLIC_KEY_FILE}; "
                      "start the risk engine signer (IntentSigner.load_or_create) before building the gateway")
            record_mode_decision(db, requested_mode=mode.value, actor=actor, result="DENIED", detail=detail)
            raise modes.ModeViolation(detail)
        def claimed_in_ledger(coid: str) -> bool:
            rec = ledger.get_record(coid)
            return rec is not None and rec.state == OrderState.SUBMITTING and rec.gateway == GatewayKind.FAKE

        kw: dict[str, Any] = {"verifier": verifier, "state_path": settings.var_dir / "fake_broker_state.json",
                              "poll_interval_s": 0.02, "claim_check": claimed_in_ledger}
        kw.update(fake_broker_kwargs)
        gw = FakeBroker(clock, quote_source, **kw)
        accept("deterministic in-process fake broker (SIMULATED fills)")
        return gw

    if kind != GatewayKind.ALPACA_PAPER:
        detail = "LEAN execution retired; use ALPACA_PAPER (SDK) or FAKE. LEAN is for research/backtesting."
        record_mode_decision(db, requested_mode=mode.value, actor=actor, result="DENIED", detail=detail)
        raise modes.ModeViolation(detail)
    if not secret_store.alpaca_paper_available():
        detail = "ALPACA_PAPER requires ALPACA_PAPER_KEY_ID and ALPACA_PAPER_SECRET_KEY"
        record_mode_decision(db, requested_mode=mode.value, actor=actor, result="DENIED", detail=detail)
        raise MissingPaperCredentials(detail)
    if verifier is None:
        detail = "no intent verification key for Alpaca gateway"
        record_mode_decision(db, requested_mode=mode.value, actor=actor, result="DENIED", detail=detail)
        raise modes.ModeViolation(detail)
    from .alpaca_gateway import AlpacaPaperGateway

    gateway = AlpacaPaperGateway(secret_store, ledger=ledger, verifier=verifier, clock=clock)
    accept("official alpaca-py SDK; paper endpoint only; no LEAN execution or QuantConnect deployment")
    return gateway
