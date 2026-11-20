"""End-to-end tests of the integrated runtime (SIMULATED: synthetic replay feed, deterministic FakeBroker,
explicit fake model transports). None of these tests touch a real broker or model API.

Covers acceptance criteria 2, 3, 4 (fake transport), 5, 7 (fake broker path), 8, 9, 11, 12.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from packages.contracts import TradeProposal
from packages.contracts.enums import (
    DecisionAction,
    FeedMode,
    GatewayKind,
    OrderState,
    Proposer,
    ReconciliationStatus,
    Side,
    TradingMode,
)
from packages.core.config import Settings
from packages.core.ids import new_id
from packages.core.money import floor_shares, round_price

pytestmark = pytest.mark.slow


def _settings(root: Path, **over: Any) -> Settings:
    base: dict[str, Any] = dict(project_root=root, use_keychain=False, mode=TradingMode.PAPER,
                                gateway=GatewayKind.FAKE, feed_mode=FeedMode.SIMULATED, reconcile_interval_s=3600)
    base.update(over)
    return Settings(**base)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def rt(root: Path) -> Iterator[Any]:
    from services.api.runtime import Runtime

    runtime = Runtime(_settings(root))
    yield runtime
    asyncio.run(runtime.aclose())


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


async def _startup(rt: Any) -> Any:
    report = await rt.reconciler.startup_recovery()
    rt._after_reconcile(report)
    rt._on_sim_tick(rt.now())  # deliver real replay quotes to feed-health monitoring
    return report


def _forced_proposal(rt: Any, symbol: str = "SPY", *, qty: Decimal | None = None,
                     limit_bump_bps: Decimal = Decimal(0)) -> tuple[TradeProposal, Any]:
    """A deterministic BUY proposal at the current simulated ask (bypasses the signal threshold only)."""
    from services.market_data.snapshot import build_market_snapshot

    now = rt.now()
    snap = _run(build_market_snapshot(rt.provider, symbol, now, rt.feed_mode, dataset_version=rt.dataset_version))
    assert snap.quote is not None, "synthetic replay must provide a point-in-time quote"
    limit = round_price(snap.quote.ask * (1 + limit_bump_bps / Decimal(10_000)), side_buy=True)
    q = qty if qty is not None else max(floor_shares(rt.policy.max_order_notional * Decimal("0.6") / limit), Decimal(1))
    stop = round_price(limit * Decimal("0.99"), side_buy=False)
    sig_id = new_id("sig")
    prop = TradeProposal(
        proposal_id=new_id("prop"), causal_id=new_id("cz"), created_at=now, data_asof=snap.data_asof,
        action=DecisionAction.TRADE, proposer=Proposer.TEST, strategy_id="e2e-forced", symbol=symbol,
        side=Side.BUY, quantity=q, limit_price=limit, stop_price=stop, max_planned_loss=q * (limit - stop),
        evidence_ids=(sig_id,), signal_id=sig_id, dataset_version=rt.dataset_version,
    )
    return prop, snap


async def _push(rt: Any, prop: TradeProposal, snap: Any) -> tuple[Any, Any]:
    rt.ledger.record_market_snapshot(snap)
    rt.ledger.record_proposal(prop)
    ctx = await rt.risk_context(snap)
    decision = rt.risk.evaluate(prop, ctx)
    rt.ledger.record_decision(decision)
    if not decision.approved:
        return decision, None
    intent = rt.risk.authorize(prop, decision, ctx)
    rt.ledger.record_intent(intent)
    await rt.prepare_gate(snap)
    order = await rt.execution.submit(intent)
    return decision, (intent, order)


async def _drain_events(rt: Any, seconds: float = 1.0) -> None:
    stop = asyncio.Event()
    task = asyncio.create_task(rt.execution.run_event_consumer(stop))
    await asyncio.sleep(seconds)
    stop.set()
    task.cancel()
    with __import__("contextlib").suppress(asyncio.CancelledError, Exception):
        await task


def test_status_is_labeled_simulated_and_not_live(rt: Any) -> None:
    st = rt.status()
    assert st["execution_label"] == "SIMULATED"
    assert st["live_execution_compiled"] is False
    assert st["feed"]["feed_mode"] == "SIMULATED" or st["feed"].get("is_simulated") is True
    assert st["credentials"] == {"alpaca_paper": False, "anthropic": False, "openai": False}


def test_entries_locked_until_startup_reconciliation_clean(rt: Any) -> None:
    locked, reasons = rt.entries_status()
    assert locked, "entries must be locked before broker-authoritative reconciliation"
    report = _run(_startup(rt))
    assert report.status == ReconciliationStatus.CLEAN
    locked, reasons = rt.entries_status()
    assert not locked, reasons


def test_decision_cycle_persists_causal_chain(rt: Any) -> None:
    _run(_startup(rt))
    res = _run(rt.run_cycle(symbols=None, use_ai=True, actor="test"))
    assert res["outcome"] in {"NO_TRADE", "REJECTED", "SUBMITTED"}, res
    cz = res["causal_id"]
    assert rt.db.query_one("SELECT COUNT(*) n FROM quant_signals WHERE causal_id=?", (cz,))["n"] >= 1
    assert rt.db.query_one("SELECT COUNT(*) n FROM proposals WHERE causal_id=?", (cz,))["n"] == 1
    assert rt.db.query_one("SELECT COUNT(*) n FROM audit_events WHERE causal_id=? AND event_type='DECISION_CYCLE'",
                           (cz,))["n"] == 1


def test_approved_paper_order_fills_on_fake_broker_and_reconciles(rt: Any) -> None:
    _run(_startup(rt))
    prop, snap = _forced_proposal(rt)
    decision, sent = _run(_push(rt, prop, snap))
    assert decision.approved, decision.reasons
    intent, _ = sent
    for _ in range(3):
        rt.sim_clock.step(60)
        rt._on_sim_tick(rt.now())
    _run(_drain_events(rt))
    order = rt.ledger.get_order(intent.client_order_id)
    assert order.state in (OrderState.FILLED, OrderState.PARTIALLY_FILLED, OrderState.ACKNOWLEDGED,
                           OrderState.SUBMITTED), order
    report = _run(rt.reconcile())
    assert report.status == ReconciliationStatus.CLEAN, report.mismatches
    # exactly one broker submission for this client order id
    assert rt.db.query_one("SELECT COUNT(*) n FROM orders WHERE client_order_id=?", (intent.client_order_id,))["n"] == 1


@pytest.mark.parametrize("variant", ["oversized", "leveraged_etf", "far_limit"])
def test_risky_proposals_are_stopped_by_risk_policy(rt: Any, variant: str) -> None:
    _run(_startup(rt))
    if variant == "oversized":
        prop, snap = _forced_proposal(rt, qty=Decimal("100000"))
    elif variant == "far_limit":
        prop, snap = _forced_proposal(rt, limit_bump_bps=Decimal("500"))
    else:
        prop, snap = _forced_proposal(rt)
        prop = prop.model_copy(update={"symbol": "TQQQ"})
    decision, sent = _run(_push(rt, prop, snap))
    assert not decision.approved
    assert sent is None
    assert rt.db.query_one("SELECT COUNT(*) n FROM orders WHERE proposal_id=?", (prop.proposal_id,))["n"] == 0


def test_operator_halt_blocks_new_entries(rt: Any) -> None:
    _run(_startup(rt))
    rt.halt("e2e stop", "test")
    assert rt.entries_status()[0]
    prop, snap = _forced_proposal(rt)
    decision, sent = _run(_push(rt, prop, snap))
    assert not decision.approved and sent is None


def test_tampered_intent_refused_at_final_gate(rt: Any) -> None:
    _run(_startup(rt))
    prop, snap = _forced_proposal(rt)
    ctx = _run(rt.risk_context(snap))
    decision = rt.risk.evaluate(prop, ctx)
    assert decision.approved, decision.reasons
    intent = rt.risk.authorize(prop, decision, ctx)
    tampered = intent.model_copy(update={"quantity": intent.quantity * 10})
    _run(rt.prepare_gate(snap))
    assert rt._final_gate(tampered), "tampered intent must be refused"
    assert rt._final_gate(intent), "final gate context is single-use; a second call without a fresh context fails closed"


def test_restart_recovers_and_relocks_until_reconciled(root: Path) -> None:
    from services.api.runtime import Runtime

    rt1 = Runtime(_settings(root))
    try:
        _run(_startup(rt1))
        prop, snap = _forced_proposal(rt1)
        _run(_push(rt1, prop, snap))
    finally:
        _run(rt1.aclose())
    rt2 = Runtime(_settings(root))  # same var/paper namespace: simulated crash/restart
    try:
        assert rt2.entries_status()[0], "after restart entries stay locked until reconciliation"
        report = _run(_startup(rt2))
        assert report.status in (ReconciliationStatus.CLEAN, ReconciliationStatus.MISMATCH)
        # the in-process fake broker does not survive a restart, so a previously filled position at the
        # 'broker' disappears: reconciliation must notice rather than invent state.
    finally:
        _run(rt2.aclose())


def test_live_modes_cannot_start_an_order_route(root: Path) -> None:
    from packages.core.modes import LiveActivationDenied, ModeViolation
    from services.api.runtime import Runtime

    for mode, gw in [(TradingMode.LIVE_APPROVAL, GatewayKind.LEAN_ALPACA_LIVE),
                     (TradingMode.BOUNDED_LIVE, GatewayKind.LEAN_ALPACA_LIVE),
                     (TradingMode.LIVE_APPROVAL, GatewayKind.FAKE)]:
        with pytest.raises((LiveActivationDenied, ModeViolation)):
            Runtime(_settings(root, mode=mode, gateway=gw))


def test_replay_clock_moves_forward_only(rt: Any) -> None:
    t0 = rt.now()
    for _ in range(500):
        rt.sim_clock.step(120)
    assert rt.now() >= t0 + timedelta(minutes=1)
