"""Position lifecycle inside the integrated runtime (SIMULATED: synthetic replay feed, deterministic FakeBroker).

The real ``Runtime`` builds the PositionManager behind ``RuntimePositionPort``: every protective stop and exit is a
POSITION_MANAGER proposal through ``execute_proposal`` (risk engine -> signed intent -> ledger -> final gate ->
FakeBroker) while the decision cycle keeps its own path. No broker or model API is touched.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Iterator
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from packages.contracts.enums import OrderState, OrderType, ReconciliationStatus, Side
from tests.e2e.api.test_e2e_pipeline import _forced_proposal, _settings, _startup

pytestmark = pytest.mark.slow


@pytest.fixture
def rt(tmp_path: Path) -> Iterator[Any]:
    from services.api.runtime import Runtime

    runtime = Runtime(_settings(tmp_path))
    yield runtime
    asyncio.run(runtime.aclose())


async def _pump(rt: Any, seconds: float = 0.3) -> None:
    """Let the (continuously running) order-event consumer apply broker events, as in the live runtime."""
    await asyncio.sleep(seconds)


async def _step(rt: Any, seconds: float) -> None:
    rt.sim_clock.step(seconds)
    rt._on_sim_tick(rt.now())
    await _pump(rt, 0.05)


async def _manage(rt: Any, ticks: int = 1) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for _ in range(ticks):
        rt._on_sim_tick(rt.now())
        out.extend(await rt.positions.tick())
        await _pump(rt, 0.05)
    return out


async def _scenario(rt: Any, prop: Any, snap: Any) -> dict[str, Any]:
    stop = asyncio.Event()
    consumer = asyncio.create_task(rt.execution.run_event_consumer(stop))
    try:
        await asyncio.wait_for(rt.execution.consumer_ready.wait(), timeout=5)
        return await _lifecycle(rt, prop, snap)
    finally:
        stop.set()
        consumer.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await consumer


async def _lifecycle(rt: Any, prop: Any, snap: Any) -> dict[str, Any]:
    entry = await rt.execute_proposal(prop, snap)
    assert entry["outcome"] == "SUBMITTED", entry
    await _pump(rt)
    for _ in range(4):
        if rt.ledger.position_quantities().get("SPY"):
            break
        await _step(rt, 30)
    assert rt.ledger.position_quantities().get("SPY") == Decimal(2)
    protect = await _manage(rt)
    assert protect and protect[0]["action"] == "PROTECT" and protect[0]["outcome"] == "SUBMITTED", protect
    prot = [r for r in rt.ledger.records() if r.is_protective and r.state == OrderState.ACKNOWLEDGED]
    assert len(prot) == 1 and prot[0].quantity == Decimal(2) and prot[0].order_type == OrderType.STOP_LIMIT
    view = rt.positions_view()
    assert view["execution_label"] == "SIMULATED" and view["open"][0]["protected_quantity"] == "2"
    status = rt.status()
    assert status["exits_locked"] is False and status["position_safety"] == []
    # jump to 9 minutes before the session close: the manager flattens
    close = rt.sim_clock.session[1]
    rt.sim_clock.set(close - timedelta(minutes=9))
    acts: list[dict[str, Any]] = []
    for _ in range(6):
        acts.extend(await _manage(rt))
        if not rt.ledger.position_quantities().get("SPY") and rt.position_store.open_plan("SPY") is None:
            break
    return {"acts": acts, "entry": entry}


def test_runtime_protects_a_filled_entry_and_flattens_before_the_close(rt: Any) -> None:
    assert asyncio.run(_startup(rt)).status == ReconciliationStatus.CLEAN
    prop, snap = _forced_proposal(rt, qty=Decimal(2))
    out = asyncio.run(_scenario(rt, prop, snap))
    purposes = [a.get("purpose") for a in out["acts"]]
    assert "EXIT_EOD" in purposes, out["acts"]
    assert rt.ledger.position_quantities().get("SPY") is None
    plan = rt.position_store.closed_plans(limit=1)[0]
    assert plan.close_reason == "EOD" and plan.exit_quantity == Decimal(2)
    sells = [r for r in rt.ledger.records() if r.side == Side.SELL]
    assert [r.order_type for r in sells] == [OrderType.STOP_LIMIT, OrderType.LIMIT]
    assert sells[0].state == OrderState.CANCELED and sells[1].state == OrderState.FILLED
    assert asyncio.run(rt.reconcile()).status == ReconciliationStatus.CLEAN
    # every manager order is labeled and audited like any other order
    n = rt.db.query_one("SELECT COUNT(*) AS n FROM proposals WHERE proposer='POSITION_MANAGER'")["n"]
    assert n == 2
    assert rt.db.query_one("SELECT COUNT(*) AS n FROM audit_events WHERE event_type='POSITION_CLOSED'")["n"] == 1
