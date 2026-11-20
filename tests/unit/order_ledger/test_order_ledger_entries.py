"""EntriesLock: fail-closed entry gating, flag epochs, unlock rules, process-start guard, sleep watchdog."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from packages.contracts.enums import GatewayKind, ReconciliationStatus, TradingMode
from packages.contracts.models import ReconciliationReport
from packages.core.clock import ManualClock
from packages.core.db import Database
from services.order_ledger.entries import (
    PROCESS_GUARD_REASON,
    EntriesLock,
    Flag,
    SuspensionWatchdog,
)
from services.order_ledger.errors import UnlockDenied
from services.order_ledger.reconcile import Reconciler
from tests.unit.order_ledger.ol_support import T0, Harness


async def _clean(h: Harness) -> ReconciliationReport:
    rec = Reconciler(h.db, h.ledger, h.broker, h.entries, h.clock, confirm_delay_s=0)
    report = await rec.reconcile()
    assert report.status == ReconciliationStatus.CLEAN, report.mismatches
    return report


def test_fresh_ledger_is_locked_with_explicit_reasons(h: Harness) -> None:
    locked, reasons = h.entries.is_locked()
    assert locked
    assert PROCESS_GUARD_REASON in reasons
    assert "no CLEAN broker reconciliation has completed yet" in reasons


async def test_clean_reconciliation_unlocks(h: Harness) -> None:
    report = await _clean(h)
    assert not report.entries_locked_after
    assert h.entries.is_locked() == (False, [])
    assert h.entries.process_reconciled()


@pytest.mark.parametrize("flag", list(Flag))
async def test_every_flag_locks_and_needs_a_clean_run_started_after_it(h: Harness, flag: Flag) -> None:
    await _clean(h)
    old_clean = await _clean(h)
    h.entries.set_flag(flag, "test reason", "test")
    locked, reasons = h.entries.is_locked()
    assert locked and any(r.startswith(flag.value) for r in reasons)
    # a CLEAN run that started before the flag was raised cannot clear it
    with pytest.raises(UnlockDenied, match="raised after"):
        h.entries.clear_flag(flag, report=old_clean)
    assert h.entries.is_flag_active(flag)
    # a non-CLEAN report never clears anything
    with pytest.raises(UnlockDenied):
        h.entries.clear_flag(flag, report=old_clean.model_copy(update={"status": ReconciliationStatus.MISMATCH}))
    # an unpersisted report is not accepted either
    with pytest.raises(UnlockDenied):
        h.entries.clear_flag(flag, report=old_clean.model_copy(update={"run_id": "recon_forged"}))
    await _clean(h)  # a fresh CLEAN run clears it
    assert not h.entries.is_flag_active(flag)
    assert h.entries.is_locked() == (False, [])


async def test_flag_raised_during_reconciliation_survives_it(h: Harness) -> None:
    await _clean(h)

    class RaisingReader:
        kind = GatewayKind.FAKE
        is_paper = True

        def __init__(self, inner, entries: EntriesLock) -> None:
            self.inner, self.entries = inner, entries

        async def get_account(self):
            self.entries.set_flag(Flag.BROKER_DISCONNECTED, "dropped mid-run", "test")
            return await self.inner.get_account()

        async def get_positions(self):
            return await self.inner.get_positions()

        async def get_order_by_client_id(self, coid: str):
            return await self.inner.get_order_by_client_id(coid)

        async def list_orders(self, **kw):
            return await self.inner.list_orders(**kw)

    rec = Reconciler(h.db, h.ledger, RaisingReader(h.broker, h.entries), h.entries, h.clock, confirm_delay_s=0)
    report = await rec.reconcile()
    assert report.status == ReconciliationStatus.CLEAN
    assert report.entries_locked_after
    assert h.entries.is_flag_active(Flag.BROKER_DISCONNECTED)


async def test_operator_halt_needs_operator_clearance_and_clean_reconciliation(h: Harness) -> None:
    halt_id = h.entries.lock("operator pressed stop", "dashboard")
    with pytest.raises(UnlockDenied, match="CLEAN"):
        h.entries.clear_halt(halt_id, by="operator", reason="resume")
    await _clean(h)
    assert h.entries.is_locked()[0], "a CLEAN reconciliation does not clear operator halts"
    assert h.entries.clear_halt(halt_id, by="operator", reason="resume")
    assert h.entries.is_locked() == (False, [])


def test_unknown_order_locks_even_without_flag(h: Harness) -> None:
    intent = h.factory.make()
    h.ledger.claim_for_submission(intent.client_order_id, "t")
    h.db.write(lambda c: c.execute("UPDATE orders SET state='UNKNOWN' WHERE client_order_id=?", (intent.client_order_id,)))
    locked, why = h.entries.is_locked()
    assert locked and any("unknown outcome" in r for r in why)


@pytest.mark.parametrize("mode", [TradingMode.RESEARCH_ONLY, TradingMode.LIVE_APPROVAL, TradingMode.BOUNDED_LIVE])
def test_non_trading_modes_are_always_locked(tmp_path: Path, mode: TradingMode) -> None:
    db = Database(tmp_path / "l.sqlite")
    try:
        locked, reasons = EntriesLock(db, mode).is_locked()
        assert locked and any(mode.value in r for r in reasons)
    finally:
        db.close()


def test_unreadable_state_fails_closed(tmp_path: Path) -> None:
    db = Database(tmp_path / "l.sqlite")
    lock = EntriesLock(db, TradingMode.PAPER)
    db.close()
    db.path.unlink()  # state gone / unreadable
    locked, reasons = lock.is_locked()
    assert locked and "unreadable" in reasons[0]


async def test_process_guard_is_per_database_instance(h: Harness) -> None:
    await _clean(h)
    assert not h.entries.is_locked()[0]
    # another EntriesLock on the same open Database shares the guard
    assert EntriesLock(h.db, TradingMode.PAPER).is_locked() == (False, [])
    # a restarted process (new Database on the same file) is locked until its own CLEAN run
    restarted = h.reopen()
    try:
        locked, reasons = restarted.entries.is_locked()
        assert locked and reasons == [PROCESS_GUARD_REASON]
        rec = Reconciler(restarted.db, restarted.ledger, restarted.broker, restarted.entries, restarted.clock,
                         confirm_delay_s=0)
        report = await rec.reconcile()
        assert report.status == ReconciliationStatus.CLEAN
        assert restarted.entries.is_locked() == (False, [])
    finally:
        restarted.db.close()


def test_mark_process_reconciled_requires_persisted_clean_run(h: Harness) -> None:
    forged = ReconciliationReport(
        run_id="recon_forged", started_at=T0, finished_at=T0, status=ReconciliationStatus.CLEAN,
        gateway=GatewayKind.FAKE, broker_source="x", mismatches=(), entries_locked_after=False,
    )
    with pytest.raises(UnlockDenied):
        h.entries.mark_process_reconciled(forged)
    assert not h.entries.mark_process_reconciled(forged.model_copy(update={"status": ReconciliationStatus.MISMATCH}))
    assert not h.entries.process_reconciled()


def test_suspension_watchdog_detects_clock_jump(h: Harness) -> None:
    wall = ManualClock(T0)
    dog = SuspensionWatchdog(h.entries, wall, max_gap_s=30)
    assert not dog.tick()
    wall.advance(5)
    assert not dog.tick()
    wall.advance(3600)  # the Mac slept for an hour
    assert dog.tick()
    assert h.entries.is_flag_active(Flag.PROCESS_SUSPENDED)
    wall.set(T0)  # clock went backwards
    assert dog.tick()


def test_heartbeat_observation(h: Harness) -> None:
    assert h.entries.observe_heartbeat("lean", T0 - timedelta(seconds=5), max_age_s=15)
    assert not h.entries.observe_heartbeat("lean", T0 - timedelta(seconds=60), max_age_s=15)
    assert h.entries.is_flag_active(Flag.ENGINE_HEARTBEAT_STALE)
    assert not h.entries.observe_heartbeat("lean", None, max_age_s=15)


def test_flag_audit_trail(h: Harness) -> None:
    epoch0 = h.entries.epoch()
    h.entries.set_flag(Flag.BROKER_DISCONNECTED, "x", "t")
    h.entries.set_flag(Flag.BROKER_DISCONNECTED, "again", "t")  # re-raise bumps the epoch, audits once
    assert h.entries.epoch() == epoch0 + 2
    rows = h.db.query("SELECT payload_json FROM audit_events WHERE event_type='ENTRIES_FLAG_SET'")
    assert len(rows) == 1
