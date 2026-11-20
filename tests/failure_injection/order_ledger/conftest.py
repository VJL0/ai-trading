"""Failure-injection fixtures: ledger + deterministic FakeBroker + ExecutionService + Reconciler (all offline)."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from packages.contracts.enums import ReconciliationStatus
from packages.contracts.models import ReconciliationReport
from services.order_ledger.execution import ExecutionService
from services.order_ledger.reconcile import Reconciler
from tests.unit.order_ledger.ol_support import Harness, make_harness, pass_gate


@dataclass
class System:
    h: Harness
    svc: ExecutionService
    rec: Reconciler

    def pump(self) -> int:
        """Deliver every event the broker emitted since the last pump through the consumer path."""
        evs = self.h.broker.drain_events()
        for ev in evs:
            self.svc.handle_event(ev)
        return len(evs)

    async def startup(self) -> ReconciliationReport:
        report = await self.rec.startup_recovery()
        return report

    async def ready(self) -> None:
        report = await self.startup()
        assert report.status == ReconciliationStatus.CLEAN, report.mismatches
        assert not self.h.entries.is_locked()[0]


def build_system(h: Harness, **rec_kw: Any) -> System:
    rec_kw.setdefault("confirm_delay_s", 0)
    svc = ExecutionService(h.ledger, h.broker, pass_gate, h.clock, entries_lock=h.entries, submit_timeout_s=0.5,
                           lookup_timeout_s=0.5, reconnect_backoff_s=(0.01,), watchdog_interval_s=3600)
    rec = Reconciler(h.db, h.ledger, h.broker, h.entries, h.clock, **rec_kw)
    return System(h, svc, rec)


@pytest.fixture
def system(tmp_path: Path) -> Iterator[System]:
    h = make_harness(tmp_path)
    sysm = build_system(h)
    yield sysm
    sysm.h.db.close()


@pytest.fixture
def h(tmp_path: Path) -> Iterator[Harness]:
    harness = make_harness(tmp_path)
    yield harness
    harness.db.close()
