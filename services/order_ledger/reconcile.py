"""Broker-authoritative reconciliation and startup recovery (spec §7, §11; acceptance 8 and 11).

``reconcile()``:
  1. reads the broker (account, positions, open + recent orders). Any read failure => ERROR,
     ``broker_disconnected`` raised, entries stay locked;
  2. resolves in-doubt orders (UNKNOWN, and SUBMITTING beyond the grace period or at startup) by client
     order id: found => broker state applied (source RECONCILE); not found after ``unknown_grace_s`` =>
     RECONCILED "not present at broker". **No fills are ever invented and nothing is ever resubmitted**;
  3. applies broker state of every known order (catch-up of missed fills from broker cumulative quantity);
  4. compares broker positions with ``ledger_positions`` and broker orders with the ledger;
  5. persists account/position snapshots and an append-only ``reconciliation_runs`` row. CLEAN clears the
     reconciliation-gated entries flags raised before the run started; MISMATCH raises
     ``reconciliation_mismatch`` and, for unexpected state (foreign position/order, qty mismatch, live
     account), an operator halt.

``startup_recovery()`` raises ``startup_unreconciled``, turns every SUBMITTING order into UNKNOWN (the process
may have crashed after the broker accepted), reconciles, and leaves entries locked unless the run is CLEAN.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sqlite3
from collections.abc import Callable
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

from packages.contracts.enums import EventSource, GatewayKind, OrderState, ReconciliationStatus
from packages.contracts.interfaces import BrokerStateReader
from packages.contracts.models import (
    AccountSnapshot,
    BrokerOrder,
    PositionSnapshot,
    ReconciliationMismatch,
    ReconciliationReport,
)
from packages.core.audit import add_halt, append_audit, record_failure, set_state
from packages.core.canonical import canonical_json
from packages.core.clock import Clock
from packages.core.db import Database, utc_iso
from packages.core.ids import new_id
from packages.core.logging import log_event

from .entries import EntriesLock, Flag, clear_flags_after_clean_conn, set_flag_conn
from .ledger import OrderLedger, OrderRecord
from .state_machine import TERMINAL
from .timeutil import iso

log = logging.getLogger("lat.order_ledger.reconcile")
ZERO = Decimal(0)

CRITICAL_KINDS = frozenset({
    "UNEXPECTED_POSITION", "QTY_MISMATCH", "MISSING_POSITION", "SHORT_POSITION", "UNKNOWN_ORDER",
    "ACCOUNT_NOT_PAPER", "BROKER_ORDER_MISMATCH", "FILL_QTY_MISMATCH",
})
"""Mismatch kinds that add an operator halt (cleared only by the operator after a CLEAN run)."""


def engine_health_of(reader: object) -> tuple[bool, str] | None:
    """``(ok, detail)`` from a gateway exposing ``engine_health()`` (LEAN bridge), else None. Errors => not ok."""
    fn: Callable[[], tuple[bool, str]] | None = getattr(reader, "engine_health", None)
    if fn is None or not callable(fn):
        return None
    try:
        ok, detail = fn()
    except Exception as exc:  # fail closed
        return False, f"engine health unreadable: {type(exc).__name__}"
    return bool(ok), str(detail)


def _foreign_gateway(rec: OrderRecord, reader_kind: GatewayKind) -> ReconciliationMismatch:
    return ReconciliationMismatch(
        kind="GATEWAY_MISMATCH", client_order_id=rec.client_order_id, symbol=rec.symbol,
        local=f"{rec.state.value} via {rec.gateway.value}", broker=f"reader {reader_kind.value}",
        detail=f"open order was routed through {rec.gateway.value}; the {reader_kind.value} broker cannot resolve it "
               "(gateway switched within one ledger namespace?)",
    )


class Reconciler:
    def __init__(
        self,
        db: Database,
        ledger: OrderLedger,
        reader: BrokerStateReader,
        entries_lock: EntriesLock,
        clock: Clock,
        *,
        unknown_grace_s: float = 60.0,
        read_timeout_s: float = 30.0,
        order_lookback: timedelta = timedelta(days=5),
        confirm_delay_s: float = 0.5,
    ) -> None:
        self.db = db
        self.ledger = ledger
        self.reader = reader
        self.entries = entries_lock
        self.clock = clock
        self.unknown_grace = timedelta(seconds=unknown_grace_s)
        self.read_timeout_s = read_timeout_s
        self.order_lookback = order_lookback
        self.confirm_delay_s = confirm_delay_s
        self._lock = asyncio.Lock()

    @property
    def broker_source(self) -> str:
        label = getattr(self.reader, "source_label", None)
        return str(label) if label else self.reader.kind.value

    async def _read(self, coro: Any) -> Any:
        return await asyncio.wait_for(coro, timeout=self.read_timeout_s)

    async def _fetch(self) -> tuple[AccountSnapshot, list[PositionSnapshot], dict[str, BrokerOrder], set[str]]:
        account = await self._read(self.reader.get_account())
        positions = await self._read(self.reader.get_positions())
        open_orders = await self._read(self.reader.list_orders(status="open"))
        recent = await self._read(self.reader.list_orders(status="all", after=self.clock.now() - self.order_lookback))
        by_coid: dict[str, BrokerOrder] = {o.client_order_id: o for o in recent}
        by_coid.update({o.client_order_id: o for o in open_orders})
        return account, list(positions), by_coid, {o.client_order_id for o in open_orders}

    # ------------------------------------------------------------------ public
    async def startup_recovery(self) -> ReconciliationReport:
        """Process start: lock entries, treat SUBMITTING as UNKNOWN, reconcile. Unlocks only if CLEAN."""
        async with self._lock:
            return await self._startup_locked()

    async def reconcile(self, *, startup: bool = False) -> ReconciliationReport:
        """Broker-authoritative reconciliation.

        Until a CLEAN run has completed in this process, every call runs the startup path: any SUBMITTING
        order found then was left by a previous process (this process cannot submit while entries are locked).
        """
        async with self._lock:
            if startup or not self.entries.process_reconciled():
                return await self._startup_locked()
            return await self._reconcile(startup=False)

    async def _startup_locked(self) -> ReconciliationReport:
        self.entries.set_flag(Flag.STARTUP_UNRECONCILED, "process start: broker-authoritative reconciliation pending", "startup")
        recovered = self.ledger.recover_in_doubt_at_startup()
        at = self.clock.now()
        self.db.write(lambda c: append_audit(
            c, event_type="STARTUP_RECOVERY", source="reconciler", mode=self.ledger.mode.value, ts=at,
            payload={"in_doubt": recovered, "note": "SUBMITTING treated as UNKNOWN; nothing resubmitted"},
        ))
        return await self._reconcile(startup=True)

    # ------------------------------------------------------------------ core
    async def _reconcile(self, *, startup: bool) -> ReconciliationReport:
        started = self.clock.now()
        run_id = new_id("recon")
        observed_epoch = self.entries.epoch()
        keep: set[str] = set()
        notes: list[str] = []
        with contextlib.suppress(Exception):
            expired = self.ledger.expire_stale_intents()
            if expired:
                notes.append(f"expired {len(expired)} unsent intent(s)")
        health = engine_health_of(self.reader)
        if health is not None:
            ok, detail = health
            if not ok:
                self.entries.set_flag(Flag.ENGINE_HEARTBEAT_STALE, str(detail), "reconciler")
                keep.add(Flag.ENGINE_HEARTBEAT_STALE.value)
                notes.append(f"engine not healthy: {detail}")
        try:
            account, positions, by_coid, open_coids = await self._fetch()
        except Exception as exc:
            return self._error_report(run_id, started, observed_epoch, startup,
                                      f"broker state unavailable: {type(exc).__name__}: {exc}")

        resolved: list[str] = []
        mismatches: list[ReconciliationMismatch] = []
        try:
            for attempt in range(2):
                mismatches = await self._compare(account, positions, by_coid, open_coids, startup=startup, resolved=resolved)
                if not any(m.kind in CRITICAL_KINDS for m in mismatches) or attempt == 1:
                    break
                # Confirm against a fresh broker read before declaring a critical mismatch (event/REST race).
                await asyncio.sleep(self.confirm_delay_s)
                account, positions, by_coid, open_coids = await self._fetch()
        except Exception as exc:
            return self._error_report(run_id, started, observed_epoch, startup,
                                      f"reconciliation failed: {type(exc).__name__}: {exc}")

        status = ReconciliationStatus.CLEAN if not mismatches else ReconciliationStatus.MISMATCH
        detail = "; ".join(
            [f"{len(mismatches)} mismatch(es)" if mismatches else "broker and ledger agree", *notes,
             *(f"{m.kind}: {m.detail}" for m in mismatches[:10])]
        )
        return self._persist(
            run_id=run_id, started=started, observed_epoch=observed_epoch, startup=startup, status=status,
            mismatches=mismatches, resolved=resolved, detail=detail, account=account, positions=positions,
            keep=frozenset(keep),
        )

    async def _lookup(self, coid: str, by_coid: dict[str, BrokerOrder]) -> BrokerOrder | None:
        if coid in by_coid:
            return by_coid[coid]
        found = await self._read(self.reader.get_order_by_client_id(coid))
        if found is not None:
            by_coid[coid] = found
        return found

    async def _compare(
        self,
        account: AccountSnapshot,
        positions: list[PositionSnapshot],
        by_coid: dict[str, BrokerOrder],
        open_coids: set[str],
        *,
        startup: bool,
        resolved: list[str],
    ) -> list[ReconciliationMismatch]:
        now = self.clock.now()
        out: list[ReconciliationMismatch] = []
        checked: set[str] = set()

        reader_kind = self.reader.kind

        # 1. in-doubt orders: resolve by client order id; never resubmit, never invent fills
        for rec in self.ledger.in_doubt_orders():
            coid = rec.client_order_id
            if rec.gateway != reader_kind:
                # Another gateway's broker cannot vouch for this order (absent there proves nothing): stay locked.
                out.append(_foreign_gateway(rec, reader_kind))
                continue
            if rec.state == OrderState.SUBMITTING:
                if self.ledger.engine_queue_status(coid) == "QUEUED":
                    continue  # waiting for engine pickup: never sent, not in doubt
                claimed = rec.claimed_at or rec.updated_at
                if not startup and now - claimed < self.unknown_grace:
                    continue  # submission in flight in this process
                self.ledger.mark_unknown(coid, "still SUBMITTING beyond the grace period; outcome unknown")
                rec = self.ledger.get_record(coid) or rec
            try:
                found = await self._lookup(coid, by_coid)
            except Exception as exc:
                out.append(ReconciliationMismatch(kind="LOOKUP_FAILED", client_order_id=coid,
                                                  detail=f"lookup by client order id failed: {type(exc).__name__}: {exc}"))
                continue
            if found is not None:
                out.extend(self._apply_and_check(coid, found))
                checked.add(coid)
                after = self.ledger.get_record(coid)
                if after is not None and after.state not in (OrderState.UNKNOWN, OrderState.SUBMITTING) and coid not in resolved:
                    resolved.append(coid)
                continue
            ref = rec.unknown_since or rec.claimed_at or rec.updated_at
            if now - ref >= self.unknown_grace:
                self.ledger.mark_reconciled_absent(
                    coid, f"not present at broker {int(self.unknown_grace.total_seconds())}s after submission; no fills invented")
                if coid not in resolved:
                    resolved.append(coid)
            else:
                out.append(ReconciliationMismatch(
                    kind="UNRESOLVED_UNKNOWN_ORDER", client_order_id=coid, local=rec.state.value, broker="not found (yet)",
                    detail=f"outcome unknown; broker has no such order yet (grace {int(self.unknown_grace.total_seconds())}s)",
                ))

        # 2. every broker order known to the ledger: apply broker truth (missed events / fills catch-up)
        for coid, bo in sorted(by_coid.items()):
            rec = self.ledger.get_record(coid)
            if rec is None:
                if coid in open_coids or bo.state not in TERMINAL:
                    out.append(ReconciliationMismatch(
                        kind="UNKNOWN_ORDER", client_order_id=coid, symbol=bo.symbol, broker=bo.broker_status,
                        detail="open broker order unknown to the ledger (not placed through the signed-intent gateway)",
                    ))
                elif bo.filled_quantity > 0:
                    out.append(ReconciliationMismatch(
                        kind="UNKNOWN_ORDER", client_order_id=coid, symbol=bo.symbol, broker=bo.broker_status,
                        detail=f"filled broker order ({bo.filled_quantity}) unknown to the ledger",
                    ))
                continue
            if coid in checked:
                continue  # just applied and checked in step 1
            out.extend(self._apply_and_check(coid, bo))

        # 3. ledger working orders the broker does not know
        for rec in self.ledger.working_orders():
            if rec.client_order_id in by_coid:
                continue
            if rec.gateway != reader_kind:
                out.append(_foreign_gateway(rec, reader_kind))
                continue
            try:
                found = await self._lookup(rec.client_order_id, by_coid)
            except Exception as exc:
                out.append(ReconciliationMismatch(kind="LOOKUP_FAILED", client_order_id=rec.client_order_id,
                                                  detail=f"{type(exc).__name__}: {exc}"))
                continue
            if found is None:
                out.append(ReconciliationMismatch(
                    kind="MISSING_ORDER", client_order_id=rec.client_order_id, symbol=rec.symbol, local=rec.state.value,
                    broker="absent", detail="ledger shows a working order the broker does not report",
                ))
            else:
                out.extend(self._apply_and_check(rec.client_order_id, found))

        # 4. positions (broker-authoritative)
        ledger_pos = self.ledger.position_quantities()
        broker_pos: dict[str, Decimal] = {}
        for p in positions:
            if p.quantity != 0:
                broker_pos[p.symbol] = broker_pos.get(p.symbol, ZERO) + p.quantity
        for sym in sorted(set(ledger_pos) | set(broker_pos)):
            lq, bq = ledger_pos.get(sym, ZERO), broker_pos.get(sym, ZERO)
            if bq < 0:
                out.append(ReconciliationMismatch(kind="SHORT_POSITION", symbol=sym, local=str(lq), broker=str(bq),
                                                  detail="broker reports a short position (long-only system)"))
                continue
            if lq == bq:
                continue
            kind = "UNEXPECTED_POSITION" if lq == 0 else ("MISSING_POSITION" if bq == 0 else "QTY_MISMATCH")
            out.append(ReconciliationMismatch(kind=kind, symbol=sym, local=format(lq, "f"), broker=format(bq, "f"),
                                              detail=f"ledger {lq} vs broker {bq}"))

        # 5. account sanity
        if not account.is_paper:
            out.append(ReconciliationMismatch(kind="ACCOUNT_NOT_PAPER", detail="broker account is not a paper account"))
        if account.trading_blocked or account.account_blocked or account.status.strip().upper() != "ACTIVE":
            out.append(ReconciliationMismatch(
                kind="ACCOUNT_BLOCKED",
                detail=f"account status {account.status}, trading_blocked={account.trading_blocked}, "
                       f"account_blocked={account.account_blocked}",
            ))
        return out

    @staticmethod
    def _anomaly_mismatches(coid: str, anomalies: tuple[str, ...]) -> list[ReconciliationMismatch]:
        return [ReconciliationMismatch(kind="BROKER_ORDER_MISMATCH", client_order_id=coid, detail=a[:500]) for a in anomalies]

    def _apply_and_check(self, coid: str, bo: BrokerOrder) -> list[ReconciliationMismatch]:
        """Apply broker truth, then re-check consistency explicitly (independent of event dedupe)."""
        res = self.ledger.apply_broker_order(bo, EventSource.RECONCILE)
        out = self._anomaly_mismatches(coid, res.anomalies)
        rec = self.ledger.get_record(coid)
        if rec is None:
            return out
        diffs: list[str] = []
        if bo.symbol != rec.symbol or bo.side != rec.side or bo.quantity != rec.quantity or bo.order_type != rec.order_type:
            diffs.append(f"broker {bo.symbol}/{bo.side.value}/{bo.quantity}/{bo.order_type.value} vs ledger "
                         f"{rec.symbol}/{rec.side.value}/{rec.quantity}/{rec.order_type.value}")
        if bo.limit_price is not None and rec.limit_price is not None and bo.limit_price != rec.limit_price:
            diffs.append(f"limit {bo.limit_price} vs {rec.limit_price}")
        if bo.state in TERMINAL and rec.state not in TERMINAL:
            diffs.append(f"broker state {bo.state.value} but ledger {rec.state.value}")
        if diffs and not res.anomalies:
            out.append(ReconciliationMismatch(kind="BROKER_ORDER_MISMATCH", client_order_id=coid, local=rec.state.value,
                                              broker=bo.broker_status, detail="; ".join(diffs)[:500]))
        if bo.filled_quantity != rec.filled_quantity:
            out.append(ReconciliationMismatch(
                kind="FILL_QTY_MISMATCH", client_order_id=coid, symbol=rec.symbol, local=format(rec.filled_quantity, "f"),
                broker=format(bo.filled_quantity, "f"), detail="cumulative filled quantity differs after reconciliation",
            ))
        return out

    # ------------------------------------------------------------------ persistence
    def _persist(
        self,
        *,
        run_id: str,
        started: datetime,
        observed_epoch: int,
        startup: bool,
        status: ReconciliationStatus,
        mismatches: list[ReconciliationMismatch],
        resolved: list[str],
        detail: str,
        account: AccountSnapshot | None,
        positions: list[PositionSnapshot],
        keep: frozenset[str] = frozenset(),
    ) -> ReconciliationReport:
        finished = self.clock.now()
        mode = self.ledger.mode.value
        gateway = self.reader.kind

        def build(locked: bool) -> ReconciliationReport:
            return ReconciliationReport(
                run_id=run_id, started_at=started, finished_at=finished, status=status, gateway=gateway,
                broker_source=self.broker_source, mismatches=tuple(mismatches), resolved_unknown_orders=tuple(resolved),
                entries_locked_after=locked, detail=detail[:4000],
            )

        def write_all(conn: sqlite3.Connection, report: ReconciliationReport) -> list[str]:
            if account is not None:
                conn.execute(
                    "INSERT OR IGNORE INTO account_snapshots(snapshot_id, asof, source, is_paper, equity, payload_json)"
                    " VALUES (?,?,?,?,?,?)",
                    (account.snapshot_id, iso(account.asof), account.source, int(account.is_paper), format(account.equity, "f"),
                     canonical_json(account)),
                )
            for p in positions:
                conn.execute(
                    "INSERT INTO position_snapshots(snapshot_group, symbol, asof, source, payload_json) VALUES (?,?,?,?,?)",
                    (run_id, p.symbol, iso(p.asof), p.source, canonical_json(p)),
                )
            conn.execute(
                "INSERT INTO reconciliation_runs(run_id, started_at, finished_at, status, gateway, entries_locked_after,"
                " payload_json) VALUES (?,?,?,?,?,?,?)",
                (run_id, iso(started), iso(finished), status.value, gateway.value, int(report.entries_locked_after),
                 canonical_json(report)),
            )
            conn.execute("INSERT INTO reconciliation_meta(run_id, observed_epoch, startup) VALUES (?,?,?)",
                         (run_id, observed_epoch, int(startup)))
            cleared: list[str] = []
            if status == ReconciliationStatus.CLEAN:
                cleared = clear_flags_after_clean_conn(conn, run_id=run_id, observed_epoch=observed_epoch, mode=mode,
                                                       at=finished, keep=keep)
            elif status == ReconciliationStatus.MISMATCH:
                summary = ", ".join(sorted({m.kind for m in mismatches}))
                set_flag_conn(conn, Flag.RECONCILIATION_MISMATCH, reason=f"reconciliation {run_id}: {summary}",
                              source="reconciler", mode=mode, at=finished)
                critical = [m for m in mismatches if m.kind in CRITICAL_KINDS]
                if critical:
                    active = conn.execute(
                        "SELECT 1 FROM trading_halts WHERE cleared_at IS NULL AND source='reconciler' LIMIT 1"
                    ).fetchone()
                    if active is None:
                        add_halt(conn, reason=f"broker/ledger mismatch ({summary}); operator review required",
                                 source="reconciler", mode=mode)
            else:
                set_flag_conn(conn, Flag.BROKER_DISCONNECTED, reason=detail[:300], source="reconciler", mode=mode, at=finished)
            append_audit(
                conn, event_type="RECONCILIATION_RUN", source="reconciler", mode=mode, ts=finished,
                payload={"run_id": run_id, "status": status.value, "startup": startup, "mismatches": len(mismatches),
                         "kinds": sorted({m.kind for m in mismatches}), "resolved_unknown": resolved, "cleared_flags": cleared,
                         "broker_source": self.broker_source},
            )
            set_state(conn, "last_reconciliation", {"run_id": run_id, "status": status.value, "finished_at": utc_iso(finished)})
            set_state(conn, "broker_connected", status != ReconciliationStatus.ERROR)
            return cleared

        clean = status == ReconciliationStatus.CLEAN

        def _w(conn: sqlite3.Connection) -> ReconciliationReport:
            # The report row is append-only, so compute entries_locked_after inside a savepoint first.
            conn.execute("SAVEPOINT recon_persist")
            guess = build(True)
            write_all(conn, guess)
            locked = bool(self.entries.reasons_conn(conn, assume_process_reconciled=clean))
            if locked:
                conn.execute("RELEASE recon_persist")
                return guess
            conn.execute("ROLLBACK TO recon_persist")
            final = build(False)
            write_all(conn, final)
            conn.execute("RELEASE recon_persist")
            return final

        report = self.db.write(_w)
        if clean:
            self.entries.mark_process_reconciled(report)
        log_event(log, "reconciliation finished", status=status.value, run_id=run_id, mismatches=len(mismatches),
                  entries_locked_after=report.entries_locked_after)
        return report

    def _error_report(self, run_id: str, started: datetime, epoch: int, startup: bool, detail: str) -> ReconciliationReport:
        with contextlib.suppress(Exception):
            self.db.write(lambda c: record_failure(c, component="reconciler", kind="broker_unavailable", detail=detail))
        return self._persist(
            run_id=run_id, started=started, observed_epoch=epoch, startup=startup, status=ReconciliationStatus.ERROR,
            mismatches=[ReconciliationMismatch(kind="BROKER_UNAVAILABLE", detail=detail[:500])], resolved=[], detail=detail,
            account=None, positions=[],
        )
