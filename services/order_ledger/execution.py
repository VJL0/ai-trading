"""ExecutionService: the single order path from a signed intent to the execution gateway (spec §7).

``submit(intent)``:
  1. the intent must already be in the ledger (RISK_APPROVED / HUMAN_APPROVED) and identical to the record;
  2. entries lock (the narrower exit lock for risk-reducing SELLs) +
     ``final_gate(intent)`` (signature, expiry, mode, policy, health ...). Any violation =>
     order REJECTED locally, audited, *nothing sent*;
  3. ``ledger.claim_for_submission`` (atomic CAS to SUBMITTING; an intent is never claimed twice);
  4. ``gateway.submit`` under an asyncio timeout.
     * ``GatewayRejected`` (incl. HTTP 429 ``RateLimited``) => REJECTED. Never retried.
     * timeout / ``SubmitTimeout`` / connection error / anything unexpected => UNKNOWN, entries locked
       (``unknown_orders_present``), then ONE read-only lookup by client order id: found => broker state
       applied; not found => stays UNKNOWN for reconciliation. **Never resubmitted automatically.**
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from packages.contracts.enums import EventSource, OrderState, Side
from packages.contracts.interfaces import ExecutionGateway, GatewayRejected, SubmitTimeout
from packages.contracts.models import ApprovedOrderIntent, BrokerOrder, OrderEvent
from packages.core.audit import append_audit, record_failure
from packages.core.clock import Clock, SystemClock
from packages.core.logging import log_event

from .entries import EntriesLock, Flag, SuspensionWatchdog
from .errors import CancelNotSupported
from .ledger import OrderLedger
from .reconcile import engine_health_of
from .state_machine import PRE_SUBMISSION, TERMINAL

log = logging.getLogger("lat.order_ledger.execution")

FinalGate = Callable[[ApprovedOrderIntent], list[str]]


class ExecutionService:
    def __init__(
        self,
        ledger: OrderLedger,
        gateway: ExecutionGateway,
        final_gate: FinalGate,
        clock: Clock,
        *,
        submit_timeout_s: float = 10.0,
        entries_lock: EntriesLock | None = None,
        lookup_timeout_s: float = 10.0,
        reconnect_backoff_s: tuple[float, ...] = (0.5, 1.0, 2.0, 5.0),
        on_reconnect: Callable[[], Awaitable[Any]] | None = None,
        wall_clock: Clock | None = None,
        watchdog_interval_s: float = 5.0,
        suspend_gap_s: float = 30.0,
        engine_heartbeat_check: bool = True,
    ) -> None:
        if submit_timeout_s <= 0 or lookup_timeout_s <= 0:
            raise ValueError("timeouts must be positive")
        self.ledger = ledger
        self.gateway = gateway
        self.final_gate = final_gate
        self.clock = clock
        self.submit_timeout_s = submit_timeout_s
        self.lookup_timeout_s = lookup_timeout_s
        self.entries = entries_lock or EntriesLock(ledger.db, ledger.mode, clock=clock)
        self.reconnect_backoff_s = reconnect_backoff_s or (1.0,)
        self.on_reconnect = on_reconnect
        self.consumer_ready = asyncio.Event()
        self.events_applied = 0
        self.events_duplicate = 0
        self.disconnects = 0
        # Watchdog (runs alongside the event consumer). Wall clock on purpose: replay/simulation clocks run
        # faster than real time, and the Mac's monotonic clock does not advance during sleep.
        self.wall_clock: Clock = wall_clock or SystemClock()
        self.watchdog_interval_s = watchdog_interval_s
        self.engine_heartbeat_check = engine_heartbeat_check
        self.suspension = SuspensionWatchdog(self.entries, self.wall_clock, max_gap_s=watchdog_interval_s + suspend_gap_s)
        # The LEAN bridge gateway streams events that ``ingest_engine_event`` already applied (they are ledger rows);
        # re-applying them would only inflate the duplicate counters.
        self._events_preapplied = bool(getattr(gateway, "events_preapplied", False))

    # ------------------------------------------------------------------ helpers
    def _audit(self, event_type: str, payload: dict[str, Any], causal_id: str | None = None) -> None:
        at = self.clock.now()
        self.ledger.db.write(lambda c: append_audit(
            c, event_type=event_type, source="execution", mode=self.ledger.mode.value, payload=payload,
            causal_id=causal_id, ts=at,
        ))

    def _failure(self, kind: str, detail: str, causal_id: str | None = None) -> None:
        self.ledger.db.write(lambda c: record_failure(c, component="execution", kind=kind, detail=detail, causal_id=causal_id))

    # ------------------------------------------------------------------ submit
    async def submit(self, intent: ApprovedOrderIntent) -> BrokerOrder | None:
        coid = intent.client_order_id
        rec = self.ledger.get_record(coid)
        if rec is None:
            self._audit("SUBMIT_REFUSED", {"client_order_id": coid, "reason": "intent not recorded in the ledger"},
                        intent.causal_id)
            return None
        if rec.state not in (OrderState.RISK_APPROVED, OrderState.HUMAN_APPROVED):
            # Already claimed/submitted/terminal: never submit twice.
            self._audit("SUBMIT_REFUSED", {"client_order_id": coid, "reason": f"order is {rec.state.value}"}, intent.causal_id)
            return None
        violations: list[str] = []
        if rec.intent_json != intent.model_dump_json():
            violations.append("intent differs from the ledger record (tampered or re-signed)")
        if intent.side == Side.SELL:
            # Long-only: a SELL always reduces risk. Only state uncertainty / an ALL-scope halt blocks it.
            locked, reasons = self.entries.exits_locked()
            if locked:
                violations.extend(f"exits locked: {r}" for r in reasons)
        else:
            locked, reasons = self.entries.is_locked()
            if locked:
                violations.extend(f"entries locked: {r}" for r in reasons)
        try:
            violations.extend(self.final_gate(intent))
        except Exception as exc:  # fail closed
            violations.append(f"final gate error ({type(exc).__name__}: {exc})")
        if violations:
            self.ledger.reject_locally(coid, violations)
            self._audit("FINAL_GATE_REJECTED", {"client_order_id": coid, "violations": violations[:20]}, intent.causal_id)
            return None
        claimed = self.ledger.claim_for_submission(coid, f"execution:{self.gateway.kind.value}")
        if claimed is None:
            after = self.ledger.get_record(coid)
            self._audit("SUBMIT_REFUSED", {"client_order_id": coid, "reason": "claim refused (expired, already claimed,"
                        " or required human approval missing)", "state": after.state.value if after else None},
                        intent.causal_id)
            return None
        try:
            order = await asyncio.wait_for(self.gateway.submit(claimed), timeout=self.submit_timeout_s)
        except GatewayRejected as exc:
            self.ledger.mark_rejected(coid, f"{type(exc).__name__}: {exc}")
            self._audit("SUBMIT_REJECTED", {"client_order_id": coid, "reason": str(exc)[:500], "retried": False},
                        intent.causal_id)
            return None
        except asyncio.CancelledError:
            self.ledger.mark_unknown(coid, "submission task cancelled; outcome unknown")
            raise
        except (SubmitTimeout, TimeoutError, ConnectionError, OSError) as exc:
            return await self._handle_unknown(claimed, f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__)
        except Exception as exc:  # unexpected: the broker may still have accepted => UNKNOWN (fail closed)
            return await self._handle_unknown(claimed, f"unexpected {type(exc).__name__}: {exc}")
        try:
            self.ledger.mark_submitted(coid, order)
        except Exception as exc:  # a response we cannot record is an unknown outcome, never a retry
            return await self._handle_unknown(claimed, f"submit response not recordable: {type(exc).__name__}: {exc}")
        return order

    async def _handle_unknown(self, intent: ApprovedOrderIntent, reason: str) -> BrokerOrder | None:
        coid = intent.client_order_id
        self.ledger.mark_unknown(coid, reason)
        self._audit("SUBMIT_OUTCOME_UNKNOWN", {"client_order_id": coid, "reason": reason[:500], "resubmitted": False},
                    intent.causal_id)
        log_event(log, "submission outcome unknown; not retrying", logging.WARNING, client_order_id=coid, reason=reason)
        try:
            found = await asyncio.wait_for(self.gateway.get_order_by_client_id(coid), timeout=self.lookup_timeout_s)
        except Exception as exc:
            self._failure("unknown_lookup_failed", f"{coid}: {type(exc).__name__}: {exc}", intent.causal_id)
            return None
        if found is None:
            self._audit("UNKNOWN_LOOKUP_NOT_FOUND", {"client_order_id": coid}, intent.causal_id)
            return None
        self.ledger.apply_broker_order(found, EventSource.POLL)
        return found

    # ------------------------------------------------------------------ cancel
    async def cancel(self, client_order_id: str) -> OrderState | None:
        rec = self.ledger.get_record(client_order_id)
        if rec is None:
            return None
        if rec.state in PRE_SUBMISSION:
            self.ledger.cancel_local(client_order_id)
            return self._state(client_order_id)
        if rec.state in TERMINAL:
            return rec.state
        self.ledger.request_cancel(client_order_id)
        try:
            await asyncio.wait_for(self.gateway.cancel(client_order_id), timeout=self.submit_timeout_s)
        except CancelNotSupported as exc:
            # Never transmitted: undo the local CANCEL_PENDING so the ledger does not claim a pending cancel.
            self.ledger.revert_cancel_request(client_order_id, f"cancel not transmitted: {exc}")
            self._audit("CANCEL_NOT_SUPPORTED", {"client_order_id": client_order_id, "reason": str(exc)[:500]},
                        rec.causal_id)
        except GatewayRejected as exc:
            self._audit("CANCEL_REFUSED", {"client_order_id": client_order_id, "reason": str(exc)[:500]}, rec.causal_id)
            try:
                found = await asyncio.wait_for(self.gateway.get_order_by_client_id(client_order_id),
                                               timeout=self.lookup_timeout_s)
            except Exception as lookup_exc:
                self._failure("cancel_lookup_failed", f"{client_order_id}: {lookup_exc}", rec.causal_id)
                found = None
            if found is not None:
                self.ledger.apply_broker_order(found, EventSource.POLL)
        except asyncio.CancelledError:
            self.entries.set_flag(Flag.BROKER_DISCONNECTED, "cancel task interrupted; reconcile broker state", "execution")
            raise
        except Exception as exc:  # outcome of the cancel unknown: broker events / reconciliation decide
            self.entries.set_flag(Flag.BROKER_DISCONNECTED, "cancel outcome unknown; reconciliation required", "execution")
            self._failure("cancel_outcome_unknown", f"{client_order_id}: {type(exc).__name__}: {exc}", rec.causal_id)
        return self._state(client_order_id)

    def _state(self, coid: str) -> OrderState | None:
        rec = self.ledger.get_record(coid)
        return None if rec is None else rec.state

    # ------------------------------------------------------------------ event consumer
    def handle_event(self, ev: OrderEvent) -> None:
        if self._events_preapplied:
            self.events_applied += 1  # already applied (and deduplicated) when the engine posted it
            return
        try:
            result = self.ledger.apply_event(ev)
        except Exception as exc:  # never drop silently: record + lock entries (the event itself may be malformed)
            ident = f"{getattr(ev, 'client_order_id', '?')}/{getattr(ev, 'event_id', type(ev).__name__)}"
            detail = f"{ident}: {type(exc).__name__}: {exc}"[:1000]
            with contextlib.suppress(Exception):
                self._failure("event_apply_failed", detail)
            with contextlib.suppress(Exception):
                self.entries.set_flag(Flag.LEDGER_ANOMALY, f"failed to apply broker event {detail}", "execution")
            return
        if result.duplicate:
            self.events_duplicate += 1
        else:
            self.events_applied += 1

    async def run_event_consumer(self, stop_event: asyncio.Event) -> None:
        """Consume ``gateway.order_events()`` into the ledger until ``stop_event`` is set.

        A dropped stream raises ``broker_disconnected`` (entries locked until a CLEAN reconciliation; events
        missed during the outage are recovered from broker REST state, never invented) and reconnects with
        bounded backoff.
        """
        watchdog = asyncio.create_task(self._watchdog_loop(stop_event), name="order-ledger-watchdog")
        try:
            await self._consume(stop_event)
        finally:
            watchdog.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await watchdog

    # ------------------------------------------------------------------ watchdog
    def watchdog_tick(self) -> list[str]:
        """One watchdog pass. Returns the flags raised (Mac sleep / suspension, stale LEAN heartbeat)."""
        raised: list[str] = []
        try:
            if self.suspension.tick():
                raised.append(Flag.PROCESS_SUSPENDED.value)
        except Exception as exc:  # never let the watchdog die silently
            self._failure("watchdog_failed", f"suspension check: {type(exc).__name__}: {exc}")
        if self.engine_heartbeat_check:
            health = engine_health_of(self.gateway)
            if health is not None and not health[0]:
                try:
                    self.entries.set_flag(Flag.ENGINE_HEARTBEAT_STALE, health[1], "watchdog")
                    raised.append(Flag.ENGINE_HEARTBEAT_STALE.value)
                except Exception as exc:
                    self._failure("watchdog_failed", f"heartbeat flag: {type(exc).__name__}: {exc}")
        return raised

    async def _watchdog_loop(self, stop_event: asyncio.Event) -> None:
        while not stop_event.is_set():
            self.watchdog_tick()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(stop_event.wait(), timeout=self.watchdog_interval_s)

    # ------------------------------------------------------------------ consumer loop
    async def _consume(self, stop_event: asyncio.Event) -> None:
        attempt = 0
        while not stop_event.is_set():
            try:
                stream = self.gateway.order_events()
                iterator = stream.__aiter__()
                self.consumer_ready.set()
                while not stop_event.is_set():
                    nxt: asyncio.Future[OrderEvent] = asyncio.ensure_future(iterator.__anext__())
                    stopper = asyncio.ensure_future(stop_event.wait())
                    done, _ = await asyncio.wait({nxt, stopper}, return_when=asyncio.FIRST_COMPLETED)
                    if nxt not in done:
                        nxt.cancel()
                        with contextlib.suppress(asyncio.CancelledError, StopAsyncIteration, Exception):
                            await nxt
                        return
                    stopper.cancel()
                    ev = nxt.result()  # may raise StopAsyncIteration / disconnect
                    attempt = 0
                    self.handle_event(ev)
            except asyncio.CancelledError:
                raise
            except StopAsyncIteration:
                reason = "order-update stream ended"
            except Exception as exc:
                reason = f"order-update stream error: {type(exc).__name__}: {exc}"
            else:
                return
            self.consumer_ready.clear()
            self.disconnects += 1
            with contextlib.suppress(Exception):
                self.entries.set_flag(Flag.BROKER_DISCONNECTED, reason, "execution")
            delay = self.reconnect_backoff_s[min(attempt, len(self.reconnect_backoff_s) - 1)]
            attempt += 1
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(stop_event.wait(), timeout=delay)
            if not stop_event.is_set() and self.on_reconnect is not None:
                try:
                    await self.on_reconnect()
                except Exception as exc:
                    self._failure("on_reconnect_failed", f"{type(exc).__name__}: {exc}")
