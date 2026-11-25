"""In-process pub/sub for Server-Sent Events, fed by a DB tailer so changes made by any writer
(API, CLI decision cycles, reconciliation) reach structured stream subscribers."""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator
from typing import Any

from packages.core.db import Database


class EventBus:
    def __init__(self, max_queue: int = 1000) -> None:
        self._subs: set[asyncio.Queue[tuple[str, Any]]] = set()
        self._max = max_queue

    def publish(self, event: str, data: Any) -> None:
        for q in list(self._subs):
            if q.full():
                with contextlib.suppress(asyncio.QueueEmpty):
                    q.get_nowait()  # drop oldest for slow consumers; status is re-sent periodically
            q.put_nowait((event, data))

    async def subscribe(self) -> AsyncIterator[tuple[str, Any]]:
        q: asyncio.Queue[tuple[str, Any]] = asyncio.Queue(self._max)
        self._subs.add(q)
        try:
            while True:
                yield await q.get()
        finally:
            self._subs.discard(q)

    @property
    def subscriber_count(self) -> int:
        return len(self._subs)


def sse_format(event: str, data: Any) -> str:
    payload = json.dumps(data, default=str, separators=(",", ":"))
    return f"event: {event}\ndata: {payload}\n\n"


class DbTailer:
    """Polls append-only tables and publishes new rows (audit, order events) plus changed orders."""

    def __init__(self, db: Database, bus: EventBus) -> None:
        self._db = db
        self._bus = bus
        self._audit_seq = self._max("audit_events")
        self._event_seq = self._max("order_events")

    def _max(self, table: str) -> int:
        row = self._db.query_one(f"SELECT COALESCE(MAX(seq), 0) AS m FROM {table}")  # noqa: S608 - fixed names
        return int(row["m"]) if row else 0

    def poll_once(self) -> int:
        published = 0
        for r in self._db.query("SELECT * FROM audit_events WHERE seq > ? ORDER BY seq LIMIT 500", (self._audit_seq,)):
            self._audit_seq = r["seq"]
            r["payload"] = json.loads(r.pop("payload_json"))
            self._bus.publish("audit", r)
            published += 1
        rows = self._db.query("SELECT * FROM order_events WHERE seq > ? ORDER BY seq LIMIT 500", (self._event_seq,))
        touched: set[str] = set()
        for r in rows:
            self._event_seq = r["seq"]
            r["payload"] = json.loads(r.pop("payload_json"))
            self._bus.publish("order_event", r)
            touched.add(r["client_order_id"])
            published += 1
        for coid in touched:
            order = self._db.query_one("SELECT * FROM orders WHERE client_order_id=?", (coid,))
            if order:
                order.pop("intent_json", None)
                self._bus.publish("order", order)
        return published

    async def run(self, stop: asyncio.Event, interval_s: float = 1.0) -> None:
        while not stop.is_set():
            try:
                await asyncio.to_thread(self.poll_once)
            except Exception as exc:  # keep tailing; surface the failure on the stream
                self._bus.publish("failure", {"component": "db_tailer", "detail": str(exc)})
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=interval_s)
