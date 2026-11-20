"""Per-agent-session memory kept by the tool handlers (in the API process, next to the runtime).

* ``IssuedIds`` remembers every citable id a tool output handed to an agent session (signal, prediction,
  snapshot, bars source, model registry version), with its symbol. ``propose_paper_trade`` accepts only
  evidence ids that this session actually obtained (or persisted signal ids), so an agent cannot invent
  citations. Bounded (LRU) and time-limited (wall clock).
* ``WindowLimiter`` is a wall-clock sliding-window limiter for agent-initiated quant cycles (the per-hour
  proposal limit itself is enforced by the runtime from persisted proposals).

State is keyed by the runtime object and the sanitized actor; it lives only in memory and is lost on restart
(after which agents simply call the read tools again).
"""

from __future__ import annotations

import threading
import time
import weakref
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Any, Final

ISSUED_ID_TTL_S: Final = 6 * 3600.0
MAX_IDS_PER_SESSION: Final = 2000
MAX_SESSIONS: Final = 64


@dataclass(frozen=True)
class IssuedId:
    kind: str
    symbol: str | None
    issued_at: float


class IssuedIds:
    def __init__(self, ttl_s: float = ISSUED_ID_TTL_S, max_ids: int = MAX_IDS_PER_SESSION) -> None:
        self._ttl = ttl_s
        self._max = max_ids
        self._ids: OrderedDict[str, IssuedId] = OrderedDict()
        self._lock = threading.Lock()

    def add(self, record_id: str | None, kind: str, symbol: str | None = None) -> None:
        if not record_id:
            return
        with self._lock:
            self._ids.pop(record_id, None)
            self._ids[record_id] = IssuedId(kind, symbol, time.monotonic())
            while len(self._ids) > self._max:
                self._ids.popitem(last=False)

    def get(self, record_id: str) -> IssuedId | None:
        with self._lock:
            item = self._ids.get(record_id)
            if item is None:
                return None
            if time.monotonic() - item.issued_at > self._ttl:
                self._ids.pop(record_id, None)
                return None
            return item

    def __len__(self) -> int:
        with self._lock:
            return len(self._ids)


class WindowLimiter:
    def __init__(self, window_s: float = 3600.0) -> None:
        self._window = window_s
        self._events: deque[float] = deque()
        self._lock = threading.Lock()

    def try_acquire(self, limit: int) -> bool:
        now = time.monotonic()
        with self._lock:
            while self._events and now - self._events[0] > self._window:
                self._events.popleft()
            if limit <= 0 or len(self._events) >= limit:
                return False
            self._events.append(now)
            return True


class _RuntimeState:
    def __init__(self) -> None:
        self.sessions: OrderedDict[str, IssuedIds] = OrderedDict()
        self.cycle_limiter = WindowLimiter()
        self.lock = threading.Lock()

    def ids_for(self, actor: str) -> IssuedIds:
        with self.lock:
            ids = self.sessions.pop(actor, None) or IssuedIds()
            self.sessions[actor] = ids
            while len(self.sessions) > MAX_SESSIONS:
                self.sessions.popitem(last=False)
            return ids


_states: weakref.WeakKeyDictionary[Any, _RuntimeState] = weakref.WeakKeyDictionary()
_fallback: dict[int, _RuntimeState] = {}
_states_lock = threading.Lock()


def state_for(runtime: Any) -> _RuntimeState:
    with _states_lock:
        try:
            st = _states.get(runtime)
            if st is None:
                st = _RuntimeState()
                _states[runtime] = st
            return st
        except TypeError:  # runtime not weak-referenceable
            return _fallback.setdefault(id(runtime), _RuntimeState())
