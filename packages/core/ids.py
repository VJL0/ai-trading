"""Time-ordered unique identifiers (ULID-style) with readable prefixes.

``causal_id`` ties every artifact of one decision cycle together (data snapshot → signal →
model runs → proposal → risk decision → intent → orders → fills → audit events).
"""

from __future__ import annotations

import os
import threading
import time

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_lock = threading.Lock()
_last_ms = 0
_last_rand = 0


def _encode(value: int, length: int) -> str:
    out: list[str] = []
    for _ in range(length):
        out.append(_CROCKFORD[value & 31])
        value >>= 5
    return "".join(reversed(out))


def ulid() -> str:
    """26-char Crockford base32 ULID; strictly monotonic within the process (monotonic-ULID rule)."""
    global _last_ms, _last_rand
    with _lock:
        ms = int(time.time() * 1000)
        if ms <= _last_ms:
            ms = _last_ms
            rand = _last_rand + 1
            if rand >= 1 << 80:  # overflow within one millisecond: borrow the next millisecond
                ms += 1
                rand = int.from_bytes(os.urandom(10), "big") >> 1
        else:
            rand = int.from_bytes(os.urandom(10), "big") >> 1  # headroom for increments
        _last_ms, _last_rand = ms, rand
    return _encode(ms, 10) + _encode(rand, 16)


def new_id(prefix: str) -> str:
    if not prefix.isidentifier():
        raise ValueError(f"bad id prefix {prefix!r}")
    return f"{prefix}_{ulid()}"


def new_causal_id() -> str:
    return new_id("cz")


def client_order_id(mode_tag: str) -> str:
    """Broker client order id (Alpaca limit: 128 chars). Mode tag is embedded for isolation checks."""
    tag = mode_tag.lower()
    if tag not in {"sim", "paper", "live", "replay"}:
        raise ValueError(f"unknown mode tag {mode_tag!r}")
    return f"lat-{tag}-{ulid().lower()}"
