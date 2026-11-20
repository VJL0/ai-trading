"""Minimal stand-in for the local API's LEAN engine bridge (docs/architecture/api-contract.md, "Engine bridge").

Used by the SIMULATED live-path tests and ``scripts/lean_live_sim.sh``. Bound to 127.0.0.1 only. It holds intents
signed by the REAL ``packages.core.signing.IntentSigner`` (or deliberately forged/tampered ones), hands each out at
most once (atomic claim), checks the ``X-LAT-Engine-Token`` header on every route, rejects JSON floats in engine
payloads, and records everything the engine sends for assertions. It is a test double, not the product API.
"""

from __future__ import annotations

import hmac
import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from packages.contracts.models import ApprovedOrderIntent

INTENTS = "/api/v1/engine/intents/next"
EVENTS = "/api/v1/engine/events"
HEARTBEAT = "/api/v1/engine/heartbeat"
REJECTIONS = "/api/v1/engine/rejections"


class FloatInPayloadError(ValueError):
    pass


def _no_float(value: str) -> Decimal:
    raise FloatInPayloadError(f"binary float {value!r} in engine payload (decimal strings required)")


@dataclass
class QueuedIntent:
    intent_id: str
    client_order_id: str
    signing_string: str
    signature: str
    key_id: str
    signature_alg: str = "ECDSA_P256_SHA256"
    label: str = ""
    claimed_by: str | None = None
    claimed_at: float | None = None

    def wire(self) -> dict[str, str]:
        return {"intent_id": self.intent_id, "client_order_id": self.client_order_id, "signing_string": self.signing_string,
                "signature": self.signature, "key_id": self.key_id, "signature_alg": self.signature_alg}


@dataclass
class StubState:
    token: str
    intents: list[QueuedIntent] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    heartbeats: list[dict[str, Any]] = field(default_factory=list)
    rejections: list[dict[str, Any]] = field(default_factory=list)
    auth_failures: int = 0
    protocol_errors: list[str] = field(default_factory=list)
    # fault injection: path -> number of upcoming POSTs to answer with HTTP 503 (nothing recorded for them)
    fail_posts: dict[str, int] = field(default_factory=dict)
    injected_failures: list[str] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)
    _seen_events: set[tuple[str, int, int]] = field(default_factory=set)


class _Handler(BaseHTTPRequestHandler):
    @property
    def state(self) -> StubState:
        srv = self.server
        assert isinstance(srv, _Server)
        return srv.state

    def log_message(self, format: str, *args: Any) -> None:  # stdlib signature
        return

    def _send(self, code: int, body: dict[str, Any]) -> None:
        raw = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _authorised(self) -> bool:
        presented = self.headers.get("X-LAT-Engine-Token") or ""
        ok = hmac.compare_digest(presented.encode(), self.state.token.encode())
        if not ok:
            with self.state.lock:
                self.state.auth_failures += 1
            self._send(401, {"detail": "engine token required"})
        return ok

    def do_GET(self) -> None:
        if not self._authorised():
            return
        url = urlparse(self.path)
        if url.path != INTENTS:
            self._send(404, {"detail": "not found"})
            return
        q = parse_qs(url.query)
        engine_id = (q.get("engine_id") or [""])[0]
        n = max(1, min(int((q.get("max") or ["1"])[0]), 10))
        st = self.state
        out: list[dict[str, str]] = []
        with st.lock:
            for it in st.intents:
                if len(out) >= n:
                    break
                if it.claimed_by is None:  # atomic claim: never handed out twice
                    it.claimed_by, it.claimed_at = engine_id, time.time()
                    out.append(it.wire())
        self._send(200, {"intents": out})

    def do_POST(self) -> None:
        if not self._authorised():
            return
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length)
        st = self.state
        path = urlparse(self.path).path
        with st.lock:
            remaining = st.fail_posts.get(path, 0)
            if remaining > 0:
                st.fail_posts[path] = remaining - 1
                st.injected_failures.append(path)
        if remaining > 0:
            self._send(503, {"detail": "injected failure"})
            return
        try:
            body = json.loads(raw, parse_float=_no_float)
            if not isinstance(body, dict):
                raise ValueError("payload must be an object")
        except (ValueError, FloatInPayloadError) as exc:
            with st.lock:
                st.protocol_errors.append(f"{self.path}: {exc}")
            self._send(422, {"detail": str(exc)})
            return
        body["_received_at"] = time.time()
        with st.lock:
            if path == EVENTS:
                key = (str(body.get("client_order_id")), int(body.get("lean_order_id", -1)), int(body.get("lean_event_id", -1)))
                duplicate = key in st._seen_events
                st._seen_events.add(key)
                st.events.append(body)
                resp: dict[str, Any] = {"accepted": True, "state": str(body.get("status")), "duplicate": duplicate}
            elif path == HEARTBEAT:
                st.heartbeats.append(body)
                resp = {"ok": True}
            elif path == REJECTIONS:
                st.rejections.append(body)
                resp = {"ok": True, "order_rejected": True}
            else:
                self._send(404, {"detail": "not found"})
                return
        self._send(200, resp)


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, state: StubState, port: int) -> None:
        super().__init__(("127.0.0.1", port), _Handler)
        self.state = state


class StubBridge:
    """``with StubBridge(token) as stub: stub.add_intent(...); ...; stub.url``"""

    def __init__(self, token: str, port: int = 0) -> None:
        if len(token) < 16:
            raise ValueError("engine token too short")
        self.state = StubState(token=token)
        self._server = _Server(self.state, port)
        host, real_port = self._server.server_address[:2]
        assert host == "127.0.0.1", "stub bridge must bind loopback only"
        self.port = int(real_port)
        self._thread = threading.Thread(target=self._server.serve_forever, name="stub-bridge", daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> StubBridge:
        self._thread.start()
        return self

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def __enter__(self) -> StubBridge:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def add_intent(self, intent: ApprovedOrderIntent, *, label: str = "", signing_string: str | None = None,
                   signature: str | None = None) -> QueuedIntent:
        q = QueuedIntent(intent_id=intent.intent_id, client_order_id=intent.client_order_id,
                         signing_string=signing_string if signing_string is not None else intent.signing_string(),
                         signature=signature if signature is not None else intent.signature,
                         key_id=intent.key_id, signature_alg=intent.signature_alg, label=label)
        with self.state.lock:
            self.state.intents.append(q)
        return q

    def snapshot(self) -> dict[str, Any]:
        with self.state.lock:
            return {
                "intents": [{**i.wire(), "label": i.label, "claimed_by": i.claimed_by} for i in self.state.intents],
                "events": list(self.state.events),
                "heartbeats": list(self.state.heartbeats),
                "rejections": list(self.state.rejections),
                "auth_failures": self.state.auth_failures,
                "protocol_errors": list(self.state.protocol_errors),
                "injected_failures": list(self.state.injected_failures),
            }

    def inject_post_failures(self, path: str, count: int) -> None:
        """Answer the next ``count`` POSTs to ``path`` with HTTP 503 (transient API outage)."""
        with self.state.lock:
            self.state.fail_posts[path] = count

    def wait_for(self, predicate: Callable[[dict[str, Any]], bool], timeout_s: float, poll_s: float = 0.25,
                 abort: Callable[[], bool] | None = None) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if predicate(self.snapshot()):
                return True
            if abort is not None and abort():
                return predicate(self.snapshot())
            time.sleep(poll_s)
        return predicate(self.snapshot())
