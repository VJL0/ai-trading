"""The 127.0.0.1 stub of the engine-bridge contract used by the SIMULATED LEAN tests behaves like the contract says."""

from __future__ import annotations

import http.client
import json
import secrets
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit

import pytest

from packages.contracts.enums import Side
from packages.core.signing import IntentSigner
from tests.integration.lean.live_sim_harness import make_intent
from tests.integration.lean.stub_bridge import EVENTS, HEARTBEAT, INTENTS, REJECTIONS, StubBridge


def _req(url: str, token: str, body: Any = None, raw: bytes | None = None) -> tuple[int, dict[str, Any]]:
    """Plain HTTP to the loopback stub (``url`` = http://127.0.0.1:<port><path>)."""
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    parsed = urlsplit(url)
    assert parsed.hostname == "127.0.0.1" and parsed.port
    conn = http.client.HTTPConnection("127.0.0.1", parsed.port, timeout=5)
    try:
        path = parsed.path + (f"?{parsed.query}" if parsed.query else "")
        conn.request("POST" if data is not None else "GET", path, body=data,
                     headers={"X-LAT-Engine-Token": token, "Content-Type": "application/json"})
        resp = conn.getresponse()
        return resp.status, json.loads(resp.read())
    finally:
        conn.close()


@pytest.fixture
def stub() -> Any:
    token = secrets.token_hex(16)
    with StubBridge(token) as s:
        s.token = token  # type: ignore[attr-defined]
        yield s


def test_binds_loopback_only(stub: StubBridge) -> None:
    assert stub.url.startswith("http://127.0.0.1:")


def test_every_route_requires_the_engine_token(stub: StubBridge) -> None:
    for path in (INTENTS + "?engine_id=e&max=1",):
        assert _req(stub.url + path, "wrong")[0] == 401
    for path in (EVENTS, HEARTBEAT, REJECTIONS):
        assert _req(stub.url + path, "wrong", {"engine_id": "e"})[0] == 401
    assert stub.snapshot()["auth_failures"] == 4


def test_intents_are_claimed_atomically_and_at_most_once(stub: StubBridge) -> None:
    signer = IntentSigner.generate()
    a = make_intent(signer, "unit", 1, side=Side.BUY, qty="1", limit="10.00")
    b = make_intent(signer, "unit", 2, side=Side.BUY, qty="2", limit="10.00", created=datetime.now(UTC))
    stub.add_intent(a)
    stub.add_intent(b)
    tok = stub.token  # type: ignore[attr-defined]
    code, first = _req(stub.url + INTENTS + "?engine_id=e1&max=1", tok)
    assert code == 200 and [i["client_order_id"] for i in first["intents"]] == [a.client_order_id]
    assert first["intents"][0]["signing_string"] == a.signing_string()
    assert signer.verifier.verify(first["intents"][0]["signing_string"], first["intents"][0]["signature"])
    _, second = _req(stub.url + INTENTS + "?engine_id=e2&max=5", tok)
    assert [i["client_order_id"] for i in second["intents"]] == [b.client_order_id]
    assert _req(stub.url + INTENTS + "?engine_id=e1&max=5", tok)[1] == {"intents": []}
    claimed = {i["client_order_id"]: i["claimed_by"] for i in stub.snapshot()["intents"]}
    assert claimed == {a.client_order_id: "e1", b.client_order_id: "e2"}


def test_float_money_is_rejected_and_duplicates_flagged(stub: StubBridge) -> None:
    tok = stub.token  # type: ignore[attr-defined]
    ev = {"engine_id": "e", "client_order_id": "c", "lean_order_id": 1, "lean_event_id": 2, "status": "Filled",
          "fill_quantity": "1", "fill_price": "10.01"}
    assert _req(stub.url + EVENTS, tok, ev) == (200, {"accepted": True, "state": "Filled", "duplicate": False})
    assert _req(stub.url + EVENTS, tok, ev)[1]["duplicate"] is True
    code, _ = _req(stub.url + EVENTS, tok, raw=b'{"engine_id":"e","fill_price":10.01}')
    assert code == 422
    assert stub.snapshot()["protocol_errors"]


def test_injected_post_failures_are_transient(stub: StubBridge) -> None:
    tok = stub.token  # type: ignore[attr-defined]
    stub.inject_post_failures(REJECTIONS, 2)
    body = {"engine_id": "e", "client_order_id": None, "intent_id": None, "reason": "r"}
    assert [_req(stub.url + REJECTIONS, tok, body)[0] for _ in range(3)] == [503, 503, 200]
    snap = stub.snapshot()
    assert len(snap["rejections"]) == 1 and snap["injected_failures"] == [REJECTIONS, REJECTIONS]
