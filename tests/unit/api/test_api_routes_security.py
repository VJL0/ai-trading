"""API route + guard tests with a stub runtime (no services): CSRF/origin/operator/engine guards,
loopback-only binding, read queries over an empty and a populated ledger, SSE first frame."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from packages.contracts.enums import ExecutionLabel
from packages.core.audit import add_halt, append_audit
from packages.core.config import Settings
from packages.core.db import Database
from services.api.app import create_app
from services.api.events import EventBus
from services.api.security import AGENT_TOKEN_FILE, ENGINE_TOKEN_FILE, OPERATOR_TOKEN_FILE, ensure_token, read_token


class StubRuntime:
    def __init__(self, settings: Settings) -> None:
        settings.ensure_dirs()
        ensure_token(settings.keys_dir, OPERATOR_TOKEN_FILE)
        ensure_token(settings.keys_dir, ENGINE_TOKEN_FILE)
        ensure_token(settings.keys_dir, AGENT_TOKEN_FILE)
        self.settings = settings
        self.db = Database(settings.db_path)
        self.bus = EventBus()
        self.execution_label = ExecutionLabel.SIMULATED
        self.calls: list[tuple[str, Any]] = []

    def now_iso(self) -> str:
        return "2026-11-05T15:00:00Z"

    def status(self) -> dict[str, Any]:
        return {"mode": "PAPER", "execution_label": "SIMULATED", "entries_locked": True}

    def health_report(self) -> dict[str, Any]:
        return {"overall": "DEGRADED", "components": [
            {"name": name, "status": "DEGRADED", "detail": "test runtime"}
            for name in ("database", "scheduler", "broker")
        ]}

    def market_bars(self, symbol: str, timeframe: str, limit: int) -> dict[str, Any]:
        return {"symbol": symbol, "bars": [], "is_simulated": True}

    def market_quality(self) -> dict[str, Any]:
        return {"feed_mode": "SIMULATED", "events": []}

    def risk_view(self) -> dict[str, Any]:
        return {"policy": {}, "utilization": []}

    def halt(self, reason: str, source: str) -> str:
        self.calls.append(("halt", reason))
        return self.db.write(lambda c: add_halt(c, reason=reason, source=source, mode="PAPER"))

    def resume(self, halt_id: str, reason: str, actor: str) -> dict[str, Any]:
        self.calls.append(("resume", halt_id))
        return {"cleared": True}

    def bridge_next(self, engine_id: str, max_n: int) -> list[dict[str, Any]]:
        self.calls.append(("next", engine_id))
        return []

    def bridge_event(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(("event", payload))
        return {"accepted": True, "state": "SUBMITTED", "duplicate": False}

    def bridge_heartbeat(self, payload: dict[str, Any]) -> None:
        self.calls.append(("hb", payload))

    def bridge_rejection(self, payload: dict[str, Any]) -> None:
        self.calls.append(("rej", payload))

    async def run_cycle(self, *, symbols: list[str] | None, use_ai: bool, actor: str) -> dict[str, Any]:
        return {"outcome": "NO_TRADE"}

    async def start(self, stop: asyncio.Event) -> list[asyncio.Task[Any]]:
        return []

    async def aclose(self) -> None:
        self.db.close()


@pytest.fixture
def client(tmp_path: Path) -> Any:
    settings = Settings(project_root=tmp_path)
    holder: dict[str, StubRuntime] = {}

    def factory(s: Settings) -> StubRuntime:
        holder["rt"] = StubRuntime(s)
        return holder["rt"]

    app = create_app(settings, runtime_factory=factory)  # type: ignore[arg-type]
    with TestClient(app, base_url="http://127.0.0.1:8765") as c:
        c.rt = holder["rt"]  # type: ignore[attr-defined]
        c.op = read_token(settings.keys_dir, OPERATOR_TOKEN_FILE)  # type: ignore[attr-defined]
        c.eng = read_token(settings.keys_dir, ENGINE_TOKEN_FILE)  # type: ignore[attr-defined]
        yield c


def test_health_and_empty_reads(client: Any) -> None:
    health = client.get("/api/v1/health").json()
    assert health["execution_label"] == "SIMULATED"
    assert {c["name"] for c in health["components"]} >= {"database", "scheduler", "broker"}
    assert client.get("/api/v1/health/live").json()["status"] == "ok"
    routes = [r.path for r in client.app.routes if getattr(r, "path", None) == "/api/v1/health"]
    assert len(routes) == 1
    for path in ("/api/v1/orders", "/api/v1/proposals", "/api/v1/signals", "/api/v1/research/runs",
                 "/api/v1/reconciliation", "/api/v1/failures", "/api/v1/portfolio", "/api/v1/backtests"):
        r = client.get(path)
        assert r.status_code == 200, path
    assert client.get("/api/v1/orders/nope").status_code == 404
    assert client.get("/api/v1/audit").json()["chain"]["ok"] is True


def test_halt_requires_csrf_header_and_local_origin(client: Any) -> None:
    assert client.post("/api/v1/control/halt", json={"reason": "x"}).status_code == 403
    r = client.post("/api/v1/control/halt", json={"reason": "x"},
                    headers={"X-LAT-Request": "1", "Origin": "https://evil.example"})
    assert r.status_code == 403
    r = client.post("/api/v1/control/halt", json={"reason": "manual stop"},
                    headers={"X-LAT-Request": "1", "Origin": "http://127.0.0.1:5173"})
    assert r.status_code == 200 and r.json()["entries_locked"] is True
    assert client.rt.calls[-1] == ("halt", "manual stop")


def test_operator_endpoints_need_token(client: Any) -> None:
    h = {"X-LAT-Request": "1"}
    assert client.post("/api/v1/control/resume", json={"halt_id": "h", "reason": "r"}, headers=h).status_code == 401
    bad = {**h, "X-LAT-Operator-Token": "wrong"}
    assert client.post("/api/v1/pipeline/run-cycle", json={}, headers=bad).status_code == 401
    ok = {**h, "X-LAT-Operator-Token": client.op}
    assert client.post("/api/v1/pipeline/run-cycle", json={}, headers=ok).json()["outcome"] == "NO_TRADE"
    # operator token without CSRF header is still refused
    assert client.post("/api/v1/control/resume", json={"halt_id": "h", "reason": "r"},
                       headers={"X-LAT-Operator-Token": client.op}).status_code == 403


def test_engine_bridge_needs_engine_token_and_keeps_decimals(client: Any) -> None:
    assert client.get("/api/v1/engine/intents/next?engine_id=e1").status_code == 401
    assert client.get("/api/v1/engine/intents/next?engine_id=e1",
                      headers={"X-LAT-Engine-Token": client.op}).status_code == 401  # operator token != engine token
    r = client.get("/api/v1/engine/intents/next?engine_id=e1", headers={"X-LAT-Engine-Token": client.eng})
    assert r.json() == {"intents": []}
    body = b'{"client_order_id":"c","fill_price":101.10,"fill_quantity":"2"}'
    r = client.post("/api/v1/engine/events", content=body, headers={"X-LAT-Engine-Token": client.eng})
    assert r.json()["accepted"] is True
    payload = client.rt.calls[-1][1]
    assert str(payload["fill_price"]) == "101.10"  # parsed as Decimal, not float


def test_populated_audit_and_halt_visible(client: Any) -> None:
    client.rt.db.write(lambda c: append_audit(c, event_type="X", source="t", mode="PAPER", payload={"a": 1}))
    ev = client.get("/api/v1/audit?limit=5").json()
    assert ev["events"][0]["event_type"] == "X" and ev["chain"]["rows"] == 1


def test_backtests_reads_real_summary_files(tmp_path: Path) -> None:
    from services.api.queries import backtests

    d = tmp_path / "experiments" / "lean" / "r1"
    d.mkdir(parents=True)
    (d / "summary.json").write_text(json.dumps({"run_id": "r1", "engine": "LEAN", "started_at": "2026-11-05T00:00:00Z"}))
    (tmp_path / "experiments" / "junk").mkdir()
    (tmp_path / "experiments" / "junk" / "summary.json").write_text("not json")
    runs = backtests(tmp_path / "experiments")["runs"]
    assert [r["run_id"] for r in runs] == ["r1"]


def test_sse_stream_first_frame_is_status(tmp_path: Path) -> None:
    """Real uvicorn server on an ephemeral loopback port (ASGITransport buffers streams)."""
    import socket
    import threading
    import time

    import httpx
    import uvicorn

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    settings = Settings(project_root=tmp_path, api_port=port)
    app = create_app(settings, runtime_factory=StubRuntime)  # type: ignore[arg-type]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    try:
        for _ in range(100):
            if server.started:
                break
            time.sleep(0.05)
        first = ""
        with httpx.stream("GET", f"http://127.0.0.1:{port}/api/v1/stream", timeout=5) as resp:
            assert resp.headers["content-type"].startswith("text/event-stream")
            for line in resp.iter_lines():
                if line.startswith("event:"):
                    first = line
                    break
        assert first == "event: status"
    finally:
        server.should_exit = True
        t.join(timeout=10)


def test_loopback_only_setting() -> None:
    with pytest.raises(ValueError):
        Settings(api_host="192.168.1.10")
    assert Settings(api_host="localhost").api_host == "127.0.0.1"




def test_agent_tool_routes_need_agent_token(client: Any) -> None:
    assert client.get("/api/v1/agent/tools").status_code == 401
    for wrong in (client.op, client.eng):  # operator/engine tokens never work as agent tokens
        assert client.get("/api/v1/agent/tools", headers={"X-LAT-Agent-Token": wrong}).status_code == 401
    assert client.post("/api/v1/agent/tools/request_halt", json={"reason": "x"}).status_code == 401
