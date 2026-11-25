"""Repeatable real-process MCP -> API -> session -> ledger smoke, isolated and SIMULATED only.

Run: uv run --frozen python -m scripts.smoke_paper
No model calls, credentials, external network, existing runtime state or broker connections.
"""
from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import httpx
from mcp import ClientSession, StdioServerParameters, stdio_client

from packages.core.audit import verify_chain
from packages.core.db import Database

ROOT = Path(__file__).resolve().parents[1]


async def smoke(root: Path) -> dict[str, Any]:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = {k: v for k, v in os.environ.items() if not k.startswith("LAT_")}
    env.update(LAT_PROJECT_ROOT=str(root), LAT_API_PORT=str(port), LAT_MODE="PAPER", LAT_GATEWAY="FAKE",
               LAT_FEED_MODE="SIMULATED", LAT_AI_RUNTIME="NONE", LAT_AI_TRANSPORT="fake", LAT_USE_KEYCHAIN="false",
               LAT_AGENT_NAME="smoke", LAT_RECONCILE_INTERVAL_S="2")
    params = StdioServerParameters(command=sys.executable, args=["-m", "services.mcp_server"], env=env, cwd=str(ROOT))
    with (root / "api.log").open("w") as log, (root / "mcp.log").open("w") as errlog:
        proc = await asyncio.to_thread(subprocess.Popen, [sys.executable, "-m", "services.api"], cwd=ROOT, env=env,
                                stdout=log, stderr=log, stdin=subprocess.DEVNULL)
        try:
            async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=5) as http:
                for _ in range(60):
                    try:
                        status = (await http.get("/api/v1/status")).json()
                        if status["startup"]["phase"] == "READY":
                            break
                    except (httpx.HTTPError, KeyError):
                        pass
                    if proc.poll() is not None:
                        raise RuntimeError(f"API exited; inspect {root / 'api.log'}")
                    await asyncio.sleep(0.5)
                else:
                    raise RuntimeError("API never became ready")
                assert status["execution_label"] == "SIMULATED" and not status["live_execution_compiled"]
                monitor = (await http.get("/monitor/status")).raise_for_status().json()
                assert monitor["mode"] == "Paper" and monitor["equity"] is not None
                assert (await http.get("/monitor")).status_code == 200

                async def call(client: ClientSession, name: str, args: dict[str, Any]) -> dict[str, Any]:
                    result = await client.call_tool(name, args)
                    assert not result.is_error, result.content
                    assert result.structured_content is not None
                    return result.structured_content

                async with (stdio_client(params, errlog=errlog) as (read, write),
                            ClientSession(read, write, read_timeout_seconds=30) as client):
                    await client.initialize()
                    started = await call(client, "start_paper_session", {
                        "duration_minutes": 3, "cadence_minutes": 1, "symbols": ["SPY"], "use_ai": False,
                    })
                    sid = started["session"]["session_id"]
                # The operator disconnects. The API process must complete the session independently.
                report: dict[str, Any] = {}
                for _ in range(40):
                    report = (await http.get(f"/api/v1/sessions/{sid}/report")).json()
                    if report["session"]["state"] == "COMPLETED":
                        break
                    await asyncio.sleep(0.5)
                else:
                    raise AssertionError(f"session failed to complete: {report}")
                async with (stdio_client(params, errlog=errlog) as (read, write),
                            ClientSession(read, write, read_timeout_seconds=30) as client):
                    await client.initialize()
                    result = await call(client, "get_session_report", {"session_id": sid})
                    assert result["cycles"] >= 1 and result["api_cost_usd"] == "0"
                    assert result["session"]["state"] == "COMPLETED"
                    health = await call(client, "get_system_health", {})
                    assert next(c for c in health["components"] if c["name"] == "scheduler")["status"] == "OK"
                    activity = await call(client, "get_activity", {"limit": 20})
                    assert activity["events"]
                    # Read-only stream endpoint must emit actual SSE data.
                    async with http.stream("GET", "/api/v1/stream") as stream:
                        async for line in stream.aiter_lines():
                            if line.startswith("data:"):
                                break
                    result["transport"] = "MCP stdio -> real loopback HTTP API"
                    result["operator_disconnected_during_session"] = True
        finally:
            proc.terminate()
            try:
                await asyncio.to_thread(proc.wait, 10)
            except subprocess.TimeoutExpired:
                proc.kill()
                await asyncio.to_thread(proc.wait)
    db = Database(root / "var/paper/ledger.sqlite")
    try:
        ok, rows, detail = verify_chain(db)
        assert ok, detail
        result["audit_verified_rows"] = rows
    finally:
        db.close()
    return result


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="lat-paper-smoke-") as tmp:
        print(json.dumps(asyncio.run(smoke(Path(tmp))), indent=2))


if __name__ == "__main__":
    main()
