"""`run --once` against a real API process in a temporary SIMULATED workspace (built like `make smoke`).

Real path: autopilot -> HttpToolBackend (agent token from the temp workspace, actor `autopilot`) -> loopback API ->
session manager. The autopilot's own clock is set to a pre-open instant; the API runs its own simulated clock.
No credentials, broker, model or network beyond 127.0.0.1. Everything is SIMULATED (fake broker, synthetic data).
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from services.autopilot import cli
from tests.unit.autopilot.fakes import FakeClock, et

REPO = Path(__file__).resolve().parents[3]


def _env(root: Path, port: int) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("LAT_")}
    env.update(LAT_PROJECT_ROOT=str(root), LAT_API_PORT=str(port), LAT_MODE="PAPER", LAT_GATEWAY="FAKE",
               LAT_FEED_MODE="SIMULATED", LAT_AI_RUNTIME="NONE", LAT_AI_TRANSPORT="fake", LAT_USE_KEYCHAIN="false",
               LAT_RECONCILE_INTERVAL_S="2")
    return env


@pytest.fixture
def api(tmp_path: Path) -> Iterator[tuple[Path, int]]:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with (tmp_path / "api.log").open("w") as log:
        proc = subprocess.Popen([sys.executable, "-m", "services.api"], cwd=REPO, env=_env(tmp_path, port),
                                stdout=log, stderr=log, stdin=subprocess.DEVNULL)
        try:
            deadline = time.monotonic() + 60
            while True:
                try:
                    status = httpx.get(f"http://127.0.0.1:{port}/api/v1/status", timeout=2).json()
                    if status["startup"]["phase"] == "READY":
                        break
                except (httpx.HTTPError, KeyError, ValueError):
                    pass
                if proc.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError(f"API not ready: {(tmp_path / 'api.log').read_text()[-2000:]}")
                time.sleep(0.5)
            assert status["execution_label"] == "SIMULATED" and not status["live_execution_compiled"]
            yield tmp_path, port
        finally:
            proc.terminate()
            try:
                proc.wait(10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


def test_run_once_ends_with_a_paper_session_and_a_report(api: tuple[Path, int], monkeypatch: pytest.MonkeyPatch,
                                                         capsys: pytest.CaptureFixture[str]) -> None:
    root, port = api
    for k in list(os.environ):
        if k.startswith("LAT_"):
            monkeypatch.delenv(k)
    for k, v in _env(root, port).items():
        if k.startswith("LAT_"):
            monkeypatch.setenv(k, v)
    monkeypatch.setenv("LAT_AUTOPILOT_ENABLED", "true")
    # The lead agent runs as a real child process, with simulated providers (no CLI, no subscription quota).
    monkeypatch.setenv("LAT_AUTOPILOT_LEAD_FAKE", "true")
    clock = FakeClock(et(2026, 11, 30, 8, 45))  # Monday pre-open (autopilot clock; the API keeps its own)

    assert cli.main(["run", "--once"], clock=clock) == 0
    capsys.readouterr()

    sessions = httpx.get(f"http://127.0.0.1:{port}/api/v1/sessions", timeout=5)
    state = root / "experiments" / "autopilot"
    assert cli.main(["status", "--json"], clock=clock) == 0
    st = json.loads(capsys.readouterr().out)
    pre = next(j for j in st["jobs"] if j["job"] == "preopen")
    assert pre["status"] == "DONE", st
    rep = json.loads((state / "reports" / "2026-11-30.json").read_text())
    assert rep["session"]["outcome"] in ("STARTED", "SCHEDULED")
    assert rep["session"]["session_id"]
    assert rep["simulated"] is True and rep["data_provenance"]["execution_labels"] == ["SIMULATED"]
    assert (state / "reports" / "2026-11-30.md").read_text().count("SIMULATED") >= 1
    # The session was requested by the deterministic pre-open step before the lead child was even launched.
    from services.autopilot.store import Store as _Store

    events = _Store(state / "autopilot.sqlite3", readonly=True).events(kind="ACTION")
    order = [e["job"] for e in events]
    assert order.index("preopen") < order.index("lead")
    lead = next(j for j in st["jobs"] if j["job"] == "lead")
    assert lead["status"] == "PENDING"
    lead_rows = [e for e in _Store(state / "autopilot.sqlite3", readonly=True).events(kind="RESULT")
                 if e["job"] == "lead"]
    assert lead_rows and lead_rows[-1]["data"]["lead_outcome"] == "OK", lead_rows
    assert lead_rows[-1]["data"]["delegated"] and lead_rows[-1]["data"]["ran"]
    assert (root / "experiments" / "agents" / "crew-fake.sqlite3").is_file()  # demo memory, never the real crew store
    assert not (root / "experiments" / "agents" / "crew.sqlite3").exists()
    briefing = json.loads((state / "lead" / "briefing.json").read_text())
    assert briefing["provenance"]["is_simulated"] is True and "token" not in json.dumps(briefing).lower()
    if sessions.status_code == 200:  # cross-check on the API side when the read endpoint exists
        body = json.dumps(sessions.json())
        assert rep["session"]["session_id"] in body and "agent:autopilot" in body

    # A second tick is idempotent: no duplicate session request.
    clock.advance(minutes=10)
    assert cli.main(["run", "--once"], clock=clock) == 0
    capsys.readouterr()
    from services.autopilot.store import Store

    store = Store(state / "autopilot.sqlite3", readonly=True)
    actions = [e for e in store.events(kind="ACTION") if e["job"] == "preopen"]
    assert len(actions) == 1
    assert len([e for e in store.events(kind="ACTION") if e["job"] == "lead"]) == 1  # min interval not reached
