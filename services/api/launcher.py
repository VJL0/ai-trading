"""Start the configured local API once, without changing its trading authority."""

from __future__ import annotations

import fcntl
import hashlib
import os
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx

from packages.contracts.enums import GatewayKind
from packages.core.config import PROJECT_ROOT, Settings
from packages.core.modes import LIVE_EXECUTION_COMPILED, OperatorTradingMode


class ApiStartupError(RuntimeError):
    pass


@contextmanager
def api_process_lock(run_dir: Path) -> Iterator[None]:
    """Hold the workspace API lock for the lifetime of the server."""
    run_dir.mkdir(parents=True, exist_ok=True)
    if api_pid(run_dir) is not None:
        raise ApiStartupError("another API process already owns this workspace")
    with (run_dir / "api-process.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ApiStartupError("another API process already owns this workspace") from None
        pidfile = run_dir / "api.pid"
        pidfile.write_text(str(os.getpid()))
        try:
            yield
        finally:
            pidfile.unlink(missing_ok=True)


def workspace_id(settings: Settings) -> str:
    return hashlib.sha256(os.fsencode(settings.project_root.resolve())).hexdigest()


def api_pid(run_dir: Path) -> int | None:
    """Find the locked workspace API, or a verified process from the previous launcher."""
    try:
        pid = int((run_dir / "api.pid").read_text())
    except (OSError, ValueError):
        return None
    if pid <= 1:
        return None
    try:
        with (run_dir / "api-process.lock").open("rb") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return pid
    except OSError:
        pass  # A process started before the upgrade may have no lock file.
    try:
        if os.getpgid(pid) != pid:
            return None
        command = subprocess.run(["ps", "-p", str(pid), "-o", "command="], capture_output=True,
                                 text=True, timeout=2, check=False).stdout.split()
    except (OSError, subprocess.TimeoutExpired):
        return None
    if command and Path(command[0]).name == "uv" and command[1:] == ["run", "--frozen", "python", "-m", "services.api"]:
        return pid
    return None


def _health(client: httpx.Client) -> dict[str, Any] | None:
    try:
        response = client.get("/api/v1/health/live")
    except httpx.ConnectError:
        return None
    except httpx.HTTPError as exc:
        raise ApiStartupError(f"local API health check failed ({type(exc).__name__})") from None
    try:
        data = response.json()
    except ValueError:
        data = None
    if response.status_code != 200 or not isinstance(data, dict) or data.get("service") != "local-ai-trader":
        raise ApiStartupError("the configured API address is in use by another service or an older API process")
    return data


def ensure_api(settings: Settings, *, require_gateway_match: bool = True) -> dict[str, Any]:
    """Start or reuse the API. Only this caller's fixed runtime settings can select a gateway."""
    if settings.trading_mode == OperatorTradingMode.LIVE and not LIVE_EXECUTION_COMPILED:
        raise ApiStartupError("Live trading is unavailable in this build")
    settings.run_dir.mkdir(parents=True, exist_ok=True)
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    log_path = settings.log_dir / "api.log"
    expected_workspace = workspace_id(settings)
    with (settings.run_dir / "api-start.lock").open("a") as lock, httpx.Client(
        base_url=settings.api_url, timeout=2, trust_env=False, follow_redirects=False,
    ) as client:
        fcntl.flock(lock, fcntl.LOCK_EX)
        health = _health(client)
        process = None
        if health is None and api_pid(settings.run_dir) is None:
            with log_path.open("ab") as log:
                process = subprocess.Popen(
                    [sys.executable, "-m", "services.api"], cwd=PROJECT_ROOT, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                    env={**os.environ, "PYTHONUNBUFFERED": "1"},
                )

        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if health is not None:
                if health.get("workspace_id") != expected_workspace:
                    raise ApiStartupError("the running API belongs to a different workspace")
                if health.get("mode") != settings.mode.value:
                    raise ApiStartupError("the running API has a different operating mode")
                gateway = health.get("gateway")
                if require_gateway_match and gateway != settings.gateway.value:
                    raise ApiStartupError(f"the running API uses gateway {gateway}; stop it before switching")
                if not require_gateway_match and (gateway not in (GatewayKind.FAKE.value, GatewayKind.ALPACA_PAPER.value)
                                                  or health.get("execution_label") not in ("PAPER", "SIMULATED")):
                    raise ApiStartupError("the running API is outside the supported paper execution paths")
                if health.get("startup_phase") in ("READY", "LOCKED"):
                    return health
            if process is not None and process.poll() is not None:
                raise ApiStartupError(f"the API exited during startup. See {log_path}")
            time.sleep(0.25)
            health = _health(client)
        raise ApiStartupError(f"the API has not finished startup reconciliation. See {log_path}")
