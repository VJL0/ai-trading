"""Minimal local process supervisor (spec §11: local macOS process supervisor / make scripts).

Commands:
  start [api|research|agents|lean-sim ...] start detached services (PIDs in var/run, logs in var/logs)
  stop  [names...]                      SIGTERM process groups, SIGKILL after a grace period
  status                                 show process state + API /status summary
Services bind to 127.0.0.1 only. No hosted services, queues or daemons are installed.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from pydantic import ValidationError
from pydantic_settings import SettingsError

from packages.contracts.enums import ExecutionLabel
from packages.core.config import Settings
from packages.core.modes import operator_trading_mode
from services.api.launcher import ApiStartupError, api_pid, ensure_api

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "var" / "run"
LOGS = ROOT / "var" / "logs"

SERVICES: dict[str, list[str]] = {
    "research": [str(ROOT / ".venv" / "bin" / "python"), "-m", "services.research.autonomous"],
    # Research crew (Claude Code + Codex on the operator's own sign-in); no execution authority.
    "agents": [str(ROOT / ".venv" / "bin" / "python"), "-m", "services.agents", "run"],
    # Autonomous daily orchestrator (paper only; agent token only; see docs/architecture/autopilot.md).
    "autopilot": [str(ROOT / ".venv" / "bin" / "python"), "-m", "services.autopilot", "run"],
    "api": [],  # managed by the shared API launcher
    "lean-sim": ["bash", "scripts/lean_live_sim.sh", "--attach-api"],
}
DEFAULT = ["api"]


def _pidfile(name: str) -> Path:
    if name == "api":
        try:
            return Settings().run_dir / "api.pid"
        except (SettingsError, ValidationError):
            return Path(os.environ.get("LAT_PROJECT_ROOT", ROOT)) / "var" / "run" / "api.pid"
    return RUN / f"{name}.pid"


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read_pid(name: str) -> int | None:
    p = _pidfile(name)
    if name == "api":
        return api_pid(p.parent)
    if not p.exists():
        return None
    try:
        pid = int(p.read_text().strip())
    except (OSError, ValueError):
        return None
    return pid if pid > 1 and _alive(pid) else None


def _http_json(api_url: str, path: str, timeout: float = 2.0) -> dict[str, object] | None:
    try:
        with urllib.request.urlopen(f"{api_url}{path}", timeout=timeout) as r:  # noqa: S310 - Settings validates loopback
            return json.loads(r.read())
    except Exception:
        return None


def start(names: list[str]) -> int:
    settings = Settings()
    RUN.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    for name in names:
        if name not in SERVICES:
            print(f"unknown service {name}")
            return 2
        if name == "api":
            try:
                health = ensure_api(settings)
            except (ApiStartupError, OSError) as exc:
                print(f"api: {exc}")
                return 2
            mode = operator_trading_mode(settings.mode, ExecutionLabel(health["execution_label"]))
            print(f"api: {settings.api_url} mode={mode.value.title()} startup={health['startup_phase']}")
            print(f"monitor: {settings.api_url}/monitor")
            continue
        if (pid := _read_pid(name)) is not None:
            print(f"{name}: already running (pid {pid})")
            continue
        child_env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        if name == "research":
            child_env = {k: v for k, v in child_env.items() if k in ("PATH", "HOME", "TMPDIR", "LANG", "PYTHONUNBUFFERED")}
            if "LAT_PROJECT_ROOT" in os.environ:
                child_env["LAT_PROJECT_ROOT"] = os.environ["LAT_PROJECT_ROOT"]
        elif name == "agents":
            # The provider CLIs use their own subscription sign-in; no API keys or app tokens are passed on.
            keep = ("PATH", "HOME", "USER", "LANG", "TMPDIR", "TERM", "SHELL", "CODEX_HOME", "CLAUDE_CONFIG_DIR",
                    "LAT_PROJECT_ROOT", "LAT_USE_KEYCHAIN", "PYTHONUNBUFFERED")
            child_env = {k: v for k, v in child_env.items() if k in keep or k.startswith("LAT_AGENTS_")}
        with (LOGS / f"{name}.log").open("ab") as log:
            proc = subprocess.Popen(
                SERVICES[name], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                start_new_session=True, env=child_env,
            )
        _pidfile(name).write_text(str(proc.pid))
        print(f"{name}: started pid {proc.pid} (log var/logs/{name}.log)")
    return 0


def stop(names: list[str]) -> int:
    for name in names or list(SERVICES):
        pid = _read_pid(name)
        if pid is None:
            if name != "api":
                _pidfile(name).unlink(missing_ok=True)
            continue
        try:
            send_signal = os.killpg if os.getpgid(pid) == pid else os.kill
            send_signal(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        for _ in range(40):
            if not _alive(pid):
                break
            time.sleep(0.25)
        if _read_pid(name) == pid:
            with contextlib.suppress(ProcessLookupError):
                send_signal(pid, signal.SIGKILL)
        if name != "api":
            _pidfile(name).unlink(missing_ok=True)
        print(f"{name}: stopped")
    return 0


def status() -> int:
    for name in SERVICES:
        pid = _read_pid(name)
        print(f"{name:10s} {'running pid ' + str(pid) if pid else 'stopped'}")
    st = _http_json(Settings().api_url, "/monitor/status")
    if st is None:
        print("api status: unreachable (service down). Broker-held orders may still be live at the broker;"
              " run `make reconcile` after restart.")
        return 0
    print(json.dumps(st, indent=2))
    return 0


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    cmd, rest = argv[0], argv[1:]
    if cmd == "start":
        return start(rest or DEFAULT)
    if cmd == "stop":
        return stop(rest)
    if cmd == "status":
        return status()
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
