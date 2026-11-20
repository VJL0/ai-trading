"""Minimal local process supervisor (spec §11: local macOS process supervisor / make scripts).

Commands:
  start [api|dashboard|research|lean-sim ...]   start detached services (PIDs in var/run, logs in var/logs)
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

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "var" / "run"
LOGS = ROOT / "var" / "logs"
API_PORT = int(os.environ.get("LAT_API_PORT", "8765"))
DASH_PORT = int(os.environ.get("LAT_DASHBOARD_PORT", "5173"))

SERVICES: dict[str, list[str]] = {
    "research": [str(ROOT / ".venv" / "bin" / "python"), "-m", "services.research.autonomous"],
    "api": ["uv", "run", "--frozen", "python", "-m", "services.api"],
    "dashboard": ["pnpm", "--dir", "apps/dashboard", "exec", "vite", "--host", "127.0.0.1", "--port", str(DASH_PORT),
                  "--strictPort"],
    "lean-sim": ["bash", "scripts/lean_live_sim.sh", "--attach-api"],
}
DEFAULT = ["api", "dashboard"]


def _pidfile(name: str) -> Path:
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
    if not p.exists():
        return None
    try:
        pid = int(p.read_text().strip())
    except ValueError:
        return None
    return pid if _alive(pid) else None


def _http_json(path: str, timeout: float = 2.0) -> dict[str, object] | None:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{API_PORT}{path}", timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return None


def start(names: list[str]) -> int:
    RUN.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    for name in names:
        if name not in SERVICES:
            print(f"unknown service {name}")
            return 2
        if (pid := _read_pid(name)) is not None:
            if name == "api" and os.environ.get("LAT_GATEWAY"):
                status = _http_json("/api/v1/status") or {}
                expected = os.environ["LAT_GATEWAY"]
                if status.get("gateway") != expected:
                    print(f"api: existing gateway {status.get('gateway', 'unverified')} differs from requested {expected}; "
                          "check sessions/orders and stop the existing service before switching")
                    return 2
            print(f"{name}: already running (pid {pid})")
            continue
        log = open(LOGS / f"{name}.log", "ab")  # noqa: SIM115 - handed to the child
        child_env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        if name == "research":
            child_env = {k: v for k, v in child_env.items() if k in ("PATH", "HOME", "TMPDIR", "LANG", "PYTHONUNBUFFERED")}
        if name == "dashboard":
            child_env.setdefault("LAT_DASHBOARD_API_TARGET", f"http://127.0.0.1:{API_PORT}")
        proc = subprocess.Popen(
            SERVICES[name], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            start_new_session=True, env=child_env,
        )
        _pidfile(name).write_text(str(proc.pid))
        print(f"{name}: started pid {proc.pid} (log var/logs/{name}.log)")
    if "api" in names:
        for _ in range(60):
            if _http_json("/api/v1/health"):
                break
            time.sleep(0.5)
        h = _http_json("/api/v1/health")
        if not h:
            print("api: did not become healthy within 30s — see var/logs/api.log")
            return 1
        print(f"api: healthy at http://127.0.0.1:{API_PORT}  mode={h.get('mode')} label={h.get('execution_label')}")
    if "dashboard" in names:
        print(f"dashboard: http://127.0.0.1:{DASH_PORT}")
    return 0


def stop(names: list[str]) -> int:
    for name in names or list(SERVICES):
        pid = _read_pid(name)
        if pid is None:
            _pidfile(name).unlink(missing_ok=True)
            continue
        with contextlib.suppress(ProcessLookupError):
            os.killpg(pid, signal.SIGTERM)
        for _ in range(40):
            if not _alive(pid):
                break
            time.sleep(0.25)
        if _alive(pid):
            os.killpg(pid, signal.SIGKILL)
        _pidfile(name).unlink(missing_ok=True)
        print(f"{name}: stopped")
    return 0


def status() -> int:
    for name in SERVICES:
        pid = _read_pid(name)
        print(f"{name:10s} {'running pid ' + str(pid) if pid else 'stopped'}")
    st = _http_json("/api/v1/status")
    if st is None:
        print("api status: unreachable (service down). Broker-held orders may still be live at the broker;"
              " run `make reconcile` after restart.")
        return 0
    keys = ("mode", "execution_label", "gateway", "entries_locked", "lock_reasons", "policy_version")
    print(json.dumps({k: st.get(k) for k in keys}, indent=2))
    for section in ("startup", "feed", "engine", "broker", "credentials", "ai"):
        print(f"{section}: {json.dumps(st.get(section))}")
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
