"""``python -m services.autopilot {run [--once] | status [--json] | plan [--json] [--at ISO]}``.

* ``run``: the supervisor loop (single instance, ``LAT_AUTOPILOT_ENABLED=true`` required). SIGINT/SIGTERM stop it
  between actions; a running API session is never stopped by the autopilot.
* ``run --once``: one tick, then exit.
* ``status``: read-only state (lock holder, phase, jobs, stand-down, alerts, next actions). No API calls.
* ``plan``: dry run of the next tick: the actions it would take and why. No API calls, no writes.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import signal
import sqlite3
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any

from .backend import AuthorityError, ToolPort
from .config import AutopilotConfig, ConfigError
from .research import Runner, run_child
from .store import AutopilotLocked, Store, instance_lock, lock_holder
from .supervisor import Autopilot, Clock, SystemClock, status_snapshot

EXIT_DISABLED = 3
EXIT_LOCKED = 4
EXIT_REFUSED = 5
EXIT_STATE_ERROR = 6


def _emit(data: Any, as_json: bool, text: str) -> None:
    print(json.dumps(data, indent=1, sort_keys=True, default=str) if as_json else text)


def _parse_at(text: str | None) -> datetime | None:
    if not text:
        return None
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise SystemExit("--at needs a timezone, e.g. 2026-11-27T08:45:00-05:00")
    return dt.astimezone(UTC)


def _readonly_store(cfg: AutopilotConfig) -> Store | None:
    try:
        return Store(cfg.store_path, readonly=True)
    except FileNotFoundError:
        return None


def cmd_status(args: argparse.Namespace, cfg: AutopilotConfig, clock: Clock) -> int:
    snap = status_snapshot(cfg, _readonly_store(cfg), clock.now(), lock_pid=lock_holder(cfg.lock_path))
    lines = [f"autopilot: {'enabled' if snap['enabled'] else 'DISABLED (LAT_AUTOPILOT_ENABLED=false)'}; "
             f"running pid: {snap['running_pid'] or 'none'}", f"phase: {snap['phase']} {snap.get('trading_day') or ''}"]
    if snap.get("standdown"):
        lines.append(f"STAND-DOWN: {snap['standdown']}")
    for j in snap.get("jobs", []):
        lines.append(f"  {j['day']} {j['job']:<15} {j['status']:<10} attempts={j['attempts']}")
    for a in snap.get("recent_alerts", []):
        lines.append(f"  ALERT {a['ts']} {a['message']}")
    for a in (snap.get("next") or {}).get("actions", []):
        lines.append(f"  next: {a['kind']} {a['day'] or ''} - {a['reason']}")
    _emit(snap, args.json, "\n".join(lines))
    return 0


def cmd_plan(args: argparse.Namespace, cfg: AutopilotConfig, clock: Clock) -> int:
    now = _parse_at(args.at) or clock.now()
    store = _readonly_store(cfg)
    if store is None:
        # Fresh install: plan against an empty in-memory view (nothing is written to the state directory).
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory(prefix="lat-autopilot-plan-") as tmp:
            plan = Autopilot(cfg, Store(Path(tmp) / "plan.sqlite3"), None).decide(now).public()
    else:
        plan = Autopilot(cfg, store, None).decide(now).public()
    lines = [f"plan at {plan['now']}: phase {plan['phase']}"]
    if plan["trading_day"]:
        td = plan["trading_day"]
        lines.append(f"  trading day {td['day']}: open {td['open']} close {td['close']}"
                     f"{' (early close)' if td['early_close'] else ''}, entry cutoff {td['entry_cutoff']}")
    lines += [f"  would: {a['kind']} {a['day'] or ''} - {a['reason']}" for a in plan["actions"]]
    lines += [f"  note: {n}" for n in plan["notes"]]
    _emit(plan, args.json, "\n".join(lines))
    return 0


def _settings_and_backend(backend_factory: Callable[[Any], ToolPort] | None) -> tuple[Any, ToolPort, bool, bool]:
    from packages.contracts.enums import FeedMode
    from packages.core.config import Settings
    from packages.core.secrets import SecretStore

    from .backend import make_http_backend, require_paper_runtime

    settings = Settings()
    require_paper_runtime(settings)
    # Presence check only (the API makes the same decision); no credential value is read.
    feed = settings.market_data_feed(SecretStore(settings.project_root, use_keychain=settings.use_keychain))
    backend = backend_factory(settings) if backend_factory else make_http_backend(settings)
    return settings, backend, bool(settings.quantlab_daily_enabled), feed == FeedMode.SIMULATED


def cmd_run(args: argparse.Namespace, cfg: AutopilotConfig, clock: Clock, *,
            backend_factory: Callable[[Any], ToolPort] | None = None, runner: Runner = run_child,
            ensure_api: Callable[[Any], Any] | None = None) -> int:
    if not cfg.enabled:
        _emit({"error": "disabled"}, True, "")
        print("autopilot is disabled; set LAT_AUTOPILOT_ENABLED=true to let it run", file=sys.stderr)
        return EXIT_DISABLED
    try:
        settings, backend, api_owns_quantlab, simulated = _settings_and_backend(backend_factory)
    except AuthorityError as exc:
        print(f"autopilot refused to start: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    starter: Callable[[], Any] | None = None
    if cfg.start_api:
        from .backend import ensure_api_running

        launch = ensure_api or ensure_api_running

        def _start_api() -> Any:
            return launch(settings)

        starter = _start_api
    try:
        with instance_lock(cfg.lock_path):
            store = Store(cfg.store_path)
            ap = Autopilot(cfg, store, backend, clock=clock, runner=runner, api_owns_quantlab=api_owns_quantlab,
                           simulated_feed=simulated, ensure_api=starter)
            asyncio.run(_run(ap, once=args.once))
    except AutopilotLocked as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_LOCKED
    return 0


STOP_GRACE_S = 20.0


async def _run(ap: Autopilot, *, once: bool) -> None:
    """SIGINT/SIGTERM stop the loop between actions. An action still running after ``STOP_GRACE_S`` (a long
    research child) is cancelled: the child is killed, and its job resumes idempotently on the next start."""
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    task = asyncio.current_task()

    def on_signal() -> None:
        stop.set()
        if task is not None:
            loop.call_later(STOP_GRACE_S, task.cancel)

    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError, RuntimeError):
            loop.add_signal_handler(sig, on_signal)
    try:
        await ap.run(stop, once=once)
    except asyncio.CancelledError:
        if not stop.is_set():
            raise
    finally:
        closer = getattr(ap.backend, "aclose", None)
        if closer is not None:
            await closer()


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m services.autopilot", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="supervisor loop (or one tick with --once)")
    run.add_argument("--once", action="store_true", help="one tick, then exit")
    st = sub.add_parser("status", help="read-only state, no API calls")
    st.add_argument("--json", action="store_true")
    pl = sub.add_parser("plan", help="dry run: what the next tick would do and why")
    pl.add_argument("--json", action="store_true")
    pl.add_argument("--at", help="evaluate at this ISO time instead of now (dry run only)")
    return ap


def main(argv: Sequence[str] | None = None, *, clock: Clock | None = None,
         backend_factory: Callable[[Any], ToolPort] | None = None, runner: Runner = run_child,
         ensure_api: Callable[[Any], Any] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        cfg = AutopilotConfig.from_env()
    except ConfigError as exc:
        print(f"autopilot configuration error: {exc}", file=sys.stderr)
        return 2
    clk = clock or SystemClock()
    try:
        if args.cmd == "status":
            return cmd_status(args, cfg, clk)
        if args.cmd == "plan":
            return cmd_plan(args, cfg, clk)
        return cmd_run(args, cfg, clk, backend_factory=backend_factory, runner=runner, ensure_api=ensure_api)
    except (sqlite3.DatabaseError, json.JSONDecodeError) as exc:
        print(f"autopilot state error at {cfg.store_path}: {exc}", file=sys.stderr)
        return EXIT_STATE_ERROR
