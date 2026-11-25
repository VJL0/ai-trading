"""``python -m services.agents {run [--once] [--fake] | status | notes | enqueue | providers}``.

Research crew CLI. It never starts the API, never trades and never touches the trading ledger.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path
from typing import Any

from . import mechanisms
from .config import ConfigError, CrewConfig
from .contracts import Provider, TaskKind, TaskState
from .store import CrewLocked, CrewStore, day_start

FAKE_STORE_NAME = "crew-fake.sqlite3"


def _emit(data: Any, as_json: bool, text: str | None = None) -> None:
    if as_json or text is None:
        print(json.dumps(data, indent=1, sort_keys=True, default=str))
    else:
        print(text)


def _config(args: argparse.Namespace) -> CrewConfig:
    cfg = CrewConfig.from_env(project_root=Path(args.root).resolve() if args.root else None)
    store = Path(args.store) if args.store else None
    if store is None and getattr(args, "fake", False):
        store = cfg.store_path.with_name(FAKE_STORE_NAME)  # never mix demo notes into the real crew memory
    if store is not None:
        from dataclasses import replace

        cfg = replace(cfg, store_path=store if store.is_absolute() else cfg.project_root / store)
    return cfg


def cmd_run(args: argparse.Namespace) -> int:
    from .crew import Crew, build_providers

    cfg = _config(args)
    store = CrewStore(cfg.store_path)
    if args.fake:
        from .fake import demo_experiment_runner, demo_provider, fake_mcp_spec

        providers: dict[Provider, Any] = {p: demo_provider(p) for p in cfg.providers}
        crew = Crew(cfg, store, providers, experiment_runner=demo_experiment_runner, mcp_spec_factory=fake_mcp_spec)
    else:
        crew = Crew(cfg, store, build_providers(cfg))
    previous_root = os.environ.get("LAT_PROJECT_ROOT")
    if not args.fake:
        # QuantLab's child process resolves its Settings from the environment.
        os.environ["LAT_PROJECT_ROOT"] = str(cfg.project_root.resolve())
    try:
        history = crew.run(once=args.once)
    except CrewLocked as exc:
        _emit({"error": str(exc)}, True)
        return 4
    finally:
        if not args.fake:
            if previous_root is None:
                os.environ.pop("LAT_PROJECT_ROOT", None)
            else:
                os.environ["LAT_PROJECT_ROOT"] = previous_root
    _emit({"mode": "FAKE (simulated providers, nothing evaluated)" if args.fake else "CLI subscriptions",
           "store": str(cfg.store_path), "ticks": history}, args.json, None)
    return 0


def _status(cfg: CrewConfig, store: CrewStore) -> dict[str, Any]:
    from .crew import utcnow

    now = utcnow()
    providers = {}
    for p in cfg.providers:
        st = store.provider_state(p).public(now)
        st["runs_today"] = store.runs_today(p, now)
        st["daily_cap"] = cfg.daily_runs_per_provider
        providers[p.value] = st
    runs = store.runs_since(day_start(now))
    return {"store": str(cfg.store_path), "asof": now.isoformat(), "queue": store.counts(),
            "providers": providers,
            "runs_today": [{k: r.get(k) for k in ("id", "task_id", "provider", "started", "ended", "outcome")}
                           for r in runs[-20:]],
            "last_notes": [n.public() for n in store.notes(limit=5)],
            "evidence_note": "crew notes are AGENT_NOTE proposals; experiment rows copy the development trial "
                             "ledger; nothing here is validation or a trading signal"}


def cmd_status(args: argparse.Namespace) -> int:
    cfg = _config(args)
    data = _status(cfg, CrewStore(cfg.store_path))
    lines = [f"store: {data['store']}", "queue:"]
    for kind, states in sorted(data["queue"].items()):
        lines.append(f"  {kind}: " + ", ".join(f"{s}={n}" for s, n in sorted(states.items())))
    lines.append("providers:")
    for name, st in data["providers"].items():
        lines.append(f"  {name}: runs_today={st['runs_today']}/{st['daily_cap']} cooldown_until={st['cooldown_until']}"
                     f" disabled={st['disabled_reason']}")
    lines.append("last notes:")
    for n in data["last_notes"]:
        lines.append(f"  #{n['seq']} {n['at'][:16]} {n['provider'] or 'crew'} {n['kind']} {n['mechanism'] or ''}")
    _emit(data, args.json, "\n".join(lines))
    return 0


def cmd_notes(args: argparse.Namespace) -> int:
    cfg = _config(args)
    notes = [n.public() for n in CrewStore(cfg.store_path).notes(kind=args.kind, limit=args.limit)]
    text = "\n".join(f"#{n['seq']} {n['at'][:16]} {n['provider'] or 'crew'} {n['kind']} [{n['mechanism'] or '-'}] "
                     f"({n['evidence_status']}) {json.dumps(n['body'], default=str)[:300]}" for n in notes)
    _emit({"notes": notes}, args.json, text or "no notes")
    return 0


def cmd_enqueue(args: argparse.Namespace) -> int:
    from .crew import utcnow
    from .planner import PRIORITY, _review_payload, completed_experiments

    cfg = _config(args)
    try:
        payload = json.loads(args.payload) if args.payload else {}
    except ValueError as exc:
        _emit({"error": f"--payload is not JSON: {exc}"}, True)
        return 2
    if not isinstance(payload, dict):
        _emit({"error": "--payload must be a JSON object"}, True)
        return 2
    kind = TaskKind(args.kind)
    store = CrewStore(cfg.store_path)
    now = utcnow()
    exclude: Provider | None = None
    parent: str | None = None
    expires = None
    if kind is TaskKind.HYPOTHESIS:
        mech = payload.get("mechanism")
        if mech not in mechanisms.BY_KEY:
            _emit({"error": f"hypothesis payload needs mechanism, one of {list(mechanisms.KEYS)}"}, True)
            return 2
    if kind is TaskKind.REVIEW:
        parent = str(payload.get("hypothesis_task_id") or "")
        hyp = store.get_task(parent) if parent else None
        if (hyp is None or hyp.kind is not TaskKind.HYPOTHESIS or hyp.state is not TaskState.DONE
                or hyp.author_provider is None or not completed_experiments(hyp.result or {})):
            _emit({"error": "review payload needs hypothesis_task_id of an authored hypothesis with a completed experiment"},
                  True)
            return 2
        payload = _review_payload(hyp.id, hyp.author_provider, hyp.result or {})
        exclude = hyp.author_provider
        expires = now + timedelta(hours=cfg.review_wait_hours)
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
    key = f"review:{parent}" if kind is TaskKind.REVIEW else f"manual:{kind.value}:{digest}"
    tid, created = store.enqueue(kind, key, payload, now=now, priority=PRIORITY[kind] + 5,
                                 exclude_provider=exclude, max_attempts=cfg.max_attempts, expires_at=expires,
                                 parent_id=parent)
    _emit({"task_id": tid, "created": created, "kind": kind.value}, args.json,
          f"{'enqueued' if created else 'already queued'}: {kind.value} {tid}")
    return 0


def cmd_providers(args: argparse.Namespace) -> int:
    cfg = _config(args)
    out: dict[str, Any] = {}
    try:
        from .crew import build_providers

        providers = build_providers(cfg)
    except Exception as exc:
        _emit({"error": f"providers unavailable: {type(exc).__name__}: {exc}"}, True)
        return 3
    for p, prov in providers.items():
        try:
            ok, reason = prov.availability()
        except Exception as exc:
            ok, reason = False, f"availability raised {type(exc).__name__}"
        out[p.value] = {"available": ok, "reason": reason, "binary": cfg.binary_for(p),
                        "model": cfg.model_for(p) or "(CLI default)"}
    _emit({"providers": out, "note": "availability only; no quota was spent"}, args.json,
          "\n".join(f"{k}: {'OK' if v['available'] else 'UNAVAILABLE'} ({v['reason']})" for k, v in out.items()))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m services.agents", description=__doc__)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", help="structured JSON output")
    common.add_argument("--store", help="crew store path (default LAT_AGENTS_STORE or experiments/agents/crew.sqlite3)")
    common.add_argument("--root", help="project root (default LAT_PROJECT_ROOT or the repository)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", parents=[common], help="run the crew loop")
    run.add_argument("--once", action="store_true", help="one tick, then exit")
    run.add_argument("--fake", action="store_true",
                     help="simulated providers and experiment runner (demo; separate store, nothing evaluated)")
    run.set_defaults(fn=cmd_run)
    sub.add_parser("status", parents=[common], help="queue, provider cooldowns, today's runs").set_defaults(fn=cmd_status)
    notes = sub.add_parser("notes", parents=[common], help="crew notes, newest first")
    notes.add_argument("--kind")
    notes.add_argument("--limit", type=int, default=20)
    notes.set_defaults(fn=cmd_notes)
    enq = sub.add_parser("enqueue", parents=[common], help="enqueue a task by hand (idempotent per payload)")
    enq.add_argument("kind", choices=[k.value for k in TaskKind])
    enq.add_argument("--payload", help="JSON object")
    enq.set_defaults(fn=cmd_enqueue)
    sub.add_parser("providers", parents=[common], help="CLI availability (no quota spent)").set_defaults(
        fn=cmd_providers)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.fn(args))
    except ConfigError as exc:
        _emit({"error": str(exc)}, True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
