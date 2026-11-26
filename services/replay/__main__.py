"""CLI: ``python -m services.replay {evaluate,parity,compare-paper,strategies}``.

Every command runs in a disposable replay workspace (its own ledger, keys and FakeBroker state); nothing touches the
operator's ledger, policy, broker or running API. ``--policy active`` only READS the active paper policy.

Examples::

    python -m services.replay evaluate --policy active --equity 5000 --strategy tod-SPY-1000-60m --bars bars.csv
    python -m services.replay evaluate --policy my_profile.json --equity 5000 --strategy tod-SPY-1000-60m \\
        --synthetic --sessions 8 --parity
    python -m services.replay compare-paper --paper-db var/paper/ledger.sqlite --start 2026-11-23 \\
        --end 2026-11-25 --bars bars.csv --strategy runtime-baseline --policy active
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from packages.core.config import PROJECT_ROOT

from .broker import FillModel
from .engine import ReplayConfig, ReplayEngine, ReplayResult
from .policy import PolicyProfileError, load_policy
from .snapshot import BarSnapshot, SnapshotError, SyntheticSpec, synthetic_snapshot
from .strategies import get_strategy, list_strategies

HEADLINE = ("sessions", "initial_equity", "final_equity_net", "gross_pl", "modelled_costs_usd", "net_pl", "cagr",
            "sharpe", "max_drawdown", "turnover", "cost_drag", "capital_exposure", "rejections_by_check",
            "order_rejections", "decisions", "entry_matching", "round_trips")


def _data_args(p: argparse.ArgumentParser) -> None:
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--bars", type=Path, help="5-minute bars (csv/parquet: t,symbol,o,h,l,c,v[,vw]); real data must "
                                                "lie outside the protected QuantLab holdout")
    src.add_argument("--synthetic", action="store_true", help="seeded synthetic bars (SYNTHETIC, not market data)")
    p.add_argument("--symbols", default="SPY", help="synthetic symbols, comma separated")
    p.add_argument("--sessions", type=int, default=6, help="synthetic sessions (incl. warm-up)")
    p.add_argument("--synthetic-start", type=date.fromisoformat, default=date(2024, 3, 4))
    p.add_argument("--seed", type=int, default=7, help="synthetic data seed")
    p.add_argument("--start", type=date.fromisoformat, default=None)
    p.add_argument("--end", type=date.fromisoformat, default=None)
    p.add_argument("--warmup-sessions", type=int, default=1)


def _run_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--strategy", required=True)
    p.add_argument("--policy", default="default", help="default | active (read-only) | path to a RiskPolicy JSON")
    p.add_argument("--equity", type=Decimal, default=Decimal("5000"))
    p.add_argument("--fill-seed", type=int, default=0)
    p.add_argument("--nonfill", type=float, default=0.0, help="entry non-fill probability when marketable")
    p.add_argument("--partial", type=float, default=0.0, help="entry partial-fill probability")
    p.add_argument("--quote-step", type=int, default=5, help="seconds between synthesized intrabar quotes")
    p.add_argument("--out", type=Path, default=None, help="output dir (default experiments/replay/<run>)")
    p.add_argument("--workspace", type=Path, default=None, help="replay workspace (default: a new temp dir)")
    p.add_argument("--resume", action="store_true", help="resume an interrupted replay from --workspace/--out")


def _snapshot(args: argparse.Namespace) -> BarSnapshot:
    if args.synthetic:
        syms = tuple(s.strip().upper() for s in args.symbols.split(",") if s.strip())
        return synthetic_snapshot(SyntheticSpec(symbols=syms, start=args.synthetic_start, sessions=args.sessions,
                                                seed=args.seed))
    return BarSnapshot.load(args.bars)


def _config(args: argparse.Namespace) -> ReplayConfig:
    policy, source = load_policy(args.policy)
    get_strategy(args.strategy)  # fail fast on an unknown id
    return ReplayConfig(strategy_id=args.strategy, equity=args.equity, policy=policy, policy_source=source,
                        fill=FillModel(seed=args.fill_seed, entry_nonfill_prob=args.nonfill,
                                       entry_partial_prob=args.partial, quote_step_s=args.quote_step),
                        start=args.start, end=args.end, warmup_sessions=args.warmup_sessions)


def _dirs(args: argparse.Namespace) -> tuple[Path, Path]:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = args.out or PROJECT_ROOT / "experiments" / "replay" / f"{stamp}-{args.strategy}"
    ws = args.workspace or Path(tempfile.mkdtemp(prefix="lat-replay-"))
    if args.resume and (args.workspace is None or args.out is None):
        raise SystemExit("--resume needs the --workspace and --out of the interrupted run")
    return out, ws


def _replay(args: argparse.Namespace) -> tuple[ReplayEngine, ReplayResult, Path]:
    snap = _snapshot(args)
    cfg = _config(args)
    out, ws = _dirs(args)
    engine = ReplayEngine(snap, cfg, ws, journal_path=out / "journal.jsonl")
    result = asyncio.run(engine.run(resume=args.resume))
    return engine, result, out


def _headline(result: ReplayResult, engine: ReplayEngine, out: Path) -> dict[str, Any]:
    s = result.summary
    return {"execution_label": "REPLAY", "is_simulated": True, "strategy": engine.strategy.strategy_id,
            "policy": engine.config.policy_source, "equity": str(engine.config.equity),
            "provenance": engine.provenance, **{k: s.get(k) for k in HEADLINE}, "journal": str(result.journal_path),
            "summary": str(result.summary_path), "out": str(out), "workspace": str(engine.workspace)}


def cmd_evaluate(args: argparse.Namespace) -> int:
    engine, result, out = _replay(args)
    report = _headline(result, engine, out)
    if args.parity:
        report["parity"] = _parity(engine, result, out)
    print(json.dumps(report, indent=1, default=str))
    return 0


def _parity(engine: ReplayEngine, result: ReplayResult, out: Path) -> dict[str, Any]:
    from packages.contracts.policy import RiskPolicy

    from .parity import compare

    rep = compare(result.summary, engine.snapshot, engine.strategy, engine.days, engine.config.policy or RiskPolicy(),
                  engine.config.equity)
    _write_comparison(out / "parity.json", engine, rep)
    return {k: v for k, v in rep.items() if k != "trades"}


def cmd_parity(args: argparse.Namespace) -> int:
    engine, result, out = _replay(args)
    print(json.dumps(_parity(engine, result, out), indent=1, default=str))
    return 0


def cmd_compare_paper(args: argparse.Namespace) -> int:
    from .paper import compare_paper, export_paper_ledger

    engine, result, out = _replay(args)
    start = datetime.combine(args.period_start, time(0), tzinfo=UTC)
    end = datetime.combine(args.period_end + timedelta(days=1), time(0), tzinfo=UTC)
    paper = export_paper_ledger(args.paper_db, start, end, mode=args.paper_mode or None)
    rep = compare_paper(paper, result.records)
    _write_comparison(out / "paper_comparison.json", engine, rep)
    print(json.dumps({k: v for k, v in rep.items() if k != "matched"}, indent=1, default=str))
    return 0


def _write_comparison(path: Path, engine: ReplayEngine, report: dict[str, Any]) -> None:
    report.update(execution_label="REPLAY", runtime_label="SIMULATED", is_simulated=True, provenance=engine.provenance)
    with path.open("x", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, default=str)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m services.replay",
                                 description="Replay the real decision and execution stack over bar snapshots (REPLAY).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("strategies", help="list replay strategies")
    ev = sub.add_parser("evaluate", help="replay a strategy under one policy and account size")
    _data_args(ev)
    _run_args(ev)
    ev.add_argument("--parity", action="store_true", help="also compare with the research simulator")
    pa = sub.add_parser("parity", help="replay vs services.quantlab.intraday for the same snapshot and strategy")
    _data_args(pa)
    _run_args(pa)
    cp = sub.add_parser("compare-paper", help="compare exported paper-ledger orders/fills with the replay")
    _data_args(cp)
    _run_args(cp)
    cp.add_argument("--paper-db", type=Path, required=True, help="paper ledger SQLite (opened read-only)")
    cp.add_argument("--period-start", type=date.fromisoformat, required=True)
    cp.add_argument("--period-end", type=date.fromisoformat, required=True)
    cp.add_argument("--paper-mode", default="PAPER", help="orders.mode filter ('' = any)")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "strategies":
            print(json.dumps(list_strategies(), indent=1))
            return 0
        return {"evaluate": cmd_evaluate, "parity": cmd_parity, "compare-paper": cmd_compare_paper}[args.cmd](args)
    except (PolicyProfileError, SnapshotError, FileExistsError, KeyError, ValueError) as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
