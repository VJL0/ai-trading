"""Shared helpers for replay integration and failure-injection tests (synthetic data, disposable workspaces)."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

from packages.contracts.policy import RiskPolicy
from services.replay.broker import FillModel
from services.replay.engine import ReplayConfig, ReplayEngine, ReplayHooks, ReplayResult
from services.replay.snapshot import BarSnapshot, SyntheticSpec, synthetic_snapshot


def snapshot(sessions: int = 3, **kw: Any) -> BarSnapshot:
    return synthetic_snapshot(SyntheticSpec(sessions=sessions, **kw))


def config(strategy_id: str = "tod-SPY-1000-60m", *, policy: RiskPolicy | None = None, fill: FillModel | None = None,
           equity: str = "5000", start: date | None = None) -> ReplayConfig:
    return ReplayConfig(strategy_id=strategy_id, equity=Decimal(equity), policy=policy, fill=fill or FillModel(),
                        start=start)


async def replay(tmp: Path, snap: BarSnapshot, cfg: ReplayConfig, *, hooks: ReplayHooks | None = None,
                 journal: Path | None = None, resume: bool = False, stop_after: Any = None
                 ) -> tuple[ReplayEngine, ReplayResult]:
    engine = ReplayEngine(snap, cfg, tmp / "ws", journal_path=journal or tmp / "out" / "journal.jsonl", hooks=hooks)
    result = await engine.run(resume=resume, stop_after=stop_after)
    return engine, result


def of(records: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    return [r for r in records if r["kind"] == kind]


def trade_key(t: dict[str, Any]) -> tuple[Any, ...]:
    return (t["symbol"], t["entry_at"], t["entry_quantity"], t["entry_avg_price"], t["exit_avg_price"],
            t["exit_reason"], t["close_reason"])
