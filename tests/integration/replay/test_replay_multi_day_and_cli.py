"""Multi-day holding (overnight, DAY stop re-arm), policy profiles via the CLI, and the paper comparison hook on a
real-schema PAPER ledger produced by the real components (SIMULATED; fixtures only)."""

from __future__ import annotations

import hashlib
import io
import json
from contextlib import redirect_stdout
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from packages.contracts.enums import TradingMode
from packages.contracts.policy import RiskPolicy
from packages.core.db import Database
from services.replay.__main__ import main
from services.replay.engine import ReplayEngine
from services.replay.paper import compare_paper, export_paper_ledger
from services.replay.parity import compare
from services.risk.operator_auth import OperatorAuth
from services.risk.policy_store import PolicyStore
from tests.integration.positions.pm_rig import build_rig
from tests.integration.replay.replay_support import config, of, replay, snapshot

MULTI = "tod-SPY-1000-2d"
OVERNIGHT = RiskPolicy(name="overnight-test", allow_overnight_positions=True, max_holding_sessions=2,
                       multi_day_strategy_allowlist=(MULTI,))


async def test_multi_day_hold_rearms_protection_and_exits_on_the_holding_limit(tmp_path: Path) -> None:
    snap = snapshot(sessions=4)
    engine, res = await replay(tmp_path, snap, config(MULTI, policy=OVERNIGHT))
    recs = res.records
    first = res.summary["trades"][0]
    assert first["exit_reason"] == "HOLDING_LIMIT"
    assert first["opened_at"][:10] == "2024-03-05" and first["exit_at"].startswith("2024-03-06T20:50")
    expired = [o for o in of(recs, "ORDER") if o["order_type"] == "STOP_LIMIT" and o["to_state"] == "EXPIRED"]
    assert expired and expired[0]["at"].startswith("2024-03-05T21:00")  # DAY protection expired at the close
    placed = [s for s in of(recs, "STOP") if s["outcome"] == "SUBMITTED"]
    assert {p["at"][:10] for p in placed} >= {"2024-03-05", "2024-03-06"}  # re-armed at the next open
    approvals = [r for r in of(recs, "RISK_DECISION") if r["strategy_id"] == MULTI]
    assert approvals and all(r["approved"] for r in approvals)
    rep = compare(res.summary, snap, engine.strategy, engine.days, OVERNIGHT, Decimal(5000))
    assert rep["matched"] >= 1 and rep["trades"][0]["exit_category_research"] == "EOD"
    assert rep["trades"][0]["exit_category_replay"] == "EOD"


async def test_multi_day_strategy_is_rejected_under_the_default_policy(tmp_path: Path) -> None:
    _, res = await replay(tmp_path, snapshot(sessions=2), config(MULTI))
    rej = of(res.records, "REJECTION")
    assert rej and "holding_period_authorized" in rej[0]["failed_checks"]
    assert res.summary["round_trips"] == 0 and res.summary["rejections_by_check"]["holding_period_authorized"] >= 1


async def test_workspace_alias_cannot_write_operator_var(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "operator"
    ledger = _paper_project(root, RiskPolicy(name="operator"))
    before = ledger.read_bytes()
    monkeypatch.setenv("LAT_PROJECT_ROOT", str(root))
    workspace = tmp_path / "replay"
    (workspace / "var").mkdir(parents=True)
    (workspace / "var" / "replay").symlink_to(ledger.parent, target_is_directory=True)
    engine = ReplayEngine(snapshot(), config(), workspace)
    with pytest.raises(ValueError, match="empty disposable workspace"):
        await engine.run()
    assert ledger.read_bytes() == before
    assert not engine.journal_path.exists()


def _paper_project(root: Path, policy: RiskPolicy) -> Path:
    db_path = root / "var" / "paper" / "ledger.sqlite"
    db = Database(db_path)
    auth = OperatorAuth(db_path.parent / "keys")
    auth.ensure_token()
    store = PolicyStore(db, TradingMode.PAPER, operator_auth=auth)
    store.ensure_default()
    store.apply_change(policy, operator_token=auth.token_path.read_text().strip(), actor="operator", reason="fixture")
    db.close()
    return db_path


def test_cli_evaluate_with_policy_file_and_active_policy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prof = tmp_path / "profile.json"
    prof.write_text(RiskPolicy(name="tight", max_concentration_pct=Decimal("0.10")).model_dump_json())
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["evaluate", "--policy", str(prof), "--equity", "5000", "--strategy", "tod-SPY-1000-60m",
                   "--synthetic", "--sessions", "3", "--out", str(tmp_path / "o1"), "--workspace", str(tmp_path / "w1")])
    assert rc == 0
    out = json.loads(buf.getvalue())
    assert out["execution_label"] == "REPLAY" and out["round_trips"] == 0
    assert out["rejections_by_check"]["max_concentration"] == 2  # a $1000 order is 20% of $5000 equity > 10%
    assert (tmp_path / "o1" / "summary.json").is_file() and (tmp_path / "o1" / "journal.jsonl").is_file()

    active = RiskPolicy(name="operator-active", max_order_notional=Decimal("1500.00"))
    db_path = _paper_project(tmp_path / "proj", active)
    before = hashlib.sha256(db_path.read_bytes()).hexdigest()
    monkeypatch.setenv("LAT_PROJECT_ROOT", str(tmp_path / "proj"))
    monkeypatch.setenv("LAT_API_HOST", "invalid-api-host")  # Replay is independent of API settings.
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["evaluate", "--policy", "active", "--equity", "5000", "--strategy", "tod-SPY-1000-60m",
                   "--synthetic", "--sessions", "3", "--out", str(tmp_path / "o2"), "--workspace", str(tmp_path / "w2"),
                   "--parity"])
    assert rc == 0
    out = json.loads(buf.getvalue())
    assert active.version in out["policy"] and out["round_trips"] == 2
    assert {t["entry_quantity"] for t in json.loads((tmp_path / "o2" / "summary.json").read_text())["trades"]} == {"1"}
    assert out["parity"]["all_within_tolerance"] is True
    assert out["parity"]["execution_label"] == "REPLAY" and out["parity"]["provenance"] == out["provenance"]
    assert hashlib.sha256(db_path.read_bytes()).hexdigest() == before  # the operator's active policy never changes
    reader_store = PolicyStore(Database(db_path), TradingMode.PAPER)
    assert reader_store.active().version == active.version


def test_cli_refuses_unknown_strategy_and_invalid_policy(tmp_path: Path) -> None:
    assert main(["evaluate", "--strategy", "nope", "--synthetic", "--out", str(tmp_path)]) == 2
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({**json.loads(RiskPolicy().model_dump_json()), "allow_margin": True}))
    assert main(["evaluate", "--strategy", "tod-SPY-1000-60m", "--policy", str(bad), "--synthetic",
                 "--out", str(tmp_path)]) == 2


async def test_paper_comparison_hook_on_a_real_schema_paper_ledger(tmp_path: Path) -> None:
    rig = build_rig(tmp_path / "paper")  # real PAPER-mode ledger + FakeBroker at 2026-11-05 10:00 New York
    assert (await rig.reconciler.startup_recovery()).status.value == "CLEAN"
    out = await rig.buy("2", stop="490.00")
    assert out["outcome"] == "SUBMITTED"
    await rig.tick(2)
    rig.db.close()
    paper_db = tmp_path / "paper" / "var" / "paper" / "ledger.sqlite"
    before = hashlib.sha256(paper_db.read_bytes()).hexdigest()
    snap = snapshot(sessions=2, start=date(2026, 11, 4))  # synthetic (the period lies in the research holdout)
    _, res = await replay(tmp_path / "replay", snap, config("tod-SPY-1000-60m"))
    exp = export_paper_ledger(paper_db, datetime(2026, 11, 5, tzinfo=UTC), datetime(2026, 11, 6, tzinfo=UTC))
    rep = compare_paper(exp, res.records, match_window_s=120)
    assert hashlib.sha256(paper_db.read_bytes()).hexdigest() == before
    assert rep["paper_legs"] == 1 and rep["matched_legs"] == 1
    m = rep["matched"][0]
    assert m["side"] == "BUY" and m["paper_avg_price"] == "500.04" and abs(m["time_delta_s"]) <= 2
    assert m["quantity_delta"] == "0" and rep["entries_paper_only"] == []
    assert [e["side"] for e in rep["exits_replay_only"]] == ["SELL"]  # the replay's TIME exit; paper still holds
