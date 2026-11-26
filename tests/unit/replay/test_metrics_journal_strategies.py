"""Replay metrics, the append-only journal, strategy ids and policy profiles (offline)."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from packages.contracts.enums import TradingMode
from packages.contracts.policy import RiskPolicy
from packages.core.db import Database
from services.replay.journal import Journal
from services.replay.metrics import summarize
from services.replay.policy import PolicyProfileError, load_policy
from services.replay.strategies import TimeOfDayModel, get_strategy, list_strategies
from services.risk.operator_auth import OperatorAuth
from services.risk.policy_store import PolicyStore


def _r(kind: str, at: str, **kw: Any) -> dict[str, Any]:
    return {"kind": kind, "at": at, **kw}


def test_summarize_computes_executable_portfolio_metrics() -> None:
    recs = [
        _r("MARK", "2024-03-05T15:30:00.000000Z", equity="5000", cash="4000", position_value="1000"),
        _r("FILL", "2024-03-05T15:00:01.000000Z", symbol="SPY", side="BUY", quantity="2", price="500"),
        _r("LEG_COST", "2024-03-05T15:00:01.000000Z", symbol="SPY", charged_usd="0.5", notional="1000"),
        _r("FILL", "2024-03-05T16:00:06.000000Z", symbol="SPY", side="SELL", quantity="2", price="505"),
        _r("LEG_COST", "2024-03-05T16:00:06.000000Z", symbol="SPY", charged_usd="0.5", notional="1010"),
        _r("MARK", "2024-03-05T21:00:00.000000Z", equity="5010", cash="5010", position_value="0", session_end=True),
        _r("MARK", "2024-03-06T21:00:00.000000Z", equity="4990", cash="4990", position_value="0", session_end=True),
        _r("REJECTION", "2024-03-06T15:00:00.000000Z", side="BUY", failed_checks=["max_concentration",
                                                                                  "max_allowable_quantity"]),
        _r("REJECTION", "2024-03-06T15:10:00.000000Z", side="SELL", failed_checks=["max_spread"]),
        _r("ORDER", "2024-03-06T15:00:00.000000Z", to_state="REJECTED", from_state="RISK_APPROVED",
           reason="final gate: policy changed"),
        _r("DECISION", "2024-03-05T15:00:00.000000Z", outcome="SUBMITTED"),
        _r("DECISION", "2024-03-06T15:00:00.000000Z", outcome="REJECTED"),
        _r("ROUND_TRIP", "2024-03-05T16:00:11.000000Z", plan_id="p1", symbol="SPY", opened_at="2024-03-05T15:00:01.000000Z",
           closed_at="2024-03-05T16:00:11.000000Z", entry_quantity="2", entry_avg_price="500", exit_avg_price="505",
           exit_reason="TIME", close_reason="TIME", realized_pl="10"),
    ]
    s = summarize(recs, equity=Decimal("5000"))
    assert s["sessions"] == 2 and s["modelled_costs_usd"] == "1.0"
    assert s["final_equity_net"] == "4989.0" and s["gross_pl"] == "-10" and s["net_pl"] == "-11.0"
    assert s["max_drawdown"] == pytest.approx(4989 / 5009 - 1, abs=1e-6)
    assert s["sharpe"] is not None and s["cagr"] is not None and s["cagr"] < 0
    assert s["turnover"]["traded_notional"] == "2010"
    assert s["cost_drag"]["bps_of_traded_notional"] == pytest.approx(1 / 2010 * 1e4, rel=1e-6)
    assert s["capital_exposure"]["max"] == pytest.approx(0.2)
    assert s["rejections_by_check"] == {"max_allowable_quantity": 1, "max_concentration": 1, "max_spread": 1}
    assert s["rejections_by_check_exits"] == {"max_spread": 1}
    assert s["order_rejections"] == {"final_gate: final gate: policy changed": 1}
    assert s["decisions"] == {"REJECTED": 1, "SUBMITTED": 1}
    (t,) = s["trades"]
    assert t["net_pl"] == "9.0" and t["gross_bps"] == pytest.approx(100.0)


def test_journal_is_append_only_and_resumable(tmp_path: Path) -> None:
    path = tmp_path / "j.jsonl"
    at = datetime(2024, 3, 5, 15, 0, tzinfo=UTC)
    j = Journal(path, {"dataset_version": "ds"})
    j.write("RUN_START", at, a=1, execution_label="PAPER", provenance={})
    j.write("MARK", at, equity="1")
    j.close()
    with path.open("a") as fh:
        fh.write('{"seq": 3, "kind": "TRUNC')  # a line cut by a crash
    j2 = Journal(path, {"dataset_version": "ds"})
    rec = j2.write("RESUMED", at)
    j2.close()
    rows = list(Journal.read(path))
    assert [r["seq"] for r in rows] == [1, 2, 3] and rec["seq"] == 3
    assert all(r["execution_label"] == "REPLAY" and r["is_simulated"] and r["provenance"] == {"dataset_version": "ds"}
               for r in rows)
    assert path.read_text().count("TRUNC") == 1 and j2.truncated_lines == 1  # never rewritten


def test_strategy_ids_and_time_of_day_model() -> None:
    s = get_strategy("tod-SPY-1000-60m")
    assert s.horizon_minutes == 60 and s.holding_sessions == 1 and s.decision_clocks == ("10:00",)
    m = get_strategy("tod-QQQ-1030-3d")
    assert m.holding_sessions == 3 and not m.horizon_within_session and m.proposal_strategy_id == "tod-QQQ-1030-3d"
    for bad in ("tod-SPY-1000-7m", "tod-SPY-1000-1d", "nope"):
        with pytest.raises((KeyError, ValueError)):
            get_strategy(bad)
    assert next(x["strategy_id"] for x in list_strategies()) == "runtime-baseline"
    model = s.model_factory()
    assert isinstance(model, TimeOfDayModel) and model.horizon_bars == 12
    ends = pd.to_datetime(["2024-03-05T14:55:00Z", "2024-03-05T15:00:00Z"], utc=True)
    df = pd.DataFrame({"symbol": ["SPY", "SPY"], "end_ts": ends})
    assert model.predict_latest(df.iloc[:1]) is None
    pred = model.predict_latest(df)
    assert pred is not None and pred.expected_return_bps > 0 and pred.basis == "GROSS"
    assert model.predict_latest(df.assign(symbol="QQQ")) is None


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_policy_profiles_default_file_and_active_read_only(tmp_path: Path) -> None:
    assert load_policy("default")[0] == RiskPolicy()
    prof = tmp_path / "p.json"
    prof.write_text(RiskPolicy(name="lean", max_order_notional=Decimal("1500.00")).model_dump_json())
    pol, src = load_policy(str(prof))
    assert pol.max_order_notional == Decimal("1500.00") and pol.version in src
    prof.write_text(json.dumps({**json.loads(RiskPolicy().model_dump_json()), "max_order_notional": "90000"}))
    with pytest.raises(PolicyProfileError, match=r"ceiling|max_order_notional"):
        load_policy(str(prof))
    with pytest.raises(PolicyProfileError, match="no paper ledger"):
        load_policy("active", project_root=tmp_path)
    db_path = tmp_path / "var" / "paper" / "ledger.sqlite"
    db = Database(db_path)
    auth = OperatorAuth(db_path.parent / "keys")
    auth.ensure_token()
    store = PolicyStore(db, TradingMode.PAPER, operator_auth=auth)
    store.ensure_default()
    custom = RiskPolicy(name="operator-active", max_positions=1, max_order_notional=Decimal("1800.00"))
    store.apply_change(custom, operator_token=auth.token_path.read_text().strip(), actor="operator", reason="test")
    db.close()
    before = _sha(db_path)
    pol, src = load_policy("active", project_root=tmp_path)
    assert pol == custom and "read-only" in src
    assert _sha(db_path) == before  # read through SQLite mode=ro: the operator's store is untouched
    escaped = tmp_path / "project#?"
    escaped.mkdir()
    (tmp_path / "var").rename(escaped / "var")
    assert load_policy("active", project_root=escaped)[0] == custom
    assert _sha(escaped / "var" / "paper" / "ledger.sqlite") == before


def test_journal_recovers_a_truncated_utf8_record(tmp_path: Path) -> None:
    path = tmp_path / "journal.jsonl"
    at = datetime(2024, 3, 5, 15, 0, tzinfo=UTC)
    j = Journal(path, {})
    j.write("RUN_START", at)
    j.close()
    with path.open("ab") as fh:
        fh.write(b'{"partial":"\xc3')
    before = path.read_bytes()
    j = Journal(path, {})
    j.write("RESUMED", at)
    j.close()
    assert path.read_bytes().startswith(before)
    assert [r["kind"] for r in Journal.read(path)] == ["RUN_START", "RESUMED"]
    assert j.truncated_lines == 1


def test_partial_summary_marks_equity_and_counts_each_bar_once() -> None:
    recs = [
        _r("MARK", "2024-03-05T20:55:00Z", equity="5010", position_value="1000"),
        _r("MARK", "2024-03-05T21:00:00Z", equity="5015", position_value="1000"),
        _r("MARK", "2024-03-05T21:00:00Z", equity="5015", position_value="1000", session_end=True),
        _r("MARK", "2024-03-06T15:05:00Z", equity="5020", position_value="0"),
    ]
    s = summarize(recs, equity=Decimal("5000"))
    assert s["final_equity_net"] == "5020"
    assert s["sessions"] == 1
    assert s["capital_exposure"]["fraction_of_bars_invested"] == pytest.approx(2 / 3, abs=1e-6)
