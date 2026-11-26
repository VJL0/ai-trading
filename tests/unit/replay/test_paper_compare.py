"""Paper comparison hook on a fixture ledger (minimal orders/fills tables; read-only access)."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from services.replay.__main__ import main
from services.replay.paper import compare_paper, export_paper_ledger


def _fixture_db(path: Path) -> None:
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE orders (client_order_id TEXT PRIMARY KEY, mode TEXT, gateway TEXT, symbol TEXT, side TEXT,
          quantity TEXT, order_type TEXT, limit_price TEXT, state TEXT, filled_quantity TEXT, avg_fill_price TEXT,
          created_at TEXT, submitted_at TEXT);
        CREATE TABLE fills (fill_id TEXT PRIMARY KEY, client_order_id TEXT, symbol TEXT, side TEXT, quantity TEXT,
          price TEXT, ts TEXT);
    """)
    rows = [
        ("lat-paper-a", "PAPER", "BUY", "LIMIT", "FILLED", "2024-03-05T15:00:00.000000Z"),
        ("lat-paper-b", "PAPER", "SELL", "LIMIT", "FILLED", "2024-03-05T16:00:00.000000Z"),
        ("lat-paper-c", "PAPER", "BUY", "LIMIT", "CANCELED", "2024-03-05T18:00:00.000000Z"),
        ("lat-sim-x", "REPLAY", "BUY", "LIMIT", "FILLED", "2024-03-05T15:00:00.000000Z"),
    ]
    for coid, mode, side, otype, state, at in rows:
        con.execute("INSERT INTO orders VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (coid, mode, "ALPACA_PAPER", "SPY", side, "2", otype, "500.00", state, "2", "500.00", at, at))
    con.execute("INSERT INTO fills VALUES ('f1','lat-paper-a','SPY','BUY','1','500.00','2024-03-05T15:00:02.000000Z')")
    con.execute("INSERT INTO fills VALUES ('f2','lat-paper-a','SPY','BUY','1','500.10','2024-03-05T15:00:03.000000Z')")
    con.execute("INSERT INTO fills VALUES ('f3','lat-paper-b','SPY','SELL','2','501.00','2024-03-05T16:00:09.000000Z')")
    con.execute("INSERT INTO fills VALUES ('f4','lat-sim-x','SPY','BUY','2','499.00','2024-03-05T15:00:01.000000Z')")
    con.commit()
    con.close()


def _replay_fill(coid: str, side: str, qty: str, px: str, at: str) -> dict[str, Any]:
    return {"kind": "FILL", "at": at, "client_order_id": coid, "symbol": "SPY", "side": side, "quantity": qty,
            "price": px, "order_type": "LIMIT"}


@pytest.mark.parametrize("filename", ["paper.sqlite", "paper#?ledger.sqlite"])
def test_export_is_read_only_and_filters_period_and_mode(tmp_path: Path, filename: str) -> None:
    db = tmp_path / filename
    _fixture_db(db)
    before = db.read_bytes()
    exp = export_paper_ledger(db, datetime(2024, 3, 5, tzinfo=UTC), datetime(2024, 3, 6, tzinfo=UTC))
    assert {o["client_order_id"] for o in exp.orders} == {"lat-paper-a", "lat-paper-b", "lat-paper-c"}
    assert {f["fill_id"] for f in exp.fills} == {"f1", "f2", "f3"}
    assert db.read_bytes() == before
    with pytest.raises(FileNotFoundError):
        export_paper_ledger(tmp_path / "missing.sqlite", datetime(2024, 3, 5, tzinfo=UTC), datetime(2024, 3, 6, tzinfo=UTC))


def test_compare_matches_legs_and_reports_one_sided_entries(tmp_path: Path) -> None:
    db = tmp_path / "paper.sqlite"
    _fixture_db(db)
    exp = export_paper_ledger(db, datetime(2024, 3, 5, tzinfo=UTC), datetime(2024, 3, 6, tzinfo=UTC))
    replay = [
        _replay_fill("lat-replay-1", "BUY", "2", "500.05", "2024-03-05T15:00:01.000000Z"),
        _replay_fill("lat-replay-2", "SELL", "2", "500.50", "2024-03-05T16:00:06.000000Z"),
        _replay_fill("lat-replay-3", "BUY", "2", "501.00", "2024-03-05T19:00:01.000000Z"),
        {"kind": "ORDER", "at": "2024-03-05T15:00:00.000000Z", "client_order_id": "lat-replay-1"},
    ]
    rep = compare_paper(exp, replay, match_window_s=120)
    assert rep["matched_legs"] == 2 and rep["paper_legs"] == 2 and rep["replay_legs"] == 3
    buy = next(m for m in rep["matched"] if m["side"] == "BUY")
    assert buy["paper_avg_price"] == "500.05" and buy["price_delta_bps"] == pytest.approx(0.0)
    assert buy["quantity_delta"] == "0"
    sell = next(m for m in rep["matched"] if m["side"] == "SELL")
    assert sell["price_delta_bps"] == pytest.approx((500.5 / 501.0 - 1) * 1e4, abs=1e-3)
    assert [e["client_order_id"] for e in rep["entries_replay_only"]] == ["lat-replay-3"]
    assert rep["entries_paper_only"] == [] and rep["paper_order_states"] == {"CANCELED": 1, "FILLED": 2}
    assert rep["replay_orders"] == 1 and "PAPER" in rep["labels"]["paper"]


def test_compare_paper_cli_includes_the_entire_last_date(tmp_path: Path) -> None:
    db = tmp_path / "paper.sqlite"
    _fixture_db(db)
    with closing(sqlite3.connect(db)) as conn, conn:
        conn.execute("UPDATE fills SET ts = '2024-03-05T23:59:59.500000Z' WHERE fill_id = 'f3'")
    out = tmp_path / "out"
    assert main(["compare-paper", "--strategy", "tod-SPY-1000-60m", "--synthetic", "--sessions", "2",
                 "--out", str(out), "--workspace", str(tmp_path / "workspace"), "--paper-db", str(db),
                 "--period-start", "2024-03-05", "--period-end", "2024-03-05"]) == 0
    report = json.loads((out / "paper_comparison.json").read_text())
    assert report["paper_legs"] == 2
    assert report["execution_label"] == "REPLAY" and report["provenance"]["quote_feed"] == "SYNTHETIC"
