"""Prospective (forward-only) evaluation of pre-registered intraday specs — the only honest test for strategies whose
inputs have no usable history (news-confirmed ORB: Alpaca news on this plan starts ~2026-04, inside the protected
holdout).

* ``register`` freezes a list of specs once (append-only registry with code hash and timestamp). A spec only earns
  evidence for sessions that OPEN strictly after its registration time, so nothing it is scored on existed when it
  was chosen.
* ``record_session`` simulates every registered spec on one completed session using only that session's bars and
  pre-open news plus earlier non-holdout sessions (for the opening-range volume normalizer), and appends one row
  per spec to an append-only journal.
* ``summary`` reports per-spec forward performance and the paired news-vs-no-news difference (t-stat on daily
  differences); ``PROSPECTIVE_GATE`` sets the minimum evidence before a spec may even enter the development gate.
The protected holdout window is never simulated or scored here.
"""

from __future__ import annotations

import json
import math
import sqlite3
from contextlib import closing
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np

from .ledger import code_hash, spec_hash

PROSPECTIVE_GATE: Final[dict[str, float]] = {"min_sessions": 60, "min_net_sharpe": 0.5, "paired_t_min": 2.0}

DEFAULT_ORB_SPECS: Final[tuple[dict[str, Any], ...]] = tuple(
    {"or_minutes": 5, "n": 5, "min_relvol": 2.0, "stop": "or_low", "filter": "candle_vwap_spy", "target_r": 0.0,
     "side": "long", "catalyst": cat}
    for cat in ("none", "news")
)
"""The development-best ORB construction with and without pre-open news confirmation (paired comparison)."""


class ProspectiveJournal:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._db()) as db, db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS specs (spec_hash TEXT PRIMARY KEY, family TEXT NOT NULL, spec TEXT NOT NULL,
                    registered_at TEXT NOT NULL, code_hash TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS sessions (spec_hash TEXT NOT NULL, session TEXT NOT NULL, net REAL NOT NULL,
                    gross REAL NOT NULL, trades INTEGER NOT NULL, recorded_at TEXT NOT NULL,
                    PRIMARY KEY (spec_hash, session));
                CREATE TRIGGER IF NOT EXISTS specs_no_update BEFORE UPDATE ON specs
                    BEGIN SELECT RAISE(ABORT, 'registered specs are immutable'); END;
                CREATE TRIGGER IF NOT EXISTS specs_no_delete BEFORE DELETE ON specs
                    BEGIN SELECT RAISE(ABORT, 'registered specs are immutable'); END;
                CREATE TRIGGER IF NOT EXISTS sessions_no_update BEFORE UPDATE ON sessions
                    BEGIN SELECT RAISE(ABORT, 'prospective results are append-only'); END;
                CREATE TRIGGER IF NOT EXISTS sessions_no_delete BEFORE DELETE ON sessions
                    BEGIN SELECT RAISE(ABORT, 'prospective results are append-only'); END;
            """)

    def _db(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        return db

    def register(self, family: str, specs: list[dict[str, Any]] | tuple[dict[str, Any], ...],
                 now: datetime | None = None) -> list[str]:
        now = now or datetime.now(UTC)
        out = []
        with closing(self._db()) as db, db:
            for sp in specs:
                h = spec_hash(sp)
                if db.execute("SELECT 1 FROM specs WHERE spec_hash=?", (h,)).fetchone() is None:
                    db.execute("INSERT INTO specs VALUES (?,?,?,?,?)",
                               (h, family, json.dumps(sp, sort_keys=True), now.isoformat(), code_hash()))
                out.append(h)
        return out

    def specs(self) -> list[dict[str, Any]]:
        with closing(self._db()) as db:
            rows = [dict(r) for r in db.execute("SELECT * FROM specs ORDER BY registered_at, spec_hash")]
        for r in rows:
            r["spec"] = json.loads(r["spec"])
        return rows

    def eligible(self, spec_row: dict[str, Any], session_open: datetime) -> bool:
        """A session counts only if it opened strictly after the spec was registered."""
        return session_open > datetime.fromisoformat(spec_row["registered_at"])

    def recorded(self, spec_hash_: str) -> set[str]:
        with closing(self._db()) as db:
            return {r[0] for r in db.execute("SELECT session FROM sessions WHERE spec_hash=?", (spec_hash_,))}

    def record(self, spec_hash_: str, session: date, net: float, gross: float, trades: int) -> bool:
        with closing(self._db()) as db, db:
            try:
                db.execute("INSERT INTO sessions VALUES (?,?,?,?,?,?)",
                           (spec_hash_, session.isoformat(), float(net), float(gross), int(trades),
                            datetime.now(UTC).isoformat()))
            except sqlite3.IntegrityError:
                return False
        return True

    def series(self, spec_hash_: str) -> dict[str, float]:
        with closing(self._db()) as db:
            return {r[0]: float(r[1]) for r in db.execute(
                "SELECT session, net FROM sessions WHERE spec_hash=? ORDER BY session", (spec_hash_,))}

    def summary(self) -> dict[str, Any]:
        specs = self.specs()
        out: dict[str, Any] = {"gate": PROSPECTIVE_GATE, "specs": []}
        series = {s["spec_hash"]: self.series(s["spec_hash"]) for s in specs}
        for s in specs:
            r = np.array(list(series[s["spec_hash"]].values()), dtype=float)
            n = len(r)
            sd = float(r.std(ddof=1)) if n > 1 else 0.0
            out["specs"].append({
                "spec_hash": s["spec_hash"], "family": s["family"], "spec": s["spec"],
                "registered_at": s["registered_at"], "sessions": n,
                "net_total": float(np.prod(1 + r) - 1) if n else 0.0,
                "net_sharpe": float(r.mean() / sd * math.sqrt(252)) if sd > 0 else 0.0,
                "meets_gate": bool(n >= PROSPECTIVE_GATE["min_sessions"] and sd > 0
                                   and r.mean() / sd * math.sqrt(252) >= PROSPECTIVE_GATE["min_net_sharpe"]),
            })
        # paired news-vs-none comparison for otherwise identical specs
        by_key: dict[str, dict[str, str]] = {}
        for s in specs:
            base = {k: v for k, v in s["spec"].items() if k != "catalyst"}
            by_key.setdefault(json.dumps(base, sort_keys=True), {})[str(s["spec"].get("catalyst"))] = s["spec_hash"]
        pairs = []
        for base, m in by_key.items():
            if "news" in m and "none" in m:
                a, b = series[m["news"]], series[m["none"]]
                common = sorted(set(a) & set(b))
                d = np.array([a[k] - b[k] for k in common], dtype=float)
                t = float(d.mean() / (d.std(ddof=1) / math.sqrt(len(d)))) if len(d) > 2 and d.std(ddof=1) > 0 else 0.0
                pairs.append({"base": json.loads(base), "sessions": len(d),
                              "mean_daily_diff": float(d.mean()) if len(d) else 0.0, "paired_t": t,
                              "news_adds_value": bool(len(d) >= PROSPECTIVE_GATE["min_sessions"]
                                                      and t >= PROSPECTIVE_GATE["paired_t_min"])})
        out["paired_news_vs_none"] = pairs
        return out
