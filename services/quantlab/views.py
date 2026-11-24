"""Read-only views of quantlab artifacts for the API, dashboard and agent tools (files only; no computation that
could touch the holdout, no broker access). ``run_agent_experiment`` launches a bounded development-only
evaluation of a declarative factor spec in a separate process."""

from __future__ import annotations

import json
import subprocess
import sys
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .factors import FAMILY as AGENT_FAMILY
from .factors import SpecError, validate_spec
from .paths import results_root

KINDS = ("capabilities", "research", "opportunities", "strategies", "trials", "options")
AGENT_EXPERIMENTS_PER_HOUR = 6
EXPERIMENT_TIMEOUT_S = 600


def _read(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def capabilities(root: Path) -> dict[str, Any]:
    snap = _read(root / "datasets" / "quantlab" / "capabilities" / "latest.json")
    if not snap:
        return {"status": "MISSING", "note": "run `python -m services.capabilities probe`"}
    acct = snap.get("account", {})
    return {
        "status": "OK", "asof": snap.get("asof"), "content_hash": snap.get("content_hash"),
        "account": {k: acct.get(k) for k in ("status", "type", "crypto_status", "crypto_tier", "options_trading_level",
                                             "multiplier", "shorting_enabled", "pdt_fields_reported", "pdt_note")},
        "assets": {k: snap.get("assets", {}).get(k) for k in ("us_equity_tradable", "us_equity_inactive",
                                                              "by_kind_tradable", "crypto_pairs")},
        "option_underlyings": {u: {k: v.get(k) for k in ("contracts_first_page", "roots", "styles")}
                               for u, v in (snap.get("option_underlyings") or {}).items()},
        "market_data": snap.get("market_data"),
        "asset_classes": snap.get("asset_classes"),
    }


def research(root: Path, family: str | None = None) -> dict[str, Any]:
    out = results_root(root)
    latest = _read(out / "latest_research.json")
    if not latest:
        return {"status": "MISSING", "families": [], "note": "run `python -m services.quantlab.cli research`"}
    fams = latest.get("families", {})
    if family:
        info = fams.get(family)
        if not info:
            return {"status": "UNKNOWN_FAMILY", "families": sorted(fams)}
        detail = _read(out / "runs" / str(info.get("run_id")) / f"{family}.json") or {}
        detail.pop("variants", None)
        return {"status": "OK", "summary": info, "detail": detail}
    looks = []
    holdouts = out / "holdout_results"
    if holdouts.is_dir():
        looks = [r for p in sorted(holdouts.glob("*.json")) if (r := _read(p))]
    return {"status": "OK", "updated_at": latest.get("updated_at"), "holdout": latest.get("holdout"),
            "gate": latest.get("gate"), "no_edge_note": latest.get("no_edge_note"),
            "families": sorted(fams.values(), key=lambda f: (not f.get("passed_dev_gate"), -(f.get("dsr") or 0))),
            "holdout_results": looks}


def opportunities(root: Path) -> dict[str, Any]:
    out = results_root(root)
    scan = _read(out / "latest_scan.json") or {}
    sel = _read(out / "latest_selection.json") or {}
    options = _read(out / "latest_options_scan.json") or {}
    crypto = _read(out / "latest_crypto_scan.json") or {}
    return {"status": "OK" if scan or sel else "MISSING", "scan": scan, "selection": sel, "options": options,
            "crypto": crypto}


def strategies(root: Path) -> dict[str, Any]:
    from .registry import StrategyRegistry

    path = results_root(root) / "registry.sqlite3"
    if not path.is_file():
        return {"status": "MISSING", "strategies": [], "transitions": []}
    reg = StrategyRegistry(path)
    rows = reg.list()
    for r in rows:
        r["shadow_sessions"] = reg.shadow_sessions(r["strategy_id"])
    return {"status": "OK", "strategies": rows, "transitions": reg.transitions(limit=50)}


def trials(root: Path, family: str | None = None, limit: int = 200) -> dict[str, Any]:
    from .ledger import TrialLedger

    path = results_root(root) / "trials.sqlite3"
    if not path.is_file():
        return {"status": "MISSING", "trials": []}
    led = TrialLedger(path)
    rows = led.rows(family, limit)
    for r in rows:
        r.pop("spec", None)
        m = json.loads(r.pop("metrics") or "{}")
        r.update({k: m.get(k) for k in ("cagr", "max_drawdown", "turnover_annual", "cost_drag_annual")})
    counts: dict[str, int] = {}
    with closing(__import__("sqlite3").connect(path)) as db:
        for fam, n in db.execute("SELECT family, COUNT(*) FROM trials WHERE stage='DEV' GROUP BY family"):
            counts[str(fam)] = int(n)
    return {"status": "OK", "trial_counts": counts, "trials": rows}


def view(root: Path, kind: str, family: str | None = None) -> dict[str, Any]:
    if kind == "capabilities":
        return capabilities(root)
    if kind == "research":
        return research(root, family)
    if kind == "opportunities":
        return opportunities(root)
    if kind == "strategies":
        return strategies(root)
    if kind == "trials":
        return trials(root, family)
    if kind == "options":
        return _read(results_root(root) / "latest_options_scan.json") or {"status": "MISSING"}
    raise KeyError(kind)


def run_agent_experiment(root: Path, spec: dict[str, Any], actor: str) -> dict[str, Any]:
    """Development-only evaluation of an agent factor spec (separate process, timeout, hourly budget)."""
    try:
        clean = validate_spec(spec)
    except SpecError as exc:
        return {"accepted": False, "error": str(exc)}
    from .ledger import TrialLedger

    led = TrialLedger(results_root(root) / "trials.sqlite3")
    since = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
    recent = [r for r in led.rows(AGENT_FAMILY, 50) if r["at"] >= since and r["stage"] == "DEV"]
    if len(recent) >= AGENT_EXPERIMENTS_PER_HOUR:
        return {"accepted": False, "error": f"agent experiment budget exhausted ({AGENT_EXPERIMENTS_PER_HOUR}/hour)"}
    cmd = [sys.executable, "-m", "services.quantlab.cli", "experiment", "--spec", json.dumps(clean),
           "--actor", actor[:64]]
    try:
        proc = subprocess.run(cmd, cwd=root, capture_output=True, text=True, timeout=EXPERIMENT_TIMEOUT_S,
                              check=False)
    except subprocess.TimeoutExpired:
        return {"accepted": True, "status": "TIMEOUT", "spec": clean}
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
    if proc.returncode != 0 or not lines:
        return {"accepted": True, "status": "ERROR", "spec": clean, "error": proc.stderr[-400:]}
    return {"accepted": True, "status": "DONE", "spec": clean, "result": json.loads(lines[-1])}
