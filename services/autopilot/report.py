"""Daily autopilot report: ``<report_dir>/YYYY-MM-DD.json`` and ``.md``.

Assembled from the autopilot's own store (jobs, alerts, crew runs, promotions) and read-only tool outputs
(session report, performance, orders, research results). Every number carries its provenance: the execution
labels and the ``is_simulated`` flag of the outputs it came from. A simulated day is labelled SIMULATED at the top.
Backtest and shadow figures are research artifacts, never realized P&L.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Final, cast

SCHEMA: Final = "lat-autopilot-report-v1"


def _dict(v: Any) -> dict[str, Any]:
    return cast(dict[str, Any], v) if isinstance(v, dict) else {}


def _list(v: Any) -> list[Any]:
    return cast(list[Any], v) if isinstance(v, list) else []


def build(*, day: str, now: datetime, final: bool, jobs: list[dict[str, Any]], alerts: list[dict[str, Any]],
          session_report: dict[str, Any] | None, performance: dict[str, Any] | None, orders: list[dict[str, Any]],
          order_output: dict[str, Any] | None, simulated_feed: bool,
          research: dict[str, Any] | None, shadow: dict[str, Any] | None, standdown: dict[str, Any] | None,
          trading_day: dict[str, Any] | None, errors: list[str]) -> dict[str, Any]:
    by_job = {j["job"]: j for j in jobs}
    pre = _dict(by_job.get("preopen", {}).get("detail"))
    pre_provenance = _dict(pre.get("provenance"))
    feed_mode = pre_provenance.get("feed_mode") or ("SIMULATED" if simulated_feed else None)
    outputs = [o for o in (pre, pre_provenance, session_report, performance, order_output, research) if o]
    labels = {str(o["execution_label"]) for o in (*outputs, *orders) if o.get("execution_label")}
    labels.update(str(label) for label in _list(pre_provenance.get("execution_labels")))
    simulated = feed_mode == "SIMULATED" or "SIMULATED" in labels or any(bool(o.get("is_simulated")) for o in outputs)
    sess = _dict(_dict(session_report).get("session"))
    rejections = _list(_dict(session_report).get("top_risk_rejections"))
    perf = _dict(performance)
    crew = _dict(by_job.get("crew", {}).get("detail"))
    holdout = _dict(by_job.get("holdout", {}).get("detail"))
    promo_checks = _list(_dict(_dict(shadow).get("equity")).get("promotion_checks")) + \
        _list(_dict(_dict(shadow).get("crypto")).get("promotion_checks"))
    families = [{k: _dict(f).get(k) for k in ("family", "passed_dev_gate", "n_trials", "dsr", "registry_state")}
                for f in _list(_dict(research).get("families"))]
    return {
        "schema": SCHEMA, "day": day, "generated_at": now.isoformat(), "status": "FINAL" if final else "PARTIAL",
        "simulated": simulated,
        "data_provenance": {
            "execution_labels": sorted(labels), "is_simulated": simulated,
            "feed_mode": feed_mode,
            "feed": pre_provenance.get("feed"),
            "note": ("SIMULATED: simulated execution or synthetic data." if simulated
                     else "PAPER: Alpaca paper execution." if "PAPER" in labels else "Execution provenance unavailable."),
        },
        "trading_day": trading_day,
        "session": {
            "preopen_status": by_job.get("preopen", {}).get("status"), "outcome": pre.get("outcome"),
            "reasons": pre.get("blocked_reasons") or pre.get("reason"), "session_id": pre.get("session_id"),
            "state": sess.get("state"), "started_at": sess.get("started_at"), "ends_at": sess.get("ends_at"),
            "cycles": _dict(session_report).get("cycles"), "decisions": _dict(session_report).get("decisions"),
            "top_no_trade_reasons": _list(_dict(session_report).get("top_no_trade_reasons"))[:5],
            "narrative": _dict(session_report).get("narrative"),
        },
        "equity": {"portfolio_value": perf.get("portfolio_value"), "cash": perf.get("cash"),
                   "mode_label": perf.get("mode_label")},
        "pnl": {k: perf.get(k) for k in ("daily_pnl", "total_pnl", "realized_pnl", "unrealized_pnl", "max_drawdown")},
        "orders": {"count": len(orders), "items": orders[:50]},
        "rejections_by_check": {str(_dict(r).get("check")): _dict(r).get("count") for r in rejections},
        "research": {
            "quantlab_daily": by_job.get("quantlab_daily", {}).get("status"),
            "quantlab_detail": _dict(by_job.get("quantlab_daily", {}).get("detail")),
            "families": families,
            "note": "development-period backtests after costs; not validation and not realized results",
        },
        "crew_runs": {"status": by_job.get("crew", {}).get("status"), "ticks_done": crew.get("ticks_done", 0),
                      "ticks": _list(crew.get("ticks"))[:12],
                      "note": "agent notes are proposals; only the trial ledger's gates decide"},
        "promotions": {
            "holdout": {"status": by_job.get("holdout", {}).get("status"), **holdout},
            "shadow": by_job.get("shadow", {}).get("status"),
            "shadow_promotion_checks": promo_checks[:20],
            "note": "holdout looks and SHADOW->PAPER transitions are decided by the quantlab vault and registry gates",
        },
        "standdown": standdown,
        "alerts": [{"ts": a.get("ts"), "job": a.get("job"), "message": a.get("message")} for a in alerts][-50:],
        "jobs": [{"job": j["job"], "status": j["status"], "attempts": j["attempts"], "updated_at": j["updated_at"]}
                 for j in jobs],
        "errors": errors[:20],
    }


def markdown(rep: dict[str, Any]) -> str:
    prov = _dict(rep.get("data_provenance"))
    sess = _dict(rep.get("session"))
    lines = [f"# Autopilot daily report {rep['day']} ({rep['status']})", ""]
    if rep.get("simulated"):
        lines += ["> **SIMULATED** · This report includes simulated execution or synthetic data.", ""]
    lines += [f"- Execution labels: {', '.join(prov.get('execution_labels') or []) or 'unknown'}; "
              f"simulated: {prov.get('is_simulated')}; feed: {prov.get('feed_mode')} {prov.get('feed') or ''}".rstrip(),
              f"- Generated: {rep['generated_at']}", ""]
    lines += ["## Session", f"- Pre-open: {sess.get('preopen_status')} / {sess.get('outcome')}",
              f"- Session: {sess.get('session_id')} state {sess.get('state')} "
              f"({sess.get('started_at')} -> {sess.get('ends_at')}), cycles {sess.get('cycles')}"]
    if sess.get("reasons"):
        lines.append(f"- Reasons: {sess.get('reasons')}")
    eq, pnl = _dict(rep.get("equity")), _dict(rep.get("pnl"))
    lines += ["", "## Equity and P&L", f"- Portfolio value {eq.get('portfolio_value')} ({eq.get('mode_label')}), "
              f"cash {eq.get('cash')}",
              f"- Daily {pnl.get('daily_pnl')}, total {pnl.get('total_pnl')}, realized {pnl.get('realized_pnl')}, "
              f"unrealized {pnl.get('unrealized_pnl')}", "", "## Orders and rejections",
              f"- Orders: {_dict(rep.get('orders')).get('count')}"]
    for check, n in _dict(rep.get("rejections_by_check")).items():
        lines.append(f"- Rejected by `{check}`: {n}")
    res, crew, promo = _dict(rep.get("research")), _dict(rep.get("crew_runs")), _dict(rep.get("promotions"))
    lines += ["", "## Research", f"- QuantLab daily: {res.get('quantlab_daily')}",
              f"- Crew: {crew.get('status')} ({crew.get('ticks_done')} ticks)",
              f"- Holdout: {_dict(promo.get('holdout')).get('status')} "
              f"{_dict(promo.get('holdout')).get('family') or ''}".rstrip(),
              f"- Shadow journal: {promo.get('shadow')}"]
    for f in _list(res.get("families"))[:12]:
        d = _dict(f)
        lines.append(f"  - {d.get('family')}: dev gate {'PASS' if d.get('passed_dev_gate') else 'fail'}, "
                     f"state {d.get('registry_state')}")
    alerts = _list(rep.get("alerts"))
    lines += ["", "## Alerts"] + ([f"- {_dict(a).get('ts')} {_dict(a).get('message')}" for a in alerts] or ["- none"])
    if rep.get("standdown"):
        lines += ["", "## Stand-down", f"- {rep['standdown']}"]
    lines += ["", "_Backtest and shadow figures are research artifacts, not realized results._", ""]
    return "\n".join(lines)


def write(report_dir: Path, rep: dict[str, Any]) -> tuple[Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    jp, mp = report_dir / f"{rep['day']}.json", report_dir / f"{rep['day']}.md"
    for path, text in ((jp, json.dumps(rep, indent=1, default=str, sort_keys=True)), (mp, markdown(rep))):
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=report_dir, delete=False) as file:
            tmp = Path(file.name)
            try:
                file.write(text)
                file.close()
                os.replace(tmp, path)
            finally:
                tmp.unlink(missing_ok=True)
    return jp, mp
