"""Why did the system not trade? Attribution of every NO_TRADE decision cycle (roadmap Phase 4).

For each decision cycle the top-ranked signal (the best opportunity the engine saw) is decomposed with the shared
decision contract (``services.research.decision``): gross forecast, live round-trip cost, uncertainty buffer and
threshold. The primary cause is the first of these that applies:

``LOCKED``            entries were locked (state, halt, unprotected position) - no opportunity could be taken;
``DATA``              a data/eligibility blocker (stale, illiquid/unknown liquidity, gap, extreme move, horizon past close);
``NO_FORECAST``       the model produced no usable prediction;
``PREDICTION``        the gross forecast was not positive (no predicted opportunity at all);
``COSTS``             gross > threshold, but the round trip consumed the edge;
``UNCERTAINTY``       net of costs cleared the threshold, the explicit uncertainty buffer did not;
``THRESHOLD``         a small positive net edge below the threshold;
``RISK``              a candidate was proposed and the independent risk engine rejected it (failed checks counted);
``NO_SIGNALS``        nothing was scored (no eligible symbols/data).

The report counts causes, lists near misses (closest to the threshold) and gives cost/edge distributions. It reads
only persisted records (``quant_signals``, ``risk_decisions``, ``audit_events``); it never changes a threshold.
"""

from __future__ import annotations

import json
from collections import Counter
from decimal import Decimal
from statistics import median
from typing import Any

DATA_MARKERS = ("stale data", "liquidity", "ADV $", "data gap", "extreme 1-bar move", "horizon", "insufficient history")
CAUSES = ("LOCKED", "DATA", "NO_FORECAST", "PREDICTION", "COSTS", "UNCERTAINTY", "THRESHOLD", "RISK", "NO_SIGNALS")


def _d(v: Any) -> float:
    try:
        return float(Decimal(str(v)))
    except Exception:
        return float("nan")


def classify_signal(sig: dict[str, Any], *, threshold_bps: float) -> str:
    reason = str(sig.get("reason") or "")
    if any(m in reason for m in DATA_MARKERS):
        return "DATA"
    if "no prediction" in reason or "no expected return" in reason or "output invalid" in reason:
        return "NO_FORECAST"
    gross = _d(sig.get("expected_return_bps"))
    cost = _d(sig.get("est_spread_bps")) + _d(sig.get("est_fees_bps")) + _d(sig.get("est_slippage_bps"))
    unc = _d(sig.get("uncertainty_bps") or 0)
    if not gross > 0:
        return "PREDICTION"
    net = gross - cost
    if gross > threshold_bps and net <= threshold_bps:
        return "COSTS"
    if net > threshold_bps and net - unc <= threshold_bps:
        return "UNCERTAINTY"
    return "THRESHOLD"


def attribute_no_trades(db: Any, *, since: str | None = None, threshold_bps: float = 5.0, limit: int = 5000
                        ) -> dict[str, Any]:
    """Attribution over persisted cycles (``since`` = ISO UTC lower bound on signal generation time)."""
    params: tuple[Any, ...] = (since or "", limit)
    rows = db.query("SELECT causal_id, payload_json FROM quant_signals WHERE generated_at >= ? "
                    "ORDER BY generated_at DESC LIMIT ?", params)
    by_cycle: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_cycle.setdefault(str(r["causal_id"]), []).append(json.loads(r["payload_json"]))
    cycles = db.query("SELECT causal_id, payload_json FROM audit_events WHERE event_type='DECISION_CYCLE' AND ts >= ? "
                      "ORDER BY ts DESC LIMIT ?", params)
    causes: Counter[str] = Counter()
    risk_checks: Counter[str] = Counter()
    near: list[dict[str, Any]] = []
    costs: list[float] = []
    gross_pos: list[float] = []
    traded = 0
    for c in cycles:
        brief = json.loads(c["payload_json"])
        outcome = str(brief.get("outcome") or "")
        cid = str(c["causal_id"] or brief.get("causal_id") or "")
        if outcome == "SUBMITTED":
            traded += 1
            continue
        sigs = sorted(by_cycle.get(cid, []), key=lambda s: int(s.get("rank") or 10**6))
        if outcome == "REJECTED":
            causes["RISK"] += 1
            for d in db.query("SELECT payload_json FROM risk_decisions WHERE causal_id=?", (cid,)):
                for chk in json.loads(d["payload_json"]).get("checks", []):
                    if not chk.get("passed"):
                        risk_checks[str(chk.get("name"))] += 1
            continue
        reason = str(brief.get("reason") or "")
        if "locked" in reason.lower() and not sigs:
            causes["LOCKED"] += 1
            continue
        if not sigs:
            causes["NO_SIGNALS"] += 1
            continue
        top = sigs[0]
        cause = classify_signal(top, threshold_bps=threshold_bps)
        causes[cause] += 1
        gross = _d(top.get("expected_return_bps"))
        cost = _d(top.get("est_spread_bps")) + _d(top.get("est_fees_bps")) + _d(top.get("est_slippage_bps"))
        if cost == cost:
            costs.append(cost)
        if gross > 0:
            gross_pos.append(gross)
        near.append({"causal_id": cid, "symbol": top.get("symbol"), "cause": cause, "gross_bps": round(gross, 3),
                     "cost_bps": round(cost, 3), "net_edge_bps": round(_d(top.get("net_edge_bps")), 3),
                     "generated_at": top.get("generated_at")})
    near.sort(key=lambda x: -x["net_edge_bps"] if x["net_edge_bps"] == x["net_edge_bps"] else 1e9)
    total = sum(causes.values())
    return {
        "cycles": total + traded, "submitted": traded, "no_trade": total,
        "causes": {k: causes.get(k, 0) for k in CAUSES},
        "cause_share": {k: round(causes.get(k, 0) / total, 4) if total else 0.0 for k in CAUSES},
        "risk_rejection_checks": dict(risk_checks.most_common(10)),
        "near_misses": near[:10], "threshold_bps": threshold_bps,
        "median_round_trip_cost_bps": round(median(costs), 3) if costs else None,
        "median_positive_gross_bps": round(median(gross_pos), 3) if gross_pos else None,
        "note": "NO_TRADE without a demonstrated edge is correct behaviour; this report explains it, it never lowers "
                "a threshold or a risk limit",
    }
