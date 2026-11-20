"""SIMULATED live path through the REAL self-hosted LEAN engine (live mode, GatedPaperBrokerage, simulated data queue).

Each test starts the LEAN Launcher as a subprocess against the 127.0.0.1 stub bridge (no credentials, no network).
Skipped cleanly when LEAN has not been built (scripts/lean_build.sh).
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest

from tests.integration.lean import live_sim_harness as harness
from trading.lean.tools import lat_lean

pytestmark = [
    pytest.mark.lean,
    pytest.mark.slow,
    pytest.mark.skipif(not lat_lean.lean_built(), reason="LEAN not built (run scripts/lean_build.sh)"),
]


def _events(report: dict[str, Any], coid: str) -> list[dict[str, Any]]:
    return [e for e in report["stub"]["events"] if e.get("client_order_id") == coid]


@pytest.fixture(scope="module")
def bridge_report() -> dict[str, Any]:
    return harness.run("bridge", exit_after_s=25, timeout_s=120)


@pytest.fixture(scope="module")
def rogue_report() -> dict[str, Any]:
    if not harness.lat_lean.TEST_ALGOS_DLL.is_file():
        pytest.skip("LatTestAlgorithms not built")
    return harness.run("rogue", exit_after_s=20, timeout_s=100)


@pytest.mark.timeout(240)
def test_signed_intent_is_claimed_submitted_filled_and_reported(bridge_report: dict[str, Any]) -> None:
    rep = bridge_report
    assert rep["execution_label"] == "SIMULATED"
    assert rep["completed"], f"scenario did not complete; see {rep['launcher_log']}"
    assert rep["clean_stop"], "LEAN did not stop via the configured run time"
    valid = next(x for x in rep["expectations"] if x["label"] == "valid_buy")
    coid = valid["client_order_id"]
    claimed = [i for i in rep["stub"]["intents"] if i["client_order_id"] == coid and i["label"] == "valid_buy"]
    assert claimed and claimed[0]["claimed_by"], "intent was never claimed by LEAN"

    evs = _events(rep, coid)
    statuses = [e["status"] for e in evs]
    assert "Submitted" in statuses and statuses[-1] == "Filled", statuses
    fill = next(e for e in evs if e["status"] == "Filled")
    assert fill["intent_id"] == valid["intent_id"]
    assert Decimal(fill["fill_quantity"]) == Decimal(valid["quantity"]) == Decimal("10")
    assert Decimal(fill["cum_filled_quantity"]) == Decimal("10")
    # LatFillModel: base fill + half-spread + impact, rounded adversely to the tick, never above the signed limit
    px = Decimal(fill["fill_price"])
    assert Decimal("450.00") < px <= Decimal(valid["limit_price"])
    assert fill["broker_order_id"], "PaperBrokerage broker id must be reported"
    assert isinstance(fill["lean_order_id"], int) and isinstance(fill["lean_event_id"], int)
    assert fill["event_ts"].endswith("Z")
    # exactly one LEAN order ever carried this client order id (the duplicate delivery was not resubmitted)
    assert {e["lean_order_id"] for e in evs} == {fill["lean_order_id"]}
    assert not rep["stub"]["protocol_errors"], rep["stub"]["protocol_errors"]
    assert rep["stub"]["auth_failures"] == 0


@pytest.mark.timeout(240)
def test_engine_talks_to_loopback_only(bridge_report: dict[str, Any]) -> None:
    socks = bridge_report["process_sockets"]
    assert any("127.0.0.1" in s for s in socks), "lsof sampling saw no bridge connection (sampler broken?)"
    assert bridge_report["non_loopback_sockets"] == [], bridge_report["non_loopback_sockets"]


@pytest.mark.timeout(240)
@pytest.mark.parametrize("label", ["tampered_quantity", "forged_other_key", "expired", "short_sale",
                                   "signed_market_order", "wrong_gateway"])
def test_invalid_intents_are_refused_by_the_gate_and_reported(bridge_report: dict[str, Any], label: str) -> None:
    x = next(e for e in bridge_report["expectations"] if e["label"] == label)
    rej = bridge_report["stub"]["rejections"]
    assert harness.refusal_found(rej, client_order_id=x["client_order_id"], layer="[GatedBrokerageModel]",
                                 reason=x["reason_contains"]), (label, rej)
    evs = _events(bridge_report, x["client_order_id"])
    assert evs and all(e["status"] == "Invalid" for e in evs), evs
    assert all(e["fill_quantity"] is None for e in evs)


@pytest.mark.timeout(240)
def test_duplicate_delivery_is_not_resubmitted_and_report_is_unkeyed(bridge_report: dict[str, Any]) -> None:
    x = next(e for e in bridge_report["expectations"] if e["label"] == "duplicate_delivery")
    assert x["keyed"] is False
    assert harness.refusal_found(bridge_report["stub"]["rejections"], reason=x["reason_contains"], unkeyed=True)


@pytest.mark.timeout(240)
def test_heartbeats_and_assembly_identity(bridge_report: dict[str, Any]) -> None:
    hbs = bridge_report["stub"]["heartbeats"]
    assert len(hbs) >= 3
    assert {h["gateway"] for h in hbs} == {"LEAN_PAPER_SIM"} and {h["mode"] for h in hbs} == {"PAPER"}
    assert hbs[-1]["status"] == "STOPPED"
    assert any(h["status"] == "RUNNING" for h in hbs)
    assert hbs[0]["lean_commit"] == lat_lean.pins()["lean"]["commit"]
    checks = bridge_report["assembly_check"]
    common = [c for c in checks if c.startswith("QuantConnect.Common ")]
    assert common and "copies=1" in common[0] and "/Lean/Launcher/bin/Release/" in common[0]
    assert any(c.startswith("QuantConnect.Brokerages.Alpaca copies=0") for c in checks)


@pytest.mark.timeout(240)
def test_strategy_originated_orders_are_refused_at_both_boundaries(rogue_report: dict[str, Any]) -> None:
    rep = rogue_report
    assert rep["completed"], f"rogue scenario did not complete; see {rep['launcher_log']}"
    rej = rep["stub"]["rejections"]
    for layer, why, order_type in harness.ROGUE_REFUSALS:
        assert harness.refusal_found(rej, layer=layer, reason=why, order_type=order_type), (layer, why, rej)
    assert harness.refusal_found(rej, layer="[LatIntentBridgeAlgorithm]", reason="brokerage model replaced")
    assert not any(e.get("status") in ("Filled", "PartiallyFilled", "Submitted") for e in rep["stub"]["events"])
    # the strategy's own orders carry no intent ids, so nothing could be attributed to a ledger order
    gate_rej = [r for r in rej if "unsigned order" in str(r.get("reason"))]
    assert gate_rej and all(r["client_order_id"] is None and r["intent_id"] is None for r in gate_rej)
    assert any(h["status"] == "HALTED" for h in rep["stub"]["heartbeats"])


@pytest.mark.timeout(240)
def test_engine_payloads_satisfy_the_order_ledger_bridge_contract(bridge_report: dict[str, Any]) -> None:
    """Contract check against the consumer (services.order_ledger.bridge, read-only use): every payload the real
    engine sent parses there (no floats, required fields) and every LEAN status/heartbeat status is known."""
    import json

    from services.order_ledger.bridge import HEALTHY_ENGINE_STATUSES, LEAN_STATUS_MAP, parse_engine_payload

    def wire(d: dict[str, Any]) -> str:
        return json.dumps({k: v for k, v in d.items() if not k.startswith("_")})

    for e in bridge_report["stub"]["events"]:
        data = parse_engine_payload(wire(e))
        assert data["status"].replace("_", "").lower() in LEAN_STATUS_MAP
        for key in ("engine_id", "client_order_id", "intent_id", "event_ts"):
            assert isinstance(data[key], str) and data[key]
    for r in bridge_report["stub"]["rejections"]:
        assert parse_engine_payload(wire(r))["reason"]
    statuses = {h["status"] for h in bridge_report["stub"]["heartbeats"]}
    assert statuses <= {"STARTING", "RUNNING", "HALTED", "STOPPED"}
    assert "RUNNING" in HEALTHY_ENGINE_STATUSES and not ({"STARTING", "HALTED", "STOPPED"} & HEALTHY_ENGINE_STATUSES)
