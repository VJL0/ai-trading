"""Failure injection against the REAL self-hosted LEAN engine (live mode, SIMULATED PaperBrokerage + data queue).

* transient API outage on /engine/events  -> the fill event is retried and delivered, the local outbox keeps it;
* engine restart + API re-delivering an intent that was already executed -> refused by the durable intent journal
  (replay protection at the final boundary), reported WITHOUT ids, no order event posted under the original ids;
* wrong engine token -> nothing is claimed, no order is ever created;
* missing intent public key -> the engine refuses to start (fail closed).

Skipped when LEAN is not built. Everything here is SIMULATED and stays on 127.0.0.1.
"""

from __future__ import annotations

import json
import secrets
from collections.abc import Callable
from decimal import Decimal
from typing import Any

import pytest

from packages.contracts.enums import Side
from tests.integration.lean import live_sim_harness as harness
from tests.integration.lean.stub_bridge import EVENTS, StubBridge
from trading.lean.tools import lat_lean

pytestmark = [
    pytest.mark.lean,
    pytest.mark.slow,
    pytest.mark.skipif(not lat_lean.lean_built(), reason="LEAN not built (run scripts/lean_build.sh)"),
]


def _filled(coid: str) -> Callable[[dict[str, Any]], bool]:
    def pred(s: dict[str, Any]) -> bool:
        return any(e.get("status") == "Filled" and e.get("client_order_id") == coid for e in s["events"])
    return pred


@pytest.mark.timeout(300)
def test_event_outage_is_retried_and_restart_cannot_replay_an_executed_intent() -> None:
    run_id, run_dir, pub, signer = harness.new_sim_run("lean-fi-restart-a")
    journal = run_dir / "shared-used-intents.tsv"
    token = secrets.token_hex(24)
    buy = harness.make_intent(signer, run_id[-6:], 1, side=Side.BUY, qty="5", limit="450.10")

    # ---- run A: intent executes while the API rejects the first 3 event posts (transient outage)
    with StubBridge(token) as stub:
        stub.add_intent(buy, label="valid_buy")
        stub.inject_post_failures(EVENTS, 3)
        cfg = lat_lean.write_live_sim_config(lat_lean.LiveSimSpec(
            run_id=run_id, bridge_url=stub.url, pubkey_path=pub, sim_quotes=harness.SIM_QUOTES,
            exit_after_seconds=20, journal_path=journal))
        eng = harness.launch_engine(run_dir, cfg, token, stub, _filled(buy.client_order_id), exit_after_s=20, timeout_s=90)
        snap_a = stub.snapshot()
    assert eng.completed, f"run A did not complete; see {eng.stdout_path}"
    assert snap_a["injected_failures"] == [EVENTS] * 3
    fills = [e for e in snap_a["events"] if e["client_order_id"] == buy.client_order_id and e["status"] == "Filled"]
    assert len(fills) == 1 and Decimal(fills[0]["fill_quantity"]) == Decimal("5")
    statuses = [e["status"] for e in snap_a["events"] if e["client_order_id"] == buy.client_order_id]
    assert statuses == ["Submitted", "Filled"], f"events must arrive once each and in order after retries: {statuses}"
    outbox = [json.loads(line) for line in (run_dir / "bridge-outbox.jsonl").read_text().splitlines()]
    assert any(o["path"] == EVENTS and o["body"]["status"] == "Filled" for o in outbox)
    journal_lines = journal.read_text().splitlines()
    assert journal_lines and all(line.split("\t")[0] == buy.intent_id for line in journal_lines)

    # ---- run B: engine "restarts" (new process, LEAN order ids restart at 1); the API wrongly re-delivers the intent
    run_id_b = lat_lean.new_run_id("lean-fi-restart-b")
    run_dir_b = lat_lean.RUNS_DIR / run_id_b
    run_dir_b.mkdir(parents=True)
    with StubBridge(token) as stub:
        stub.add_intent(buy, label="replayed_after_restart")
        cfg_b = lat_lean.write_live_sim_config(lat_lean.LiveSimSpec(
            run_id=run_id_b, bridge_url=stub.url, pubkey_path=pub, sim_quotes=harness.SIM_QUOTES,
            exit_after_seconds=15, journal_path=journal))

        def refused(s: dict[str, Any]) -> bool:
            return harness.refusal_found(s["rejections"], reason="earlier engine run", unkeyed=True)

        eng_b = harness.launch_engine(run_dir_b, cfg_b, token, stub, refused, exit_after_s=15, timeout_s=60)
        snap_b = stub.snapshot()
    assert eng_b.completed, f"run B did not refuse the replay; see {eng_b.stdout_path}"
    assert harness.refusal_found(snap_b["rejections"], layer="[GatedBrokerageModel]", reason="replayed intent", unkeyed=True)
    # no order event under the original ids (its order is FILLED in the ledger) and certainly no second fill
    assert not [e for e in snap_b["events"] if e.get("client_order_id") == buy.client_order_id], snap_b["events"]
    assert "LAT bridge: order 1 refused as a duplicate/replay" in eng_b.log_text


@pytest.mark.timeout(200)
def test_wrong_engine_token_claims_nothing_and_creates_no_order() -> None:
    run_id, run_dir, pub, signer = harness.new_sim_run("lean-fi-token")
    good, wrong = secrets.token_hex(24), secrets.token_hex(24)
    buy = harness.make_intent(signer, run_id[-6:], 1, side=Side.BUY, qty="1", limit="450.10")
    with StubBridge(good) as stub:
        stub.add_intent(buy, label="valid_buy")
        cfg = lat_lean.write_live_sim_config(lat_lean.LiveSimSpec(
            run_id=run_id, bridge_url=stub.url, pubkey_path=pub, sim_quotes=harness.SIM_QUOTES, exit_after_seconds=12))
        eng = harness.launch_engine(run_dir, cfg, wrong, stub, lambda s: s["auth_failures"] >= 6,
                                    exit_after_s=12, timeout_s=50)
        snap = stub.snapshot()
    assert eng.completed
    assert snap["auth_failures"] >= 6
    assert snap["intents"][0]["claimed_by"] is None
    assert snap["events"] == [] and snap["rejections"] == [] and snap["heartbeats"] == []
    assert "Status: Submitted" not in eng.log_text and "Status: Filled" not in eng.log_text


@pytest.mark.timeout(150)
def test_missing_intent_public_key_fails_closed_at_startup() -> None:
    run_id, run_dir, pub, signer = harness.new_sim_run("lean-fi-nokey")
    pub.unlink()
    token = secrets.token_hex(24)
    buy = harness.make_intent(signer, run_id[-6:], 1, side=Side.BUY, qty="1", limit="450.10")
    with StubBridge(token) as stub:
        stub.add_intent(buy, label="valid_buy")
        cfg = lat_lean.write_live_sim_config(lat_lean.LiveSimSpec(
            run_id=run_id, bridge_url=stub.url, pubkey_path=pub, sim_quotes=harness.SIM_QUOTES, exit_after_seconds=10))
        eng = harness.launch_engine(run_dir, cfg, token, stub, lambda s: False, exit_after_s=10, timeout_s=40)
        snap = stub.snapshot()
    assert eng.exit_code not in (None, 0)
    assert snap["intents"][0]["claimed_by"] is None and snap["events"] == []
    assert "intent public key not found" in eng.log_text
