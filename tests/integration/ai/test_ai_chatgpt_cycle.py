"""Whole Sign-in-with-ChatGPT cycle without network: sign-in (real 127.0.0.1 loopback listener) -> inference with
read-only tools -> token refresh -> inference -> logout/revocation, persisted to SQLite; plus the integrator's real
``services.api.runtime.Runtime`` driving the same AiRuntime (SIMULATED feed, FakeBroker, real risk engine)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from packages.contracts.enums import AiRuntimeMode, DecisionAction, FeedMode, GatewayKind, ModelRunStatus, TradingMode
from packages.core.config import Settings
from packages.core.db import Database
from services.ai.chatgpt_fake import fake_browser_sign_in
from services.ai.token_store import ACCOUNT_REFRESH_TOKEN

pytestmark = pytest.mark.timeout(180)


async def test_sign_in_inference_refresh_logout_cycle(chatgpt_factory: Any, db: Database, clock: Any) -> None:
    h = chatgpt_factory(db=db)
    await h.sign_in()
    first = await h.research(causal_id="cz_cycle_1")
    assert first.proposal.action == DecisionAction.TRADE
    rt1 = h.store.get(ACCOUNT_REFRESH_TOKEN)
    clock.advance(hours=2)  # access token expired -> refresh before the next inference
    second = await h.research(causal_id="cz_cycle_2")
    assert second.proposal.action == DecisionAction.TRADE
    assert h.store.get(ACCOUNT_REFRESH_TOKEN) != rt1
    grants = [r.body.get("grant_type") for r in h.server.recorded if r.path.endswith("/oauth/token")]
    assert grants == ["authorization_code", "refresh_token"]
    out = await h.rt.chatgpt_logout()
    assert out["revoked"] is True and h.store.get(ACCOUNT_REFRESH_TOKEN) is None
    third = await h.research(causal_id="cz_cycle_3")
    assert third.proposal.action == DecisionAction.NO_TRADE and third.model_runs[0].status == ModelRunStatus.ERROR
    assert h.server.violations == [] and h.mutating_calls() == []
    # persisted: model runs, evidence and one AI_PIPELINE_RESULT audit event per turn
    runs = db.query("SELECT run_id, transport, status, estimated_cost_usd FROM model_runs ORDER BY started_at")
    assert [(r["transport"], r["status"]) for r in runs] == [("FAKE", "OK"), ("FAKE", "OK"), ("FAKE", "ERROR")]
    assert all(r["estimated_cost_usd"] == "0" for r in runs)
    audits = db.query("SELECT source, payload_json FROM audit_events WHERE event_type='AI_PIPELINE_RESULT'")
    assert len(audits) == 3 and {a["source"] for a in audits} == {"ai:chatgpt"}
    assert all(json.loads(a["payload_json"])["ai_runtime"] == "CHATGPT_OAUTH" for a in audits)
    failures = db.query("SELECT kind FROM failures WHERE component='ai'")
    assert [f["kind"] for f in failures] == ["MODEL_ERROR"]
    blob = json.dumps([dict(r) for r in db.query("SELECT payload_json FROM model_runs")])
    grant = next(iter(h.server.grants.values()))
    assert grant.access_token not in blob and grant.refresh_token not in blob


def _api_settings(root: Path, **over: Any) -> Settings:
    base: dict[str, Any] = dict(project_root=root, use_keychain=False, mode=TradingMode.PAPER, gateway=GatewayKind.FAKE,
                                feed_mode=FeedMode.SIMULATED, reconcile_interval_s=3600, ai_transport="fake",
                                ai_runtime=AiRuntimeMode.CHATGPT_OAUTH, chatgpt_oauth_port=0)
    base.update(over)
    return Settings(**base)


async def test_integrator_runtime_drives_the_chatgpt_runtime(tmp_path: Path) -> None:
    from services.ai.runtime_modes import reset_shared_runtimes
    from services.api.runtime import Runtime

    reset_shared_runtimes()
    rt = Runtime(_api_settings(tmp_path))
    try:
        report = await rt.reconciler.startup_recovery()
        rt._after_reconcile(report)  # pyright: ignore[reportPrivateUsage]
        assert rt.sim_clock is not None
        for _ in range(2):
            rt.sim_clock.step(60)
            rt._on_sim_tick(rt.now())  # pyright: ignore[reportPrivateUsage]
        status = rt.ai_runtime_status()
        assert status["mode"] == "CHATGPT_OAUTH" and status["quant_engine_independent"] is True
        assert {"api_keys", "chatgpt", "mcp", "limits"} <= set(status)
        assert status["chatgpt"]["signed_in"] is False
        login = await rt.chatgpt_login()
        assert set(login) >= {"authorize_url", "callback"} and login["callback"].startswith("http://127.0.0.1:")
        ai_rt = rt._ai_rt()  # pyright: ignore[reportPrivateUsage]
        assert await ai_rt.auth.wait_login(10) is None  # FAKE transport auto-completes locally
        assert rt.ai_runtime_status()["chatgpt"]["signed_in"] is True
        server = ai_rt.fake_server
        assert server is not None
        # 1) the synthetic tape has no candidate right now: NO_TRADE without any model request
        out = await rt.chatgpt_research("SPY")
        assert out["outcome"] == "NO_TRADE" and "no selected quant candidate" in out["reason"]
        assert out["model_runs"] == [] and server.responses_bodies == []
        # 2) force the deterministic engine's top signal to be a candidate: the turn runs on the real runtime's tools
        original = rt.signal_engine.generate

        def generate(*a: Any, **kw: Any) -> list[Any]:
            sigs = original(*a, **kw)
            return [s.model_copy(update={"is_candidate": True}) if i == 0 else s for i, s in enumerate(sigs)]

        rt.signal_engine.generate = generate
        out = await rt.chatgpt_research(None)
        assert out["outcome"] in ("NO_TRADE", "REJECTED", "SUBMITTED", "BLOCKED", "AWAITING_HUMAN_APPROVAL"), out
        assert out["model_runs"] and all(r["transport"] == "FAKE" for r in out["model_runs"])
        assert out["model_runs"][0]["status"] == "OK", out["model_runs"][0]
        assert server.violations == [] and len(server.responses_bodies) == 2
        outputs = [i for i in server.responses_bodies[1]["input"] if i["type"] == "function_call_output"]
        assert outputs and not any('"error"' in o["output"] for o in outputs)  # real tool outputs validated
        if out["outcome"] != "NO_TRADE":  # a TRADE went through the real deterministic risk engine
            assert out.get("decision") is not None
        # the research cycle in CHATGPT_OAUTH mode uses the same (shared) session, never per tick
        assert rt.ai.chatgpt_runtime().auth is ai_rt.auth
        logout = await rt.chatgpt_logout()
        assert logout == {"signed_in": False}
        assert rt.ai_runtime_status()["chatgpt"]["signed_in"] is False
    finally:
        await rt.aclose()
        reset_shared_runtimes()


async def test_fake_login_url_never_points_at_openai(chatgpt_factory: Any) -> None:
    h = chatgpt_factory()
    info = await h.rt.chatgpt_login()
    assert "openai.com" not in info["authorize_url"].split("?", 1)[0]
    resp = await fake_browser_sign_in(h.server, info["authorize_url"])
    assert resp.status_code == 200
