"""The chat client against the REAL local API (``services.api.app`` + ``services.api.runtime.Runtime``, SIMULATED:
synthetic replay feed, FakeBroker, real risk engine, real agent auth guard), reached through the real
``HttpToolBackend`` over ``httpx.ASGITransport`` (in-process, no socket, no network). The model is the scripted FAKE
ChatGPT-plan stand-in. Shows: the live catalog is whatever the API serves, read tools run without prompts, a
mutating tool runs exactly once and only after the human confirms, and the agent token never reaches the model."""

from __future__ import annotations

import os
import secrets
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from packages.contracts.enums import AiRuntimeMode, FeedMode, GatewayKind, TradingMode
from packages.core.clock import ManualClock
from packages.core.config import Settings
from packages.core.secrets import SecretStore
from services.ai.chat import Console, build_session
from services.ai.chatgpt_fake import FakeChatGptServer
from services.ai.chatgpt_oauth import ChatGptOAuthConfig
from services.ai.runtime_modes import AiRuntime
from services.ai.token_store import MemoryTokenStore
from services.tools.backends import HttpToolBackend
from tests.unit.ai.chat_helpers import ScriptedConsole
from tests.unit.ai.conftest import T0

pytestmark = pytest.mark.timeout(180)


def _settings(root: Path) -> Settings:
    return Settings(project_root=root, use_keychain=False, mode=TradingMode.PAPER, gateway=GatewayKind.FAKE,
                    feed_mode=FeedMode.SIMULATED, reconcile_interval_s=3600, ai_runtime=AiRuntimeMode.CHATGPT_OAUTH,
                    ai_transport="fake", chatgpt_oauth_port=0)


@pytest.fixture
async def real_api(tmp_path: Path) -> AsyncIterator[tuple[Any, Any, str, Path]]:
    from services.api.app import create_app
    from services.api.runtime import Runtime

    settings = _settings(tmp_path)
    rt = Runtime(settings)
    report = await rt.reconciler.startup_recovery()
    rt._after_reconcile(report)  # pyright: ignore[reportPrivateUsage]
    assert rt.sim_clock is not None
    for _ in range(2):  # fresh point-in-time quotes on the replay clock
        rt.sim_clock.step(60)
        rt._on_sim_tick(rt.now())  # pyright: ignore[reportPrivateUsage]
    app = create_app(settings)
    token = secrets.token_urlsafe(32)
    app.state.runtime = rt  # what the lifespan does, without starting the background loops
    app.state.agent_token = token
    app.state.operator_token = secrets.token_urlsafe(32)
    app.state.engine_token = secrets.token_urlsafe(32)
    token_path = tmp_path / "chat-agent.token"
    fd = os.open(token_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write(token)
    try:
        yield app, rt, token, token_path
    finally:
        await rt.aclose()


async def test_chat_against_the_real_local_api(real_api: tuple[Any, Any, str, Path], tmp_path: Path) -> None:
    app, rt, token, token_path = real_api
    cycles: list[dict[str, Any]] = []
    real_run_cycle = rt.run_cycle

    async def counting_run_cycle(**kw: Any) -> dict[str, Any]:
        cycles.append(kw)
        return await real_run_cycle(**kw)

    rt.run_cycle = counting_run_cycle  # observe the runtime's real cycle path (risk engine + paper gateway)
    settings = rt.settings
    clock = ManualClock(T0)
    oauth = ChatGptOAuthConfig.fake()
    server = FakeChatGptServer(oauth, now=lambda: clock.now().timestamp())
    store = SecretStore(tmp_path, use_keychain=False, env={}, dotenv_path=tmp_path / "no.env")
    ai = AiRuntime(settings, store, clock=clock, wall_clock=clock, token_store=MemoryTokenStore(), fake_server=server,
                   oauth_config=oauth)
    io = ScriptedConsole(["y"])
    backend = HttpToolBackend("http://127.0.0.1:8765", token_path, actor="chat",
                              transport=httpx.ASGITransport(app=app))
    session = build_session(settings, store, Console(read=io.read, write=io.write), backend=backend, ai_runtime=ai,
                            run_log=False)
    try:
        assert await session.refresh_tools() is None, io.text
        await session.handle("/login")
        perf = await session.ask("Show my paper account")
        cyc = await session.ask("Run one quant cycle now")
        why = await session.ask("Why didn't you trade?")
    finally:
        await ai.aclose()
        await backend.aclose()

    from services.tools.registry import catalog

    served = {t["name"] for t in catalog()}
    assert {t.name for t in session.tools} == served  # exactly the catalog the API served
    assert perf is not None and "equity" in perf.text.lower(), io.text  # read tool on the real runtime
    assert cyc is not None and "outcome" in cyc.text, io.text
    assert len(cycles) == 1 and cycles[0].get("use_ai") is False  # the confirmed action ran once, quant only
    assert len(io.prompts) == 1 and "Run it?" in io.prompts[0]
    assert why is not None and why.text
    assert server.violations == []
    assert all(token not in str(b) for b in server.responses_bodies)
    assert token not in io.text
