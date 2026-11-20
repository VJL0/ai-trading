"""Harness for the Sign-in-with-ChatGPT tests: AiRuntime wired to an explicit FakeChatGptServer (no network,
in-memory credential store, shared ManualClock), plus the shared tool interface over a FakeToolRuntime."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from packages.contracts import RiskPolicy
from packages.contracts.enums import AiRuntimeMode
from packages.core.clock import ManualClock
from packages.core.config import Settings
from packages.core.db import Database
from packages.core.secrets import SecretStore
from services.ai.chatgpt_fake import ChatGptScenario, FakeChatGptServer, fake_browser_sign_in
from services.ai.chatgpt_oauth import ChatGptOAuthConfig
from services.ai.chatgpt_responses import ChatGptConfig
from services.ai.config import AiConfig
from services.ai.evidence import build_internal_evidence
from services.ai.runtime_modes import AiRuntime
from services.ai.token_store import MemoryTokenStore
from services.tools.backends import InProcessToolBackend
from tests.unit.tools.fake_runtime import FakeToolRuntime

from .conftest import make_signal, make_snapshot

# Mutating runtime members of the shared tool interface: none may ever be reached from a research turn.
MUTATING_RUNTIME_CALLS = ("submit_agent_proposal", "halt", "cancel", "run_cycle")


class ChatGptHarness:
    def __init__(
        self,
        root: Path,
        clock: ManualClock,
        *,
        scenario: ChatGptScenario = ChatGptScenario.TRADE,
        mode: AiRuntimeMode = AiRuntimeMode.CHATGPT_OAUTH,
        db: Database | None = None,
        config: ChatGptConfig | None = None,
        ai_config: AiConfig | None = None,
        oauth_config: ChatGptOAuthConfig | None = None,
        store: Any | None = None,
        http_client: Any | None = None,
        **server_kw: Any,
    ) -> None:
        self.clock = clock
        self.settings = Settings(project_root=root, ai_transport="fake", use_keychain=False, ai_runtime=mode,
                                 chatgpt_oauth_port=0)
        self.secrets = SecretStore(root, use_keychain=False, env={}, dotenv_path=root / "no.env")
        oauth = oauth_config or ChatGptOAuthConfig.fake()
        self.server = FakeChatGptServer(oauth, scenario=scenario, now=lambda: clock.now().timestamp(), **server_kw)
        self.store = store if store is not None else MemoryTokenStore()
        self.rt = AiRuntime(
            self.settings, self.secrets, db=db, clock=clock, wall_clock=clock,
            ai_config=ai_config or AiConfig(min_wall_interval_per_symbol_s=0, min_interval_per_symbol_s=0),
            chatgpt_config=config, token_store=self.store, fake_server=self.server, oauth_config=oauth,
            http_client=http_client, auto_complete_fake_login=False,
        )
        self.tools = FakeToolRuntime()
        self.backend = InProcessToolBackend(self.tools, actor="chatgpt")

    async def sign_in(self) -> dict[str, Any]:
        info = await self.rt.chatgpt_login()
        resp = await fake_browser_sign_in(self.server, info["authorize_url"])
        assert resp.status_code == 200, resp.text
        assert await self.rt.auth.wait_login(5) is None
        return info

    async def research(self, *, signal: Any = None, snapshot: Any = None, evidence: list[Any] | None = None,
                       extra_evidence: list[Any] | None = None, backend: Any = "default",
                       causal_id: str = "cz_chatgpt", symbol: str | None = "SPY") -> Any:
        now = self.clock.now()  # fresh point-in-time inputs at the (manual) decision clock
        sig = make_signal(at=now) if signal is None else signal
        snap = make_snapshot(at=now) if snapshot is None else snapshot
        pol = RiskPolicy()
        ev = build_internal_evidence(sig, snap, pol) if evidence is None else evidence
        return await self.rt.chatgpt_research(
            symbol, tool_backend=self.backend if backend == "default" else backend, candidate=sig, snapshot=snap,
            evidence=[*ev, *(extra_evidence or [])], policy=pol, causal_id=causal_id)

    def mutating_calls(self) -> list[str]:
        return [c for c in self.tools.calls if c in MUTATING_RUNTIME_CALLS]

    async def aclose(self) -> None:
        await self.rt.aclose()
