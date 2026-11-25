"""Research tools share dispatch and can run without the trading API."""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

import mcp_types as types
from mcp import Client, ClientSession, StdioServerParameters, stdio_client

from packages.core.config import Settings
from services.agents.contracts import RESEARCH_TOOL_NAMES
from services.mcp_server.research import RESEARCH_INSTRUCTIONS, research_backend
from services.mcp_server.server import build_server
from services.memory.store import Memory
from services.quantlab.paths import results_root
from services.tools.registry import TOOL_REGISTRY

ROOT = Path(__file__).resolve().parents[3]


def _settings(tmp_path: Path) -> Settings:
    return Settings(project_root=tmp_path, use_keychain=False)


async def test_research_profile_tools_and_refusals(tmp_path: Path) -> None:
    server = build_server(research_backend(_settings(tmp_path)), tool_names=RESEARCH_TOOL_NAMES,
                          instructions=RESEARCH_INSTRUCTIONS)
    async with Client(server) as client:
        assert client.instructions == RESEARCH_INSTRUCTIONS
        tools = (await client.list_tools()).tools
        assert tuple(tool.name for tool in tools) == RESEARCH_TOOL_NAMES
        for tool in tools:
            assert tool.input_schema == TOOL_REGISTRY[tool.name].input_schema()
            assert tool.annotations is not None and tool.annotations.read_only_hint is True
        for name in ("propose_paper_trade", "run_research_experiment", "start_paper_session"):
            result = await client.call_tool(name, {})
            assert result.is_error is True
            block = result.content[0]
            assert isinstance(block, types.TextContent) and block.text.startswith("error 404:")
        research = await client.call_tool("get_research_results", {})
        assert research.is_error is False
        assert research.structured_content["status"] == "MISSING"
        assert research.structured_content["execution_label"] == "SIMULATED"
        capabilities = await client.call_tool("get_capabilities", {})
        assert capabilities.is_error is False
        opportunities = await client.call_tool("get_opportunities", {})
        assert opportunities.is_error is False
        memory = await client.call_tool("search_memory", {"query": "SPY"})
        assert memory.is_error is False and memory.structured_content["memory"]["hits"] == []
    assert not _settings(tmp_path).db_path.parent.exists()


async def test_research_profile_reads_existing_memory(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    path = settings.db_path.parent / "memory.sqlite3"
    memory = Memory(path)
    memory.append(source="test", source_id="one", body={"note": "prior"}, available_at=datetime.now(UTC).isoformat())
    original = path.read_bytes()
    async with Client(build_server(research_backend(settings), tool_names=RESEARCH_TOOL_NAMES)) as client:
        result = await client.call_tool("search_memory", {"query": "prior"})
        assert result.is_error is False
        hit, = result.structured_content["memory"]["hits"]
        assert hit["source_id"] == "one" and hit["untrusted_data"] is True
    assert path.read_bytes() == original


async def test_research_profile_leaves_unmigrated_strategy_registry_untouched(tmp_path: Path) -> None:
    path = results_root(tmp_path) / "registry.sqlite3"
    path.parent.mkdir(parents=True)
    with closing(sqlite3.connect(path)) as db, db:
        db.executescript("""
            CREATE TABLE strategies (strategy_id TEXT PRIMARY KEY, created TEXT, spec TEXT);
            CREATE TABLE transitions (id INTEGER PRIMARY KEY, strategy_id TEXT, to_state TEXT);
            INSERT INTO strategies VALUES ('test-strategy', '2026-01-01', '{}');
            INSERT INTO transitions VALUES (1, 'test-strategy', 'RESEARCH');
        """)
    original = path.read_bytes()
    async with Client(build_server(research_backend(_settings(tmp_path)), tool_names=RESEARCH_TOOL_NAMES)) as client:
        result = await client.call_tool("get_research_results", {})
        assert result.is_error is False
    assert path.read_bytes() == original


async def test_research_cli_stdio_without_api_or_agent_token(tmp_path: Path) -> None:
    env = {
        "LAT_PROJECT_ROOT": str(tmp_path), "LAT_MODE": "PAPER", "LAT_USE_KEYCHAIN": "false",
        "LAT_FEED_MODE": "SIMULATED", "LAT_API_PORT": "1",
    }
    params = StdioServerParameters(command=sys.executable, args=["-m", "services.mcp_server", "--profile", "research"],
                                   env=env, cwd=str(ROOT))
    async with (stdio_client(params) as (read, write), ClientSession(read, write) as client):
        initialized = await client.initialize()
        assert initialized.instructions == RESEARCH_INSTRUCTIONS
        assert tuple(t.name for t in (await client.list_tools()).tools) == RESEARCH_TOOL_NAMES
        res = await client.call_tool("get_research_results", {})
        assert res.is_error is False and res.structured_content["status"] == "MISSING"


def test_research_profile_refuses_non_paper_or_replay_mode(tmp_path: Path) -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "services.mcp_server", "--profile", "research"], cwd=ROOT, capture_output=True,
        env={**os.environ, "LAT_PROJECT_ROOT": str(tmp_path), "LAT_MODE": "RESEARCH_ONLY", "LAT_USE_KEYCHAIN": "false"},
        text=True, timeout=10, check=False,
    )
    assert proc.returncode == 2 and "PAPER/REPLAY only" in proc.stderr


def test_research_import_excludes_execution_services() -> None:
    proc = subprocess.run(
        [sys.executable, "-c", """
import sys
import services.mcp_server.research
blocked = ('services.api', 'services.order_ledger', 'services.risk', 'services.positions', 'services.sessions')
found = [name for name in sys.modules if any(name == prefix or name.startswith(prefix + '.') for prefix in blocked)]
unexpected = set(found) - {'services.sessions', 'services.sessions.errors'}
assert not unexpected, unexpected
"""],
        cwd=ROOT, capture_output=True, text=True, timeout=10, check=False,
    )
    assert proc.returncode == 0, proc.stderr


async def test_research_profile_withholds_completed_holdout_outcomes(tmp_path: Path) -> None:
    import json

    out = results_root(tmp_path)
    (out / "holdout_results").mkdir(parents=True)
    (out / "holdout_results" / "A2.json").write_text(json.dumps({"family": "A2", "sharpe": -0.4}))
    (out / "latest_research.json").write_text(json.dumps({
        "updated_at": "2026-11-25T00:00:00+00:00", "holdout": ["2025-01-01", "2025-12-31"],
        "families": {"A2": {"family": "A2", "dsr": 0.1, "passed_dev_gate": False, "holdout_outcome": "FAILED"}},
    }))
    server = build_server(research_backend(_settings(tmp_path)), tool_names=RESEARCH_TOOL_NAMES,
                          instructions=RESEARCH_INSTRUCTIONS)
    async with Client(server) as client:
        research = await client.call_tool("get_research_results", {})
    assert research.is_error is False
    body = research.structured_content
    assert body["holdout_results"] == []
    assert "FAILED" not in json.dumps(body) and "-0.4" not in json.dumps(body)
    assert body["holdout"] == ["2025-01-01", "2025-12-31"]
