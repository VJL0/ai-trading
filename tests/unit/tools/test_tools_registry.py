"""Registry tests: one registry -> identical exports (catalog, MCP, OpenAI Responses, PydanticAI), strict
schemas, and a closed surface with no dangerous capability."""

from __future__ import annotations

import ast
import asyncio
import inspect
import json
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from services.tools import handlers, registry
from services.tools.backends import InProcessToolBackend
from services.tools.protocol import MUTATING_METHODS, READ_METHODS, ToolRuntime
from services.tools.registry import (
    FORBIDDEN_FIELD_RE,
    FORBIDDEN_NAME_TOKENS,
    MUTATING_TOOL_NAMES,
    TOOL_NAMES,
    TOOL_REGISTRY,
    catalog,
    mcp_tool_definitions,
    openai_function_tools,
    pydantic_ai_tools,
)

from .fake_runtime import FakeToolRuntime
from .test_tools_schemas import VALID_ARGS

EXPECTED_TOOLS = {
    "search_memory",
    "get_market_snapshot", "get_bars", "get_feed_quality", "get_predictions", "get_model_report", "get_signals",
    "get_portfolio", "get_orders", "get_order", "get_risk_status", "propose_paper_trade", "request_halt",
    "cancel_paper_order", "run_quant_cycle",
    # PAPER sessions + operations (services.sessions; see tests/unit/tools/test_tools_sessions_*.py)
    "start_paper_session", "stop_session", "schedule_session", "cancel_schedule", "get_session_status",
    "list_schedules", "get_session_report", "explain_decisions", "get_performance", "get_activity", "get_system_health",
}
SERVICES_TOOLS = Path(__file__).resolve().parents[3] / "services" / "tools"


def test_exact_tool_set_and_categories() -> None:
    assert set(TOOL_NAMES) == EXPECTED_TOOLS
    assert set(MUTATING_TOOL_NAMES) == {"propose_paper_trade", "request_halt", "cancel_paper_order", "run_quant_cycle",
                                        "start_paper_session", "stop_session", "schedule_session", "cancel_schedule"}
    for t in TOOL_REGISTRY.values():
        assert t.category in registry.CATEGORIES
        assert (t.category == "paper_trading") == t.mutating
    assert set(handlers._HANDLERS) == set(TOOL_REGISTRY)  # pyright: ignore[reportPrivateUsage]


def test_all_exports_generated_from_the_same_registry() -> None:
    cat = {t["name"]: t for t in catalog()}
    mcp = {t.name: t for t in mcp_tool_definitions()}
    oai = {t["name"]: t for t in openai_function_tools()}
    pai = {t.name: t for t in pydantic_ai_tools(InProcessToolBackend(FakeToolRuntime(), "registry-test"))}
    assert list(cat) == list(mcp) == list(oai) == list(pai) == list(TOOL_NAMES)
    for name, spec in TOOL_REGISTRY.items():
        canonical = spec.input_schema()
        assert cat[name]["input_schema"] == canonical
        assert mcp[name].input_schema == canonical
        assert oai[name]["parameters"] == canonical
        assert pai[name].tool_def.parameters_json_schema == canonical
        assert cat[name]["description"] == mcp[name].description == oai[name]["description"] == spec.description
        assert pai[name].tool_def.description == spec.description
        assert cat[name]["mutating"] is spec.mutating
        assert mcp[name].annotations is not None and mcp[name].annotations.read_only_hint is (not spec.mutating)
        assert mcp[name].output_schema == cat[name]["output_schema"] == spec.output_schema()
        assert pai[name].metadata == {"category": spec.category, "mutating": spec.mutating,
                                      "registry_version": registry.REGISTRY_VERSION}


def test_openai_function_tools_are_strict_function_tools_only() -> None:
    tools = openai_function_tools()
    assert {t["type"] for t in tools} == {"function"}  # no hosted/web/file/computer/code tools
    for t in tools:
        assert set(t) == {"type", "name", "description", "parameters", "strict"}
        assert t["strict"] is True
        _assert_strict_object(t["parameters"], t["name"])
    json.dumps(tools)  # serialisable as-is for the Responses API


def _assert_strict_object(node: Any, where: str) -> None:
    if isinstance(node, dict):
        assert "$ref" not in node and "$defs" not in node and "default" not in node and "title" not in node, where
        if node.get("type") == "object":
            assert node.get("additionalProperties") is False, where
            assert sorted(node.get("required", [])) == sorted(node.get("properties", {})), where
        for k, v in node.items():
            if k == "properties":
                for pname, sub in v.items():
                    _assert_strict_object(sub, f"{where}.{pname}")
            else:
                _assert_strict_object(v, where)
    elif isinstance(node, list):
        for v in node:
            _assert_strict_object(v, where)


def test_pydantic_ai_tools_are_strict_and_mutating_ones_sequential() -> None:
    tools = pydantic_ai_tools(InProcessToolBackend(FakeToolRuntime(), "x"))
    for t in tools:
        assert t.strict is True
        assert t.sequential is TOOL_REGISTRY[t.name].mutating
    ro = pydantic_ai_tools(InProcessToolBackend(FakeToolRuntime(), "x"), include_mutating=False)
    assert {t.name for t in ro} == set(TOOL_NAMES) - set(MUTATING_TOOL_NAMES)


def test_input_and_output_schemas_are_valid_json_schema() -> None:
    for spec in TOOL_REGISTRY.values():
        jsonschema.Draft202012Validator.check_schema(spec.input_schema())
        jsonschema.Draft202012Validator.check_schema(spec.output_schema())
        assert spec.input_schema()["type"] == "object" and spec.output_schema()["type"] == "object"


def test_canonical_schema_accepts_valid_args() -> None:
    for name, args in VALID_ARGS.items():
        full = {k: args.get(k) for k in TOOL_REGISTRY[name].input_schema()["properties"]}  # strict: all keys present
        jsonschema.validate(full, TOOL_REGISTRY[name].input_schema())
        TOOL_REGISTRY[name].input_model.model_validate(full)  # and the enforcing model agrees


def test_dispatch_outputs_validate_against_advertised_output_schema() -> None:
    rt = FakeToolRuntime()

    async def run() -> None:
        for name in ("get_signals", "get_predictions", "get_market_snapshot"):  # issue citable ids first
            await handlers.dispatch(rt, name, VALID_ARGS[name] if name != "get_market_snapshot" else {"symbol": "SPY"},
                                    actor="schema")
        for name, args in VALID_ARGS.items():
            out = await handlers.dispatch(rt, name, args, actor="schema")
            jsonschema.validate(out, TOOL_REGISTRY[name].output_schema())

    asyncio.run(run())


# ------------------------------------------------------------------------------------------ dangerous surface
def test_no_tool_name_implies_admin_execution_or_exfiltration() -> None:
    for name in TOOL_NAMES:
        tokens = set(name.split("_"))
        assert not (tokens & FORBIDDEN_NAME_TOKENS), name
    # Sanity of the guard itself: the capabilities we must never expose would be caught.
    for bad in ("resume_trading", "set_risk_limits", "change_mode", "submit_live_order", "read_secret", "get_api_key",
                "read_file", "run_shell_command", "approve_order", "update_policy", "fetch_url", "reconcile_now"):
        assert set(bad.split("_")) & FORBIDDEN_NAME_TOKENS, bad


def _property_names(node: Any) -> set[str]:
    out: set[str] = set()
    if isinstance(node, dict):
        for name, sub in (node.get("properties") or {}).items():
            out.add(name)
            out |= _property_names(sub)
        for k, v in node.items():
            if k != "properties":
                out |= _property_names(v)
    elif isinstance(node, list):
        for v in node:
            out |= _property_names(v)
    return out


def _enum_values(node: Any) -> set[str]:
    out: set[str] = set()
    if isinstance(node, dict):
        out |= {str(v) for v in node.get("enum", [])}
        if "const" in node:
            out.add(str(node["const"]))
        for v in node.values():
            out |= _enum_values(v)
    elif isinstance(node, list):
        for v in node:
            out |= _enum_values(v)
    return out


def test_no_input_field_can_select_mode_gateway_secrets_files_or_overrides() -> None:
    for spec in TOOL_REGISTRY.values():
        schema = spec.input_schema()
        for prop in _property_names(schema):
            assert not FORBIDDEN_FIELD_RE.search(prop), (spec.name, prop)
        enums = _enum_values(schema)
        assert not enums & {"LIVE", "LIVE_APPROVAL", "BOUNDED_LIVE", "MARKET", "LEAN_ALPACA_LIVE"}, spec.name
    assert FORBIDDEN_FIELD_RE.search("mode") and FORBIDDEN_FIELD_RE.search("agent_token")
    assert FORBIDDEN_FIELD_RE.search("file_path") and FORBIDDEN_FIELD_RE.search("override_risk")


def test_runtime_protocol_has_no_admin_members() -> None:
    members = {n for n in vars(ToolRuntime) if not n.startswith("_")}
    assert members == set(READ_METHODS) | set(MUTATING_METHODS) | {"settings", "policy", "dataset_version",
                                                                     "execution_label"}
    for forbidden in ("resume", "apply_policy", "reconcile", "secrets", "db", "gateway", "execution", "ledger", "risk",
                      "chatgpt_login", "bridge_next", "signer"):
        assert forbidden not in members


def test_handler_module_cannot_touch_filesystem_shell_network_or_secrets() -> None:
    src = (SERVICES_TOOLS / "handlers.py").read_text()
    tree = ast.parse(src)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0] if node.level == 0 else "." + node.module)
    assert not imported & {"os", "subprocess", "shutil", "pathlib", "socket", "httpx", "requests", "urllib", "keyring"}
    called = {n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", "")
              for n in ast.walk(tree) if isinstance(n, ast.Call)}
    assert not called & {"open", "system", "popen", "run", "exec", "eval", "read_text", "write_text", "unlink"}
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert not attrs & {"resume", "apply_policy", "reconcile", "secrets", "db", "gateway", "execution", "ledger",
                        "signer", "risk", "bridge_next", "chatgpt_login", "read_text", "write_text"}
    for name in ("SecretStore", "keyring", "keychain", "agent.token", "operator.token"):
        assert name not in src, name


@pytest.mark.parametrize("name", sorted(EXPECTED_TOOLS))
def test_descriptions_are_compact_and_paper_scoped(name: str) -> None:
    desc = TOOL_REGISTRY[name].description
    assert 40 <= len(desc) <= 700
    if TOOL_REGISTRY[name].mutating:
        assert "PAPER" in desc.upper() or "halt" in desc.lower()


def test_catalog_is_token_economical() -> None:
    blob = json.dumps(openai_function_tools(), separators=(",", ":"))
    # The whole tool surface stays a few thousand tokens (budget 14,000 chars for the original 14 tools; 16,000 since
    # the 11 session/operations tools were added).
    assert len(blob) < 16_000
    sig = inspect.signature(registry.catalog)
    assert not sig.parameters
