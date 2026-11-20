# services/tools + services/mcp_server — progress (checkpoint log)

Attempt 3 (2026-11-07). Inventory at start: no files existed in owned paths (previous attempts left nothing here).

- R1 schemas.py: strict in/out models, decimal-string-only money gate over Dec, symbol/id patterns, sanitized text.
  Smoke: every tool dispatched against FakeToolRuntime OK. Tests: pending.
- R2 registry.py: ToolSpec/TOOL_REGISTRY/catalog + MCP/OpenAI/PydanticAI exports from one strict schema. Tests: pending.
- R3 handlers.py: dispatch (404/403/422/502/503/500-UNKNOWN), evidence-id session ledger, conservative max loss. Tests: pending.
- R4 backends.py: InProcess + Http (loopback-only, token 0600, no retries except 401 re-read). Tests: pending.
- R5 mcp_server: low-level mcp 2.3.0 Server over stdio; raw-stdio probe listed 14 tools and returned a clear 503
  when the token/API was absent. Tests: pending.
- R6 registrations / R7 tests+runbook: pending.
- R1-R3 TESTED: `uv run --frozen pytest tests/unit/tools tests/unit/mcp_server -q` -> 234 passed (schemas, registry
  parity + dangerous-surface, dispatch for all 14 tools, propose w/ real RiskEngine, sanitize, in-process MCP Client).
  Left: HTTP backend parity (ASGI), stdio e2e, failure injection, registrations, runbook, README.
- R4 TESTED: `uv run --frozen pytest tests/integration/tools -q` -> 31 passed (real create_app over ASGITransport,
  read/mutating/error parity, 401 token re-read, loopback-only, timeout=UNKNOWN no retry, output re-validation).
- R5 TESTED: `uv run --frozen pytest tests/integration/mcp_server -q` -> 5 passed (real stdio subprocess + SDK
  ClientSession vs uvicorn stub API; API-down / missing-token errors; refuses non-PAPER; token never in stderr/output).
- R6 .mcp.json + .codex/config.toml written; register script, registration tests, failure injection, docs: pending.
- R6 TESTED: .mcp.json, .codex/config.toml, scripts/register_mcp.sh; `pytest tests/unit/mcp_server -q` (registrations
  + script behaviour w/ stub CLIs). `claude mcp get local-ai-trader` -> "Scope: Project config (shared via .mcp.json),
  Status: Pending approval" (approval left to the user). Codex CLI absent -> script prints install/login steps only.
- R7 tests: failure_injection/tools (32 w/ registrations) + PydanticAI Agent run (2) passing. Left: README, runbook,
  full quality gate, self-review.

Attempt 4 (2026-11-07): inventory found R1-R7 code+tests from attempt 3; re-ran: 300 passed (all owned test dirs);
ruff: 10 test-only lint errors (SIM117, RUF001); pyright src clean. Now: real-Runtime shape check, self-review.
- Attempt 4 fixes: halt reason neutralized (never refused), int prices pattern-checked, 5xx on mutating HTTP call
  -> UNKNOWN guidance, IPv6 base URL, escaped invisible chars, get_bars desc no longer claims citable source_id,
  test lint/pyright fixed. New tests/integration/tools/test_tools_real_runtime.py (7: every tool vs REAL Runtime,
  real risk engine/rate limit/audit, real trained tiny registry) passing. Codex config keys checked vs official
  config reference; input schemas < 5,000 B (Codex default tool_input_schema_max_bytes) test added.
  `claude mcp get/list local-ai-trader` -> Project config, Pending approval. Left: docs update, final gate.
- Attempt 4 DONE: + tests/failure_injection/mcp_server (2). Final gate: ruff clean, pyright 0 errors (src+tests),
  pytest unit/tools 237, unit/mcp_server 12, integration/tools 41, integration/mcp_server 5,
  failure_injection/tools 21, failure_injection/mcp_server 2 = 318 passed. Docs updated. Nothing left in scope.
