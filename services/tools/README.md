# services/tools — the ONE validated agent tool interface

Every AI runtime uses this package: Claude Code and Codex through MCP, the in-app API-key adapters, and the
Sign-in-with-ChatGPT research turn. There is one registry, one schema per tool and one executor (`dispatch`).

## Public API

```python
from services.tools.registry import (TOOL_REGISTRY, TOOL_NAMES, ToolSpec, catalog,
                                     mcp_tool_definitions, openai_function_tools, pydantic_ai_tools)
from services.tools.handlers import dispatch, ToolError          # async dispatch(runtime, name, args, *, actor) -> dict
from services.tools.backends import ToolBackend, InProcessToolBackend, HttpToolBackend
from services.tools.protocol import ToolRuntime                  # what dispatch may touch on the runtime
```

* `catalog() -> list[dict]` returns name, description, category, mutating, read_only, input_schema, output_schema
  and registry_version. It backs `GET /api/v1/agent/tools`.
* `await dispatch(runtime, name, args, actor=...) -> dict` validates input, calls the runtime and validates the
  output, then returns JSON-safe data. It raises `ToolError(status, detail)`: 404 for an unknown tool or order,
  422 for invalid input, 403 outside PAPER/REPLAY or with a LIVE label, 502 when output breaks the contract
  (withheld), 503 when a read failed and 500 when a mutating call failed (outcome UNKNOWN, never retried).
  `POST /api/v1/agent/tools/{name}` calls it.
* `mcp_tool_definitions()` returns `mcp_types.Tool` objects. `openai_function_tools()` returns Responses API
  `{"type":"function","name","description","parameters","strict":true}`. `pydantic_ai_tools(backend,
  include_mutating=True)` returns PydanticAI `Tool`s bound to `backend.call`: a 422 becomes `ModelRetry`, any
  other error becomes `ToolFailed`. All of these use the identical canonical strict schema (`strict_schema.py`).
* `InProcessToolBackend(runtime, actor)` and
  `HttpToolBackend(base_url="http://127.0.0.1:8765", agent_token_path=None, actor="mcp", *, transport=None, lazy_token=False)`
  share the interface `async call(name, args) -> dict` / `async catalog()` and return the same validated outputs.
  The HTTP backend accepts loopback addresses only. It reads the 0600 token file and never logs it. It never
  retries, except once after a 401, when the token file has changed. A timeout or 5xx on a mutating call is
  reported with an UNKNOWN outcome and a do-not-retry instruction.

## Tools

The read tools are `get_market_snapshot`, `get_bars`, `get_feed_quality`, `get_predictions`, `get_model_report`,
`get_signals`, `get_portfolio`, `get_orders`, `get_order` and `get_risk_status`.

Four tools are mutating and risk-gated. Each one reaches the runtime only through its listed method:

* `propose_paper_trade` -> `submit_agent_proposal`
* `request_halt` -> `halt`
* `cancel_paper_order` -> `cancel`
* `run_quant_cycle` -> `run_cycle(use_ai=False)`

The registry has no tool that resumes trading, changes limits or mode, submits live orders, reads secrets, or
touches files or the shell.

## Contracts

* Every input and output model uses `extra="forbid"`.
* Money, price and quantity inputs are `Dec` behind a decimal-string gate. Python floats are rejected, and so
  are JSON numbers with fractions, which the HTTP layer parses to `Decimal`. Whole-share quantities may be ints.
* Integer and boolean inputs are strict.
* `null` for an optional field means "use the default".
* Free text is cleaned (NFKC; control, zero-width and bidi characters removed), length-capped and screened for
  instruction-shaped content (`sanitize.py`). Thesis and counterarguments that match the screen are refused
  (422). A halt reason that matches is replaced by an inert placeholder instead, because a halt must never be
  refused.
* Outputs are compact and always carry `execution_label`, `is_simulated` and `asof`, plus the data's own
  timestamps.

## propose_paper_trade

The proposal sent to the risk engine is built as follows:

* `proposer=AGENT_TOOL`, `strategy_id=f"agent:{actor}"`, with the actor name sanitised.
* `data_asof` and `dataset_version` come from a fresh `market_snapshot`, whose `snapshot_id` is always attached
  as evidence.
* The limit is rounded to a passive tick (BUY rounds down, SELL rounds up). The stop is rounded up and must be
  below a BUY limit. `stop_price` is rejected on a SELL.
* `max_planned_loss = qty × (limit − min(stop, policy default stop))`. It is deterministic, never understates
  the loss, and the risk engine recomputes it anyway.
* Cited `evidence_ids` must have been returned to this actor's session by a tool output, within 6 h and
  LRU-bounded, or be persisted signal ids. At least one must belong to the traded symbol.

## Tests

```bash
uv run --frozen pytest tests/unit/tools tests/unit/mcp_server tests/integration/tools tests/integration/mcp_server tests/failure_injection/tools -q
```

`tests/integration/tools/test_tools_real_runtime.py` runs every tool against the real `services.api.runtime.Runtime`
in SIMULATED mode (synthetic replay feed, FakeBroker, real risk engine, real per-hour proposal limit, audit
trail and a tiny trained model registry). The other tests use the deterministic `FakeToolRuntime`
(`tests/unit/tools/fake_runtime.py`), which evaluates proposals with the real `RiskEngine`.

## Limitations

* Read-tool calls are not written to the SQLite audit chain, because the `ToolRuntime` facade has no audit
  member. They are logged as structured JSON (`lat.tools`). Mutating calls are audited by the runtime paths
  (AGENT_PROPOSAL, halts, cycles).
* Issued-evidence ids live in memory and are lost when the API restarts. After a restart, agents re-run the read
  tools (persisted signal ids stay citable).
* The injection screen is a deterministic pattern list: it reduces risk but does not detect injection
  exhaustively. The actual boundaries are the risk engine, the signed-intent gateway and the absence of any
  dangerous tool.
