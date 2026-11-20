# Local API contract (v1)

FastAPI, bound to `127.0.0.1:8765` only. All money/qty/price values are **decimal strings**. All
timestamps are ISO-8601 UTC strings ending in `Z`. Chart bar `time` values are UNIX seconds (int).
Every response that exposes market data or execution state carries explicit labels
(`execution_label`, `feed_mode`, `is_simulated`, `limitations`) so the dashboard never shows
unlabeled simulated data as live.

## Auth classes
| Class | Header | Who |
|---|---|---|
| public-local | `X-LAT-Request: 1` (CSRF guard on POST) | dashboard |
| operator | `X-LAT-Request: 1` + `X-LAT-Operator-Token: <token from var/<ns>/keys/operator.token>` | human operator only (never models/GUI agent) |
| engine | `X-LAT-Engine-Token: <token from var/<ns>/keys/engine.token>` | local LEAN process only |

POST without `X-LAT-Request: 1` → 403. Requests whose `Origin` is not `http://127.0.0.1:<dashboard_port>`/`http://localhost:<dashboard_port>` → 403.

## Read endpoints (public-local)

### `GET /api/v1/health`
Detailed `overall` and `components` health (see live operations below), with compatibility fields
`status`, `mode`, `execution_label`, `version`, and `time`. `status=ok` means the API responds;
use `overall` and component statuses to assess readiness.

### `GET /api/v1/health/live`
`{"status":"ok","mode":"PAPER","execution_label":"SIMULATED","version":"0.1.0","time":"…Z"}`
Lightweight liveness only; it does not assert trading readiness.

### `GET /api/v1/status`
```json
{
  "mode": "PAPER", "execution_label": "SIMULATED", "gateway": "FAKE",
  "live_execution_compiled": false,
  "entries_locked": true, "lock_reasons": ["startup reconciliation pending"],
  "halts": [{"halt_id":"…","reason":"…","source":"…","created_at":"…Z"}],
  "startup": {"phase": "RECONCILING|READY|LOCKED", "started_at": "…Z", "detail": "…"},
  "feed": {"feed_mode":"SIMULATED","provider":"replay:synthetic-v1","is_simulated":true,
           "last_bar_ts":"…Z|null","last_quote_ts":"…Z|null","staleness_s":"12.5|null",
           "status":"OK|STALE|DISCONNECTED|BACKFILLING","limitations":["…"]},
  "engine": {"kind":"FAKE|LEAN_PAPER_SIM|LEAN_ALPACA_PAPER","status":"UP|DOWN|UNKNOWN",
             "last_heartbeat":"…Z|null","detail":"…"},
  "broker": {"kind":"FAKE","is_paper":true,"connected":true,
             "last_reconciled_at":"…Z|null","reconciliation_status":"CLEAN|MISMATCH|ERROR|SKIPPED|null"},
  "credentials": {"alpaca_paper": false, "anthropic": false, "openai": false},
  "ai": {"transport":"FAKE|REAL","claude_model":"claude-opus-5-5","sol_model":"gpt-6.1-sol"},
  "policy_version": "pol_…",
  "server_time": "…Z"
}
```

### `GET /api/v1/market/bars?symbol=SPY&timeframe=1Min&limit=500`
`{"symbol","timeframe","feed","feed_mode","is_simulated","source_id","dataset_version","limitations":[…],
  "bars":[{"time":1730000000,"open":"…","high":"…","low":"…","close":"…","volume":"…","available_at":"…Z"}]}`

### `GET /api/v1/market/quality`
`{"feed_mode","provider","is_simulated","limitations":[…],"symbols":[{"symbol","last_bar_ts","last_quote_ts",
  "staleness_s","spread_bps","gaps_today":0,"rejected_today":0}],"events":[{"ts","symbol","feed","kind","severity","detail"}]}`

### `GET /api/v1/signals?limit=50`
`{"signals":[QuantSignal…], "latest_cycle": {"causal_id","ran_at","candidate_symbol|null","outcome"}}`

### `GET /api/v1/research/runs?limit=50`
`{"runs":[ModelRunRecord…],"findings":[ResearchFinding…],
  "totals":{"runs":0,"input_tokens":0,"output_tokens":0,"estimated_cost_usd":"0","fake_runs":0,"real_runs":0}}`

### `GET /api/v1/proposals?limit=50`
`{"proposals":[{"proposal":TradeProposal,"decision":RiskDecision|null,"intent_client_order_id":"…|null"}]}`

### `GET /api/v1/orders?limit=100`
`{"orders":[{"client_order_id","broker_order_id","intent_id","symbol","side","quantity","filled_quantity",
  "avg_fill_price","limit_price","state","gateway","execution_label","created_at","updated_at"}]}`

### `GET /api/v1/orders/{client_order_id}`
`{"order":{…as above…},"intent":ApprovedOrderIntent,"events":[OrderEvent + {"applied":bool,"apply_note"}],
  "transitions":[{"from_state","to_state","at","reason"}],"fills":[{"fill_id","quantity","price","ts"}]}`

### `GET /api/v1/portfolio`
`{"execution_label","account":AccountSnapshot|null,"broker_positions":[PositionSnapshot…],
  "ledger_positions":[{"symbol","quantity","avg_cost","realized_pl"}],
  "pnl":{"daily_realized":"…","daily_unrealized":"…","start_of_day_equity":"…|null"},
  "equity_curve":[{"time":1730000000,"equity":"…"}]}`

### `GET /api/v1/risk`
`{"policy":RiskPolicy,"policy_version","history":[{"policy_version","created_at","created_by","is_active"}],
  "utilization":[{"name":"order_notional","used":"…","limit":"…","pct":"…"}],
  "recent_decisions":[RiskDecision…]}`

### `GET /api/v1/backtests`
`{"runs":[{"run_id","engine":"LEAN|PYTHON_BASELINE","strategy","dataset","is_simulated","started_at",
  "metrics":{"total_return":"…","sharpe":"…","max_drawdown":"…","trades":0,"fees":"…"},
  "equity_curve":[{"time":…,"equity":"…"}],"source_path":"experiments/…","notes":"…"}]}`

### `GET /api/v1/reconciliation?limit=20`
`{"runs":[ReconciliationReport…]}`

### `GET /api/v1/audit?limit=200&event_type=`
`{"events":[{"seq","event_id","ts","event_type","causal_id","source","mode","policy_version","payload"}],
  "chain":{"ok":true,"rows":123,"detail":"ok"}}`

### `GET /api/v1/failures?limit=100`
`{"failures":[{"ts","component","kind","detail","causal_id"}]}`

### `GET /api/v1/stream` (Server-Sent Events)
Event names: `status` (same body as /status, at least every 5 s), `order` (order row), `order_event`,
`audit` (audit row), `cycle` (decision cycle summary), `heartbeat` (`{"time":"…Z"}`, every 2 s).

## Control endpoints
- `POST /api/v1/control/halt` (public-local) body `{"reason":"…"}` → `{"halt_id","entries_locked":true}`.
  Stopping new orders is always allowed (fail-safe direction).
- `POST /api/v1/control/resume` (operator) body `{"halt_id":"…","reason":"…"}` → requires last reconciliation CLEAN.
- `POST /api/v1/control/reconcile` (operator) → ReconciliationReport.
- `POST /api/v1/pipeline/run-cycle` (operator) body `{"symbols":["SPY"],"use_ai":true}` → cycle summary
  `{"causal_id","outcome":"NO_TRADE|REJECTED|SUBMITTED|BLOCKED","proposal","decision","order"}`.
- `POST /api/v1/risk/policy` (operator) body `{"policy":RiskPolicy,"reason":"…"}` → `{"policy_version"}`.
- `POST /api/v1/orders/{client_order_id}/cancel` (operator).

## Engine bridge (engine auth; used only by the local LEAN process)
- `GET /api/v1/engine/intents/next?engine_id=…&max=1` →
  `{"intents":[{"intent_id","client_order_id","signing_string","signature","key_id","signature_alg"}]}`.
  Claiming is atomic: claimed intents move to `SUBMITTING` and are never handed out twice.
- `POST /api/v1/engine/events` body
  `{"engine_id","client_order_id","intent_id","lean_order_id":int,"broker_order_id":"…|null",
    "status":"Submitted|PartiallyFilled|Filled|Canceled|Invalid|CancelPending|UpdateSubmitted|New",
    "fill_quantity":"…","fill_price":"…","cum_filled_quantity":"…","event_ts":"…Z","lean_event_id":int,
    "message":"…"}` → `{"accepted":true,"state":"…","duplicate":false}`.
- `POST /api/v1/engine/heartbeat` body `{"engine_id","mode","gateway","algorithm","lean_commit","ts","status"}`.
- `POST /api/v1/engine/rejections` body `{"engine_id","client_order_id|null","intent_id|null","reason"}`
  (orders refused by the C# `GatedBrokerageModel`).

## Quant model suite (public-local, read)
- `GET /api/v1/models` →
  `{"registry_version","trained_at","dataset_version","is_simulated","selection_metric":"net_sharpe",
    "targets":[{"target":"ret_5m|ret_15m|ret_60m|success|rank|vol|regime|cost","selected_model","candidates":[{"model_id",
    "net_sharpe","net_return","turnover","hit_rate","brier","ic","n_test","selected":bool,"rejected_reason"}],
    "calibration":{"method","brier_raw","brier_calibrated","reliability":[{"bin","pred","obs","n"}]}}],
    "walk_forward":{"folds","purge_minutes","embargo_minutes","frozen_test":{"start","end","used":bool}},
    "multiple_testing":{"method","alpha","survivors":[…]},"notes":[…]}`
- `GET /api/v1/predictions?symbol=SPY` → `{"predictions":[PredictionSet…],"asof","is_simulated"}`
  (all symbols when `symbol` omitted). Predictions are point-in-time; never generated from data after `asof`.

## AI runtime (public-local read; operator for changes)
- `GET /api/v1/ai/runtime` →
  `{"mode":"NONE|API_KEYS|CLAUDE_CODE|CODEX|CHATGPT_OAUTH","quant_engine_independent":true,
    "api_keys":{"transport":"FAKE|REAL","anthropic_key":bool,"openai_key":bool,"claude_model","sol_model"},
    "mcp":{"server":"local-ai-trader","command":"uv run --frozen python -m services.mcp_server",
           "claude_code_registered":bool,"codex_registered":bool,"tools":[{"name","description","category","mutating":bool}]},
    "chatgpt":{"signed_in":bool,"plan_usage_granted":bool,"email_masked":"…|null","model","expires_at":"…|null",
               "last_error":"…|null","usage_limit_hit":bool},
    "limits":{"agent_max_proposals_per_hour":6,"agent_proposals_last_hour":0}}`
- `POST /api/v1/ai/chatgpt/login` (operator) → `{"authorize_url","callback":"http://127.0.0.1:1455/auth/callback"}`
  starts the official PKCE loopback flow; the human completes sign-in in their own browser.
- `POST /api/v1/ai/chatgpt/logout` (operator) → revokes the refresh token at the official revocation endpoint and
  deletes it from the Keychain.
- `POST /api/v1/ai/chatgpt/research` (operator) body `{"symbol":"SPY"}` → runs one ChatGPT-plan research turn
  (function tools = the validated tool interface) and returns its proposal/NO_TRADE + model-run record.

## Agent tool interface (agent auth; used by the MCP server and by in-app AI runtimes)
Header `X-LAT-Agent-Token: <var/<ns>/keys/agent.token>` (held by the MCP server process / in-app runtime, never shown
to a model). Every call is validated against the Pydantic schemas in `services/tools/schemas.py` and audited.
Refused (403) unless mode is PAPER/REPLAY and the execution label is not LIVE.
- `GET  /api/v1/agent/tools` → the tool catalog (name, description, JSON input/output schema, category, mutating).
- `POST /api/v1/agent/tools/{name}` body = tool input → tool output, e.g.
  `get_market_snapshot`, `get_bars`, `get_feed_quality`, `get_predictions`, `get_model_report`, `get_signals`,
  `get_portfolio`, `get_orders`, `get_order`, `get_risk_status`, `propose_paper_trade`, `request_halt`,
  `cancel_paper_order`, `run_quant_cycle`.
  Mutating tools never bypass the risk engine: `propose_paper_trade` creates a `TradeProposal(proposer=AGENT_TOOL)`
  that goes through `RiskEngine.evaluate` → signed intent → final gate → paper gateway, rate-limited per hour.
  There is no tool to resume trading, change risk limits, change mode, submit live orders or read secrets.

## Live operations (read-only dashboard; update request 2026-11-08)
The dashboard is **read-only**: it only issues GETs and subscribes to the SSE stream. All actions are taken by the AI
agent through MCP tools (or by the operator CLI). Every event below is persisted in `activity_events` (append-only) and
published on `/api/v1/stream` with the same body.

Pipeline stages (fixed order): `MARKET_DATA → FEATURES → MODELS → GPT_REVIEW → RISK_ENGINE → ORDER_DECISION → BROKER →
RECONCILIATION`. Stage statuses: `idle | running | passed | rejected | error | skipped`.

SSE events (in addition to `status`, `order`, `order_event`, `audit`, `heartbeat`):
- `stage`: `{"seq","ts","cycle_id","causal_id","session_id|null","stage","status","item":"SPY|SPY,QQQ|null",
  "duration_ms":int|null,"detail":"…"}` — emitted on every stage start (`running`) and finish.
- `activity`: `{"seq","ts","kind":"scan|predict|evaluate|gpt_call|risk|reject|submit|fill|reconcile|session|agent_tool|
  error|info","actor":"system|session:<id>|agent:<name>|operator","message":"human sentence","causal_id","session_id",
  "detail":{…}}`
- `decision`: the current decision card `{"cycle_id","causal_id","ts","symbol|null","side":"BUY|SELL|null",
  "action":"BUY|SELL|NO_TRADE","status":"PROPOSED|APPROVED|REJECTED|SUBMITTED|FILLED|NO_TRADE|ERROR",
  "expected_return_bps","expected_cost_bps","net_edge_bps","confidence":"0..1|null","confidence_calibrated":bool,
  "volatility_bps","regime","reason":"exact sentence","risk_failures":["check: detail"],"proposer","is_simulated"}`
- `session`: session object (see below) whenever it changes.

Read endpoints:
- `GET /api/v1/activity?limit=200&kind=` → `{"events":[activity|stage|decision rows, newest first]}`
- `GET /api/v1/activity/current` → `{"cycle":{"cycle_id","causal_id","started_at","session_id","stages":[{"stage","status",
  "item","started_at","duration_ms","detail"}]},"current_task":{"message","kind","since"}|null,"decision":<decision card>|null}`
- `GET /api/v1/performance` → `{"mode_label":"BACKTEST|PAPER|LIVE|SIMULATED","portfolio_value","cash","daily_pnl",
  "total_pnl","realized_pnl","unrealized_pnl","win_rate":"0..1|null","round_trips":int,"max_drawdown":"0..1",
  "current_drawdown":"0..1","active_positions":int,"open_orders":int,"api_cost_usd":"…","equity_curve":[{"time","equity"}],
  "asof","is_simulated"}`
- `GET /api/v1/health` → `{"overall":"OK|DEGRADED|DOWN","components":[{"name":"market_data|broker|models|gpt_runtime|
  event_stream|database|agent|engine|scheduler","status":"OK|DEGRADED|DOWN|DISABLED","detail","last_ok":"…|null"}]}`
- `GET /api/v1/sessions?limit=20` → `{"active":<session>|null,"sessions":[<session>…],"schedules":[<schedule>…]}`
  session = `{"session_id","state":"SCHEDULED|RUNNING|STOPPING|COMPLETED|STOPPED|INTERRUPTED|FAILED","started_at",
  "ends_at","ended_at","symbols":[…],"cadence_minutes","use_ai","requested_by","cycles_run","trades","stop_reason",
  "schedule_id|null"}`; schedule = `{"schedule_id","weekdays":["MON",…],"start_time_et":"09:45","duration_minutes",
  "symbols","cadence_minutes","use_ai","enabled","next_run_at","created_by"}`
- `GET /api/v1/sessions/{session_id}/report` → `{"session":…,"performance":…,"cycles":int,"decisions":{"NO_TRADE":n,
  "REJECTED":n,"SUBMITTED":n},"top_no_trade_reasons":[{"reason","count"}],"top_risk_rejections":[{"check","count"}],
  "trades":[{"client_order_id","symbol","side","qty","avg_price","state","pnl"}],"api_cost_usd","narrative":"…"}`

Sessions are run by the API process (independent of any AI session): an AI disconnect or usage-limit hit never stops a
running session; restart recovery resumes a RUNNING session only after a CLEAN startup reconciliation and only inside its
time window, otherwise marks it INTERRUPTED.
