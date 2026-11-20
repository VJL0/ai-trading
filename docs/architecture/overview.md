# Architecture overview

Everything runs on this Mac. External calls: model APIs (opt-in), Alpaca market data, Alpaca paper broker.

```mermaid
flowchart TD
  subgraph External["External APIs (only)"]
    AD[Alpaca market data<br/>IEX_FREE / SIP_PAID]
    AB[Alpaca PAPER broker]
    AN[Anthropic Messages API<br/>claude-opus-5-5]
    OA[OpenAI Responses API<br/>gpt-6.1-sol]
  end
  subgraph Mac["Local MacBook"]
    MD[services/market_data<br/>ingestion · PIT validation · backfill]
    ST[(Parquet + DuckDB<br/>datasets/ + manifests)]
    RS[services/research<br/>baselines · cost model · LightGBM/CatBoost · Qlib env]
    AI[services/ai<br/>PydanticAI adapters · evidence · injection filter]
    RK[services/risk<br/>deterministic policy gate · ECDSA intent signer]
    LG[services/order_ledger<br/>state machine · execution · reconciliation]
    DB[(SQLite WAL<br/>single writer · hash-chained audit)]
    API[services/api FastAPI 127.0.0.1:8765<br/>SSE stream]
    UI[apps/dashboard React 19 read-only live operations<br/>127.0.0.1:5173]
    LEAN[LEAN engine C#<br/>GatedBrokerageModel · IntentBridge algorithm]
    GUI[apps/mac-agent<br/>Peekaboo + Playwright MCP · allowlist]
  end
  AD --> MD --> ST --> RS
  RS -- selected candidate only --> AI
  AN <-. sanitized read-only context .-> AI
  OA <-. Responses tool calls (read-only tools) .-> AI
  AI -- TradeProposal / NO_TRADE --> RK
  RS -- quant baseline proposal --> RK
  RK -- signed ApprovedOrderIntent --> LG
  LG -- FakeBroker (SIMULATED) --> LG
  LG -- claimable intents via engine bridge --> LEAN
  LEAN -- Alpaca brokerage plugin (paper) --> AB
  LEAN -- OnOrderEvent --> API --> LG
  AB -. read-only account/orders/positions (alpaca-py) .-> LG
  LG --> DB
  MD --> DB
  AI --> DB
  RK --> DB
  DB --> API --> UI
  GUI -. optional, deferred .-> UI
```

## Order path and trust boundaries

```mermaid
sequenceDiagram
  participant P as DecisionCycle
  participant R as RiskEngine (private key)
  participant L as OrderLedger (SQLite)
  participant X as ExecutionService
  participant G as Gateway (FakeBroker | LeanBridge)
  participant E as LEAN GatedBrokerageModel (public key)
  participant B as Alpaca PAPER
  P->>R: TradeProposal + RiskContext
  R-->>P: RiskDecision (recomputed qty/notional/loss)
  R->>L: ApprovedOrderIntent (ECDSA-signed, TTL)
  X->>R: final_gate(intent, fresh ctx)  [signature, expiry, mode, active policy, halts, health, duplicates]
  X->>L: claim_for_submission (atomic -> SUBMITTING)
  X->>G: submit(intent)
  G->>E: engine claims intent (engine token)
  E->>E: verify signature, parse order from signed string, refuse anything else
  E->>B: LimitOrder DAY (paper)
  B-->>E: order updates
  E-->>L: /engine/events -> apply_event (idempotent, out-of-order safe)
  Note over X,G: timeout => UNKNOWN, query by client_order_id, never resubmit
```

## Security boundaries
| Boundary | Enforcement |
|---|---|
| Models ↔ broker | Models get sanitized read-only tools only; no order/limit tools exist (tool-surface test). No credentials in prompts. |
| Proposal → order | Only `RiskEngine.authorize` can sign (private key 0600 in `var/<ns>/keys`). Gateway + LEAN verify with the public key. |
| LEAN strategy → broker | `GatedBrokerageModel.CanSubmitOrder` refuses any order not matching a valid signed intent. |
| Paper ↔ live | Compile-time `LIVE_EXECUTION_COMPILED=False`; separate namespaces, keys, DBs, secret names; live gateway always denied; paper key ids `PK…` required. |
| Operator ↔ automation | Operator token (human only) for resume/policy/cycle; engine token (LEAN only) for the bridge; neither given to models or the GUI agent. |
| GUI agent | App/domain allowlist, element-level denial of order controls, scrubbed subprocess env, no shell access. |
| Network | API and dashboard bind `127.0.0.1`; CSRF header + origin checks on POST. |
| Restart/sleep | Entries locked until broker-authoritative reconciliation is CLEAN; SUBMITTING-at-crash ⇒ UNKNOWN ⇒ resolved by client order id, never resubmitted. |
