-- 0001_core: operational ledger shared by all services. Money/qty/price columns are TEXT decimals.

-- Append-only, hash-chained audit log. Every action carries source, causal id and version stamps.
CREATE TABLE audit_events (
  seq INTEGER PRIMARY KEY AUTOINCREMENT,
  event_id TEXT NOT NULL UNIQUE,
  ts TEXT NOT NULL,
  event_type TEXT NOT NULL,
  causal_id TEXT,
  source TEXT NOT NULL,
  mode TEXT NOT NULL,
  dataset_version TEXT,
  model_version TEXT,
  policy_version TEXT,
  payload_json TEXT NOT NULL,
  prev_hash TEXT NOT NULL,
  hash TEXT NOT NULL
);
CREATE INDEX ix_audit_causal ON audit_events(causal_id);
CREATE INDEX ix_audit_type_ts ON audit_events(event_type, ts);
CREATE TRIGGER audit_events_no_update BEFORE UPDATE ON audit_events
BEGIN SELECT RAISE(ABORT, 'audit_events is append-only'); END;
CREATE TRIGGER audit_events_no_delete BEFORE DELETE ON audit_events
BEGIN SELECT RAISE(ABORT, 'audit_events is append-only'); END;

-- Key/value system state (entries lock, startup phase, engine heartbeat, last reconciliation ...).
CREATE TABLE system_state (
  key TEXT PRIMARY KEY,
  value_json TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

-- Stop-new-orders halts. Entries are allowed only when no halt is active.
CREATE TABLE trading_halts (
  halt_id TEXT PRIMARY KEY,
  reason TEXT NOT NULL,
  source TEXT NOT NULL,
  created_at TEXT NOT NULL,
  cleared_at TEXT,
  cleared_by TEXT,
  clear_reason TEXT
);
CREATE INDEX ix_halts_active ON trading_halts(cleared_at);

-- Immutable versioned risk policies. policy_json can never change; activation is a separate row flag.
CREATE TABLE risk_policies (
  policy_version TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  policy_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  created_by TEXT NOT NULL,
  approved_by TEXT NOT NULL,
  supersedes TEXT,
  is_active INTEGER NOT NULL DEFAULT 0 CHECK (is_active IN (0, 1))
);
CREATE UNIQUE INDEX ux_one_active_policy ON risk_policies(is_active) WHERE is_active = 1;
CREATE TRIGGER risk_policies_immutable BEFORE UPDATE OF policy_json, policy_version, name, created_at, created_by
ON risk_policies
BEGIN SELECT RAISE(ABORT, 'risk policy content is immutable; create a new version'); END;
CREATE TRIGGER risk_policies_no_delete BEFORE DELETE ON risk_policies
BEGIN SELECT RAISE(ABORT, 'risk policies are never deleted'); END;

CREATE TABLE market_snapshots (
  snapshot_id TEXT PRIMARY KEY,
  symbol TEXT NOT NULL,
  data_asof TEXT NOT NULL,
  feed TEXT NOT NULL,
  is_simulated INTEGER NOT NULL,
  payload_json TEXT NOT NULL
);

CREATE TABLE quant_signals (
  signal_id TEXT PRIMARY KEY,
  causal_id TEXT,
  strategy_id TEXT NOT NULL,
  symbol TEXT NOT NULL,
  generated_at TEXT NOT NULL,
  is_candidate INTEGER NOT NULL,
  net_edge_bps TEXT NOT NULL,
  dataset_version TEXT NOT NULL,
  payload_json TEXT NOT NULL
);
CREATE INDEX ix_signals_time ON quant_signals(generated_at);

CREATE TABLE evidence (
  evidence_id TEXT PRIMARY KEY,
  causal_id TEXT,
  source_type TEXT NOT NULL,
  source_id TEXT NOT NULL,
  trust TEXT NOT NULL,
  content_sha256 TEXT NOT NULL,
  available_at TEXT NOT NULL,
  injection_flags TEXT NOT NULL DEFAULT '[]',
  payload_json TEXT NOT NULL
);

CREATE TABLE model_runs (
  run_id TEXT PRIMARY KEY,
  causal_id TEXT NOT NULL,
  provider TEXT NOT NULL,
  model_id TEXT NOT NULL,
  transport TEXT NOT NULL,
  status TEXT NOT NULL,
  started_at TEXT NOT NULL,
  latency_ms INTEGER NOT NULL,
  input_tokens INTEGER NOT NULL,
  output_tokens INTEGER NOT NULL,
  estimated_cost_usd TEXT NOT NULL,
  payload_json TEXT NOT NULL
);
CREATE INDEX ix_model_runs_causal ON model_runs(causal_id);

CREATE TABLE research_findings (
  finding_id TEXT PRIMARY KEY,
  causal_id TEXT NOT NULL,
  model_run_id TEXT NOT NULL,
  symbol TEXT NOT NULL,
  recommendation TEXT NOT NULL,
  created_at TEXT NOT NULL,
  payload_json TEXT NOT NULL
);

CREATE TABLE proposals (
  proposal_id TEXT PRIMARY KEY,
  causal_id TEXT NOT NULL,
  created_at TEXT NOT NULL,
  action TEXT NOT NULL,
  proposer TEXT NOT NULL,
  symbol TEXT,
  payload_json TEXT NOT NULL
);
CREATE INDEX ix_proposals_time ON proposals(created_at);

CREATE TABLE risk_decisions (
  decision_id TEXT PRIMARY KEY,
  proposal_id TEXT NOT NULL REFERENCES proposals(proposal_id),
  causal_id TEXT NOT NULL,
  verdict TEXT NOT NULL,
  policy_version TEXT NOT NULL,
  evaluated_at TEXT NOT NULL,
  payload_json TEXT NOT NULL
);
CREATE INDEX ix_decisions_proposal ON risk_decisions(proposal_id);

-- Order ledger: one row per client order id. state follows the ledger state machine.
CREATE TABLE orders (
  client_order_id TEXT PRIMARY KEY,
  intent_id TEXT NOT NULL UNIQUE,
  proposal_id TEXT NOT NULL,
  decision_id TEXT NOT NULL,
  causal_id TEXT NOT NULL,
  mode TEXT NOT NULL,
  gateway TEXT NOT NULL,
  symbol TEXT NOT NULL,
  side TEXT NOT NULL,
  quantity TEXT NOT NULL,
  order_type TEXT NOT NULL,
  limit_price TEXT,
  time_in_force TEXT NOT NULL,
  state TEXT NOT NULL,
  broker_order_id TEXT UNIQUE,
  filled_quantity TEXT NOT NULL DEFAULT '0',
  avg_fill_price TEXT,
  intent_json TEXT NOT NULL,
  intent_fingerprint TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  submitted_at TEXT,
  claimed_by TEXT,
  claimed_at TEXT,
  last_broker_ts TEXT,
  version INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX ix_orders_state ON orders(state);
CREATE INDEX ix_orders_symbol ON orders(symbol);

CREATE TABLE order_events (
  seq INTEGER PRIMARY KEY AUTOINCREMENT,
  event_id TEXT NOT NULL UNIQUE,
  dedupe_key TEXT NOT NULL UNIQUE,
  client_order_id TEXT NOT NULL,
  broker_order_id TEXT,
  event_type TEXT NOT NULL,
  state TEXT NOT NULL,
  source TEXT NOT NULL,
  gateway TEXT NOT NULL,
  broker_ts TEXT,
  received_ts TEXT NOT NULL,
  applied INTEGER NOT NULL,
  apply_note TEXT,
  payload_json TEXT NOT NULL
);
CREATE INDEX ix_order_events_coid ON order_events(client_order_id);
CREATE TRIGGER order_events_no_update BEFORE UPDATE ON order_events
BEGIN SELECT RAISE(ABORT, 'order_events is append-only'); END;

CREATE TABLE order_state_transitions (
  seq INTEGER PRIMARY KEY AUTOINCREMENT,
  client_order_id TEXT NOT NULL,
  from_state TEXT,
  to_state TEXT NOT NULL,
  at TEXT NOT NULL,
  reason TEXT NOT NULL,
  event_id TEXT
);
CREATE INDEX ix_transitions_coid ON order_state_transitions(client_order_id);

CREATE TABLE fills (
  fill_id TEXT PRIMARY KEY,
  client_order_id TEXT NOT NULL REFERENCES orders(client_order_id),
  broker_exec_id TEXT NOT NULL UNIQUE,
  symbol TEXT NOT NULL,
  side TEXT NOT NULL,
  quantity TEXT NOT NULL,
  price TEXT NOT NULL,
  ts TEXT NOT NULL,
  gateway TEXT NOT NULL
);

-- Local position book derived only from recorded fills (never invented).
CREATE TABLE ledger_positions (
  symbol TEXT PRIMARY KEY,
  quantity TEXT NOT NULL,
  avg_cost TEXT NOT NULL,
  realized_pl TEXT NOT NULL DEFAULT '0',
  updated_at TEXT NOT NULL
);

CREATE TABLE account_snapshots (
  snapshot_id TEXT PRIMARY KEY,
  asof TEXT NOT NULL,
  source TEXT NOT NULL,
  is_paper INTEGER NOT NULL,
  equity TEXT NOT NULL,
  payload_json TEXT NOT NULL
);
CREATE INDEX ix_account_asof ON account_snapshots(asof);

CREATE TABLE position_snapshots (
  seq INTEGER PRIMARY KEY AUTOINCREMENT,
  snapshot_group TEXT NOT NULL,
  symbol TEXT NOT NULL,
  asof TEXT NOT NULL,
  source TEXT NOT NULL,
  payload_json TEXT NOT NULL
);

CREATE TABLE reconciliation_runs (
  run_id TEXT PRIMARY KEY,
  started_at TEXT NOT NULL,
  finished_at TEXT NOT NULL,
  status TEXT NOT NULL,
  gateway TEXT NOT NULL,
  entries_locked_after INTEGER NOT NULL,
  payload_json TEXT NOT NULL
);

CREATE TABLE human_approvals (
  approval_id TEXT PRIMARY KEY,
  intent_id TEXT NOT NULL,
  approver TEXT NOT NULL,
  confirmed_at TEXT NOT NULL,
  payload_json TEXT NOT NULL
);

CREATE TABLE mode_transitions (
  seq INTEGER PRIMARY KEY AUTOINCREMENT,
  at TEXT NOT NULL,
  from_mode TEXT,
  to_mode TEXT NOT NULL,
  actor TEXT NOT NULL,
  result TEXT NOT NULL,
  detail TEXT NOT NULL
);

CREATE TABLE data_quality_events (
  seq INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,
  symbol TEXT,
  feed TEXT NOT NULL,
  kind TEXT NOT NULL,
  severity TEXT NOT NULL,
  detail_json TEXT NOT NULL
);
CREATE INDEX ix_dq_ts ON data_quality_events(ts);

CREATE TABLE failures (
  seq INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,
  component TEXT NOT NULL,
  kind TEXT NOT NULL,
  detail TEXT NOT NULL,
  causal_id TEXT
);
