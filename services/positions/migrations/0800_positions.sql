-- 0800_positions: position lifecycle (reserved range 0800-0899).
-- Money/qty/price columns are TEXT decimals; timestamps are fixed-width ISO-8601 UTC strings.

-- One row per flat-to-flat round trip in a symbol. The plan (stop, deadline) is fixed when the position opens
-- and is the only input the position manager uses besides broker-authoritative ledger state; nothing here is a
-- second source of position quantity (that is always ledger_positions, i.e. booked fills).
CREATE TABLE position_plans (
  plan_id TEXT PRIMARY KEY,
  symbol TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('OPEN', 'CLOSED')),
  opened_at TEXT NOT NULL,              -- first entry fill (broker timestamp)
  trading_day TEXT NOT NULL,            -- America/New_York date of opened_at
  strategy_id TEXT,
  entry_client_order_id TEXT,
  stop_price TEXT NOT NULL,             -- planned protective stop: a trigger, not a guaranteed exit price
  stop_source TEXT NOT NULL,
  horizon_minutes INTEGER NOT NULL,
  exit_deadline TEXT NOT NULL,          -- time-based exit
  exit_reason TEXT,                     -- TIME | EOD | OVERNIGHT | STOP | STOP_BREACH | EXTERNAL
  exit_started_at TEXT,
  max_quantity TEXT NOT NULL DEFAULT '0',
  closed_at TEXT,
  close_reason TEXT,
  entry_quantity TEXT,
  entry_avg_price TEXT,
  exit_quantity TEXT,
  exit_avg_price TEXT,
  realized_pl TEXT,
  policy_version TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE UNIQUE INDEX ux_position_plans_open ON position_plans(symbol) WHERE status = 'OPEN';
CREATE INDEX ix_position_plans_closed ON position_plans(closed_at);

-- Append-only log of every position-manager decision that changed something (or was refused).
CREATE TABLE position_actions (
  seq INTEGER PRIMARY KEY AUTOINCREMENT,
  plan_id TEXT,
  symbol TEXT NOT NULL,
  at TEXT NOT NULL,
  action TEXT NOT NULL,                 -- PROTECT | TOP_UP | EXIT | CANCEL | RECONCILE | PLAN_OPENED | PLAN_CLOSED
  purpose TEXT,
  client_order_id TEXT,
  outcome TEXT NOT NULL,
  detail TEXT
);
CREATE INDEX ix_position_actions_symbol ON position_actions(symbol, seq);

CREATE TRIGGER position_actions_no_update BEFORE UPDATE ON position_actions
BEGIN SELECT RAISE(ABORT, 'position_actions is append-only'); END;
CREATE TRIGGER position_actions_no_delete BEFORE DELETE ON position_actions
BEGIN SELECT RAISE(ABORT, 'position_actions is append-only'); END;
CREATE TRIGGER position_plans_no_delete BEFORE DELETE ON position_plans
BEGIN SELECT RAISE(ABORT, 'position_plans are never deleted'); END;
