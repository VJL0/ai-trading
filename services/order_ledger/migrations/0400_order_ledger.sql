-- 0400_order_ledger: order-ledger service tables (reserved range 0400-0499).
-- Money/qty/price columns are TEXT decimals; timestamps are fixed-width ISO-8601 UTC strings.

-- Extra per-order columns used by the ledger state machine (0001_core.orders is never edited).
ALTER TABLE orders ADD COLUMN broker_status TEXT;
ALTER TABLE orders ADD COLUMN strategy_id TEXT;
ALTER TABLE orders ADD COLUMN requires_human_approval INTEGER NOT NULL DEFAULT 0;
ALTER TABLE orders ADD COLUMN human_approval_id TEXT;
ALTER TABLE orders ADD COLUMN expires_at TEXT;
ALTER TABLE orders ADD COLUMN unknown_since TEXT;
ALTER TABLE orders ADD COLUMN state_reason TEXT;
ALTER TABLE orders ADD COLUMN cancel_requested_at TEXT;

-- Per-fill accounting columns. cum_from/cum_to place an execution inside the order's cumulative filled
-- quantity, (cum_from, cum_to]; NULL when the broker did not report a cumulative quantity. synthetic=1 marks
-- a broker-authoritative catch-up fill booked by reconciliation from the broker's cumulative quantity.
ALTER TABLE fills ADD COLUMN cum_from TEXT;
ALTER TABLE fills ADD COLUMN cum_to TEXT;
ALTER TABLE fills ADD COLUMN synthetic INTEGER NOT NULL DEFAULT 0;
ALTER TABLE fills ADD COLUMN realized_pl TEXT NOT NULL DEFAULT '0';
ALTER TABLE fills ADD COLUMN trading_day TEXT;
ALTER TABLE fills ADD COLUMN event_id TEXT;

-- Exact running cost basis so average cost is order-independent (avg_cost = cost_basis / quantity).
ALTER TABLE ledger_positions ADD COLUMN cost_basis TEXT NOT NULL DEFAULT '0';

-- Intents handed to the local LEAN engine through the bridge. A row is created only after the
-- ExecutionService claimed the order (orders.state = SUBMITTING) and the final gate passed. Claiming by
-- the engine is a compare-and-set QUEUED -> CLAIMED, so an intent is never handed out twice.
CREATE TABLE engine_intent_queue (
  client_order_id TEXT PRIMARY KEY REFERENCES orders(client_order_id),
  intent_id TEXT NOT NULL UNIQUE,
  gateway TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('QUEUED', 'CLAIMED', 'EXPIRED', 'WITHDRAWN')),
  queued_at TEXT NOT NULL,
  expires_at TEXT NOT NULL,
  claimed_by TEXT,
  claimed_at TEXT,
  note TEXT
);
CREATE INDEX ix_engine_queue_status ON engine_intent_queue(status, queued_at);

-- Re-deliveries of an already recorded event (same dedupe key). Counted, never re-applied.
CREATE TABLE order_event_duplicates (
  dedupe_key TEXT PRIMARY KEY,
  client_order_id TEXT NOT NULL,
  first_event_id TEXT NOT NULL,
  deliveries INTEGER NOT NULL,
  first_seen TEXT NOT NULL,
  last_seen TEXT NOT NULL
);

-- When reconciliation booked a catch-up fill from broker cumulative quantities, an execution report
-- that arrives later for the same quantity is absorbed (not double-booked). One row per absorbed exec.
CREATE TABLE fill_absorptions (
  seq INTEGER PRIMARY KEY AUTOINCREMENT,
  client_order_id TEXT NOT NULL REFERENCES orders(client_order_id),
  broker_exec_id TEXT NOT NULL UNIQUE,
  reported_quantity TEXT NOT NULL,
  absorbed_quantity TEXT NOT NULL,
  event_id TEXT NOT NULL,
  at TEXT NOT NULL
);
CREATE INDEX ix_fill_absorptions_coid ON fill_absorptions(client_order_id);

-- Which entries-lock epoch a reconciliation run observed when it started. A CLEAN run may only clear
-- flags raised at or before that epoch (flags raised while it ran stay set).
CREATE TABLE reconciliation_meta (
  run_id TEXT PRIMARY KEY REFERENCES reconciliation_runs(run_id),
  observed_epoch INTEGER NOT NULL,
  startup INTEGER NOT NULL CHECK (startup IN (0, 1))
);

CREATE INDEX ix_fills_coid ON fills(client_order_id);
CREATE INDEX ix_fills_symbol_ts ON fills(symbol, ts);
CREATE INDEX ix_fills_day ON fills(trading_day);
CREATE INDEX ix_orders_created ON orders(created_at);
CREATE INDEX ix_orders_claimed ON orders(claimed_at);
CREATE INDEX ix_human_approvals_intent ON human_approvals(intent_id);
CREATE INDEX ix_reconciliation_runs_started ON reconciliation_runs(started_at);

CREATE TRIGGER fills_no_update BEFORE UPDATE ON fills
BEGIN SELECT RAISE(ABORT, 'fills are append-only'); END;
CREATE TRIGGER fills_no_delete BEFORE DELETE ON fills
BEGIN SELECT RAISE(ABORT, 'fills are append-only'); END;
CREATE TRIGGER order_events_no_delete BEFORE DELETE ON order_events
BEGIN SELECT RAISE(ABORT, 'order_events is append-only'); END;
CREATE TRIGGER order_transitions_no_update BEFORE UPDATE ON order_state_transitions
BEGIN SELECT RAISE(ABORT, 'order_state_transitions is append-only'); END;
CREATE TRIGGER order_transitions_no_delete BEFORE DELETE ON order_state_transitions
BEGIN SELECT RAISE(ABORT, 'order_state_transitions is append-only'); END;
CREATE TRIGGER fill_absorptions_no_update BEFORE UPDATE ON fill_absorptions
BEGIN SELECT RAISE(ABORT, 'fill_absorptions is append-only'); END;
CREATE TRIGGER reconciliation_runs_no_update BEFORE UPDATE ON reconciliation_runs
BEGIN SELECT RAISE(ABORT, 'reconciliation_runs is append-only'); END;
CREATE TRIGGER human_approvals_no_update BEFORE UPDATE ON human_approvals
BEGIN SELECT RAISE(ABORT, 'human_approvals is append-only'); END;
