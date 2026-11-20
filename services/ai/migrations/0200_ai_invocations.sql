-- services/ai: LLM invocation gate ledger (reserved range 0200-0299).
-- One row per gate decision for a selected candidate. RESERVED rows that never settle (crash mid-run) are
-- counted at their reserved amount, so a restart can never reset the daily run/cost budget (fail closed).
CREATE TABLE ai_invocations (
  invocation_id TEXT PRIMARY KEY,
  causal_id TEXT NOT NULL,
  symbol TEXT NOT NULL,
  decision_ts TEXT NOT NULL,
  wall_ts TEXT NOT NULL,
  budget_day TEXT NOT NULL,
  transport TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('RESERVED', 'SETTLED', 'DENIED')),
  reason TEXT NOT NULL DEFAULT '',
  reserved_usd TEXT NOT NULL,
  actual_usd TEXT,
  settled_at TEXT
);
CREATE INDEX ix_ai_invocations_day ON ai_invocations(budget_day, status);
CREATE INDEX ix_ai_invocations_symbol ON ai_invocations(symbol, wall_ts);
