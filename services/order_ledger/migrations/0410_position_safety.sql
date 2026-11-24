-- 0410_position_safety: exit-aware locking (reserved range 0400-0499).
--
-- Halts keep their meaning ("stop new entries") and gain a scope. ENTRIES (every existing halt) blocks new
-- entries only; risk-reducing protective exits and liquidations keep running. ALL is the operator kill switch: it
-- also blocks exits (broker-held protective orders stay at the broker; nothing new is sent).
ALTER TABLE trading_halts ADD COLUMN scope TEXT NOT NULL DEFAULT 'ENTRIES' CHECK (scope IN ('ENTRIES', 'ALL'));

-- Live per-symbol safety state written by the position manager. Any active row locks new entries (computed live by
-- EntriesLock, like UNKNOWN orders; not an epoch flag) but never blocks the protective exits that resolve it.
-- A row is cleared only when broker-authoritative ledger state shows the position protected or flat.
CREATE TABLE position_safety (
  symbol TEXT PRIMARY KEY,
  state TEXT NOT NULL CHECK (state IN ('UNPROTECTED', 'UNVERIFIED', 'BREACHED', 'OVERNIGHT', 'EXITING', 'ORPHANED')),
  reason TEXT NOT NULL,
  since TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  active INTEGER NOT NULL CHECK (active IN (0, 1)),
  cleared_at TEXT,
  cleared_reason TEXT
);
CREATE INDEX ix_position_safety_active ON position_safety(active);
