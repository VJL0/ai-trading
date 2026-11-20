-- 0700_sessions: autonomous paper trading sessions, their decision cycles and recurring schedules.
-- Written only through the single SQLite writer (Database.write). Timestamps are ISO-8601 UTC strings; they are
-- compared in Python after parsing (never as strings). Session states follow docs/architecture/api-contract.md.
CREATE TABLE trading_sessions (
  session_id TEXT PRIMARY KEY,
  state TEXT NOT NULL CHECK (state IN ('SCHEDULED','RUNNING','STOPPING','COMPLETED','STOPPED','INTERRUPTED','FAILED')),
  created_at TEXT NOT NULL,
  started_at TEXT,
  ends_at TEXT NOT NULL,
  ended_at TEXT,
  symbols_json TEXT NOT NULL,
  cadence_minutes INTEGER NOT NULL CHECK (cadence_minutes BETWEEN 1 AND 60),
  duration_minutes INTEGER NOT NULL CHECK (duration_minutes BETWEEN 1 AND 480),
  use_ai INTEGER NOT NULL CHECK (use_ai IN (0, 1)),
  requested_by TEXT NOT NULL,
  schedule_id TEXT,
  stop_reason TEXT,
  mode TEXT NOT NULL,
  execution_label TEXT NOT NULL,
  owner_process TEXT NOT NULL,
  cycles_run INTEGER NOT NULL DEFAULT 0,
  trades INTEGER NOT NULL DEFAULT 0,
  consecutive_failures INTEGER NOT NULL DEFAULT 0,
  last_cycle_at TEXT,
  next_cycle_at TEXT,
  ai_suspended_until TEXT,
  resumed_count INTEGER NOT NULL DEFAULT 0,
  updated_at TEXT NOT NULL
);
-- At most one active (RUNNING/STOPPING) session per ledger, enforced by the database itself.
CREATE UNIQUE INDEX ux_one_active_session ON trading_sessions((1)) WHERE state IN ('RUNNING','STOPPING');
CREATE INDEX ix_trading_sessions_created ON trading_sessions(created_at);
-- Terminal states are final: a finished session can never be revived.
CREATE TRIGGER trading_sessions_terminal_final BEFORE UPDATE OF state ON trading_sessions
WHEN OLD.state IN ('COMPLETED','STOPPED','INTERRUPTED','FAILED') AND NEW.state != OLD.state
BEGIN SELECT RAISE(ABORT, 'trading session state is terminal'); END;
CREATE TRIGGER trading_sessions_no_delete BEFORE DELETE ON trading_sessions
BEGIN SELECT RAISE(ABORT, 'trading sessions are never deleted'); END;

-- One row per decision cycle a session ran (causal id links to every artifact of the cycle). Append-only.
CREATE TABLE session_cycles (
  seq INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL REFERENCES trading_sessions(session_id),
  cycle_no INTEGER NOT NULL,
  scheduled_for TEXT NOT NULL,
  started_at TEXT NOT NULL,
  finished_at TEXT NOT NULL,
  causal_id TEXT,
  outcome TEXT NOT NULL,
  candidate_symbol TEXT,
  client_order_id TEXT,
  reason TEXT,
  use_ai INTEGER NOT NULL CHECK (use_ai IN (0, 1)),
  entries_locked INTEGER,
  lock_reasons_json TEXT NOT NULL DEFAULT '[]',
  note TEXT,
  UNIQUE (session_id, cycle_no)
);
CREATE INDEX ix_session_cycles_causal ON session_cycles(causal_id);
CREATE TRIGGER session_cycles_no_update BEFORE UPDATE ON session_cycles
BEGIN SELECT RAISE(ABORT, 'session_cycles is append-only'); END;
CREATE TRIGGER session_cycles_no_delete BEFORE DELETE ON session_cycles
BEGIN SELECT RAISE(ABORT, 'session_cycles is append-only'); END;

-- Recurring sessions (weekdays only, inside XNYS regular hours; start time is America/New_York wall time).
CREATE TABLE session_schedules (
  schedule_id TEXT PRIMARY KEY,
  weekdays_json TEXT NOT NULL,
  start_time_et TEXT NOT NULL,
  duration_minutes INTEGER NOT NULL CHECK (duration_minutes BETWEEN 1 AND 480),
  symbols_json TEXT NOT NULL,
  cadence_minutes INTEGER NOT NULL CHECK (cadence_minutes BETWEEN 1 AND 60),
  use_ai INTEGER NOT NULL CHECK (use_ai IN (0, 1)),
  enabled INTEGER NOT NULL DEFAULT 1 CHECK (enabled IN (0, 1)),
  next_run_at TEXT,
  created_by TEXT NOT NULL,
  created_at TEXT NOT NULL,
  cancelled_at TEXT,
  cancelled_by TEXT,
  updated_at TEXT NOT NULL
);
CREATE TRIGGER session_schedules_no_delete BEFORE DELETE ON session_schedules
BEGIN SELECT RAISE(ABORT, 'session schedules are cancelled, never deleted'); END;

-- Exactly one record per schedule and trading day: the primary key makes a scheduled start happen at most once,
-- even across restarts. STARTED = a session was created; MISSED = the window passed while the service was not
-- running (never run late); SKIPPED = due but refused (another session active, mode, symbols no longer allowed).
CREATE TABLE session_schedule_runs (
  schedule_id TEXT NOT NULL REFERENCES session_schedules(schedule_id),
  run_date TEXT NOT NULL,
  due_at TEXT NOT NULL,
  window_end TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('STARTED','MISSED','SKIPPED')),
  session_id TEXT,
  detail TEXT NOT NULL,
  recorded_at TEXT NOT NULL,
  PRIMARY KEY (schedule_id, run_date)
);
CREATE TRIGGER session_schedule_runs_no_update BEFORE UPDATE ON session_schedule_runs
BEGIN SELECT RAISE(ABORT, 'session_schedule_runs is append-only'); END;
CREATE TRIGGER session_schedule_runs_no_delete BEFORE DELETE ON session_schedule_runs
BEGIN SELECT RAISE(ABORT, 'session_schedule_runs is append-only'); END;
