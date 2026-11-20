-- 0500_activity: live operations log (pipeline stages, agent/system activity, decision cards). Append-only.
CREATE TABLE activity_events (
  seq INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,
  kind TEXT NOT NULL,
  cycle_id TEXT,
  causal_id TEXT,
  session_id TEXT,
  stage TEXT,
  status TEXT,
  item TEXT,
  actor TEXT,
  message TEXT,
  duration_ms INTEGER,
  payload_json TEXT NOT NULL
);
CREATE INDEX ix_activity_ts ON activity_events(ts);
CREATE INDEX ix_activity_kind ON activity_events(kind, seq);
CREATE INDEX ix_activity_cycle ON activity_events(cycle_id);
CREATE TRIGGER activity_events_no_update BEFORE UPDATE ON activity_events
BEGIN SELECT RAISE(ABORT, 'activity_events is append-only'); END;
CREATE TRIGGER activity_events_no_delete BEFORE DELETE ON activity_events
BEGIN SELECT RAISE(ABORT, 'activity_events is append-only'); END;
