-- services/market_data (reserved range 0100-0199).
-- Point-in-time universe / eligibility snapshots, ingestion run log and feed state transitions.
-- Analytical data (bars, quotes, corporate actions) lives in DuckDB + Parquet, not here.

CREATE TABLE md_universe_snapshots (
  snapshot_id TEXT PRIMARY KEY,
  asof_date TEXT NOT NULL,
  asof_ts TEXT NOT NULL,
  policy_version TEXT NOT NULL,
  feed TEXT,
  dataset_versions TEXT NOT NULL,
  eligible_symbols TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE INDEX ix_md_universe_asof ON md_universe_snapshots(asof_date);

CREATE TABLE md_ingestion_runs (
  run_id TEXT PRIMARY KEY,
  started_at TEXT NOT NULL,
  finished_at TEXT,
  provider TEXT NOT NULL,
  feed_mode TEXT NOT NULL,
  kind TEXT NOT NULL,
  symbols TEXT NOT NULL,
  window_start TEXT,
  window_end TEXT,
  status TEXT NOT NULL,
  rows_accepted INTEGER NOT NULL DEFAULT 0,
  rows_rejected INTEGER NOT NULL DEFAULT 0,
  gaps INTEGER NOT NULL DEFAULT 0,
  dataset_version TEXT,
  detail TEXT
);
CREATE INDEX ix_md_ingestion_started ON md_ingestion_runs(started_at);

CREATE TABLE md_feed_state_transitions (
  seq INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,
  provider TEXT NOT NULL,
  from_state TEXT,
  to_state TEXT NOT NULL,
  reason TEXT NOT NULL
);
