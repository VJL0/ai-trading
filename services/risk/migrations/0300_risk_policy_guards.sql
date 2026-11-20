-- 0300_risk_policy_guards: risk service tables and extra immutability guards (reserved range 0300-0399).
-- 0001_core.sql already makes policy_json/policy_version/name/created_at/created_by immutable and forbids
-- deletes. Here the provenance columns become immutable too, and every operator change attempt (accepted
-- or rejected) is kept in an append-only table that points at its hash-chained audit event.

CREATE TRIGGER risk_policies_provenance_immutable BEFORE UPDATE OF approved_by, supersedes ON risk_policies
BEGIN SELECT RAISE(ABORT, 'risk policy provenance is immutable; create a new version'); END;

CREATE TABLE risk_policy_changes (
  change_id TEXT PRIMARY KEY,
  at TEXT NOT NULL,
  actor TEXT NOT NULL,
  result TEXT NOT NULL CHECK (result IN ('ACCEPTED', 'REJECTED')),
  from_version TEXT,
  to_version TEXT,
  reason TEXT NOT NULL,
  diff_json TEXT NOT NULL,
  violations_json TEXT NOT NULL,
  audit_event_id TEXT NOT NULL REFERENCES audit_events(event_id)
);
CREATE INDEX ix_risk_policy_changes_at ON risk_policy_changes(at);

CREATE TRIGGER risk_policy_changes_no_update BEFORE UPDATE ON risk_policy_changes
BEGIN SELECT RAISE(ABORT, 'risk_policy_changes is append-only'); END;
CREATE TRIGGER risk_policy_changes_no_delete BEFORE DELETE ON risk_policy_changes
BEGIN SELECT RAISE(ABORT, 'risk_policy_changes is append-only'); END;
