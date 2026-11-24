-- Existing plans remain intraday. A holding period is fixed at the first fill and never extended on restart.
ALTER TABLE position_plans ADD COLUMN holding_sessions INTEGER NOT NULL DEFAULT 1 CHECK (holding_sessions BETWEEN 1 AND 20);
ALTER TABLE position_plans ADD COLUMN entry_risk_policy_version TEXT;
