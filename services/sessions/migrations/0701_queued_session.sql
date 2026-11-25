DROP INDEX ux_one_active_session;
CREATE UNIQUE INDEX ux_one_active_session ON trading_sessions((1))
WHERE state IN ('SCHEDULED', 'RUNNING', 'STOPPING');
