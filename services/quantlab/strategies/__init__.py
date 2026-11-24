"""Strategy families. Each family exposes ``variants()`` (a pre-declared, finite parameter grid — every variant is
recorded in the trial ledger) and ``targets(panel, universe, **params)`` returning (T, N) weights decided at the
close of each row's session. Nothing here reads the protected holdout: callers pass development-sliced panels."""
