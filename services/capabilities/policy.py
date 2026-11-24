"""Read-only access to the ACTIVE risk policy of the paper namespace (for capability and research reporting).

Opens the ledger SQLite file read-only (URI mode=ro), verifies the active row's integrity and audited activation
exactly as the runtime does (``services.risk.policy_store.active_policy_conn``), and returns ``None`` when it cannot
(no ledger yet, locked, corrupt): reporting then falls back to the build's default policy and says so.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

from packages.contracts.policy import RiskPolicy


def load_active_policy(project_root: Path, namespace: str = "paper") -> RiskPolicy | None:
    from services.risk.policy_store import PolicyStoreError, active_policy_conn

    path = project_root / "var" / namespace / "ledger.sqlite"
    if not path.is_file():
        return None
    try:
        with closing(sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True, timeout=5)) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("BEGIN")  # policy row and activation audit must come from the same read snapshot
            return active_policy_conn(conn)
    except (sqlite3.Error, PolicyStoreError):
        return None
