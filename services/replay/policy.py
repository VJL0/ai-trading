"""Policy profiles for replay evaluation. Read-only with respect to every operator store.

``load_policy`` resolves ``default`` (the built-in ``RiskPolicy()``), ``active`` (the operator's active PAPER policy,
read through SQLite ``mode=ro`` + ``query_only`` and the production integrity checks) or a JSON file
(``RiskPolicy`` fields; decimal strings). The resolved policy is installed only in the replay's disposable workspace,
through that workspace's own audited ``PolicyStore.apply_change``. The active policy is never changed.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import closing
from pathlib import Path

from packages.contracts.policy import RiskPolicy
from packages.core.config import PROJECT_ROOT
from services.risk.limits import validate_policy
from services.risk.policy_store import active_policy_conn


class PolicyProfileError(ValueError):
    pass


def active_policy_path(project_root: Path | None = None) -> Path:
    """The PAPER namespace ledger of the operator's project (``LAT_PROJECT_ROOT`` honoured, nothing else read)."""
    root = project_root or Path(os.environ.get("LAT_PROJECT_ROOT") or PROJECT_ROOT)
    return root / "var" / "paper" / "ledger.sqlite"


def load_policy(spec: str, *, project_root: Path | None = None) -> tuple[RiskPolicy, str]:
    """``(policy, source description)``; raises ``PolicyProfileError`` (fail closed) on anything invalid."""
    if spec == "default":
        policy, source = RiskPolicy(), "default: RiskPolicy() built-in defaults"
    elif spec == "active":
        path = active_policy_path(project_root)
        if not path.is_file():
            raise PolicyProfileError(f"no paper ledger at {path}: no active policy to read (pass a policy JSON file)")
        try:
            with closing(sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True, timeout=10)) as conn:
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA query_only=ON")
                conn.execute("BEGIN")
                policy = active_policy_conn(conn)
        except Exception as exc:  # integrity failure, missing table ...: never guess a policy
            raise PolicyProfileError(f"active policy unreadable ({type(exc).__name__}: {exc})") from exc
        source = f"active: read-only from {path}"
    else:
        path = Path(spec)
        if not path.is_file():
            raise PolicyProfileError(f"policy file {path} not found (use 'default', 'active' or a JSON file)")
        try:
            policy = RiskPolicy.model_validate_json(path.read_text())
        except Exception as exc:
            raise PolicyProfileError(f"policy file {path} is not a valid RiskPolicy: {exc}") from exc
        source = f"file: {path}"
    problems = validate_policy(policy)
    if problems:
        raise PolicyProfileError("policy violates the hard ceilings/invariants: " + "; ".join(problems))
    return policy, f"{source} (version {policy.version})"
