"""LIVE_APPROVAL / BOUNDED_LIVE release gate: documented, NON-EXECUTABLE interface (spec §8, acceptance 12).

Routing a real-money order would require *all* of the following; this build satisfies none of the
code-level ones, and ``activate()`` always raises ``LiveActivationDenied``:

1. ``LIVE_EXECUTION_COMPILED`` is True (a deliberate code change + review; it is a module constant, not config);
2. separately provisioned live credentials (``ALPACA_LIVE_KEY_ID`` / ``ALPACA_LIVE_SECRET_KEY``, distinct
   names from paper; only presence is ever checked here, values are never read);
3. a user-enabled release lock file ``var/live/keys/release.lock`` signed by the operator key
   (signature verification is intentionally not implemented in this build => never satisfied);
4. verified broker account identity (live account number confirmed out-of-band by the human);
5. a fresh signed human confirmation **per order** (``HumanApproval`` bound to the intent id);
6. independent risk approval of that order (risk-engine-signed intent for a live mode).

``BOUNDED_LIVE`` is additionally compiled off (``BOUNDED_LIVE_COMPILED``) and cannot be reached by prompting,
environment variables, CLI flags, GUI requests or model output. Every activation attempt is audited and
recorded in ``mode_transitions`` as DENIED.

CLI (for operators; always exits non-zero): ``python -m services.order_ledger.live_gate --activate``. The attempt is
recorded in the current namespace's ledger when that ledger exists (the CLI never creates a ledger).
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from dataclasses import dataclass
from typing import NoReturn

from packages.contracts.enums import GatewayKind, TradingMode
from packages.core import modes
from packages.core.audit import append_audit
from packages.core.clock import Clock, SystemClock
from packages.core.config import Settings
from packages.core.db import Database
from packages.core.secrets import SecretName, SecretStore

from .timeutil import iso


@dataclass(frozen=True)
class LiveRequirement:
    key: str
    description: str


LIVE_REQUIREMENTS: tuple[LiveRequirement, ...] = (
    LiveRequirement("compiled", "LIVE_EXECUTION_COMPILED must be True (code change; compiled off in this build)"),
    LiveRequirement("live_credentials", "separately provisioned live credentials (ALPACA_LIVE_KEY_ID/SECRET)"),
    LiveRequirement("release_lock", "operator-signed release lock file var/live/keys/release.lock"),
    LiveRequirement("broker_identity", "live broker account identity verified by the human operator"),
    LiveRequirement("per_order_confirmation", "fresh signed human confirmation for every order"),
    LiveRequirement("independent_risk", "independent risk-engine approval of every live order"),
)


def record_mode_decision(
    db: Database, *, requested_mode: str, actor: str, result: str, detail: str, clock: Clock | None = None,
    from_mode: str | None = None,
) -> None:
    """Append a ``mode_transitions`` row + audit event (same transaction)."""
    at = (clock or SystemClock()).now()

    def _w(conn: sqlite3.Connection) -> None:
        conn.execute(
            "INSERT INTO mode_transitions(at, from_mode, to_mode, actor, result, detail) VALUES (?,?,?,?,?,?)",
            (iso(at), from_mode, requested_mode, actor[:100], result, detail[:2000]),
        )
        append_audit(conn, event_type=f"MODE_{result}", source=actor[:100], mode=requested_mode, ts=at,
                     payload={"requested_mode": requested_mode, "from_mode": from_mode, "result": result,
                              "detail": detail[:2000]})

    db.write(_w)


class LiveReleaseGate:
    """Non-executable live release gate. ``evaluate()`` lists unmet requirements; ``activate()`` always raises."""

    def __init__(self, settings: Settings, secret_store: SecretStore, *, db: Database | None = None,
                 clock: Clock | None = None) -> None:
        self.settings = settings
        self.secrets = secret_store
        self.db = db
        self.clock = clock or SystemClock()

    def _live_credentials_present(self) -> bool:
        # Presence only (SecretStore.has never returns the value; SecretStore.get refuses live names).
        return self.secrets.has(SecretName.ALPACA_LIVE_KEY_ID) and self.secrets.has(SecretName.ALPACA_LIVE_SECRET_KEY)

    def _release_lock_valid(self) -> bool:
        # The operator-signed release lock format is documented but verification is deliberately not
        # implemented in this build, so a file's mere existence never counts.
        return False

    def evaluate(self, requested_mode: TradingMode | None = None, gateway: GatewayKind | None = None) -> list[str]:
        mode = requested_mode or self.settings.mode
        if mode not in (TradingMode.LIVE_APPROVAL, TradingMode.BOUNDED_LIVE):
            mode = TradingMode.LIVE_APPROVAL  # evaluate what a live activation would need
        failures = modes.live_activation_failures(
            mode, live_credentials_present=self._live_credentials_present(), release_lock_enabled=self._release_lock_valid(),
            broker_identity_verified=False,
        )
        failures.append("fresh signed human confirmation per order: no live confirmation flow exists in this build")
        failures.append("independent risk approval: the risk engine refuses to sign live intents in this build")
        if (gateway or self.settings.gateway) in modes.LIVE_GATEWAYS:
            failures.append(f"gateway {GatewayKind.LEAN_ALPACA_LIVE.value} is interface-only in this build")
        return list(dict.fromkeys(failures))

    def activate(self, *, requested_by: str, requested_mode: TradingMode = TradingMode.LIVE_APPROVAL) -> NoReturn:
        """Always denied in this build. Records the attempt, then raises ``LiveActivationDenied``."""
        failures = self.evaluate(requested_mode)
        failures.append("LiveReleaseGate.activate() is non-executable in this build")
        if self.db is not None:
            record_mode_decision(
                self.db, requested_mode=requested_mode.value, actor=f"live_gate:{requested_by}", result="DENIED",
                detail="; ".join(failures), clock=self.clock, from_mode=self.settings.mode.value,
            )
        raise modes.LiveActivationDenied(failures)

    def request_from(self, source: str, request: dict[str, object]) -> NoReturn:
        """Any live request from a GUI, model output, API body or CLI is routed here and denied."""
        raw_mode = str(request.get("mode") or request.get("requested_mode") or TradingMode.LIVE_APPROVAL.value).upper()
        try:
            requested = TradingMode(raw_mode)
        except ValueError:
            requested = TradingMode.LIVE_APPROVAL
        if requested not in (TradingMode.LIVE_APPROVAL, TradingMode.BOUNDED_LIVE):
            requested = TradingMode.LIVE_APPROVAL
        self.activate(requested_by=source, requested_mode=requested)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="services.order_ledger.live_gate", description="Live release gate (always denies).")
    parser.add_argument("--activate", action="store_true", help="attempt live activation (always denied)")
    parser.add_argument("--mode", default="LIVE_APPROVAL", choices=["LIVE_APPROVAL", "BOUNDED_LIVE"])
    args = parser.parse_args(argv)
    settings = Settings()
    # Audit into the operating ledger of the current namespace when it exists (never creates one).
    db = Database(settings.db_path) if settings.db_path.is_file() else None
    try:
        gate = LiveReleaseGate(settings, SecretStore(settings.project_root, use_keychain=False), db=db)
        if not args.activate:
            for f in gate.evaluate(TradingMode(args.mode)):
                print(f"UNMET: {f}")
            return 3
        try:
            gate.activate(requested_by="cli", requested_mode=TradingMode(args.mode))
        except modes.LiveActivationDenied as exc:
            print(f"DENIED: {exc}", file=sys.stderr)
            return 3
        return 4  # pragma: no cover - activate() never returns
    finally:
        if db is not None:
            db.close()


if __name__ == "__main__":
    sys.exit(main())
