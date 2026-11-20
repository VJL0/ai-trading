"""Operating modes and the compiled-in execution boundary (spec §8).

``LIVE_EXECUTION_COMPILED`` and ``BOUNDED_LIVE_COMPILED`` are module constants, not configuration.
No environment variable, prompt, UI action or API call can change them at runtime; enabling real
money requires a code change plus the separate human release process documented in
docs/runbooks/live-readiness.md. Every gateway constructor and the final order boundary call
``assert_gateway_allowed``.
"""

from __future__ import annotations

from typing import Final

from packages.contracts.enums import ExecutionLabel, GatewayKind, TradingMode

LIVE_EXECUTION_COMPILED: Final[bool] = False
BOUNDED_LIVE_COMPILED: Final[bool] = False

DEFAULT_MODE: Final[TradingMode] = TradingMode.PAPER

_ALLOWED: Final[dict[TradingMode, frozenset[GatewayKind]]] = {
    TradingMode.RESEARCH_ONLY: frozenset(),
    TradingMode.REPLAY: frozenset({GatewayKind.FAKE}),
    # Legacy values remain for isolated research harnesses and persisted audit records.
    # The runtime factory rejects both LEAN execution selections.
    TradingMode.PAPER: frozenset({GatewayKind.FAKE, GatewayKind.LEAN_PAPER_SIM,
                                  GatewayKind.LEAN_ALPACA_PAPER, GatewayKind.ALPACA_PAPER}),
    TradingMode.LIVE_APPROVAL: frozenset({GatewayKind.LEAN_ALPACA_LIVE}),
    TradingMode.BOUNDED_LIVE: frozenset(),
}

LIVE_GATEWAYS: Final[frozenset[GatewayKind]] = frozenset({GatewayKind.LEAN_ALPACA_LIVE})


class ModeViolation(Exception):
    """An action is not permitted in the current mode. Always fail closed."""


class LiveActivationDenied(ModeViolation):
    def __init__(self, failed_requirements: list[str]) -> None:
        self.failed_requirements = failed_requirements
        super().__init__("live activation denied: " + "; ".join(failed_requirements))


def namespace(mode: TradingMode) -> str:
    """Isolation namespace for DB files, keys and secrets. Live never shares paper state."""
    return {
        TradingMode.RESEARCH_ONLY: "research",
        TradingMode.REPLAY: "replay",
        TradingMode.PAPER: "paper",
        TradingMode.LIVE_APPROVAL: "live",
        TradingMode.BOUNDED_LIVE: "live",
    }[mode]


def client_order_tag(mode: TradingMode, gateway: GatewayKind) -> str:
    if gateway in LIVE_GATEWAYS:
        return "live"
    if mode == TradingMode.REPLAY:
        return "replay"
    if gateway in (GatewayKind.LEAN_ALPACA_PAPER, GatewayKind.ALPACA_PAPER):
        return "paper"
    return "sim"


def execution_label(mode: TradingMode, gateway: GatewayKind) -> ExecutionLabel:
    if gateway in LIVE_GATEWAYS:
        return ExecutionLabel.LIVE
    if gateway in (GatewayKind.LEAN_ALPACA_PAPER, GatewayKind.ALPACA_PAPER):
        return ExecutionLabel.PAPER
    return ExecutionLabel.SIMULATED


def live_activation_failures(
    mode: TradingMode,
    *,
    live_credentials_present: bool = False,
    release_lock_enabled: bool = False,
    broker_identity_verified: bool = False,
) -> list[str]:
    """All unmet requirements for routing a real-money order. Empty list == allowed (never in this build)."""
    failures: list[str] = []
    if mode == TradingMode.BOUNDED_LIVE:
        failures.append("BOUNDED_LIVE is compiled off and cannot be activated at runtime")
        if not BOUNDED_LIVE_COMPILED:
            failures.append("BOUNDED_LIVE_COMPILED is False in this build")
    if not LIVE_EXECUTION_COMPILED:
        failures.append("LIVE_EXECUTION_COMPILED is False in this build")
    if mode not in (TradingMode.LIVE_APPROVAL, TradingMode.BOUNDED_LIVE):
        failures.append(f"mode {mode} is not a live mode")
    if not live_credentials_present:
        failures.append("separately provisioned live credentials are absent")
    if not release_lock_enabled:
        failures.append("human-controlled release lock is not enabled")
    if not broker_identity_verified:
        failures.append("live broker account identity is not verified")
    return failures


def assert_gateway_allowed(mode: TradingMode, gateway: GatewayKind) -> None:
    """Raise unless ``gateway`` may receive orders in ``mode``. Live gateways are always refused here."""
    if gateway in LIVE_GATEWAYS:
        raise LiveActivationDenied(live_activation_failures(mode))
    if gateway not in _ALLOWED[mode]:
        raise ModeViolation(f"gateway {gateway} is not permitted in mode {mode}")


def allowed_gateways(mode: TradingMode) -> frozenset[GatewayKind]:
    if not LIVE_EXECUTION_COMPILED:
        return frozenset(g for g in _ALLOWED[mode] if g not in LIVE_GATEWAYS)
    return _ALLOWED[mode]
