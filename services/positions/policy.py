"""Position-management policy: how the position manager protects and exits positions.

These are *operating* parameters, not risk limits. The risk engine stays the independent authority for every order
(long-only, session, quote, band, rate limits, signature), so a bad value here can only make the manager ask for
orders the risk engine refuses. The policy is immutable; its version (SHA-256 of the canonical JSON) is audited at
startup and stored on every position plan.

Timings are in seconds of the runtime clock (simulated in REPLAY).
"""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field

from packages.contracts.types import Dec


class PositionPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = True
    protective_stops: bool = Field(
        default=True,
        description="rest a broker-held STOP_LIMIT SELL for the whole position (protects while this process is down; "
                    "a trigger, never a guaranteed exit price)",
    )
    stop_limit_gap_bps: Dec = Field(default=Decimal("50"), description="protective limit = stop - gap (floored to tick)")
    protect_grace_s: int = Field(default=30, ge=1, le=600,
                                 description="an open position without confirmed protection after this is UNPROTECTED")
    entry_fill_timeout_s: int = Field(default=15, ge=1, le=600,
                                      description="a working entry older than this (partial or unfilled) is canceled")
    default_horizon_minutes: int = Field(default=60, ge=1, le=390)
    flatten_before_close_s: int = Field(default=600, ge=60, le=3600,
                                        description="exit before the XNYS close: daily for intraday plans, "
                                                    "on the last holding session for authorized multi-day plans")
    exit_reprice_s: int = Field(default=10, ge=1, le=300, description="cancel and re-price an unfilled exit after this")
    breach_grace_s: int = Field(default=5, ge=0, le=300,
                                description="a triggered stop still unfilled after this is escalated to a marketable exit")
    cancel_stuck_s: int = Field(default=30, ge=1, le=600,
                                description="a cancel still pending after this requests a broker reconciliation")
    reconcile_request_interval_s: int = Field(default=15, ge=1, le=600)

    def canonical_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))

    @property
    def version(self) -> str:
        return "ppol_" + hashlib.sha256(self.canonical_json().encode()).hexdigest()[:16]
