"""Shared prediction -> trade-decision contract for research and the paper runtime (roadmap Phase 3).

Every forecast the decision layer sees declares its **basis**:

* ``GROSS``: an expected open-to-open return before any cost (the deterministic baseline);
* ``NET``: a model trained on ``net_bps = gross_bps - cost_rt_bps`` (``models.targets``), where the training cost is the
  CostModel round trip for the model's ``reference_notional`` with the decision-time quote spread.

``decide_edge`` turns either into one :class:`EdgeEstimate` that charges costs **exactly once**, at the live estimate:

* GROSS: ``net = gross - live_cost``;
* NET: ``gross = net_forecast + training_convention_cost`` (the cost the model already learned to subtract, re-priced
  with the same live inputs), then ``net = gross - live_cost``. With the order sized at the reference notional the two
  costs are identical and ``net == net_forecast``; a different order size only changes the impact term.

The decision is ``net - uncertainty_bps > threshold_bps``: an explicit, separately reported uncertainty buffer (e.g. a
validation-derived allowance) and an explicit threshold. Neither is folded silently into the forecast. The risk
engine stays independently authoritative downstream; this only decides whether a candidate is proposed.

Documented, intentional research/runtime differences: the runtime uses the live quote (research: the last quote
published by the decision time) and the live order notional (research: ``reference_notional``); everything else -
features, model, preprocessing, horizon, cost function, threshold and buffer - must be identical, which
``tests/unit/research/test_decision_parity.py`` checks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Literal

Basis = Literal["GROSS", "NET"]
GROSS: Final = "GROSS"
NET: Final = "NET"
DEFAULT_THRESHOLD_BPS: Final = 5.0
"""Minimum net edge (after costs and the uncertainty buffer) for a candidate; the signal engine's default."""


@dataclass(frozen=True)
class EdgeEstimate:
    horizon_minutes: int
    basis: Basis
    forecast_bps: float
    gross_bps: float
    assumed_cost_bps: float
    """Cost the forecast already deducted (0 for GROSS forecasts)."""
    live_cost_bps: float
    """Round-trip cost charged once at decision time (spread + fees + slippage + impact)."""
    net_bps: float
    uncertainty_bps: float
    threshold_bps: float

    @property
    def net_lower_bps(self) -> float:
        return self.net_bps - self.uncertainty_bps

    @property
    def trade(self) -> bool:
        return math.isfinite(self.net_lower_bps) and self.gross_bps > 0 and self.net_lower_bps > self.threshold_bps

    def explain(self) -> str:
        return (f"{self.basis} forecast {self.forecast_bps:.2f} bps -> gross {self.gross_bps:.2f} - cost "
                f"{self.live_cost_bps:.2f} = net {self.net_bps:.2f} bps; - uncertainty {self.uncertainty_bps:.2f} = "
                f"{self.net_lower_bps:.2f} vs threshold {self.threshold_bps:.2f} -> {'TRADE' if self.trade else 'NO_TRADE'}")


def decide_edge(forecast_bps: float, *, basis: Basis, live_cost_bps: float, assumed_cost_bps: float = 0.0,
                uncertainty_bps: float = 0.0, threshold_bps: float = DEFAULT_THRESHOLD_BPS,
                horizon_minutes: int) -> EdgeEstimate:
    """Combine a forecast with costs charged exactly once. Non-finite inputs yield a NO_TRADE estimate."""
    if basis not in (GROSS, NET):
        raise ValueError(f"unknown forecast basis {basis!r}")
    if uncertainty_bps < 0 or not math.isfinite(uncertainty_bps):
        raise ValueError("uncertainty_bps must be finite and >= 0")
    if basis == GROSS and assumed_cost_bps:
        raise ValueError("a GROSS forecast has deducted no cost")
    gross = forecast_bps + (assumed_cost_bps if basis == NET else 0.0)
    net = gross - live_cost_bps
    return EdgeEstimate(horizon_minutes=horizon_minutes, basis=basis, forecast_bps=forecast_bps, gross_bps=gross,
                        assumed_cost_bps=assumed_cost_bps if basis == NET else 0.0, live_cost_bps=live_cost_bps,
                        net_bps=net, uncertainty_bps=uncertainty_bps, threshold_bps=threshold_bps)
