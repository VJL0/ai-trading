# Paper risk profile: `paper-autonomous`

`paper-autonomous.json` is a PAPER risk policy designed for the autopilot (`services/autopilot`), which trades with
no operator commands. It passes `services.risk.limits.validate_policy` (version `pol_4227acc81a7275df` at creation).
Applying a policy is still an operator action: `PolicyStore.apply_change(..., operator_token=...)` is the only
writer, and neither the autopilot nor any agent holds the operator token.

## Why these numbers

The research results (docs/verification) produced two constraints:

* **Research and execution must evaluate the same economics.** On 2026-11-23, the cross-sectional families held
  10–20 names. The default policy allows one position at ≤25 % of a $5,000 account, so even a strategy that passed
  validation averaged 1–22 % exposure and 0.3–4.7 % CAGR. This profile allows 10 equal slots of 12 %, which matches
  the top-N rule that research evaluates.
* **Costs, not limits, are what kill the edge.** Family K's top decile earned 4–9 bps against about 13 bps of
  round-trip cost. The profile therefore restricts trading to the 200 most liquid names and tightens the spread
  cap. It does not loosen the loss limits.

| Limit | Default | paper-autonomous | Reason |
|---|---|---|---|
| Symbols | 24 | 200 (see `paper-autonomous-universe.json`) | Top 200 by median dollar volume in the development year 2024-11-20…2025-11-20, closing ≥ $10 every day of that year, no leveraged or inverse ETPs. It uses no holdout data. The 200th name trades about $506M a day. |
| Max positions | 1 | 10 | Equal to the hard ceiling and to research's top-N. |
| Order / position notional | $2,000 / $2,500 | $600 / $600 | About 12 % of a $5,000 account per slot. |
| Concentration | 25 % | 12 % | Diversifies across 10 slots. |
| Pending orders, orders/min, orders/day | 1, 3, 20 | 10, 6, 60 | Room to rebalance 10 slots plus their protective stops. |
| Max daily loss | $250 | $150 (3 %) | Tighter. More positions must not mean more loss per day. |
| Planned loss per trade | $50 | $15 | 2.5 % of a $600 slot. That covers stops up to 250 bps. |
| Max spread | 25 bps | 15 bps | Costs matter most. Liquid names quote 1–3 bps. |
| Overnight holding | off | off | No multi-day strategy has validated. Turning it on stays an operator decision tied to a validated strategy id. |

Everything else (long only, no margin or shorting, LIMIT and DAY orders only, regular session only, $5 minimum
price, quote age, signed intents, kill switch) is unchanged.

## Regenerating

The universe comes from `datasets/quantlab/equity_daily/sip_raw_20261120`. The method is recorded inside the JSON.
Re-run it when the panel is refreshed. Keep the window inside the development period so that holdout data never
shapes the trading universe.

## First replay evidence (2026-11-25)

Setup: `services/replay`, strategy `runtime-baseline` (the engine a paper session uses with no trained model),
real SIP 5-minute bars for the 24 default symbols, 2025-10-01…2025-11-19 (development period, 36 sessions), and a
$5,000 account. Quotes were synthesized from the bars, so this is REPLAY evidence, not paper fills.

| Policy | Net P&L | Gross P&L | Costs | Sharpe | Max DD | Mean exposure | Round trips | Top rejections |
|---|---|---|---|---|---|---|---|---|
| default | −$29.21 | +$22.22 | $51.43 | −1.23 | −1.3 % | 9.5 % | 130 | orders/day 1,155, concentration 700, positions 380 |
| paper-autonomous | −$56.10 | −$22.08 | $34.02 | −3.67 | −1.2 % | 5.6 % | 163 | position notional 506, concentration 510 |

What this shows:

* The baseline has no edge under either policy. Research reached the same verdict, and the policy cannot fix
  that.
* Under `paper-autonomous` the signal engine still sizes every proposal at $1,000 (`QuantSignalEngine`
  `order_notional`), so the $600 slot cap rejects about half of its entries and exposure falls. Before this profile
  can reach its intended exposure, the engine's order notional has to follow the policy's slot size. That is a
  sizing-contract change for the signal and risk path, and it has not been made.
* Losses stay small and bounded under both policies (max drawdown about 1.2 %), which is the point of the limits.
