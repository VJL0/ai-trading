/**
 * Current decision card (SSE `decision`, initial value from /activity/current). Shows the exact
 * backend values (decimal strings) and the exact reason sentence; confidence is badged
 * "uncalibrated" unless the backend says `confidence_calibrated: true`.
 */
import type { DecisionCard as Card } from "../api/types";
import { compareDecimalStrings, formatBps, formatFractionAsPercent } from "../lib/decimal";
import { decisionTone, tailId } from "../ops/labels";
import { useApp } from "../state/AppContext";
import { Badge, EmptyState, KeyValues, Panel, Stamp } from "./ui";

function signedBps(value: string | null): string {
  if (value === null) return "—";
  const c = compareDecimalStrings(value, "0");
  return c !== null && c > 0 ? `+${formatBps(value, 2)}` : formatBps(value, 2);
}

/** "check_name: detail" -> [name, detail] (the backend formats failed checks this way). */
export function splitCheck(failure: string): [string, string] {
  const i = failure.indexOf(": ");
  return i > 0 ? [failure.slice(0, i), failure.slice(i + 2)] : [failure, ""];
}

export function DecisionCardView({ card, sim, synced }: { card: Card | null; sim: boolean; synced: boolean }) {
  if (!card) {
    return (
      <Panel id="decision" title="Current decision" source="SSE decision · /activity/current" className="decision-panel">
        <EmptyState title={synced ? "No decision yet" : "Loading decision…"} testId="decision-empty">
          The card fills in when a decision cycle reaches a NO_TRADE, risk verdict or submission.
        </EmptyState>
      </Panel>
    );
  }
  const tone = decisionTone(card.status);
  const net = compareDecimalStrings(card.net_edge_bps, "0");
  return (
    <Panel
      id="decision"
      title="Current decision"
      source="SSE decision · /activity/current"
      className="decision-panel"
      meta={
        <>
          <Badge tone={card.is_simulated ? "warn" : "info"}>{card.is_simulated ? "SIMULATED" : "not simulated"}</Badge>
          <Stamp ts={card.ts} sim={sim} />
        </>
      }
    >
      {/* keyed so a new card (or status change) replays the entrance animation */}
      <div className={`decision-card tone-${tone}`} key={`${card.causal_id ?? ""}:${card.status}:${card.seq ?? ""}`} data-testid="decision-card" data-status={card.status} data-action={card.action}>
        <div className="decision-head">
          <span className="decision-action" data-testid="decision-action">
            {card.action === "NO_TRADE" ? "NO TRADE" : card.action}
          </span>
          <span className="decision-symbol" data-testid="decision-symbol">
            {card.symbol ?? "—"}
          </span>
          <Badge tone={tone} testId="decision-status">
            {card.status}
          </Badge>
        </div>
        <p className="decision-reason" data-testid="decision-reason">
          {card.reason || "—"}
        </p>
        <KeyValues
          className="decision-grid"
          rows={[
            ["Side", card.side ?? "—"],
            ["Expected return", signedBps(card.expected_return_bps)],
            ["Expected cost", formatBps(card.expected_cost_bps, 2)],
            [
              "Predicted net edge",
              <span className={net === null ? "" : net > 0 ? "fact-good" : net < 0 ? "fact-bad" : ""} data-testid="decision-net-edge">
                {signedBps(card.net_edge_bps)}
              </span>,
            ],
            [
              "Confidence",
              <span data-testid="decision-confidence">
                {formatFractionAsPercent(card.confidence, 1)}{" "}
                {card.confidence !== null && (
                  <Badge tone={card.confidence_calibrated ? "good" : "warn"} testId="confidence-calibration">
                    {card.confidence_calibrated ? "calibrated" : "uncalibrated"}
                  </Badge>
                )}
              </span>,
            ],
            ["Volatility", formatBps(card.volatility_bps, 1)],
            ["Regime", card.regime ?? "UNKNOWN"],
            ["Proposer", card.proposer ?? "—"],
            ["Causal id", <code title={card.causal_id ?? undefined}>{tailId(card.causal_id, 12)}</code>],
          ]}
        />
        {card.risk_failures.length > 0 && (
          <div className="risk-failures" data-testid="risk-failures">
            <h3 className="subhead">Failed risk checks ({card.risk_failures.length})</h3>
            <ul>
              {card.risk_failures.map((f, i) => {
                const [name, detail] = splitCheck(f);
                return (
                  <li key={i}>
                    <strong>{name}</strong>
                    {detail && <span>: {detail}</span>}
                  </li>
                );
              })}
            </ul>
          </div>
        )}
      </div>
    </Panel>
  );
}

export function DecisionCard() {
  const { ops } = useApp();
  return <DecisionCardView card={ops.decision} sim={ops.clock.sim !== null} synced={ops.synced} />;
}
