/**
 * Order lifecycle view model (pure): ledger state + transitions + signed intent ->
 *   Proposed -> Validated -> Authorized -> Submitted -> Acknowledged -> Partially filled -> outcome
 * where the outcome is Filled | Rejected | Expired | Canceled (| Cancel pending | Reconciled), and
 * UNKNOWN (submission outcome in doubt) is a WARNING on the Submitted step.
 *
 * Mapping (api-contract + services/order_ledger): an order row exists only for a risk-approved,
 * signed intent, so PROPOSED/RISK_APPROVED -> Validated; a signed intent (non-empty signature) ->
 * Authorized; SUBMITTING/SUBMITTED -> Submitted; ACKNOWLEDGED; PARTIALLY_FILLED; FILLED;
 * REJECTED/EXPIRED -> Rejected; CANCEL_PENDING/CANCELED -> Canceled; RECONCILED (in-doubt order not
 * present at the broker) -> warning outcome. A step the ledger never recorded but that a later state
 * logically requires (a fill requires broker acceptance) is shown as done and flagged `implied`.
 */
import type { OrderDetailResponse, OrderRow } from "../api/types";
import { compareDecimalStrings } from "../lib/decimal";

export type StepId = "PROPOSED" | "VALIDATED" | "AUTHORIZED" | "SUBMITTED" | "ACKNOWLEDGED" | "PARTIALLY_FILLED" | "OUTCOME";
export type StepStatus = "done" | "current" | "pending" | "skipped" | "failed" | "canceled" | "warning";
export type Outcome = "OPEN" | "FILLED" | "REJECTED" | "EXPIRED" | "CANCELED" | "CANCEL_PENDING" | "UNKNOWN" | "RECONCILED";

export interface LifecycleStep {
  id: StepId;
  label: string;
  status: StepStatus;
  at: string | null;
  note: string | null;
  implied: boolean;
}

export interface LifecycleView {
  clientOrderId: string;
  state: string;
  outcome: Outcome;
  steps: LifecycleStep[];
  /** index of the step the order is at (current / warning / last done) */
  activeIndex: number;
  terminal: boolean;
}

/** Ledger state -> lifecycle step index (0..6). */
const STATE_RANK: Record<string, number> = {
  PROPOSED: 1,
  RISK_APPROVED: 1,
  HUMAN_APPROVED: 2,
  SUBMITTING: 3,
  SUBMITTED: 3,
  UNKNOWN: 3,
  ACKNOWLEDGED: 4,
  PARTIALLY_FILLED: 5,
  CANCEL_PENDING: 6,
  FILLED: 6,
  CANCELED: 6,
  REJECTED: 6,
  EXPIRED: 6,
  RECONCILED: 6,
};

const OUTCOME_OF: Record<string, Outcome> = {
  FILLED: "FILLED",
  REJECTED: "REJECTED",
  EXPIRED: "EXPIRED",
  CANCELED: "CANCELED",
  CANCEL_PENDING: "CANCEL_PENDING",
  UNKNOWN: "UNKNOWN",
  RECONCILED: "RECONCILED",
};

export const OUTCOME_LABEL: Record<Outcome, string> = {
  OPEN: "Filled / Rejected / Canceled",
  FILLED: "Filled",
  REJECTED: "Rejected",
  EXPIRED: "Rejected (expired)",
  CANCELED: "Canceled",
  CANCEL_PENDING: "Cancel pending",
  UNKNOWN: "Outcome unknown",
  RECONCILED: "Closed by reconciliation",
};

export function mapLedgerState(state: string): { rank: number; outcome: Outcome } {
  const rank = STATE_RANK[state];
  return { rank: rank ?? 0, outcome: OUTCOME_OF[state] ?? "OPEN" };
}

function firstAt(detail: OrderDetailResponse | null | undefined, states: readonly string[]): string | null {
  for (const t of detail?.transitions ?? []) if (states.includes(t.to_state)) return t.at;
  return null;
}

export function lifecycleOf(order: OrderRow, detail?: OrderDetailResponse | null): LifecycleView {
  const state = order.state;
  const { rank, outcome } = mapLedgerState(state);
  const known = state in STATE_RANK;
  const visited = new Set<string>([state]);
  for (const t of detail?.transitions ?? []) {
    visited.add(t.to_state);
    if (t.from_state) visited.add(t.from_state);
  }
  const filledSome = (compareDecimalStrings(order.filled_quantity, "0") ?? 0) > 0;
  const filledAll = compareDecimalStrings(order.filled_quantity, order.quantity) === 0 && filledSome;
  const terminal = ["FILLED", "CANCELED", "REJECTED", "EXPIRED", "RECONCILED"].includes(state);
  const failedEarly = terminal && outcome !== "FILLED";

  const steps: LifecycleStep[] = [];
  const push = (id: StepId, label: string, status: StepStatus, at: string | null, note: string | null = null, implied = false) =>
    steps.push({ id, label, status, at, note, implied });

  // 0 Proposed: the proposal always precedes the ledger order row.
  push("PROPOSED", "Proposed", "done", null, "trade proposal recorded before risk evaluation");
  // 1 Validated: the independent risk engine approved it (the order row exists only then).
  push("VALIDATED", "Validated", known ? "done" : "warning", order.created_at || null, known ? "risk engine approved" : `unrecognized ledger state ${state}`);
  // 2 Authorized: signed intent.
  const signature = detail?.intent && typeof detail.intent.signature === "string" ? detail.intent.signature : "";
  const awaitingHuman = detail?.intent?.requires_human_approval === true && rank <= 1;
  if (signature) {
    push("AUTHORIZED", "Authorized", awaitingHuman ? "current" : "done", detail?.intent?.created_at ?? null,
      awaitingHuman ? "signed intent awaiting per-order human approval" : `signed intent (${detail?.intent?.signature_alg ?? "ECDSA"})`);
  } else if (rank >= 3) {
    push("AUTHORIZED", "Authorized", "done", null, "signed intent required by the final gate", true);
  } else {
    push("AUTHORIZED", "Authorized", rank >= 2 ? "done" : "pending", null, rank >= 2 ? "human approved" : null);
  }
  // 3 Submitted
  const submittedVisited = visited.has("SUBMITTING") || visited.has("SUBMITTED");
  const submittedAt = firstAt(detail, ["SUBMITTED"]) ?? firstAt(detail, ["SUBMITTING"]);
  if (state === "UNKNOWN") {
    push("SUBMITTED", "Submitted", "warning", submittedAt, "submission outcome UNKNOWN: reconciliation queries the broker by client order id; never resubmitted");
  } else if (state === "SUBMITTING" || state === "SUBMITTED") {
    push("SUBMITTED", "Submitted", "current", submittedAt, state === "SUBMITTING" ? "submitting to the gateway" : "sent to the gateway, awaiting broker acknowledgement");
  } else if (submittedVisited) {
    push("SUBMITTED", "Submitted", "done", submittedAt);
  } else if (rank > 3 && !failedEarly) {
    push("SUBMITTED", "Submitted", "done", null, "implied by a later broker state", true);
  } else if (failedEarly && rank === 6 && !visited.has("ACKNOWLEDGED") && !filledSome) {
    push("SUBMITTED", "Submitted", "skipped", null, detail ? "never submitted" : "not recorded (order details not loaded)");
  } else {
    push("SUBMITTED", "Submitted", "pending", null);
  }
  // 4 Acknowledged
  const ackAt = firstAt(detail, ["ACKNOWLEDGED"]);
  if (state === "ACKNOWLEDGED") push("ACKNOWLEDGED", "Acknowledged", "current", ackAt, "accepted by the broker, working");
  else if (visited.has("ACKNOWLEDGED")) push("ACKNOWLEDGED", "Acknowledged", "done", ackAt);
  else if (filledSome || state === "PARTIALLY_FILLED" || state === "FILLED") push("ACKNOWLEDGED", "Acknowledged", "done", null, "implied by a fill", true);
  else if (terminal || state === "CANCEL_PENDING") push("ACKNOWLEDGED", "Acknowledged", "skipped", null, "not acknowledged");
  else push("ACKNOWLEDGED", "Acknowledged", "pending", null);
  // 5 Partially filled
  const partialAt = firstAt(detail, ["PARTIALLY_FILLED"]);
  if (state === "PARTIALLY_FILLED") push("PARTIALLY_FILLED", "Partially filled", "current", partialAt, `${order.filled_quantity} of ${order.quantity} filled`);
  else if (visited.has("PARTIALLY_FILLED") || (filledSome && !filledAll)) push("PARTIALLY_FILLED", "Partially filled", "done", partialAt, `${order.filled_quantity} of ${order.quantity} filled`);
  else if (state === "FILLED") push("PARTIALLY_FILLED", "Partially filled", "skipped", null, "filled in one execution");
  else if (terminal || state === "CANCEL_PENDING") push("PARTIALLY_FILLED", "Partially filled", "skipped", null, null);
  else push("PARTIALLY_FILLED", "Partially filled", "pending", null);
  // 6 Outcome
  const outAt = terminal || state === "CANCEL_PENDING" ? firstAt(detail, [state]) ?? order.updated_at ?? null : null;
  const outcomeStatus: Record<Outcome, StepStatus> = {
    OPEN: "pending",
    FILLED: "done",
    REJECTED: "failed",
    EXPIRED: "failed",
    CANCELED: "canceled",
    CANCEL_PENDING: "current",
    UNKNOWN: "pending",
    RECONCILED: "warning",
  };
  const outcomeNote: Partial<Record<Outcome, string>> = {
    FILLED: order.avg_fill_price ? `${order.filled_quantity} @ avg ${order.avg_fill_price}` : `${order.filled_quantity} filled`,
    RECONCILED: "in-doubt order not present at the broker; no fill was invented",
    UNKNOWN: "waiting for broker-authoritative reconciliation",
  };
  push("OUTCOME", OUTCOME_LABEL[outcome], outcomeStatus[outcome], outAt, outcomeNote[outcome] ?? null);

  let activeIndex = 0;
  steps.forEach((s, i) => {
    if (s.status === "done" || s.status === "current" || s.status === "warning" || s.status === "failed" || s.status === "canceled") activeIndex = i;
  });
  const warnIdx = steps.findIndex((s) => s.status === "warning" || s.status === "current");
  if (warnIdx >= 0 && !terminal) activeIndex = warnIdx;
  return { clientOrderId: order.client_order_id, state, outcome, steps, activeIndex, terminal };
}
