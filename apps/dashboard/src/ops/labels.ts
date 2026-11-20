/** Display metadata for pipeline stages, stage statuses and activity kinds (text + icon, never color alone). */
import type { ActivityKind, StageId, StageStatus } from "../api/types";

export type Tone = "neutral" | "good" | "warn" | "bad" | "info" | "running" | "muted";

export type IconName =
  | "idle"
  | "running"
  | "passed"
  | "rejected"
  | "error"
  | "skipped"
  | "scan"
  | "predict"
  | "evaluate"
  | "gpt_call"
  | "risk"
  | "reject"
  | "submit"
  | "fill"
  | "reconcile"
  | "session"
  | "agent_tool"
  | "info"
  | "audit"
  | "order"
  | "decision"
  | "warning"
  | "canceled"
  | "pending";

export const STAGE_LABEL: Record<StageId, string> = {
  MARKET_DATA: "Market Data",
  FEATURES: "Features",
  MODELS: "Models",
  GPT_REVIEW: "GPT Review",
  RISK_ENGINE: "Risk Engine",
  ORDER_DECISION: "Order Decision",
  BROKER: "Broker",
  RECONCILIATION: "Reconciliation",
};

export const STAGE_STATUS_META: Record<StageStatus, { label: string; tone: Tone; icon: IconName }> = {
  idle: { label: "idle", tone: "muted", icon: "idle" },
  running: { label: "running", tone: "running", icon: "running" },
  passed: { label: "passed", tone: "good", icon: "passed" },
  rejected: { label: "rejected", tone: "bad", icon: "rejected" },
  error: { label: "error", tone: "bad", icon: "error" },
  skipped: { label: "skipped", tone: "neutral", icon: "skipped" },
};

export const ACTIVITY_KIND_META: Record<ActivityKind, { label: string; tone: Tone; icon: IconName }> = {
  scan: { label: "scan", tone: "info", icon: "scan" },
  predict: { label: "predict", tone: "info", icon: "predict" },
  evaluate: { label: "evaluate", tone: "info", icon: "evaluate" },
  gpt_call: { label: "GPT call", tone: "info", icon: "gpt_call" },
  risk: { label: "risk", tone: "warn", icon: "risk" },
  reject: { label: "reject", tone: "bad", icon: "reject" },
  submit: { label: "submit", tone: "good", icon: "submit" },
  fill: { label: "fill", tone: "good", icon: "fill" },
  reconcile: { label: "reconcile", tone: "neutral", icon: "reconcile" },
  session: { label: "session", tone: "info", icon: "session" },
  agent_tool: { label: "agent tool", tone: "info", icon: "agent_tool" },
  error: { label: "error", tone: "bad", icon: "error" },
  info: { label: "info", tone: "neutral", icon: "info" },
};

export function activityKindMeta(kind: string): { label: string; tone: Tone; icon: IconName } {
  return ACTIVITY_KIND_META[kind as ActivityKind] ?? ACTIVITY_KIND_META.info;
}

/** "system" | "session:<id>" | "agent:<name>" | "operator" | other -> {kind, label}. */
export function describeActor(actor: string | null | undefined): { kind: "system" | "session" | "agent" | "operator" | "other"; label: string } {
  const a = (actor ?? "").trim();
  if (a === "" || a === "system") return { kind: "system", label: "system" };
  if (a.startsWith("session:")) return { kind: "session", label: `session ${a.slice("session:".length)}` };
  if (a.startsWith("agent:")) return { kind: "agent", label: `agent ${a.slice("agent:".length)}` };
  if (a === "operator") return { kind: "operator", label: "operator (CLI)" };
  return { kind: "other", label: a };
}

export function decisionTone(status: string): Tone {
  switch (status) {
    case "APPROVED":
    case "SUBMITTED":
    case "FILLED":
      return "good";
    case "REJECTED":
    case "ERROR":
      return "bad";
    case "NO_TRADE":
      return "neutral";
    case "PROPOSED":
      return "info";
    default:
      return "warn";
  }
}

export function shortId(id: string | null | undefined, keep = 10): string {
  if (!id) return "—";
  return id.length > keep + 2 ? `${id.slice(0, keep)}…` : id;
}

/** Last characters of an id (ULID tails are the distinctive part). */
export function tailId(id: string | null | undefined, keep = 8): string {
  if (!id) return "—";
  return id.length > keep + 1 ? `…${id.slice(-keep)}` : id;
}
