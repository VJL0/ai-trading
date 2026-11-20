import type { LiveOpsState } from "../ops/reducer";
import { toChartNumber } from "../lib/decimal";

export const MAX_MEMORIES = 512;
export const SAMPLES = 256;
export type MemoryKind = "market" | "prediction" | "risk" | "order" | "outcome" | "error";
export const KINDS: MemoryKind[] = ["market", "prediction", "risk", "order", "outcome", "error"];
export interface Memory {
  id: string;
  at: number;
  kind: MemoryKind;
  colony: string;
  causal: string | null;
  label: string;
  detail: unknown;
  confidence: number | null;
  calibrated: boolean;
  rejected: boolean;
}
export function hash(value: string): number {
  let h = 2166136261;
  for (let i = 0; i < value.length; i++) h = Math.imul(h ^ value.charCodeAt(i), 16777619);
  // Avalanche the final word: adjacent strategy names must not occupy the same colony.
  h = Math.imul(h ^ (h >>> 16), 0x85ebca6b);
  h = Math.imul(h ^ (h >>> 13), 0xc2b2ae35);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}
export function colonyPosition(name: string): [number, number] {
  const angle = hash(name) * Math.PI * 2;
  const radius = 0.42 + hash(`${name}:radius`) * 0.25;
  return [Math.cos(angle) * radius, Math.sin(angle) * radius];
}
function confidence(value: unknown): number | null {
  const n = toChartNumber(value);
  return n !== null && n >= 0 && n <= 1 ? n : null;
}
/** Only observed facts enter the scene. Unknown confidence never creates a forecast fan. */
export function memoriesFromOps(ops: LiveOpsState): Memory[] {
  const proposers = new Map<string, string>();
  for (const row of [...ops.rows].reverse()) {
    if (row.event === "decision" && row.data.causal_id && row.data.proposer) proposers.set(row.data.causal_id, row.data.proposer);
  }
  const result: Memory[] = ops.rows.map(row => {
    const d = row.data;
    const common = { id: `activity:${d.causal_id ?? "none"}:${row.seq}:${d.ts}`, at: Date.parse(d.ts), causal: d.causal_id, detail: d };
    if (row.event === "decision") {
      const d = row.data;
      return { ...common, kind: "prediction", colony: d.proposer ?? "Unattributed decisions", label: `${d.symbol ?? "—"} · ${d.action} · ${d.status}`, confidence: confidence(d.confidence), calibrated: d.confidence_calibrated, rejected: d.status === "REJECTED" || d.risk_failures.length > 0 };
    }
    const kind = row.event === "stage" ? row.data.stage : row.data.kind;
    const rejected = row.event === "stage" ? ["rejected", "error"].includes(row.data.status) : ["reject", "error"].includes(row.data.kind);
    const category: MemoryKind = /RISK|risk|reject/.test(kind) ? "risk" : /error/.test(kind) ? "error" : /BROKER|ORDER|submit/.test(kind) ? "order" : /RECONCILIATION|fill|reconcile/.test(kind) ? "outcome" : /MODEL|GPT|predict|evaluate|gpt_call/.test(kind) ? "prediction" : "market";
    return { ...common, kind: category, colony: proposers.get(d.causal_id ?? "") ?? (row.event === "activity" ? row.data.actor : "Trading system"), label: row.event === "stage" ? `${row.data.stage} · ${row.data.status}` : row.data.message, confidence: null, calibrated: false, rejected };
  });
  for (const e of ops.orderEvents) result.push({ id: `order:${e.eventId}`, at: Date.parse(e.ts ?? ""), kind: /fill/i.test(e.eventType) ? "outcome" : "order", colony: "Execution", causal: ops.orderCausal[e.clientOrderId] ?? null, label: `${e.eventType} · ${e.state}${e.applied ? "" : " · not applied"}`, detail: e, confidence: null, calibrated: false, rejected: !e.applied || /reject|error/i.test(e.state) });
  for (const e of ops.audit) result.push({ id: `audit:${e.event_id}`, at: Date.parse(e.ts), kind: /risk|halt|reject/i.test(e.event_type) ? "risk" : /error|fail/i.test(e.event_type) ? "error" : "outcome", colony: e.source || "Audit", causal: e.causal_id, label: e.event_type, detail: e, confidence: null, calibrated: false, rejected: /halt|reject|fail|error/i.test(e.event_type) });
  return result.filter(e => Number.isFinite(e.at));
}
/** Bounded, idempotent session memory; replay uses a frozen copy while live ingestion continues. */
export function mergeMemories(previous: readonly Memory[], incoming: readonly Memory[]): Memory[] {
  const map = new Map(previous.map(e => [e.id, e]));
  for (const e of incoming) map.set(e.id, e);
  return [...map.values()].sort((a, b) => a.at - b.at || a.id.localeCompare(b.id)).slice(-MAX_MEMORIES);
}
export function packMemories(events: readonly Memory[], epoch: number): Float32Array<ArrayBuffer> {
  const out = new Float32Array(events.length * 12);
  events.forEach((e, i) => {
    const [x, y] = colonyPosition(e.colony);
    const origin = colonyPosition(e.causal ?? e.colony);
    out.set([x, y, (e.at - epoch) / 1000, KINDS.indexOf(e.kind), hash(e.id), e.confidence ?? -1, e.rejected ? 1 : 0, e.calibrated ? 1 : 0, origin[0] * 0.24, origin[1] * 0.24, 0, 0], i * 12);
  });
  return out;
}
export interface SceneSettings { time: number; zoom: number; panX: number; panY: number; selected: number; motion: boolean; active: boolean; rate: number }
export interface Metrics { fps: number; p95: number; gpuMs: number | null; particles: number; resolution: string; adapter: string }
export type WorkerInput =
  | { type: "init"; canvas: OffscreenCanvas; width: number; height: number }
  | { type: "resize"; width: number; height: number }
  | { type: "data"; data: Float32Array<ArrayBuffer>; count: number }
  | { type: "settings"; settings: SceneSettings }
  | { type: "dispose" };
export type WorkerOutput = { type: "ready" } | { type: "error"; message: string } | { type: "metrics"; metrics: Metrics };
