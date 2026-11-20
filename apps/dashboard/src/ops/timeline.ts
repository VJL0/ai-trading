/**
 * Event timeline view model (pure).
 *
 * Ordering rule (honest about clocks): in SIMULATED feed mode the live-ops log and the order ledger
 * stamp events with the simulated market clock while some audit events use the wall clock, so
 * timestamps from different sources are NOT comparable. The timeline therefore never interleaves
 * sources by timestamp:
 *   - the backbone is the live-ops log (stage / activity / decision rows), ordered by its
 *     monotonically increasing `seq` (true order), grouped by causal id (one group per decision
 *     cycle; causal-less rows form "system" groups), newest group first;
 *   - audit rows and order events that share a group's causal id are listed inside that group as
 *     its ledger records (each ordered within its own source);
 *   - everything else is listed in "unlinked" audit / order-event lists, newest first per source.
 * Every item shows its own timestamp.
 */
import type { AuditEventRow } from "../api/types";
import { activityKindMeta, decisionTone, describeActor, STAGE_LABEL, STAGE_STATUS_META, type IconName, type Tone } from "./labels";
import type { LiveOpsState, LiveRow, OrderEventItem } from "./reducer";

export type TimelineSource = "stage" | "activity" | "decision" | "audit" | "order_event";

export interface TimelineItem {
  key: string;
  source: TimelineSource;
  ts: string | null;
  title: string;
  detail: string;
  tone: Tone;
  icon: IconName;
  seq: number | null;
  actor: string | null;
}

export interface TimelineGroup {
  key: string;
  causalId: string | null;
  kind: "cycle" | "system";
  title: string;
  outcome: string | null;
  latestTs: string | null;
  maxSeq: number;
  items: TimelineItem[];
  ledger: TimelineItem[];
}

export interface Timeline {
  groups: TimelineGroup[];
  unlinkedAudit: TimelineItem[];
  unlinkedOrderEvents: TimelineItem[];
  totalItems: number;
}

export const MAX_GROUPS = 25;
export const MAX_UNLINKED = 40;

function compact(value: unknown, max = 160): string {
  if (value === null || value === undefined) return "";
  let text: string;
  if (typeof value === "string") text = value;
  else {
    try {
      text = JSON.stringify(value);
    } catch {
      text = String(value);
    }
  }
  return text.length > max ? `${text.slice(0, max - 1)}…` : text;
}

export function rowToItem(row: LiveRow): TimelineItem | null {
  switch (row.event) {
    case "stage": {
      const s = row.data;
      if (s.status === "running") return null; // the pipeline shows running stages; the timeline keeps finished transitions
      const meta = STAGE_STATUS_META[s.status];
      const dur = s.duration_ms !== null ? ` in ${s.duration_ms} ms` : "";
      return {
        key: `stage:${row.seq}`,
        source: "stage",
        ts: s.ts,
        title: `${STAGE_LABEL[s.stage]} ${meta.label}${dur}`,
        detail: [s.item, s.detail].filter(Boolean).join(" · "),
        tone: meta.tone,
        icon: meta.icon,
        seq: row.seq,
        actor: null,
      };
    }
    case "activity": {
      const a = row.data;
      const meta = activityKindMeta(a.kind);
      return {
        key: `activity:${row.seq}`,
        source: "activity",
        ts: a.ts,
        title: a.message,
        detail: meta.label,
        tone: meta.tone,
        icon: meta.icon,
        seq: row.seq,
        actor: describeActor(a.actor).label,
      };
    }
    case "decision": {
      const d = row.data;
      const what = d.action === "NO_TRADE" ? "NO_TRADE" : `${d.action} ${d.symbol ?? "?"}`;
      return {
        key: `decision:${row.seq}`,
        source: "decision",
        ts: d.ts,
        title: `Decision ${what}: ${d.status}`,
        detail: d.reason,
        tone: decisionTone(d.status),
        icon: "decision",
        seq: row.seq,
        actor: null,
      };
    }
  }
}

export function auditToItem(a: AuditEventRow): TimelineItem {
  return {
    key: `audit:${a.seq}`,
    source: "audit",
    ts: a.ts,
    title: a.event_type,
    detail: compact(a.payload),
    tone: /REJECT|FAIL|ERROR|MISMATCH|UNKNOWN|HALT/.test(a.event_type) ? "warn" : "neutral",
    icon: "audit",
    seq: a.seq,
    actor: a.source,
  };
}

export function orderEventToItem(e: OrderEventItem): TimelineItem {
  return {
    key: `order_event:${e.eventId}`,
    source: "order_event",
    ts: e.ts,
    title: `${e.eventType} → ${e.state}`,
    detail: `${e.clientOrderId} · via ${e.source}${e.applied ? "" : ` · NOT applied${e.note ? `: ${e.note}` : ""}`}`,
    tone: !e.applied ? "warn" : /REJECTED|EXPIRED|UNKNOWN/.test(e.state) ? "bad" : e.state === "FILLED" ? "good" : "neutral",
    icon: "order",
    seq: e.seq,
    actor: null,
  };
}

export function buildTimeline(state: LiveOpsState, maxGroups = MAX_GROUPS): Timeline {
  const groups: TimelineGroup[] = [];
  const byCausal = new Map<string, TimelineGroup>();
  let lastSystem: TimelineGroup | null = null;

  for (const row of state.rows) {
    const item = rowToItem(row);
    const causal = row.data.causal_id ?? null;
    if (causal) {
      let g = byCausal.get(causal);
      if (!g) {
        g = { key: `c:${causal}`, causalId: causal, kind: "cycle", title: "Decision cycle", outcome: null, latestTs: row.data.ts, maxSeq: row.seq, items: [], ledger: [] };
        byCausal.set(causal, g);
        groups.push(g);
      }
      if (row.event === "decision" && g.outcome === null) {
        const d = row.data;
        g.outcome = d.action === "NO_TRADE" ? "NO_TRADE" : `${d.action} ${d.symbol ?? "?"} · ${d.status}`;
      }
      if (item) g.items.push(item);
      lastSystem = null;
    } else {
      if (!lastSystem) {
        lastSystem = { key: `s:${row.seq}`, causalId: null, kind: "system", title: "System activity", outcome: null, latestTs: row.data.ts, maxSeq: row.seq, items: [], ledger: [] };
        groups.push(lastSystem);
      }
      if (item) lastSystem.items.push(item);
    }
  }

  const unlinkedAudit: TimelineItem[] = [];
  for (const a of state.audit) {
    const g = a.causal_id ? byCausal.get(a.causal_id) : undefined;
    if (g) g.ledger.push(auditToItem(a));
    else if (unlinkedAudit.length < MAX_UNLINKED) unlinkedAudit.push(auditToItem(a));
  }
  const unlinkedOrderEvents: TimelineItem[] = [];
  for (const e of state.orderEvents) {
    const causal = state.orderCausal[e.clientOrderId] ?? state.orders[e.clientOrderId]?.causal_id ?? null;
    const g = causal ? byCausal.get(causal) : undefined;
    if (g) g.ledger.push(orderEventToItem(e));
    else if (unlinkedOrderEvents.length < MAX_UNLINKED) unlinkedOrderEvents.push(orderEventToItem(e));
  }

  const kept = groups.filter((g) => g.items.length > 0 || g.ledger.length > 0).slice(0, maxGroups);
  const totalItems =
    kept.reduce((n, g) => n + g.items.length + g.ledger.length, 0) + unlinkedAudit.length + unlinkedOrderEvents.length;
  return { groups: kept, unlinkedAudit, unlinkedOrderEvents, totalItems };
}
