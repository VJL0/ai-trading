/**
 * System health view model (pure): the backend's component report (GET /api/v1/health) plus the
 * client-side event-stream liveness. Nothing is assumed healthy: a component the backend did not
 * report is UNKNOWN; if the API answered with the older liveness-only body, the rows that /status
 * can vouch for (feed, broker, engine) are derived from it and labelled as such.
 */
import type { StreamInfo } from "../api/stream";
import type { HealthResponse, StatusResponse } from "../api/types";
import { isHealthReport } from "../api/types";
import { formatAge } from "../lib/time";

export type HealthLevel = "OK" | "DEGRADED" | "DOWN" | "DISABLED" | "UNKNOWN";

export interface HealthRow {
  name: string;
  label: string;
  level: HealthLevel;
  detail: string;
  lastOk: string | null;
  source: "api" | "status" | "client" | "none";
}

export interface HealthModel {
  rows: HealthRow[];
  overall: HealthLevel;
  apiOverall: string | null;
  note: string | null;
}

export const HEALTH_COMPONENTS: readonly { name: string; label: string }[] = [
  { name: "market_data", label: "Market data" },
  { name: "broker", label: "Broker connection" },
  { name: "models", label: "Model health" },
  { name: "gpt_runtime", label: "GPT runtime" },
  { name: "event_stream", label: "Event stream" },
  { name: "database", label: "Database" },
  { name: "agent", label: "Agent" },
  { name: "engine", label: "Engine" },
  { name: "scheduler", label: "Scheduler" },
  { name: "reconciliation", label: "Reconciliation" },
];

/** No event for this long on an open stream (heartbeat every 2 s) = degraded. */
export const STREAM_QUIET_AFTER_S = 6;

function level(v: unknown): HealthLevel {
  return v === "OK" || v === "DEGRADED" || v === "DOWN" || v === "DISABLED" ? v : "UNKNOWN";
}

export function streamHealth(info: StreamInfo, now: number): { level: HealthLevel; detail: string } {
  const age = info.lastEventAt === null ? null : Math.max(0, (now - info.lastEventAt) / 1000);
  switch (info.state) {
    case "open":
      if (age === null) return { level: "DEGRADED", detail: "connected, no event received yet" };
      if (age > STREAM_QUIET_AFTER_S) return { level: "DEGRADED", detail: `connected, but no event for ${formatAge(age)}` };
      return { level: "OK", detail: `connected · last event ${formatAge(age)} ago` };
    case "connecting":
      return { level: "DEGRADED", detail: info.attempt > 0 ? `reconnecting (attempt ${info.attempt + 1})` : "connecting" };
    case "disconnected": {
      const retry = info.nextRetryAt === null ? null : Math.max(0, (info.nextRetryAt - now) / 1000);
      return { level: "DOWN", detail: `disconnected${retry !== null ? ` · retry in ${formatAge(retry)}` : ""}${info.lastError ? ` · ${info.lastError}` : ""}` };
    }
    case "closed":
      return { level: "DOWN", detail: "stream closed" };
    default:
      return { level: "UNKNOWN", detail: "stream not started" };
  }
}

function fromStatus(name: string, status: StatusResponse | null): Pick<HealthRow, "level" | "detail" | "lastOk"> | null {
  if (!status) return null;
  switch (name) {
    case "market_data": {
      const f = status.feed;
      const lv: HealthLevel = f?.status === "OK" ? "OK" : f?.status === "DISCONNECTED" ? "DOWN" : f?.status ? "DEGRADED" : "UNKNOWN";
      return { level: lv, detail: `${f?.status ?? "?"} · ${f?.feed_mode ?? "?"} · ${f?.provider ?? "?"}`, lastOk: f?.last_quote_ts ?? null };
    }
    case "broker":
      return {
        level: status.broker?.connected === true ? "OK" : status.broker?.connected === false ? "DOWN" : "UNKNOWN",
        detail: `${status.broker?.kind ?? "?"} · ${status.broker?.connected ? "connected" : "not connected"}`,
        lastOk: status.broker?.last_reconciled_at ?? null,
      };
    case "engine": {
      const s = status.engine?.status;
      return { level: s === "UP" ? "OK" : s === "DOWN" ? "DOWN" : "UNKNOWN", detail: `${status.engine?.kind ?? "?"} · ${s ?? "?"} · ${status.engine?.detail ?? ""}`, lastOk: status.engine?.last_heartbeat ?? null };
    }
    case "reconciliation": {
      const r = status.broker?.reconciliation_status;
      return { level: r === "CLEAN" ? "OK" : r ? "DEGRADED" : "UNKNOWN", detail: `last reconciliation ${r ?? "never"}`, lastOk: status.broker?.last_reconciled_at ?? null };
    }
    default:
      return null;
  }
}

const RANK: Record<HealthLevel, number> = { OK: 0, DISABLED: 0, UNKNOWN: 1, DEGRADED: 2, DOWN: 3 };

export function healthModel(report: HealthResponse | null, reportError: unknown, status: StatusResponse | null, stream: StreamInfo, now: number): HealthModel {
  const components = isHealthReport(report) ? report.components : null;
  const byName = new Map((components ?? []).map((c) => [c.name, c]));
  const rows: HealthRow[] = [];
  for (const { name, label } of HEALTH_COMPONENTS) {
    const api = byName.get(name);
    if (name === "event_stream") {
      const client = streamHealth(stream, now);
      const serverLevel = api ? level(api.status) : null;
      const lv = serverLevel !== null && RANK[serverLevel] > RANK[client.level] ? serverLevel : client.level;
      rows.push({ name, label, level: lv, detail: `${client.detail}${api ? ` · API: ${api.detail}` : ""}`, lastOk: api?.last_ok ?? null, source: "client" });
      continue;
    }
    if (api) {
      rows.push({ name, label, level: level(api.status), detail: api.detail, lastOk: api.last_ok ?? null, source: "api" });
      continue;
    }
    if (name === "reconciliation" && components) continue; // optional component, not reported by this build
    const derived = components ? null : fromStatus(name, status);
    if (derived) rows.push({ name, label, ...derived, source: "status" });
    else rows.push({ name, label, level: "UNKNOWN", detail: components ? "not reported by /api/v1/health" : "no component report available", lastOk: null, source: "none" });
  }
  let note: string | null = null;
  if (!components) {
    note = reportError
      ? "The /api/v1/health component report could not be loaded; rows marked “from /status” are derived from the status endpoint, the rest are UNKNOWN."
      : report
        ? "This API build answered /api/v1/health with the liveness body only (no component checks); rows marked “from /status” are derived from the status endpoint, the rest are UNKNOWN."
        : "Waiting for /api/v1/health.";
  }
  const overall = rows.reduce<HealthLevel>((worst, r) => (RANK[r.level] > RANK[worst] ? r.level : worst), "OK");
  return { rows, overall, apiOverall: isHealthReport(report) ? report.overall : null, note };
}
