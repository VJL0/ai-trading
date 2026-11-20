/**
 * Timestamp helpers. The API sends ISO-8601 UTC strings ending in "Z"; the dashboard always shows
 * them in UTC with an explicit "Z" plus an age, so data is never displayed without its time.
 */
import { useEffect, useState } from "react";

const HAS_ZONE = /(Z|[+-]\d{2}:?\d{2})$/;

/** Epoch ms of an ISO timestamp, or null when missing/unparseable/zone-less (never guess a zone). */
export function parseUtc(ts: string | null | undefined): number | null {
  if (typeof ts !== "string" || ts === "" || !HAS_ZONE.test(ts)) return null;
  const ms = Date.parse(ts);
  return Number.isFinite(ms) ? ms : null;
}

function pad(n: number, width = 2): string {
  return String(n).padStart(width, "0");
}

/** "2026-11-05 14:30:05Z" */
export function formatUtc(ts: string | number | null | undefined): string {
  const ms = typeof ts === "number" ? ts : parseUtc(ts);
  if (ms === null) return "—";
  const d = new Date(ms);
  return (
    `${d.getUTCFullYear()}-${pad(d.getUTCMonth() + 1)}-${pad(d.getUTCDate())} ` +
    `${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}:${pad(d.getUTCSeconds())}Z`
  );
}

/** "14:30:05Z" */
export function formatUtcTime(ts: string | number | null | undefined): string {
  const ms = typeof ts === "number" ? ts : parseUtc(ts);
  if (ms === null) return "—";
  const d = new Date(ms);
  return `${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}:${pad(d.getUTCSeconds())}Z`;
}

/** UNIX seconds (chart bar time) -> "2026-11-05 14:30:00Z" */
export function formatUnixSeconds(t: number | null | undefined): string {
  return typeof t === "number" && Number.isFinite(t) ? formatUtc(t * 1000) : "—";
}

/** Human age: "4s", "2m 05s", "1h 03m", "3d 2h". Negative ages (clock skew) are shown as "in …". */
export function formatAge(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return "—";
  const future = seconds < 0;
  let s = Math.floor(Math.abs(seconds));
  let text: string;
  if (s < 60) text = `${s}s`;
  else if (s < 3600) text = `${Math.floor(s / 60)}m ${pad(s % 60)}s`;
  else if (s < 86400) text = `${Math.floor(s / 3600)}h ${pad(Math.floor((s % 3600) / 60))}m`;
  else {
    const d = Math.floor(s / 86400);
    s -= d * 86400;
    text = `${d}d ${Math.floor(s / 3600)}h`;
  }
  return future ? `in ${text}` : text;
}

/** Age in seconds of `ts` relative to `nowMs` (null when ts is unknown). */
export function ageSeconds(ts: string | number | null | undefined, nowMs: number): number | null {
  const ms = typeof ts === "number" ? ts : parseUtc(ts);
  return ms === null ? null : (nowMs - ms) / 1000;
}

/** Re-render every `intervalMs` with the current epoch ms (for live age displays). */
export function useNow(intervalMs = 1000): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), intervalMs);
    return () => clearInterval(id);
  }, [intervalMs]);
  return now;
}

/** Stage durations: "840 ms", "1.24 s", "2m 05s". */
export function formatDurationMs(ms: number | null | undefined): string {
  if (ms === null || ms === undefined || !Number.isFinite(ms) || ms < 0) return "—";
  if (ms < 1000) return `${Math.round(ms)} ms`;
  if (ms < 60_000) return `${(Math.round(ms / 10) / 100).toString()} s`;
  return formatAge(ms / 1000);
}
