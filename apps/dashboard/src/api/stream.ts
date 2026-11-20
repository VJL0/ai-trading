/**
 * Server-Sent Events subscription to GET /api/v1/stream with explicit reconnect/backoff.
 *
 * The browser's EventSource retries on its own, silently. We do not rely on that: on any error the
 * source is closed, the state becomes `disconnected` (shown in the UI) and a reconnect is scheduled
 * with capped exponential backoff + jitter. A heartbeat watchdog treats a connection that stays
 * open but delivers nothing (the API sends `heartbeat` every 2 s) as dead, so a stalled proxy can
 * never leave the dashboard silently showing old data as if it were live.
 *
 * Everything time-related (timers, clock, randomness, EventSource) is injectable for tests.
 */
import { API_PREFIX } from "./client";
import { STREAM_EVENTS, type StreamEventName } from "./types";

export type StreamState = "idle" | "connecting" | "open" | "disconnected" | "closed";

export interface StreamInfo {
  state: StreamState;
  /** consecutive failed attempts since the last successful open */
  attempt: number;
  /** epoch ms of the last event received (any type), or null */
  lastEventAt: number | null;
  /** epoch ms when the current connection opened, or null */
  connectedSince: number | null;
  /** epoch ms of the next scheduled reconnect, or null */
  nextRetryAt: number | null;
  lastError: string | null;
  /** messages whose data was not valid JSON (dropped, never applied) */
  parseErrors: number;
}

/** Minimal surface of the browser EventSource that StreamClient uses. */
export interface EventSourceLike {
  onopen: ((this: EventSourceLike, ev: Event) => unknown) | null;
  onerror: ((this: EventSourceLike, ev: Event) => unknown) | null;
  addEventListener(type: string, listener: (ev: MessageEvent) => void): void;
  close(): void;
}

export type EventSourceFactory = (url: string) => EventSourceLike;

export interface StreamTimers {
  setTimeout(fn: () => void, ms: number): unknown;
  clearTimeout(handle: unknown): void;
  now(): number;
}

export interface StreamOptions {
  url?: string;
  factory?: EventSourceFactory;
  timers?: StreamTimers;
  random?: () => number;
  initialDelayMs?: number;
  maxDelayMs?: number;
  /** close + reconnect when no event at all arrives for this long after open */
  heartbeatTimeoutMs?: number;
  onEvent?: (name: StreamEventName, data: unknown) => void;
  onInfo?: (info: StreamInfo) => void;
}

export const STREAM_URL = `${API_PREFIX}/stream`;

/** Capped exponential backoff with +/-20% jitter. attempt 0 -> ~initial. */
export function backoffDelay(attempt: number, initialMs: number, maxMs: number, random: () => number): number {
  const exp = Math.min(maxMs, initialMs * 2 ** Math.max(0, attempt));
  const jitter = 0.8 + 0.4 * Math.min(1, Math.max(0, random()));
  return Math.round(Math.min(maxMs, exp * jitter));
}

const defaultTimers: StreamTimers = {
  setTimeout: (fn, ms) => globalThis.setTimeout(fn, ms),
  clearTimeout: (h) => globalThis.clearTimeout(h as ReturnType<typeof setTimeout>),
  now: () => Date.now(),
};

function defaultFactory(url: string): EventSourceLike {
  return new EventSource(url) as unknown as EventSourceLike;
}

export class StreamClient {
  private readonly url: string;
  private readonly factory: EventSourceFactory;
  private readonly timers: StreamTimers;
  private readonly random: () => number;
  private readonly initialDelayMs: number;
  private readonly maxDelayMs: number;
  private readonly heartbeatTimeoutMs: number;
  private readonly onEvent: (name: StreamEventName, data: unknown) => void;
  private readonly onInfo: (info: StreamInfo) => void;

  private source: EventSourceLike | null = null;
  private retryTimer: unknown = null;
  private watchdog: unknown = null;
  private generation = 0;
  private info: StreamInfo = {
    state: "idle",
    attempt: 0,
    lastEventAt: null,
    connectedSince: null,
    nextRetryAt: null,
    lastError: null,
    parseErrors: 0,
  };

  constructor(opts: StreamOptions = {}) {
    this.url = opts.url ?? STREAM_URL;
    this.factory = opts.factory ?? defaultFactory;
    this.timers = opts.timers ?? defaultTimers;
    this.random = opts.random ?? Math.random;
    this.initialDelayMs = opts.initialDelayMs ?? 1_000;
    this.maxDelayMs = opts.maxDelayMs ?? 30_000;
    this.heartbeatTimeoutMs = opts.heartbeatTimeoutMs ?? 12_000;
    this.onEvent = opts.onEvent ?? (() => undefined);
    this.onInfo = opts.onInfo ?? (() => undefined);
  }

  get snapshot(): StreamInfo {
    return this.info;
  }

  start(): void {
    if (this.info.state === "connecting" || this.info.state === "open") return;
    this.clearRetry();
    this.connect();
  }

  /** Permanently stop (component unmount). */
  stop(): void {
    this.generation += 1;
    this.clearRetry();
    this.clearWatchdog();
    this.closeSource();
    this.update({ state: "closed", nextRetryAt: null, connectedSince: null });
  }

  /** Drop the current connection (if any) and connect again immediately. */
  reconnectNow(): void {
    this.clearRetry();
    this.clearWatchdog();
    this.closeSource();
    this.connect();
  }

  private update(patch: Partial<StreamInfo>): void {
    this.info = { ...this.info, ...patch };
    this.onInfo(this.info);
  }

  private connect(): void {
    const gen = ++this.generation;
    this.update({ state: "connecting", nextRetryAt: null });
    let source: EventSourceLike;
    try {
      source = this.factory(this.url);
    } catch (err) {
      this.drop(gen, `cannot open stream: ${err instanceof Error ? err.message : String(err)}`);
      return;
    }
    this.source = source;
    source.onopen = () => {
      if (gen !== this.generation) return;
      this.update({ state: "open", attempt: 0, connectedSince: this.timers.now(), lastError: null });
      this.armWatchdog(gen);
    };
    source.onerror = () => {
      if (gen !== this.generation) return;
      this.drop(gen, "stream connection error");
    };
    for (const name of STREAM_EVENTS) {
      source.addEventListener(name, (ev: MessageEvent) => this.handleMessage(gen, name, ev));
    }
  }

  private handleMessage(gen: number, name: StreamEventName, ev: MessageEvent): void {
    if (gen !== this.generation) return;
    const now = this.timers.now();
    if (this.info.state !== "open") {
      // some EventSource implementations deliver before/without onopen
      this.update({ state: "open", attempt: 0, connectedSince: now, lastError: null });
    }
    this.armWatchdog(gen);
    let data: unknown;
    try {
      data = typeof ev.data === "string" && ev.data !== "" ? JSON.parse(ev.data) : null;
    } catch {
      this.update({ lastEventAt: now, parseErrors: this.info.parseErrors + 1 });
      return;
    }
    this.update({ lastEventAt: now });
    this.onEvent(name, data);
  }

  private drop(gen: number, reason: string): void {
    if (gen !== this.generation) return;
    this.generation += 1; // ignore anything the dead source still emits
    this.clearWatchdog();
    this.closeSource();
    const delay = backoffDelay(this.info.attempt, this.initialDelayMs, this.maxDelayMs, this.random);
    const retryAt = this.timers.now() + delay;
    this.update({
      state: "disconnected",
      attempt: this.info.attempt + 1,
      connectedSince: null,
      nextRetryAt: retryAt,
      lastError: reason,
    });
    this.retryTimer = this.timers.setTimeout(() => {
      this.retryTimer = null;
      this.connect();
    }, delay);
  }

  private armWatchdog(gen: number): void {
    this.clearWatchdog();
    this.watchdog = this.timers.setTimeout(() => {
      this.watchdog = null;
      this.drop(gen, `no stream events for ${Math.round(this.heartbeatTimeoutMs / 1000)} s (heartbeat missed)`);
    }, this.heartbeatTimeoutMs);
  }

  private clearWatchdog(): void {
    if (this.watchdog !== null) {
      this.timers.clearTimeout(this.watchdog);
      this.watchdog = null;
    }
  }

  private clearRetry(): void {
    if (this.retryTimer !== null) {
      this.timers.clearTimeout(this.retryTimer);
      this.retryTimer = null;
    }
  }

  private closeSource(): void {
    if (this.source) {
      const s = this.source;
      this.source = null;
      s.onopen = null;
      s.onerror = null;
      try {
        s.close();
      } catch {
        /* already closed */
      }
    }
  }
}
