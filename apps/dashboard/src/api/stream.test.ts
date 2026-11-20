import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { FakeEventSource } from "../test/helpers";
import { backoffDelay, StreamClient, STREAM_URL, type StreamInfo, type StreamTimers } from "./stream";

/** Deterministic timer queue (no real time passes). */
class ManualTimers implements StreamTimers {
  nowMs = 1_000_000;
  private seq = 0;
  private tasks = new Map<number, { at: number; fn: () => void }>();
  setTimeout(fn: () => void, ms: number): unknown {
    const id = ++this.seq;
    this.tasks.set(id, { at: this.nowMs + ms, fn });
    return id;
  }
  clearTimeout(handle: unknown): void {
    this.tasks.delete(handle as number);
  }
  now(): number {
    return this.nowMs;
  }
  pending(): number {
    return this.tasks.size;
  }
  /** Advance the clock, running due tasks in time order. */
  advance(ms: number): void {
    const target = this.nowMs + ms;
    for (;;) {
      const due = [...this.tasks.entries()].filter(([, t]) => t.at <= target).sort((a, b) => a[1].at - b[1].at)[0];
      if (!due) break;
      this.tasks.delete(due[0]);
      this.nowMs = due[1].at;
      due[1].fn();
    }
    this.nowMs = target;
  }
}

describe("backoffDelay", () => {
  it("grows exponentially with bounded jitter and caps at the maximum", () => {
    const mid = () => 0.5; // jitter factor 1.0
    expect(backoffDelay(0, 1000, 30_000, mid)).toBe(1000);
    expect(backoffDelay(1, 1000, 30_000, mid)).toBe(2000);
    expect(backoffDelay(3, 1000, 30_000, mid)).toBe(8000);
    expect(backoffDelay(10, 1000, 30_000, mid)).toBe(30_000);
    expect(backoffDelay(2, 1000, 30_000, () => 0)).toBe(3200); // -20%
    expect(backoffDelay(2, 1000, 30_000, () => 1)).toBe(4800); // +20%
    expect(backoffDelay(20, 1000, 30_000, () => 1)).toBe(30_000); // jitter never exceeds the cap
  });
});

describe("StreamClient", () => {
  let timers: ManualTimers;
  let infos: StreamInfo[];
  let events: [string, unknown][];
  let client: StreamClient;

  beforeEach(() => {
    FakeEventSource.reset();
    timers = new ManualTimers();
    infos = [];
    events = [];
    client = new StreamClient({
      factory: (url) => new FakeEventSource(url),
      timers,
      random: () => 0.5,
      initialDelayMs: 1000,
      maxDelayMs: 8000,
      heartbeatTimeoutMs: 12_000,
      onInfo: (i) => infos.push(i),
      onEvent: (name, data) => events.push([name, data]),
    });
  });

  afterEach(() => client.stop());

  const state = () => client.snapshot.state;

  it("connects to /api/v1/stream and delivers parsed events", () => {
    client.start();
    expect(state()).toBe("connecting");
    const es = FakeEventSource.latest();
    expect(es.url).toBe(STREAM_URL);
    es.open();
    expect(state()).toBe("open");
    es.emit("heartbeat", { time: "2026-11-05T14:30:00Z" });
    es.emit("status", { execution_label: "SIMULATED" });
    expect(events).toEqual([
      ["heartbeat", { time: "2026-11-05T14:30:00Z" }],
      ["status", { execution_label: "SIMULATED" }],
    ]);
    expect(client.snapshot.lastEventAt).toBe(timers.nowMs);
  });

  it("on error shows 'disconnected' and reconnects with exponential backoff, resetting after a good open", () => {
    client.start();
    FakeEventSource.latest().fail();
    expect(state()).toBe("disconnected");
    expect(client.snapshot.attempt).toBe(1);
    expect(client.snapshot.nextRetryAt).toBe(timers.nowMs + 1000);
    expect(FakeEventSource.instances[0]!.closed).toBe(true);

    timers.advance(999);
    expect(FakeEventSource.instances).toHaveLength(1);
    timers.advance(1);
    expect(FakeEventSource.instances).toHaveLength(2);
    expect(state()).toBe("connecting");

    FakeEventSource.latest().fail(); // second failure: 2 s
    expect(client.snapshot.nextRetryAt).toBe(timers.nowMs + 2000);
    timers.advance(2000);
    FakeEventSource.latest().fail(); // third: 4 s
    expect(client.snapshot.nextRetryAt).toBe(timers.nowMs + 4000);
    timers.advance(4000);
    FakeEventSource.latest().fail(); // fourth: capped at 8 s
    timers.advance(8000);
    FakeEventSource.latest().fail();
    expect(client.snapshot.nextRetryAt).toBe(timers.nowMs + 8000);
    timers.advance(8000);

    FakeEventSource.latest().open();
    expect(state()).toBe("open");
    expect(client.snapshot.attempt).toBe(0);
    expect(client.snapshot.lastError).toBeNull();
    FakeEventSource.latest().fail();
    expect(client.snapshot.nextRetryAt).toBe(timers.nowMs + 1000); // backoff restarted
  });

  it("treats a silent open connection as dead when no heartbeat arrives", () => {
    client.start();
    const es = FakeEventSource.latest();
    es.open();
    timers.advance(11_000);
    es.emit("heartbeat", {}); // keeps it alive
    timers.advance(11_000);
    expect(state()).toBe("open");
    timers.advance(1_000);
    expect(state()).toBe("disconnected");
    expect(client.snapshot.lastError).toMatch(/heartbeat missed/);
    expect(es.closed).toBe(true);
  });

  it("ignores everything a dropped connection still emits (no duplicate or stale delivery)", () => {
    client.start();
    const old = FakeEventSource.latest();
    old.open();
    old.fail();
    old.emit("order", { client_order_id: "late" });
    old.open();
    expect(events).toEqual([]);
    expect(state()).toBe("disconnected");
  });

  it("counts malformed messages without delivering them", () => {
    client.start();
    const es = FakeEventSource.latest();
    es.open();
    es.emit("status", "{not json");
    expect(events).toEqual([]);
    expect(client.snapshot.parseErrors).toBe(1);
  });

  it("stop() closes the source and cancels pending reconnects", () => {
    client.start();
    FakeEventSource.latest().fail();
    expect(timers.pending()).toBe(1);
    client.stop();
    expect(state()).toBe("closed");
    expect(timers.pending()).toBe(0);
    timers.advance(60_000);
    expect(FakeEventSource.instances).toHaveLength(1);
  });

  it("reconnectNow() replaces the connection immediately", () => {
    client.start();
    FakeEventSource.latest().fail();
    client.reconnectNow();
    expect(FakeEventSource.instances).toHaveLength(2);
    expect(state()).toBe("connecting");
    expect(timers.pending()).toBe(0);
  });

  it("a factory that throws is handled as a disconnect with retry", () => {
    let calls = 0;
    const c = new StreamClient({
      factory: () => {
        calls += 1;
        throw new Error("EventSource unsupported");
      },
      timers,
      random: () => 0.5,
      initialDelayMs: 500,
    });
    c.start();
    expect(c.snapshot.state).toBe("disconnected");
    expect(c.snapshot.lastError).toContain("EventSource unsupported");
    timers.advance(500);
    expect(calls).toBe(2);
    c.stop();
  });
});
