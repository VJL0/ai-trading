/**
 * Side effects that keep the live-ops reducer in sync with the backend. Every request is a GET.
 *
 * - resync on each stream (re)connect: /activity/current + /activity (retried while the stream is open);
 * - while the stream is down: poll the same endpoints (dropped if the stream opened meanwhile);
 * - /orders, /audit, /sessions: initial load, periodic refresh, refresh on reconnect (missed events);
 * - /orders/{id} for the latest orders whenever their stream-observed state changes.
 */
import { useEffect, useRef, useState } from "react";
import type { ApiClient } from "../api/client";
import type { StreamInfo } from "../api/stream";
import type { StreamEventName } from "../api/types";
import { latestOrders, type LiveOpsAction, type LiveOpsState } from "../ops/reducer";

export interface SyncTimings {
  pollMs: number;
  resyncRetryMs: number;
  ordersMs: number;
  auditMs: number;
  sessionsMs: number;
  detailDebounceMs: number;
  refreshDebounceMs: number;
}

export const DEFAULT_TIMINGS: SyncTimings = {
  pollMs: 5_000,
  resyncRetryMs: 3_000,
  ordersMs: 15_000,
  auditMs: 30_000,
  sessionsMs: 15_000,
  detailDebounceMs: 150,
  refreshDebounceMs: 300,
};

/** How many of the most recent orders get a lifecycle view (and therefore a detail fetch). */
export const LIFECYCLE_ORDERS = 3;

interface SyncArgs {
  api: ApiClient;
  stream: StreamInfo;
  ops: LiveOpsState;
  dispatch: (a: LiveOpsAction) => void;
  onStreamEvent: (name: StreamEventName, fn: (data: unknown) => void) => () => void;
  timings?: Partial<SyncTimings> | undefined;
}

function usePeriodic(fn: (signal: AbortSignal) => Promise<void>, intervalMs: number, deps: readonly unknown[]) {
  const fnRef = useRef(fn);
  fnRef.current = fn;
  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const controller = new AbortController();
    const tick = async () => {
      try {
        await fnRef.current(controller.signal);
      } catch {
        /* surfaced by the owning panel / retried on the next tick */
      }
      if (!cancelled && intervalMs > 0) timer = setTimeout(tick, intervalMs);
    };
    void tick();
    return () => {
      cancelled = true;
      controller.abort();
      if (timer) clearTimeout(timer);
    };
  }, [intervalMs, ...deps]);
}

export function useLiveOpsSync({ api, stream, ops, dispatch, onStreamEvent, timings }: SyncArgs): unknown {
  const t: SyncTimings = { ...DEFAULT_TIMINGS, ...(timings ?? {}) };
  const [syncError, setSyncError] = useState<unknown>(null);
  const open = stream.state === "open";
  const connKey = open ? stream.connectedSince : null;
  const openRef = useRef(open);
  openRef.current = open;

  // ---------------------------------------------------------------- resync on every (re)connect
  useEffect(() => {
    if (connKey === null) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const controller = new AbortController();
    const run = async () => {
      dispatch({ type: "resyncStart" });
      try {
        const [current, list] = await Promise.all([api.activityCurrent(controller.signal), api.activity(200, controller.signal)]);
        if (cancelled) return;
        dispatch({ type: "resyncApply", current, recent: Array.isArray(list?.events) ? list.events : [], at: Date.now() });
        setSyncError(null);
      } catch (err) {
        if (cancelled) return;
        dispatch({ type: "resyncAbort" });
        setSyncError(err);
        timer = setTimeout(run, t.resyncRetryMs);
      }
    };
    void run();
    return () => {
      cancelled = true;
      controller.abort();
      if (timer) clearTimeout(timer);
      dispatch({ type: "resyncAbort" });
    };
  }, [connKey, api, dispatch, t.resyncRetryMs]);

  // ---------------------------------------------------------------- polling fallback while the stream is down
  useEffect(() => {
    if (open) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const controller = new AbortController();
    const tick = async () => {
      try {
        const [current, list] = await Promise.all([api.activityCurrent(controller.signal), api.activity(200, controller.signal)]);
        if (!cancelled && !openRef.current) {
          dispatch({ type: "snapshot", current, at: Date.now() });
          dispatch({ type: "activityList", rows: Array.isArray(list?.events) ? list.events : [] });
          setSyncError(null);
        }
      } catch (err) {
        if (!cancelled) setSyncError(err);
      }
      if (!cancelled) timer = setTimeout(tick, t.pollMs);
    };
    void tick();
    return () => {
      cancelled = true;
      controller.abort();
      if (timer) clearTimeout(timer);
    };
  }, [open, api, dispatch, t.pollMs]);

  // ---------------------------------------------------------------- orders / audit / sessions
  usePeriodic(
    async (signal) => {
      const r = await api.orders(10, signal);
      dispatch({ type: "ordersList", orders: Array.isArray(r?.orders) ? r.orders : [] });
    },
    t.ordersMs,
    [api, connKey],
  );
  usePeriodic(
    async (signal) => {
      const r = await api.audit(150, signal);
      dispatch({ type: "auditList", rows: Array.isArray(r?.events) ? r.events : [] });
    },
    t.auditMs,
    [api, connKey],
  );
  const [sessionsNonce, setSessionsNonce] = useState(0);
  useEffect(() => {
    let pending: ReturnType<typeof setTimeout> | undefined;
    const bump = () => {
      if (pending) clearTimeout(pending);
      pending = setTimeout(() => setSessionsNonce((n) => n + 1), t.refreshDebounceMs);
    };
    const offs = [onStreamEvent("session", bump), onStreamEvent("cycle", bump)];
    return () => {
      offs.forEach((off) => off());
      if (pending) clearTimeout(pending);
    };
  }, [onStreamEvent, t.refreshDebounceMs]);
  usePeriodic(
    async (signal) => {
      const r = await api.sessions(20, signal);
      dispatch({ type: "sessions", data: r });
    },
    t.sessionsMs,
    [api, connKey, sessionsNonce],
  );

  // ---------------------------------------------------------------- order details for the lifecycle view
  const inflight = useRef(new Set<string>());
  const fetchedSig = useRef(new Map<string, string>());
  useEffect(() => {
    fetchedSig.current.clear(); // after a reconnect every detail may be stale
  }, [connKey]);
  const wanted = latestOrders(ops, LIFECYCLE_ORDERS).map((o) => {
    const streamed = ops.orderEvents.filter((e) => e.clientOrderId === o.client_order_id && e.seq !== null).length;
    return { coid: o.client_order_id, sig: `${o.updated_at}|${o.state}|${streamed}` };
  });
  const wantedKey = wanted.map((w) => `${w.coid}#${w.sig}`).join(",");
  const wantedRef = useRef(wanted);
  wantedRef.current = wanted;
  useEffect(() => {
    if (!wantedKey) return;
    const timer = setTimeout(() => {
      for (const w of wantedRef.current) {
        if (inflight.current.has(w.coid) || fetchedSig.current.get(w.coid) === w.sig) continue;
        inflight.current.add(w.coid);
        fetchedSig.current.set(w.coid, w.sig);
        api
          .order(w.coid)
          .then((detail) => dispatch({ type: "orderDetail", detail }))
          .catch(() => {
            /* lifecycle falls back to the ledger state; retried on the next state change */
          })
          .finally(() => inflight.current.delete(w.coid));
      }
    }, t.detailDebounceMs);
    return () => clearTimeout(timer);
  }, [wantedKey, connKey, api, dispatch, t.detailDebounceMs]);

  return syncError;
}
