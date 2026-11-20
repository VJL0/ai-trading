/**
 * App-wide state for the READ-ONLY operations UI: the GET-only API client, the SSE stream, the latest
 * /status (execution label) and the live-operations reducer (pipeline, activity, decision, orders,
 * timeline sources).
 *
 * Synchronisation:
 * - every SSE event is dispatched into the pure `liveOpsReducer`;
 * - each time the stream (re)opens, `/activity/current` + `/activity` are fetched and applied as a
 *   resync (events arriving meanwhile are buffered and replayed, see ops/reducer.ts);
 * - while the stream is not open, the same endpoints are polled (results are dropped if the stream
 *   opened in the meantime, so a slow poll can never roll the view back);
 * - /status is accepted in server_time order (a late poll never overrides a newer push).
 */
import { createContext, useCallback, useContext, useEffect, useMemo, useReducer, useRef, useState, type ReactNode } from "react";
import { ApiClient } from "../api/client";
import { StreamClient, type EventSourceFactory, type StreamInfo } from "../api/stream";
import type { StatusResponse, StreamEventName } from "../api/types";
import { parseUtc } from "../lib/time";
import { initialLiveOpsState, liveOpsReducer, type LiveOpsAction, type LiveOpsState } from "../ops/reducer";
import { useLiveOpsSync, type SyncTimings } from "./useLiveOpsSync";

type StreamListener = (data: unknown) => void;

export interface AppContextValue {
  api: ApiClient;
  stream: StreamInfo;
  status: StatusResponse | null;
  statusReceivedAt: number | null;
  statusError: unknown;
  ops: LiveOpsState;
  /** last error of the live-ops sync (resync / poll), null when the last attempt succeeded */
  syncError: unknown;
  onStreamEvent: (name: StreamEventName, fn: StreamListener) => () => void;
}

const AppContext = createContext<AppContextValue | null>(null);

export const STATUS_REORDER_GRACE_MS = 10_000;

export interface AppProviderProps {
  children: ReactNode;
  api?: ApiClient;
  /** EventSource factory (tests inject a fake); `null` disables the stream entirely. */
  streamFactory?: EventSourceFactory | null;
  statusPollMs?: number;
  statusPollWhileStreamingMs?: number;
  timings?: Partial<SyncTimings>;
}

const INITIAL_STREAM: StreamInfo = {
  state: "idle",
  attempt: 0,
  lastEventAt: null,
  connectedSince: null,
  nextRetryAt: null,
  lastError: null,
  parseErrors: 0,
};

export function AppProvider({
  children,
  api: apiProp,
  streamFactory,
  statusPollMs = 5_000,
  statusPollWhileStreamingMs = 30_000,
  timings,
}: AppProviderProps) {
  const api = useMemo(() => apiProp ?? new ApiClient(), [apiProp]);
  const [ops, dispatch] = useReducer(liveOpsReducer, undefined, initialLiveOpsState);
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [statusReceivedAt, setStatusReceivedAt] = useState<number | null>(null);
  const [statusError, setStatusError] = useState<unknown>(null);
  const [stream, setStream] = useState<StreamInfo>(INITIAL_STREAM);

  const listeners = useRef(new Map<StreamEventName, Set<StreamListener>>());
  const onStreamEvent = useCallback((name: StreamEventName, fn: StreamListener) => {
    let set = listeners.current.get(name);
    if (!set) {
      set = new Set();
      listeners.current.set(name, set);
    }
    set.add(fn);
    return () => {
      set.delete(fn);
    };
  }, []);

  const latest = useRef<{ serverMs: number | null; receivedAt: number }>({ serverMs: null, receivedAt: 0 });
  const acceptStatus = useCallback((s: unknown) => {
    if (!s || typeof s !== "object") return;
    const now = Date.now();
    const serverMs = parseUtc((s as Partial<StatusResponse>).server_time ?? null);
    const prev = latest.current;
    const older = serverMs !== null && prev.serverMs !== null && serverMs < prev.serverMs;
    if (older && now - prev.receivedAt < STATUS_REORDER_GRACE_MS) return;
    latest.current = { serverMs, receivedAt: now };
    setStatus(s as StatusResponse);
    setStatusReceivedAt(now);
    setStatusError(null);
  }, []);

  const send = useCallback((a: LiveOpsAction) => dispatch(a), []);

  // ---------------------------------------------------------------- SSE
  useEffect(() => {
    if (streamFactory === null) {
      setStream({ ...INITIAL_STREAM, state: "closed", lastError: "stream disabled" });
      return;
    }
    const opts: ConstructorParameters<typeof StreamClient>[0] = {
      onInfo: setStream,
      onEvent: (name, data) => {
        if (name === "status") acceptStatus(data);
        dispatch({ type: "stream", name, data, at: Date.now() });
        for (const fn of listeners.current.get(name) ?? []) {
          try {
            fn(data);
          } catch {
            /* a broken listener must not break the stream */
          }
        }
      },
    };
    if (streamFactory) opts.factory = streamFactory;
    const client = new StreamClient(opts);
    client.start();
    return () => client.stop();
  }, [streamFactory, acceptStatus]);

  // ---------------------------------------------------------------- /status polling fallback
  const streaming = stream.state === "open";
  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const controller = new AbortController();
    const tick = async () => {
      try {
        const s = await api.status(controller.signal);
        if (!cancelled) {
          acceptStatus(s);
          dispatch({ type: "status", status: s, at: Date.now() });
        }
      } catch (err) {
        if (!cancelled) setStatusError(err);
      }
      if (!cancelled) timer = setTimeout(tick, streaming ? statusPollWhileStreamingMs : statusPollMs);
    };
    void tick();
    return () => {
      cancelled = true;
      controller.abort();
      if (timer) clearTimeout(timer);
    };
  }, [api, acceptStatus, streaming, statusPollMs, statusPollWhileStreamingMs]);

  // ---------------------------------------------------------------- live-ops sync (resync, polls, orders, audit, sessions)
  const syncError = useLiveOpsSync({ api, stream, ops, dispatch: send, onStreamEvent, timings });

  const value: AppContextValue = { api, stream, status, statusReceivedAt, statusError, ops, syncError, onStreamEvent };
  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

export function useApp(): AppContextValue {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error("useApp must be used inside <AppProvider>");
  return ctx;
}
