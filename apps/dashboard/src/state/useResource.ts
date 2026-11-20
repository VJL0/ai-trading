/**
 * Poll one API resource, refetching early when relevant SSE events arrive.
 *
 * - `fetchedAt` is the client time of the last *successful* response; views show it (with age) so
 *   nothing is displayed without a timestamp.
 * - On error the previous data is kept but `error` is set, and views mark the panel STALE.
 * - Changing `key` (e.g. another symbol) drops the old data immediately so data for one symbol is
 *   never shown under another's label.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import type { StreamEventName } from "../api/types";
import { useApp } from "./AppContext";

export interface Resource<T> {
  data: T | null;
  error: unknown;
  loading: boolean;
  fetchedAt: number | null;
  refresh: () => void;
}

export interface ResourceOptions {
  intervalMs?: number;
  refreshOn?: readonly StreamEventName[];
  enabled?: boolean;
}

export function useResource<T>(
  key: string,
  fetcher: (signal: AbortSignal) => Promise<T>,
  { intervalMs = 10_000, refreshOn = [], enabled = true }: ResourceOptions = {},
): Resource<T> {
  const { onStreamEvent } = useApp();
  const [state, setState] = useState<{ key: string; data: T | null; error: unknown; loading: boolean; fetchedAt: number | null }>(
    { key, data: null, error: null, loading: true, fetchedAt: null },
  );
  const [nonce, setNonce] = useState(0);
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;

  const refresh = useCallback(() => setNonce((n) => n + 1), []);

  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const controller = new AbortController();
    setState((prev) => (prev.key === key ? { ...prev, loading: true } : { key, data: null, error: null, loading: true, fetchedAt: null }));
    const run = async () => {
      try {
        const data = await fetcherRef.current(controller.signal);
        if (!cancelled) setState({ key, data, error: null, loading: false, fetchedAt: Date.now() });
      } catch (error) {
        if (!cancelled) setState((prev) => ({ ...prev, key, error, loading: false, data: prev.key === key ? prev.data : null }));
      }
      if (!cancelled && intervalMs > 0) timer = setTimeout(run, intervalMs);
    };
    void run();
    return () => {
      cancelled = true;
      controller.abort();
      if (timer) clearTimeout(timer);
    };
  }, [key, intervalMs, enabled, nonce]);

  const refreshKey = refreshOn.join(",");
  useEffect(() => {
    if (!refreshKey) return;
    let pending: ReturnType<typeof setTimeout> | undefined;
    const debounced = () => {
      if (pending) clearTimeout(pending);
      pending = setTimeout(refresh, 250);
    };
    const offs = (refreshKey.split(",") as StreamEventName[]).map((name) => onStreamEvent(name, debounced));
    return () => {
      offs.forEach((off) => off());
      if (pending) clearTimeout(pending);
    };
  }, [refreshKey, onStreamEvent, refresh]);

  const current = state.key === key ? state : { data: null, error: null, loading: true, fetchedAt: null };
  return { data: current.data, error: current.error, loading: current.loading, fetchedAt: current.fetchedAt, refresh };
}
