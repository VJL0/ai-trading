/**
 * READ-ONLY client for the localhost API (docs/architecture/api-contract.md).
 *
 * The dashboard is a read-only operations view: every action (sessions, trades, halts) is taken by
 * the AI agent through the MCP tools or by the operator CLI (`make halt`), never from this UI. This
 * client therefore has exactly one transport method and it is hard-wired to `GET`: there is no way
 * to issue POST/PUT/PATCH/DELETE from the bundle, no request body, no operator/agent/engine token,
 * and no CSRF header (which only state-changing requests need).
 *
 * Requests are same-origin relative URLs (`/api/v1/...`); Vite proxies them to 127.0.0.1:8765.
 */
import type {
  ActivityCurrentResponse,
  ActivityListResponse,
  AuditResponse,
  HealthResponse,
  OrderDetailResponse,
  OrdersResponse,
  PerformanceResponse,
  SessionsResponse,
  StatusResponse,
} from "./types";

export const API_PREFIX = "/api/v1";
/** The only HTTP method this client can send. */
export const READ_ONLY_METHOD = "GET" as const;
const DEFAULT_TIMEOUT_MS = 15_000;

export type ApiErrorKind = "http" | "network" | "timeout" | "parse";

export class ApiError extends Error {
  readonly kind: ApiErrorKind;
  readonly status: number | null;
  readonly path: string;
  readonly detail: string;

  constructor(kind: ApiErrorKind, path: string, detail: string, status: number | null = null) {
    super(`${status !== null ? `HTTP ${status}` : kind}: ${detail} (${path})`);
    this.name = "ApiError";
    this.kind = kind;
    this.status = status;
    this.path = path;
    this.detail = detail;
  }
}

export interface ApiClientOptions {
  /** Origin prefix; "" (same origin) in the app. */
  baseUrl?: string;
  fetchImpl?: typeof fetch;
  timeoutMs?: number;
}

type Query = Record<string, string | number | boolean | null | undefined>;

/** FastAPI errors are {"detail": str | [{loc, msg}]}; render them as one readable line. */
export function describeErrorBody(body: unknown): string | null {
  if (body === null || typeof body !== "object") return typeof body === "string" && body ? body : null;
  const detail = (body as { detail?: unknown }).detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((d) => {
        if (d && typeof d === "object") {
          const item = d as { loc?: unknown; msg?: unknown };
          const loc = Array.isArray(item.loc) ? item.loc.join(".") : "";
          return `${loc ? `${loc}: ` : ""}${String(item.msg ?? JSON.stringify(d))}`;
        }
        return String(d);
      })
      .join("; ");
  }
  if (detail !== undefined) return JSON.stringify(detail);
  return null;
}

export function buildQuery(query: Query | undefined): string {
  if (!query) return "";
  const params = new URLSearchParams();
  for (const [k, v] of Object.entries(query)) {
    if (v === undefined || v === null || v === "") continue;
    params.set(k, String(v));
  }
  const s = params.toString();
  return s ? `?${s}` : "";
}

export class ApiClient {
  private readonly baseUrl: string;
  private readonly fetchImpl: typeof fetch;
  private readonly timeoutMs: number;

  constructor(opts: ApiClientOptions = {}) {
    this.baseUrl = opts.baseUrl ?? "";
    // bind: window.fetch must be called with the window as `this`
    this.fetchImpl = opts.fetchImpl ?? ((input, init) => globalThis.fetch(input, init));
    this.timeoutMs = opts.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  }

  /** GET `path` (relative to /api/v1). The method is fixed; there is no body and no auth header. */
  async get<T>(path: string, query?: Query, signal?: AbortSignal): Promise<T> {
    const fullPath = `${API_PREFIX}${path}${buildQuery(query)}`;
    const controller = new AbortController();
    const onAbort = () => controller.abort();
    signal?.addEventListener("abort", onAbort, { once: true });
    let timedOut = false;
    const timer = setTimeout(() => {
      timedOut = true;
      controller.abort();
    }, this.timeoutMs);
    let response: Response;
    try {
      response = await this.fetchImpl(`${this.baseUrl}${fullPath}`, {
        method: READ_ONLY_METHOD,
        headers: { Accept: "application/json" },
        cache: "no-store",
        credentials: "same-origin",
        signal: controller.signal,
      });
    } catch (err) {
      if (timedOut) throw new ApiError("timeout", fullPath, `no response within ${this.timeoutMs} ms`);
      if (signal?.aborted) throw new ApiError("network", fullPath, "request aborted");
      throw new ApiError("network", fullPath, err instanceof Error ? err.message : String(err));
    } finally {
      clearTimeout(timer);
      signal?.removeEventListener("abort", onAbort);
    }

    const text = await response.text();
    let body: unknown = null;
    if (text) {
      try {
        body = JSON.parse(text);
      } catch {
        if (response.ok) throw new ApiError("parse", fullPath, "response was not valid JSON", response.status);
        body = text.slice(0, 300);
      }
    }
    if (!response.ok) {
      const detail = describeErrorBody(body) ?? response.statusText ?? "request failed";
      throw new ApiError("http", fullPath, detail, response.status);
    }
    return body as T;
  }

  status(signal?: AbortSignal): Promise<StatusResponse> {
    return this.get("/status", undefined, signal);
  }
  health(signal?: AbortSignal): Promise<HealthResponse> {
    return this.get("/health", undefined, signal);
  }
  activity(limit = 200, signal?: AbortSignal): Promise<ActivityListResponse> {
    return this.get("/activity", { limit }, signal);
  }
  activityCurrent(signal?: AbortSignal): Promise<ActivityCurrentResponse> {
    return this.get("/activity/current", undefined, signal);
  }
  performance(signal?: AbortSignal): Promise<PerformanceResponse> {
    return this.get("/performance", undefined, signal);
  }
  sessions(limit = 20, signal?: AbortSignal): Promise<SessionsResponse> {
    return this.get("/sessions", { limit }, signal);
  }
  orders(limit = 20, signal?: AbortSignal): Promise<OrdersResponse> {
    return this.get("/orders", { limit }, signal);
  }
  order(clientOrderId: string, signal?: AbortSignal): Promise<OrderDetailResponse> {
    return this.get(`/orders/${encodeURIComponent(clientOrderId)}`, undefined, signal);
  }
  audit(limit = 150, signal?: AbortSignal): Promise<AuditResponse> {
    return this.get("/audit", { limit }, signal);
  }
}

export function errorMessage(err: unknown): string {
  if (err instanceof ApiError) {
    if (err.status !== null) return `HTTP ${err.status}: ${err.detail}`;
    if (err.kind === "timeout") return `API timeout: ${err.detail}`;
    if (err.kind === "parse") return `invalid response: ${err.detail}`;
    return `API unreachable: ${err.detail}`;
  }
  return err instanceof Error ? err.message : String(err);
}
