/**
 * Test helpers: a recording fake `fetch` that routes /api/v1 paths to fixture JSON, a fake
 * EventSource, and a render wrapper around <AppProvider>. No network is ever touched.
 */
import { render } from "@testing-library/react";
import type { ReactElement } from "react";
import { ApiClient } from "../api/client";
import type { EventSourceLike } from "../api/stream";
import { AppProvider } from "../state/AppContext";
import status from "../../fixtures/status.json";
import health from "../../fixtures/health.json";
import activity from "../../fixtures/activity.json";
import current from "../../fixtures/activity_current.json";
import orders from "../../fixtures/orders.json";
import orderFilled from "../../fixtures/order_lat-sim-01jbdashfixture00000filled1.json";
import audit from "../../fixtures/audit.json";
import performance from "../../fixtures/performance.json";
import sessions from "../../fixtures/sessions.json";
export const FILLED_COID = "lat-sim-01jbdashfixture00000filled1";
export function clone<T>(v: T): T { return JSON.parse(JSON.stringify(v)) as T; }
export const FIXTURES = { status, health, activity, current, orders, orderFilled, audit, performance, sessions };

export interface RecordedRequest {
  method: string;
  path: string;
  query: URLSearchParams;
  headers: Record<string, string>;
  body: unknown;
}

export type RouteResult = { status?: number; body?: unknown; text?: string } | unknown;
export type RouteHandler = (req: RecordedRequest) => RouteResult | Promise<RouteResult>;

function isResponseSpec(v: unknown): v is { status?: number; body?: unknown; text?: string } {
  return typeof v === "object" && v !== null && ("status" in v || "text" in v) && Object.keys(v).every((k) => ["status", "body", "text"].includes(k));
}

/** Default GET routes: the fixture JSON for every contract endpoint. */
export function defaultRoutes(): Record<string, RouteHandler> {
  return Object.fromEntries(Object.entries({health, status, activity, "activity/current": current, orders,
    [`orders/${FILLED_COID}`]: orderFilled, audit, performance, sessions})
    .map(([path, data]) => [`GET /api/v1/${path}`, () => clone(data)]));
}

export class FakeFetch {
  readonly requests: RecordedRequest[] = [];
  routes: Record<string, RouteHandler>;

  constructor(routes: Record<string, RouteHandler> = {}) {
    this.routes = { ...defaultRoutes(), ...routes };
  }

  readonly fetch = async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const url = new URL(String(input), "http://127.0.0.1:5173");
    const headers: Record<string, string> = {};
    new Headers(init?.headers).forEach((v, k) => {
      headers[k.toLowerCase()] = v;
    });
    const method = (init?.method ?? "GET").toUpperCase();
    let body: unknown = null;
    if (typeof init?.body === "string" && init.body) body = JSON.parse(init.body);
    const req: RecordedRequest = { method, path: url.pathname, query: url.searchParams, headers, body };
    this.requests.push(req);
    const handler = this.routes[`${method} ${url.pathname}`];
    if (!handler) return new Response(JSON.stringify({ detail: `no fake route for ${method} ${url.pathname}` }), { status: 404 });
    const result = await handler(req);
    if (isResponseSpec(result)) {
      const text = result.text ?? JSON.stringify(result.body ?? {});
      return new Response(text, { status: result.status ?? 200, headers: { "content-type": "application/json" } });
    }
    return new Response(JSON.stringify(result), { status: 200, headers: { "content-type": "application/json" } });
  };

  calls(method: string, path: string): RecordedRequest[] {
    return this.requests.filter((r) => r.method === method && r.path === path);
  }
}

export class FakeEventSource implements EventSourceLike {
  static instances: FakeEventSource[] = [];
  onopen: ((this: EventSourceLike, ev: Event) => unknown) | null = null;
  onerror: ((this: EventSourceLike, ev: Event) => unknown) | null = null;
  closed = false;
  private listeners = new Map<string, ((ev: MessageEvent) => void)[]>();

  constructor(readonly url: string) {
    FakeEventSource.instances.push(this);
  }

  static reset(): void {
    FakeEventSource.instances = [];
  }

  static latest(): FakeEventSource {
    const s = FakeEventSource.instances[FakeEventSource.instances.length - 1];
    if (!s) throw new Error("no FakeEventSource created");
    return s;
  }

  addEventListener(type: string, listener: (ev: MessageEvent) => void): void {
    const list = this.listeners.get(type) ?? [];
    list.push(listener);
    this.listeners.set(type, list);
  }

  close(): void {
    this.closed = true;
  }

  open(): void {
    this.onopen?.call(this, new Event("open"));
  }

  fail(): void {
    this.onerror?.call(this, new Event("error"));
  }

  emit(type: string, data: unknown): void {
    const ev = new MessageEvent(type, { data: typeof data === "string" ? data : JSON.stringify(data) });
    for (const l of this.listeners.get(type) ?? []) l(ev);
  }
}

export function renderWithApp(ui: ReactElement, opts: { routes?: Record<string, RouteHandler>; stream?: boolean } = {}) {
  const fake = new FakeFetch(opts.routes);
  const api = new ApiClient({ fetchImpl: fake.fetch });
  const streamFactory = opts.stream ? (url: string) => new FakeEventSource(url) : null;
  const result = render(<AppProvider api={api} streamFactory={streamFactory}>{ui}</AppProvider>);
  return { ...result, fake, api };
}
