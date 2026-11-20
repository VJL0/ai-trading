/// <reference types="node" />
import react from "@vitejs/plugin-react";
import type { IncomingMessage, ServerResponse } from "node:http";
import { defineConfig, type Plugin, type ProxyOptions } from "vite";

/**
 * Localhost-only dashboard. The browser only ever talks to this Vite server (same origin); `/api`
 * is proxied to the FastAPI service on 127.0.0.1:8765. No environment values are injected into the
 * client bundle (envPrefix is a dedicated, unused prefix) so no token can end up in built JS.
 */

const DEFAULT_API_TARGET = "http://127.0.0.1:8765";
const LOOPBACK_HOSTS = new Set(["127.0.0.1", "localhost", "[::1]"]);

export function resolveApiTarget(raw: string | undefined = process.env.LAT_DASHBOARD_API_TARGET): string {
  const url = new URL(raw && raw.trim() !== "" ? raw : DEFAULT_API_TARGET);
  if (url.protocol !== "http:" || !LOOPBACK_HOSTS.has(url.hostname)) {
    throw new Error(`LAT_DASHBOARD_API_TARGET must be an http:// loopback URL (localhost-only); got ${url.origin}`);
  }
  return url.origin;
}

const SECURITY_HEADERS: Record<string, string> = {
  "X-Content-Type-Options": "nosniff",
  "Referrer-Policy": "no-referrer",
  "X-Frame-Options": "DENY",
  "Cross-Origin-Opener-Policy": "same-origin",
  "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
};

/** CSP for the built app (dev needs inline HMR preamble, so it is only applied to build/preview). */
const CSP_DIRECTIVES = [
  "default-src 'self'",
  "script-src 'self'",
  "style-src 'self' 'unsafe-inline'",
  "img-src 'self' data:",
  "font-src 'self'",
  "connect-src 'self'",
  "object-src 'none'",
  "base-uri 'none'",
  "form-action 'self'",
];

function cspMetaPlugin(): Plugin {
  return {
    name: "lat-csp-meta",
    apply: "build",
    transformIndexHtml() {
      return [
        {
          tag: "meta",
          attrs: { "http-equiv": "Content-Security-Policy", content: CSP_DIRECTIVES.join("; ") },
          injectTo: "head-prepend",
        },
      ];
    },
  };
}

function isStream(req: IncomingMessage): boolean {
  return (req.url ?? "").startsWith("/api/v1/stream");
}

/** `/api` proxy. SSE (`/api/v1/stream`) is passed through unbuffered and uncompressed. */
export function apiProxy(target: string): Record<string, ProxyOptions> {
  return {
    "/api": {
      target,
      changeOrigin: false, // keep Host/Origin as the browser sent them: the API checks Origin
      ws: false,
      configure(proxy) {
        proxy.on("proxyReq", (proxyReq, req) => {
          if (isStream(req)) {
            proxyReq.setHeader("accept-encoding", "identity");
            proxyReq.setHeader("cache-control", "no-cache");
          }
        });
        proxy.on("proxyRes", (proxyRes, req) => {
          const type = String(proxyRes.headers["content-type"] ?? "");
          if (isStream(req) || type.includes("text/event-stream")) {
            proxyRes.headers["cache-control"] = "no-cache, no-transform";
            proxyRes.headers["x-accel-buffering"] = "no";
            delete proxyRes.headers["content-length"];
          }
        });
        proxy.on("error", (err, _req, res) => {
          const out = res as ServerResponse | undefined;
          if (out && typeof out.writeHead === "function" && !out.headersSent) {
            out.writeHead(502, { "content-type": "application/json" });
            out.end(JSON.stringify({ detail: `dashboard proxy: API unreachable at ${target} (${err.message})` }));
          }
        });
      },
    },
  };
}

const target = resolveApiTarget();

export default defineConfig({
  plugins: [react(), cspMetaPlugin()],
  envPrefix: "LAT_DASHBOARD_PUBLIC_UNUSED_",
  server: {
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,
    headers: SECURITY_HEADERS,
    proxy: apiProxy(target),
  },
  preview: {
    host: "127.0.0.1",
    port: 4173,
    strictPort: true,
    headers: { ...SECURITY_HEADERS, "Content-Security-Policy": [...CSP_DIRECTIVES, "frame-ancestors 'none'"].join("; ") },
    proxy: apiProxy(target),
  },
  build: {
    sourcemap: false,
    target: "es2022",
    chunkSizeWarningLimit: 900,
  },
});
