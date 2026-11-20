// Minimal project-specific source linter (run by `pnpm lint` after `tsc --noEmit`).
//
// ESLint is intentionally not used (one less large dependency tree to pin and audit); TypeScript's
// strict mode covers correctness and these rules cover the dashboard's safety invariants:
//
//   no-float-money     parseFloat / Number() / parseInt / .toFixed only in the allow-listed
//                      display helpers (decimal strings are formatted exactly by lib/decimal.ts)
//   no-local-storage   the operator token may live in memory or sessionStorage, never localStorage
//   no-env-injection   no import.meta.env / process.env in client code (no tokens in the bundle)
//   no-raw-html        no innerHTML / dangerouslySetInnerHTML / eval / new Function
//   no-console         no console.* in app code (tokens and payloads must never be logged)
//   no-remote-fetch    fetch / EventSource only with relative /api URLs (same origin, localhost)
//   read-only-*        the UI is read-only: no HTTP method other than GET, fetch only in api/client.ts,
//                      EventSource only in api/stream.ts, no other transports (XHR, beacon, WebSocket),
//                      and no <form>/<input>/<button>/<select>/<textarea> elements in components
//
//   node scripts/lint.mjs        exit 1 and list violations
import { readdir, readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const srcDir = path.join(root, "src");

const RULES = [
  {
    id: "no-float-money",
    re: /\bparseFloat\s*\(|\bNumber\s*\(|\bNumber\.parse(?:Float|Int)\s*\(|\bparseInt\s*\(|\.toFixed\s*\(/g,
    // display-only conversions: chart coordinates, ML metric formatting, exponent parsing, aria values
    // GPU timestamp deltas may convert bigint nanoseconds in renderer.ts; money stays in decimal.ts.
    allow: ["src/lib/decimal.ts", "src/mycelium/renderer.ts"],
  },
  { id: "no-local-storage", re: /\blocalStorage\b/g, allow: [] },
  { id: "no-env-injection", re: /\bimport\.meta\.env\b|\bprocess\.env\b/g, allow: [] },
  { id: "no-raw-html", re: /\binnerHTML\b|dangerouslySetInnerHTML|\beval\s*\(|\bnew\s+Function\s*\(/g, allow: [] },
  { id: "no-console", re: /\bconsole\.(?:log|debug|info|warn|error|trace)\s*\(/g, allow: [] },
  {
    id: "no-remote-fetch",
    re: /\b(?:fetch|EventSource)\s*\(\s*["'`]https?:/g,
    allow: [],
  },
  { id: "read-only-method", re: /["'`](?:POST|PUT|PATCH|DELETE)["'`]|\bmethod\s*:(?!\s*READ_ONLY_METHOD\b)(?!\s*["'`]GET["'`])/g, allow: [] },
  { id: "read-only-fetch", re: /\bfetch\s*\(/g, allow: ["src/api/client.ts"] },
  { id: "read-only-eventsource", re: /\bnew\s+EventSource\s*\(/g, allow: ["src/api/stream.ts"] },
  { id: "read-only-transport", re: /\bXMLHttpRequest\b|\bsendBeacon\b|\bWebSocket\b/g, allow: [] },
  // Mycelium controls only local camera, replay, and inspection state. Network rules still apply.
  { id: "read-only-controls", re: /<(?:form|input|button|select|textarea)\b/g, allow: ["src/mycelium/MarketMycelium.tsx"] },
];

/** Strip comments (block, line) so documentation may mention forbidden names. Strings are kept. */
function stripComments(text) {
  let out = "";
  let i = 0;
  let quote = null;
  while (i < text.length) {
    const c = text[i];
    const n = text[i + 1];
    if (quote) {
      out += c;
      if (c === "\\") {
        out += n ?? "";
        i += 2;
        continue;
      }
      if (c === quote) quote = null;
      i += 1;
      continue;
    }
    if (c === '"' || c === "'" || c === "`") {
      quote = c;
      out += c;
      i += 1;
      continue;
    }
    if (c === "/" && n === "*") {
      const end = text.indexOf("*/", i + 2);
      const chunk = text.slice(i, end === -1 ? text.length : end + 2);
      out += chunk.replace(/[^\n]/g, " ");
      i += chunk.length;
      continue;
    }
    if (c === "/" && n === "/") {
      const end = text.indexOf("\n", i);
      const stop = end === -1 ? text.length : end;
      out += " ".repeat(stop - i);
      i = stop;
      continue;
    }
    out += c;
    i += 1;
  }
  return out;
}

async function walk(dir) {
  const out = [];
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      if (entry.name === "generated" || entry.name === "test") continue;
      out.push(...(await walk(full)));
    } else if (/\.(ts|tsx)$/.test(entry.name) && !/\.test\.(ts|tsx)$/.test(entry.name)) {
      out.push(full);
    }
  }
  return out;
}

const violations = [];
for (const file of await walk(srcDir)) {
  const rel = path.relative(root, file).split(path.sep).join("/");
  const code = stripComments(await readFile(file, "utf8"));
  for (const rule of RULES) {
    if (rule.allow.includes(rel)) continue;
    for (const m of code.matchAll(rule.re)) {
      const line = code.slice(0, m.index).split("\n").length;
      violations.push(`${rel}:${line}  ${rule.id}  ${m[0].trim()}`);
    }
  }
}

if (violations.length) {
  console.error(violations.join("\n"));
  console.error(`\n${violations.length} lint violation(s)`);
  process.exit(1);
}
console.log("lint: no violations");
