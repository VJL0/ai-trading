// Post-build check of dist/ (run by `pnpm build`): the client bundle must contain no secrets.
//
// * no secret-shaped strings (provider API keys, private keys, bearer tokens);
// * none of the local secret VALUES known on this machine: var/*/keys/*.token files and secret-named
//   variables in the environment or the project .env. Values are compared in memory and never
//   printed -- a hit reports only the file and the secret's NAME;
// * no source maps; none of the API's state-changing routes or auth header names (read-only UI);
// * if TradingView Lightweight Charts is ever bundled again, its attribution text + link must be present.
//
//   node scripts/check-bundle.mjs [distDir]
import { existsSync } from "node:fs";
import { readdir, readFile, stat } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const appDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const projectRoot = path.resolve(appDir, "..", "..");
const dist = path.resolve(process.argv[2] ?? path.join(appDir, "dist"));

const SECRET_PATTERNS = [
  ["anthropic-key", /sk-ant-[A-Za-z0-9_-]{16,}/],
  ["openai-key", /\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{32,}/],
  ["private-key", /-----BEGIN [A-Z ]*PRIVATE KEY-----/],
  ["bearer-token", /Bearer\s+[A-Za-z0-9._~+/-]{24,}/],
  ["alpaca-key-id", /\b(?:PK|AK)[A-Z0-9]{18}\b/],
  ["operator-token-header-literal", /X-LAT-Operator-Token["']?\s*[:=]\s*["'][A-Za-z0-9_-]{16,}["']/i],
];

const SECRET_NAME = /(KEY|SECRET|TOKEN|PASSWORD)/i;

/** API routes / headers that only state-changing clients use. None may appear in the read-only bundle. */
const READ_ONLY_FORBIDDEN = [
  "/control/halt",
  "/control/resume",
  "/control/reconcile",
  "/pipeline/run-cycle",
  "/risk/policy",
  "/ai/chatgpt/",
  "/agent/tools",
  "/cancel",
  "/engine/",
  "X-LAT-Request",
  "X-LAT-Operator-Token",
  "X-LAT-Agent-Token",
  "X-LAT-Engine-Token",
];

async function walk(dir) {
  const out = [];
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) out.push(...(await walk(full)));
    else out.push(full);
  }
  return out;
}

/** name -> value of every locally known secret (values never leave this process). */
async function localSecrets() {
  const secrets = new Map();
  const add = (name, value) => {
    const v = String(value ?? "").trim();
    if (v.length >= 12) secrets.set(name, v);
  };
  for (const [k, v] of Object.entries(process.env)) if (SECRET_NAME.test(k)) add(`env:${k}`, v);
  const envFile = path.join(projectRoot, ".env");
  if (existsSync(envFile)) {
    for (const line of (await readFile(envFile, "utf8")).split("\n")) {
      const m = /^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$/.exec(line);
      if (m && SECRET_NAME.test(m[1])) add(`.env:${m[1]}`, m[2].replace(/^["']|["']$/g, ""));
    }
  }
  const varDir = path.join(projectRoot, "var");
  if (existsSync(varDir)) {
    for (const ns of await readdir(varDir, { withFileTypes: true })) {
      if (!ns.isDirectory()) continue;
      const keys = path.join(varDir, ns.name, "keys");
      if (!existsSync(keys)) continue;
      for (const f of await readdir(keys)) {
        const full = path.join(keys, f);
        if ((await stat(full)).isFile() && (await stat(full)).size < 4096) add(`var/${ns.name}/keys/${f}`, await readFile(full, "utf8"));
      }
    }
  }
  return secrets;
}

if (!existsSync(dist)) {
  console.error(`check-bundle: ${dist} does not exist (run vite build first)`);
  process.exit(2);
}

const files = await walk(dist);
const problems = [];
const secrets = await localSecrets();
let bundlesLightweightCharts = false;
let sawAttribution = false;
let sawLink = false;
for (const file of files) {
  const rel = path.relative(dist, file);
  if (rel.endsWith(".map")) problems.push(`${rel}: source map shipped`);
  if (!/\.(js|mjs|css|html|txt|json|svg)$/.test(rel)) continue;
  const text = await readFile(file, "utf8");
  for (const [name, re] of SECRET_PATTERNS) if (re.test(text)) problems.push(`${rel}: matches secret pattern '${name}'`);
  for (const [name, value] of secrets) if (text.includes(value)) problems.push(`${rel}: contains the value of local secret ${name}`);
  if (/lightweight-charts|LightweightCharts|attributionLogo/.test(text)) bundlesLightweightCharts = true;
  if (text.includes("Charts by TradingView Lightweight Charts")) sawAttribution = true;
  if (text.includes("https://www.tradingview.com/")) sawLink = true;
  // read-only UI: the bundle must not carry the API's state-changing routes or auth headers
  for (const forbidden of READ_ONLY_FORBIDDEN) if (text.includes(forbidden)) problems.push(`${rel}: contains '${forbidden}' (the dashboard is read-only)`);
}
// TradingView licence: attribution is required only if Lightweight Charts is ever bundled again.
if (bundlesLightweightCharts && !sawAttribution) problems.push("Lightweight Charts is bundled but the TradingView attribution text is missing");
if (bundlesLightweightCharts && !sawLink) problems.push("Lightweight Charts is bundled but the https://www.tradingview.com/ link is missing");

if (problems.length) {
  console.error(problems.join("\n"));
  process.exit(1);
}
console.log(`check-bundle: ${files.length} files clean (${secrets.size} local secret values compared, none present)`);
