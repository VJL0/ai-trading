"""Dependency-free local monitor. Emergency actions use the existing control API."""

from __future__ import annotations

import base64
import hashlib
from typing import Any

from fastapi.responses import HTMLResponse

from packages.core.logging import redact
from packages.core.modes import operator_trading_mode

from .runtime import Runtime

NO_STORE = {"Cache-Control": "no-store"}

_STYLE = """
body { color-scheme: light dark; font: 1rem system-ui, sans-serif; max-width: 42rem; margin: 2rem auto; padding: 0 1rem; }
dt { font-weight: 600; } dd { margin: .25rem 0 .9rem; font-variant-numeric: tabular-nums; }
button, input { font: inherit; padding: .35rem .5rem; }
label { display: block; margin-bottom: .4rem; }
section + section { margin-top: 1.6rem; }
"""

_SCRIPT = """
const byId = (id) => document.getElementById(id);
const fields = ["trading_status", "mode", "equity", "daily_pnl", "total_pnl"];
let generation = 0;
async function refresh() {
  const request = ++generation;
  try {
    const response = await fetch("/monitor/status", { cache: "no-store", signal: AbortSignal.timeout(5000) });
    if (!response.ok) throw new Error();
    const state = await response.json();
    if (request !== generation) return;
    for (const key of fields) byId(key).textContent = state[key] ?? "Unavailable";
    byId("updated").textContent = "Updated " + new Date().toLocaleTimeString();
    byId("alerts").replaceChildren(...(state.alerts.length ? state.alerts : ["No active critical alerts."]).map((text) => {
      const li = document.createElement("li");
      li.textContent = text;
      return li;
    }));
  } catch {
    if (request !== generation) return;
    for (const key of fields) byId(key).textContent = "Unavailable";
    byId("alerts").replaceChildren();
    byId("updated").textContent = "Monitor unavailable. Check the local API.";
  }
}
async function poll() {
  await refresh();
  setTimeout(poll, 15000);
}
const buttons = [...document.querySelectorAll("[data-action]")];
for (const button of buttons) {
  button.disabled = false;
  button.addEventListener("click", async () => {
    const all = button.dataset.action === "kill-switch";
    const input = byId("operator");
    const credential = all ? input.value.trim() : "";
    if (all && !credential) {
      byId("result").textContent = "Enter the operator token to use the kill switch.";
      input.focus();
      return;
    }
    if (!confirm(all ? "Halt entries and protective exits?" : "Halt new entries?")) return;
    input.value = "";
    for (const control of buttons) control.disabled = true;
    const headers = { "Content-Type": "application/json", "X-LAT-Request": "1" };
    if (all) headers["X-LAT-Operator-Token"] = credential;
    try {
      const response = await fetch("/api/v1/control/" + button.dataset.action, {
        method: "POST", headers, body: JSON.stringify({ reason: "Emergency halt from local monitor" }),
        signal: AbortSignal.timeout(8000)
      });
      byId("result").textContent = response.ok ? (all ? "Entries and exits halted." : "New entries halted.")
        : response.status === 401 ? "Operator token was refused."
        : response.status >= 500 ? "Outcome unknown. Check trading status." : "Halt request was refused.";
    } catch {
      byId("result").textContent = "Outcome unknown. Check trading status.";
    } finally {
      for (const control of buttons) control.disabled = false;
      await refresh();
    }
  });
}
poll();
"""

_HTML = """<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Trading monitor</title>
<style>""" + _STYLE + """</style>
<main>
  <h1>Trading monitor</h1>
  <p id="updated" role="status">Loading trading status.</p>
  <dl>
    <dt>Trading status</dt><dd id="trading_status">Unavailable</dd>
    <dt>Trading mode</dt><dd id="mode">Unavailable</dd>
    <dt>Account equity (USD)</dt><dd id="equity">Unavailable</dd>
    <dt>Daily P&amp;L (USD)</dt><dd id="daily_pnl">Unavailable</dd>
    <dt>Total P&amp;L (USD)</dt><dd id="total_pnl">Unavailable</dd>
  </dl>
  <section aria-labelledby="alerts-title">
    <h2 id="alerts-title">Critical alerts</h2>
    <ul id="alerts" aria-live="polite"></ul>
  </section>
  <section aria-labelledby="controls-title">
    <h2 id="controls-title">Emergency controls</h2>
    <p>Halting entries lets protective exits continue. The kill switch also blocks protective exits.
      Broker-held orders stay active.</p>
    <p><button type="button" data-action="halt" disabled>Halt new entries</button></p>
    <label for="operator">Operator token for the kill switch</label>
    <input type="password" id="operator" autocomplete="off" spellcheck="false">
    <button type="button" data-action="kill-switch" disabled>Halt entries and exits</button>
    <p id="result" role="status"></p>
    <noscript>Use the operator CLI for emergency controls when JavaScript is disabled.</noscript>
  </section>
</main>
<script>""" + _SCRIPT + """</script>
</html>
"""


def _csp_hash(content: str) -> str:
    return "'sha256-" + base64.b64encode(hashlib.sha256(content.encode()).digest()).decode() + "'"


_HEADERS = {
    **NO_STORE,
    "Content-Security-Policy": (
        f"default-src 'none'; script-src {_csp_hash(_SCRIPT)}; style-src {_csp_hash(_STYLE)}; "
        "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    ),
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
}


def page() -> HTMLResponse:
    return HTMLResponse(_HTML, headers=_HEADERS)


def snapshot(runtime: Runtime) -> dict[str, Any]:
    status = runtime.status()
    performance = runtime.performance(summary_only=True)
    manager = runtime.sessions
    session = manager.active() if manager is not None else None
    phase = status["startup"]["phase"]
    locked = status["entries_locked"]
    exits_locked = status["exits_locked"]
    alerts = []
    if phase != "READY":
        alerts.append(f"Startup reconciliation: {phase}.")
    if not status["broker"]["connected"]:
        alerts.append("Broker disconnected.")
    if locked:
        alerts.append("New entries locked.")
        alerts.extend(status["lock_reasons"])
    if exits_locked:
        alerts.append("Protective exits locked.")
        alerts.extend(status["exit_lock_reasons"])
    if status["engine"]["status"] == "DOWN":
        alerts.append("Execution gateway unavailable.")
    if session and session["state"] == "RUNNING" and status["feed"]["status"] != "OK":
        alerts.append("Market data requires attention.")
    if manager is not None and session is None:
        recent = manager.list(1)
        if recent and recent[0]["state"] in {"FAILED", "INTERRUPTED"}:
            alerts.append(f"Last session {recent[0]['state'].lower()}.")
    alerts.extend(f"{row['symbol']}: {row['state']}" for row in status["position_safety"])
    equity = performance["portfolio_value"]
    session_state = session["state"] if session else "IDLE"
    trading_status = f"{session_state} / ENTRIES LOCKED" if locked else session_state
    return {
        "trading_status": trading_status,
        "mode": operator_trading_mode(runtime.settings.mode, runtime.execution_label).value.title(),
        "equity": equity,
        "daily_pnl": performance["daily_pnl"] if equity is not None else None,
        "total_pnl": performance["total_pnl"] if equity is not None else None,
        "alerts": list(dict.fromkeys(redact(alert) for alert in alerts)),
    }
