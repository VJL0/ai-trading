"""GUI verification workflows (spec §9, acceptance criterion 10).

    uv run --frozen python -m mac_agent.verify <calculator|notes|browser|all> [--no-peekaboo-browser]

* ``calculator`` — Peekaboo: launch Calculator, inspect, click 7 + 8 =, read the display through the
  accessibility tree and confirm ``15``; screenshot the Calculator window into ``artifacts/``.
* ``notes`` — Peekaboo: create a note titled ``local-ai-trader GUI test <UTC timestamp>`` with a short
  body, then read it back through a fresh accessibility inspection. The note is left in place (user data
  is never deleted) and its title is reported.
* ``browser`` — Playwright (guarded Chromium, headless): fixture note type/save/reload/confirm, DENIED
  'Place LIVE order', DENIED https://app.alpaca.markets (pre-check, route layer, page-script request,
  redirect); Playwright MCP (pinned local @playwright/mcp behind the guarded proxy): the same flow through
  MCP tool calls; Peekaboo pass on the fixture page in a headed Chromium window (inspect -> type -> click
  Save -> confirm) with the live-order button DENIED.

Every run writes ``artifacts/verify-<target>-<ts>.json`` and appends policy decisions to
``artifacts/audit.jsonl``. Exit codes: 0 all passed, 1 a check failed, 2 BLOCKED (Peekaboo or macOS
permissions missing; the exact user steps are printed — this tool never changes privacy settings).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .audit import DEFAULT_AUDIT_PATH, AuditLog, verify_chain
from .env import minimal_env
from .peekaboo import (
    PERMISSION_STEPS,
    PeekabooController,
    PeekabooError,
    PeekabooPermissionError,
    PeekabooUnavailable,
    Snapshot,
    UIElement,
)
from .policy import DEFAULT_ARTIFACTS_DIR, MAC_AGENT_ROOT, GuiPolicy, PolicyDenied, Reason

BROWSER_DIR = MAC_AGENT_ROOT / "browser"
BROWSER_APP = "Google Chrome for Testing"
NOTE_TITLE_PREFIX = "local-ai-trader GUI test"
NOTE_BODY = "Created by apps/mac-agent GUI verification. PAPER/SIMULATED test artifact; no trading data."

PASSED, FAILED, BLOCKED = "PASSED", "FAILED", "BLOCKED"


@dataclass
class StepResult:
    name: str
    status: str
    checks: dict[str, bool] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    error: str | None = None

    def finish(self) -> StepResult:
        if self.status != BLOCKED and self.error is None:
            self.status = PASSED if self.checks and all(self.checks.values()) else FAILED
        elif self.error is not None and self.status != BLOCKED:
            self.status = FAILED
        return self

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "status": self.status, "checks": self.checks, "details": self.details,
                "artifacts": self.artifacts, "error": self.error}


def utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def normalize_display(text: str) -> str:
    """Calculator display text -> plain digits ('1,234' / '‎15' / U+2212 minus handled)."""
    t = unicodedata.normalize("NFKC", text)
    t = re.sub(r"[‎‏‪-‮⁦-⁩\s,]", "", t)
    return t.replace("−", "-")  # noqa: RUF001 - normalize calculator Unicode minus


def calculator_display(snap: Snapshot) -> str | None:
    """The result shown by Calculator, from the AX tree only (no OCR, no pixels).

    macOS 27 Calculator exposes it twice: ``AXScrollArea`` identifier ``StandardInputView;value:<v>`` and
    an ``AXStaticText`` inside it. The identifier form is authoritative; the static text is a fallback."""
    for e in snap.elements:
        m = re.fullmatch(r"StandardInputView;value:(.*)", e.identifier)
        if m:
            return normalize_display(m.group(1))
    texts = [e for e in snap.elements if e.ax_role == "AXStaticText" and e.value]
    return normalize_display(texts[-1].value) if texts else None


# ----------------------------------------------------------------------------- preflight
def gui_preflight(pk: PeekabooController) -> tuple[bool, dict[str, Any]]:
    info: dict[str, Any] = {}
    try:
        info["peekaboo_version"] = pk.ensure_version()
        perms = pk.permissions()
    except PeekabooUnavailable as exc:
        info["blocked_reason"] = str(exc)
        return False, info
    except PeekabooError as exc:
        info["blocked_reason"] = f"peekaboo permissions check failed: {exc}"
        return False, info
    info["permissions"] = {"screen_recording": perms.screen_recording, "accessibility": perms.accessibility,
                           "event_synthesizing": perms.event_synthesizing, "source": perms.source}
    if not perms.gui_ready:
        info["blocked_reason"] = f"missing macOS permission(s): {', '.join(perms.missing)}. {PERMISSION_STEPS}"
        return False, info
    return True, info


def _blocked(name: str, info: dict[str, Any]) -> StepResult:
    return StepResult(name, BLOCKED, details=info, error=str(info.get("blocked_reason")))


# ----------------------------------------------------------------------------- calculator
def verify_calculator(pk: PeekabooController) -> StepResult:
    r = StepResult("calculator", "RUNNING")
    ready, info = gui_preflight(pk)
    if not ready:
        return _blocked("calculator", info)
    r.details.update(info)
    try:
        launch = pk.launch("Calculator")
        r.details["launch_effect"] = launch.effect
        win = pk.wait_for_window("Calculator")
        r.details["window_title"] = win.title
        snap = pk.inspect("Calculator")
        r.checks["inspected_elements"] = len(snap.elements) > 0
        # clear (label toggles between "All Clear" and "Clear"), then 7 + 8 =
        for _ in range(2):
            if calculator_display(snap) == "0":
                break
            lab = "All Clear" if pk.resolve_element("Calculator", label="All Clear") else "Clear"
            pk.click("Calculator", label=lab)
            snap = pk.inspect("Calculator")
        clicks: list[dict[str, Any]] = []
        for label in ("7", "Add", "8", "Equals"):
            res = pk.click("Calculator", label=label)
            clicks.append({"label": label, "effect": res.effect, "dispatched": res.mutation_dispatched})
            snap = pk.inspect("Calculator")
        r.details["clicks"] = clicks
        shown = calculator_display(snap)
        r.details["display"] = shown
        r.checks["display_reads_15"] = shown == "15"
        # Calculator has no live-order controls; prove the denylist still blocks a denied app:
        try:
            pk.inspect("System Settings")
            r.checks["denied_app_refused"] = False
        except PolicyDenied as exc:
            r.checks["denied_app_refused"] = exc.decision.reason == Reason.APP_DENYLISTED
        r.artifacts.append(str(pk.screenshot("Calculator", "calculator-15", window_title=win.title)))
    except PeekabooPermissionError as exc:
        return _blocked("calculator", {**info, "blocked_reason": str(exc)})
    except (PeekabooError, PolicyDenied) as exc:
        r.error = f"{type(exc).__name__}: {exc}"
    return r.finish()


# ----------------------------------------------------------------------------- notes
def _empty_note_bodies(snap: Snapshot) -> list[UIElement]:
    return [e for e in snap.elements if e.ax_role == "AXTextArea" and e.identifier.startswith("Note")
            and not e.value.strip()]


def _note_bodies_with(snap: Snapshot, text: str) -> list[UIElement]:
    return [e for e in snap.elements if e.ax_role == "AXTextArea" and text in e.value]


def verify_notes(pk: PeekabooController) -> StepResult:
    r = StepResult("notes", "RUNNING")
    ready, info = gui_preflight(pk)
    if not ready:
        return _blocked("notes", info)
    r.details.update(info)
    title = f"{NOTE_TITLE_PREFIX} {utc_stamp()}"
    r.details["note_title"] = title
    try:
        pk.launch("Notes")
        win = pk.wait_for_window("Notes", timeout_s=20)
        r.details["window_title_prefix"] = win.title.split(" ")[0]
        snap = pk.inspect("Notes")
        r.checks["inspected_elements"] = len(snap.elements) > 0
        pk.click("Notes", label="New Note", roles=("button",))
        snap = pk.inspect("Notes")
        empties = _empty_note_bodies(snap)
        r.details["empty_note_bodies"] = len(empties)
        if len(empties) != 1:
            raise PeekabooError(f"expected exactly one empty new-note body after 'New Note', found {len(empties)}",
                                code="AMBIGUOUS_TARGET")
        typed = pk.type_text("Notes", f"{title}\n{NOTE_BODY}", element_id=empties[0].id, focus_app=True,
                             foreground=True)
        r.details["type_effect"] = typed.effect
        r.details["type_dispatched"] = typed.mutation_dispatched
        # Confirm persisted content through fresh accessibility observations (never by trusting the dispatch).
        found: list[UIElement] = []
        for _ in range(10):
            time.sleep(0.5)
            snap = pk.inspect("Notes", fresh=True)
            found = _note_bodies_with(snap, title)
            if found and NOTE_BODY in found[0].value:
                break
        r.checks["note_body_contains_title"] = bool(found)
        r.checks["note_body_contains_text"] = bool(found) and NOTE_BODY in found[0].value
        time.sleep(1.5)
        snap = pk.inspect("Notes", fresh=True)
        listed = [e for e in snap.elements if e.ax_role != "AXTextArea" and any(title in t for t in e.texts())]
        r.checks["note_still_present_after_autosave"] = bool(_note_bodies_with(snap, title))
        r.details["note_listed_in_sidebar"] = bool(listed)
        r.details["left_in_place"] = True
    except PeekabooPermissionError as exc:
        return _blocked("notes", {**info, "blocked_reason": str(exc)})
    except (PeekabooError, PolicyDenied) as exc:
        r.error = f"{type(exc).__name__}: {exc}"
    return r.finish()


# ----------------------------------------------------------------------------- browser
def find_node() -> str | None:
    override = os.environ.get("MAC_AGENT_NODE")
    cand = override or shutil.which("node")
    return str(Path(cand).resolve()) if cand else None


def node_env(node: str) -> dict[str, str]:
    """Node/Playwright children get the same minimal env as Peekaboo (+ the node binary's directory)."""
    return minimal_env(extra_path_dirs=[Path(node).parent])


def run_node_json(node: str, script: str, *, timeout_s: float = 240.0) -> dict[str, Any]:
    proc = subprocess.run(
        [node, script], cwd=BROWSER_DIR, env=node_env(node), stdin=subprocess.DEVNULL, capture_output=True,
        text=True, timeout=timeout_s, check=False,
    )
    lines = [ln for ln in proc.stdout.strip().splitlines() if ln.startswith("{")]
    if not lines:
        return {"ok": False, "error": f"{script} produced no JSON (rc={proc.returncode}): {proc.stderr.strip()[:400]}"}
    out = json.loads(lines[-1])
    out["returncode"] = proc.returncode
    return out


class HeldFixture:
    """Headed guarded Chromium holding the fixture open (browser/src/hold-fixture.mjs)."""

    def __init__(self, node: str) -> None:
        self.proc = subprocess.Popen(
            [node, "src/hold-fixture.mjs"], cwd=BROWSER_DIR, env=node_env(node), stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        self.ready = self._read_event("ready", timeout_s=60)

    def _read_event(self, event: str, timeout_s: float) -> dict[str, Any]:
        assert self.proc.stdout is not None
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            line = self.proc.stdout.readline()
            if not line:
                break
            if line.startswith("{"):
                obj = json.loads(line)
                if obj.get("event") == event:
                    return obj
                if obj.get("event") == "error":
                    raise RuntimeError(f"hold-fixture: {obj.get('error')}")
        raise RuntimeError(f"hold-fixture did not report {event!r}")

    def state(self) -> dict[str, Any]:
        assert self.proc.stdin is not None
        self.proc.stdin.write("state\n")
        self.proc.stdin.flush()
        return self._read_event("state", timeout_s=15)

    def reload(self) -> dict[str, Any]:
        assert self.proc.stdin is not None
        self.proc.stdin.write("reload\n")
        self.proc.stdin.flush()
        return self._read_event("reloaded", timeout_s=20)

    def close(self) -> None:
        try:
            if self.proc.stdin is not None and self.proc.poll() is None:
                self.proc.stdin.write("close\n")
                self.proc.stdin.flush()
            self.proc.wait(timeout=20)
        except (OSError, subprocess.TimeoutExpired):
            self.proc.kill()


def peekaboo_browser_pass(pk: PeekabooController, node: str) -> StepResult:
    r = StepResult("browser_peekaboo", "RUNNING")
    ready, info = gui_preflight(pk)
    if not ready:
        return _blocked("browser_peekaboo", info)
    held: HeldFixture | None = None
    try:
        held = HeldFixture(node)
        rd = held.ready
        r.details["fixture_url"] = rd.get("url")
        port = urlsplit(str(rd["url"])).port
        if port is None:
            raise RuntimeError("hold-fixture reported a URL without a port")
        pk.policy.register_fixture_port(port)  # our own Node process serves fixtures/ on this loopback port
        pk.bind_browser_page(BROWSER_APP, url=str(rd["url"]), texts=[str(t) for t in rd.get("texts", [])],
                             trading_mode_attr=str(rd.get("trading_mode", "")))
        wins: list[Any] = []
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            wins = [w for w in pk.list_windows(BROWSER_APP) if w.title.startswith("FAKE PAPER DASHBOARD")]
            if wins:
                break
            time.sleep(0.5)
        r.checks["single_fixture_window"] = len(wins) == 1
        if len(wins) != 1:
            raise PeekabooError(f"expected one fixture window, found {len(wins)}", code="AMBIGUOUS_TARGET")
        snap = pk.inspect(BROWSER_APP, window_title=wins[0].title)
        r.checks["web_content_visible"] = len(snap.web_areas()) == 1
        # browser chrome is never driven
        addr = [e for e in snap.elements if e.label == "Address and search bar"]
        if addr:
            try:
                pk.click(BROWSER_APP, element_id=addr[0].id)
                r.checks["address_bar_denied"] = False
            except PolicyDenied as exc:
                r.checks["address_bar_denied"] = exc.decision.reason in (Reason.OUTSIDE_PAGE_CONTENT,
                                                                         Reason.LIVE_ORDER_ELEMENT)
        snap = pk.inspect(BROWSER_APP, window_title=wins[0].title)
        note = f"peekaboo note {utc_stamp()}"
        typed = pk.type_text(BROWSER_APP, note, label="Paper note")
        r.details["type_effect"] = typed.effect
        r.details["type_detail"] = typed.detail
        pk.inspect(BROWSER_APP, window_title=wins[0].title)
        clicked = pk.click(BROWSER_APP, label="Save note")
        r.details["save_click_effect"] = clicked.effect
        rendered = False
        for _ in range(8):  # Chromium updates its native AX tree asynchronously
            snap = pk.inspect(BROWSER_APP, window_title=wins[0].title, fresh=True)
            rendered = any(note in t for e in snap.elements for t in e.texts() if e.ax_role != "AXTextArea")
            if rendered:
                break
            time.sleep(0.5)
        r.checks["saved_text_rendered_in_ax_tree"] = rendered
        reloaded = held.reload()
        r.details["reload"] = reloaded
        persisted = False
        for _ in range(8):
            snap = pk.inspect(BROWSER_APP, window_title=wins[0].title, fresh=True)
            persisted = any(note in t for e in snap.elements for t in e.texts() if e.ax_role != "AXTextArea")
            if persisted:
                break
            time.sleep(0.5)
        r.checks["saved_text_after_reload_in_ax_tree"] = persisted and not reloaded.get("error")
        st = held.state()
        r.checks["saved_text_in_dom"] = st.get("saved_note") == note
        r.checks["saved_text_in_localstorage"] = st.get("stored_note") == note
        try:
            pk.click(BROWSER_APP, label="Place LIVE order")
            r.checks["live_order_click_denied"] = False
        except PolicyDenied as exc:
            r.checks["live_order_click_denied"] = exc.decision.reason == Reason.LIVE_ORDER_ELEMENT
        r.checks["live_order_never_activated"] = held.state().get("live_order_clicked") is False
        r.artifacts.append(str(pk.screenshot(BROWSER_APP, "browser-peekaboo", window_title=wins[0].title)))
    except PeekabooPermissionError as exc:
        return _blocked("browser_peekaboo", {**info, "blocked_reason": str(exc)})
    except (PeekabooError, PolicyDenied, RuntimeError, KeyError) as exc:
        r.error = f"{type(exc).__name__}: {exc}"
    finally:
        if held is not None:
            held.close()
    return r.finish()


def _node_step(name: str, node: str | None, script: str) -> StepResult:
    r = StepResult(name, "RUNNING")
    if node is None:
        return StepResult(name, BLOCKED, error="node not found (install Node 24 or set MAC_AGENT_NODE)")
    if not (BROWSER_DIR / "node_modules" / "playwright").exists():
        return StepResult(
            name, BLOCKED, error="browser deps missing: cd apps/mac-agent/browser && pnpm install --frozen-lockfile"
        )
    out = run_node_json(node, script)
    r.details = {k: v for k, v in out.items() if k not in ("checks", "artifacts", "ok")}
    r.checks = {str(k): v is True for k, v in (out.get("checks") or {}).items()}
    r.artifacts = [str(a) for a in out.get("artifacts") or []]
    if out.get("error"):
        r.error = str(out["error"])
    if not out.get("ok") and not r.error:
        r.error = "node verification reported failure"
    return r.finish()


def verify_browser(pk: PeekabooController, *, peekaboo_pass: bool = True) -> list[StepResult]:
    node = find_node()
    steps = [
        _node_step("browser_playwright", node, "src/verify-fixture.mjs"),
        _node_step("browser_playwright_mcp", node, "src/mcp-smoke.mjs"),
    ]
    if peekaboo_pass:
        steps.append(peekaboo_browser_pass(pk, node) if node else
                     StepResult("browser_peekaboo", BLOCKED, error="node not found"))
    return steps


# ----------------------------------------------------------------------------- main
def run(targets: Sequence[str], *, peekaboo_browser: bool = True,
        controller_factory: Callable[[GuiPolicy], PeekabooController] | None = None) -> dict[str, Any]:
    audit = AuditLog(DEFAULT_AUDIT_PATH, writer="py-verify")
    policy = GuiPolicy(audit=audit)
    factory = controller_factory or (lambda p: PeekabooController(p))
    results: list[StepResult] = []
    with factory(policy) as pk:
        for t in targets:
            if t == "calculator":
                results.append(verify_calculator(pk))
            elif t == "notes":
                results.append(verify_notes(pk))
            elif t == "browser":
                results.extend(verify_browser(pk, peekaboo_pass=peekaboo_browser))
    statuses = [s.status for s in results]
    overall = FAILED if FAILED in statuses else BLOCKED if BLOCKED in statuses else PASSED
    try:
        chain = verify_chain(DEFAULT_AUDIT_PATH)
        chain_ok = True
    except ValueError as exc:
        chain, chain_ok = {"error": str(exc)}, False
    report = {
        "generated_at": utc_stamp(), "overall": overall, "policy_version": policy.version,
        "audit_log": str(DEFAULT_AUDIT_PATH), "audit_chain_ok": chain_ok, "audit_writers": len(chain),
        "steps": [s.as_dict() for s in results],
    }
    DEFAULT_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out = DEFAULT_ARTIFACTS_DIR / f"verify-{'-'.join(targets)}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_path"] = str(out)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m mac_agent.verify", description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("target", choices=["calculator", "notes", "browser", "all"])
    ap.add_argument("--no-peekaboo-browser", action="store_true", help="skip the native Peekaboo pass on the fixture")
    args = ap.parse_args(argv)
    targets = ["calculator", "notes", "browser"] if args.target == "all" else [args.target]
    report = run(targets, peekaboo_browser=not args.no_peekaboo_browser)
    print(json.dumps(report, indent=2))
    return {PASSED: 0, BLOCKED: 2}.get(report["overall"], 1)


if __name__ == "__main__":
    sys.exit(main())
