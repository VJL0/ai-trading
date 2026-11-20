"""Policy-gated wrapper around the Peekaboo v4.9.7 CLI (spec §9).

Only a fixed action set is exposed — list apps/windows, inspect (accessibility element tree), click
an element of the current snapshot, type text, press allowlisted keys, launch/focus an allowlisted
app and screenshot an allowlisted window into ``apps/mac-agent/artifacts``.

Hard guarantees (each is unit-tested in ``tests/unit/mac_agent``):

* Every call is decided by :class:`mac_agent.policy.GuiPolicy` *before* a process is spawned; a
  denied decision never reaches the runner.
* ``argv`` lists only, ``shell=False``; every caller-derived value is passed as one ``--opt=value``
  token (so text such as ``--help`` or ``; rm -rf ~`` is data, never an option or shell syntax).
  Backslashes in typed text are doubled because ``peekaboo type`` interprets ``\\n \\t \\b \\e``.
* Subcommands are allowlisted: never ``agent`` (free-form natural-language mode), ``mcp``,
  ``browser``, ``clipboard``, ``paste``, ``config``, ``dialog``, ``menu``, ``capture``, ``--analyze``…
* The ``--app`` value is always the canonical bundle id from ``rules.json``, never caller input.
* The child environment is :func:`mac_agent.env.minimal_env` (PATH/HOME/LANG only): no broker
  credentials, model keys or operator/engine tokens.
* ``--no-remote`` always: Peekaboo runs in-process, so the macOS privacy permissions that apply are
  the ones granted to the host app running this agent (and no Bridge/daemon host is used).
* Inspection is accessibility-tree only (``--tree --no-screenshot``): no pixels are captured unless
  :meth:`PeekabooController.screenshot` is called for an app with the screenshot capability, and
  the Peekaboo snapshot caches we create under ``~/.peekaboo/snapshots`` are removed again
  (``peekaboo clean --snapshot <id>``) so window pixels/values do not linger outside ``artifacts/``.
* Peekaboo 4.9.7 snapshots are single-use for mutations (``SNAPSHOT_STALE`` afterwards): the
  controller drops a snapshot after every click/type/press, so each mutation is evaluated against a
  fresh observation of exactly the element that will receive it.
* Unknown/malformed output, timeouts and permission errors raise (fail closed). Mutating commands
  are never retried automatically; a dispatched-but-unconfirmed mutation is reported as such.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .audit import AuditLog, NullAuditLog, scrub
from .env import SAFE_SYSTEM_PATH, minimal_env
from .policy import (
    Action,
    AppRule,
    Decision,
    ElementInfo,
    GuiPolicy,
    PolicyDenied,
    Reason,
    Surface,
    SurfaceKind,
    normalize_text,
)

PEEKABOO_PINNED_VERSION = "4.9.7"
DEFAULT_TIMEOUT_S = 30.0
SNAPSHOT_MAX_AGE_S = 120.0
BROWSER_AX_DEPTH = 60
BROWSER_AX_MAX_ELEMENTS = 2000
_ELEMENT_ID = re.compile(r"[A-Za-z0-9_.:\-]{1,96}")
_SNAPSHOT_ID = re.compile(r"ps1_[0-9a-f]{32}")
_VERSION_RE = re.compile(r"^Peekaboo\s+(\d+\.\d+\.\d+)\b")
_KEY_TOKEN = re.compile(r"[a-z0-9+=.\-*/]{1,24}")

PERMISSION_STEPS = (
    "Grant macOS privacy permissions to the HOST app that runs this agent (e.g. Claude, Terminal, "
    "iTerm or VS Code — whichever process launches `uv run`), then restart that app: "
    "System Settings > Privacy & Security > Accessibility (on macOS 27 Peekaboo 4.9.7 reports this pane as "
    "'Device Control and Data Access') -> enable the host app; and System Settings > Privacy & Security > "
    "Screen & System Audio Recording -> enable the host app. The agent never changes these settings itself."
)

# command path -> (allowed --options, allowed flags, positional args allowed)
_COMMANDS: dict[tuple[str, ...], tuple[frozenset[str], frozenset[str], bool]] = {
    ("--version",): (frozenset(), frozenset(), False),
    ("permissions", "status"): (frozenset(), frozenset(), False),
    ("app", "list"): (frozenset(), frozenset(), False),
    ("app", "launch"): (frozenset({"bundle-id"}), frozenset({"wait-ready", "foreground"}), False),
    ("app", "focus"): (frozenset({"app"}), frozenset({"foreground"}), False),
    ("window", "list"): (frozenset({"app"}), frozenset(), False),
    ("see",): (frozenset({"app", "window-title", "path", "format", "timeout", "max-elements", "depth"}),
               frozenset({"no-elements", "tree", "no-screenshot", "fresh"}), False),
    ("click",): (frozenset({"on", "snapshot"}), frozenset(), False),
    ("type",): (frozenset({"text", "snapshot"}), frozenset({"clear", "foreground"}), False),
    ("press",): (frozenset({"snapshot"}), frozenset(), True),
    ("clean",): (frozenset({"snapshot"}), frozenset(), False),
}
FORBIDDEN_WORDS = frozenset({"agent", "mcp", "browser", "clipboard", "paste", "config", "dialog", "dock", "menu",
                             "menubar", "space", "capture", "learn", "tools", "daemon", "bridge", "drag",
                             "move", "scroll", "set-value", "select-text", "action", "visualizer", "analyze",
                             "quit", "relaunch", "hide", "unhide", "close", "verify", "screen", "completions"})
_MUTATING = frozenset({("app", "launch"), ("app", "focus"), ("click",), ("type",), ("press",)})


# ----------------------------------------------------------------------------- errors
class PeekabooError(RuntimeError):
    """Peekaboo failed or returned something we cannot trust. State is unknown -> fail closed."""

    def __init__(self, message: str, *, code: str = "PEEKABOO_ERROR", mutation_dispatched: bool | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.mutation_dispatched = mutation_dispatched


class PeekabooUnavailable(PeekabooError):
    """Binary missing, wrong version, or not runnable."""


class PeekabooPermissionError(PeekabooError):
    """macOS Accessibility / Screen Recording permission is missing for the host app."""

    def __init__(self, message: str, *, missing: Sequence[str] = (), code: str = "PERMISSION_ERROR") -> None:
        super().__init__(f"{message}. {PERMISSION_STEPS}", code=code, mutation_dispatched=False)
        self.missing = tuple(missing)


class PeekabooTimeout(PeekabooError):
    """The command did not finish. A mutating command's outcome is UNKNOWN and must not be retried."""


# ----------------------------------------------------------------------------- runner
@dataclass(frozen=True)
class RunResult:
    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    duration_s: float


Runner = Callable[[Sequence[str], Mapping[str, str], float], RunResult]


def subprocess_runner(argv: Sequence[str], env: Mapping[str, str], timeout: float) -> RunResult:
    """The only place a Peekaboo process is spawned: argv list, no shell, minimal env, no stdin."""
    if not argv or not Path(argv[0]).is_absolute():
        raise PeekabooUnavailable("peekaboo binary path must be absolute")
    started = time.monotonic()
    try:
        proc = subprocess.run(
            list(argv), shell=False, env=dict(env), stdin=subprocess.DEVNULL, capture_output=True,
            text=True, timeout=timeout, check=False, close_fds=True,
        )
    except subprocess.TimeoutExpired as exc:
        raise PeekabooTimeout(f"peekaboo {argv[1] if len(argv) > 1 else ''} timed out after {timeout:.0f}s",
                              code="TIMEOUT") from exc
    except OSError as exc:
        raise PeekabooUnavailable(f"could not execute peekaboo: {exc.strerror}", code="EXEC_FAILED") from exc
    return RunResult(tuple(argv), proc.returncode, proc.stdout, proc.stderr, time.monotonic() - started)


def find_peekaboo() -> str | None:
    """Resolve ``peekaboo`` on the fixed system PATH only (never the caller's PATH)."""
    found = shutil.which("peekaboo", path=":".join(SAFE_SYSTEM_PATH))
    return str(Path(found).resolve()) if found else None


# ----------------------------------------------------------------------------- argv
def escape_type_text(text: str) -> str:
    """``peekaboo type`` interprets ``\\n \\t \\b \\e \\\\``; doubling backslashes makes every one literal."""
    return text.replace("\\", "\\\\")


def build_argv(binary: str, command: Sequence[str], *, options: Sequence[tuple[str, str]] = (),
               flags: Sequence[str] = (), positionals: Sequence[str] = ()) -> list[str]:
    """Build a validated argv. Raises ValueError for anything outside the fixed command surface."""
    cmd = tuple(command)
    spec = _COMMANDS.get(cmd)
    if spec is None or any(w in FORBIDDEN_WORDS for w in cmd):
        raise ValueError(f"peekaboo command {' '.join(cmd)!r} is not in the allowlisted action set")
    allowed_opts, allowed_flags, allow_pos = spec
    if not Path(binary).is_absolute():
        raise ValueError("binary must be an absolute path")
    argv = [binary, *cmd]
    for name, value in options:
        if name not in allowed_opts:
            raise ValueError(f"option --{name} not allowed for {' '.join(cmd)}")
        if not isinstance(value, str) or "\x00" in value or len(value) > 8192:
            raise ValueError(f"invalid value for --{name}")
        if name == "snapshot" and not _SNAPSHOT_ID.fullmatch(value):
            raise ValueError("invalid snapshot id")
        if name == "on" and not _ELEMENT_ID.fullmatch(value):
            raise ValueError("invalid element id")
        argv.append(f"--{name}={value}")
    for flag in flags:
        if flag not in allowed_flags:
            raise ValueError(f"flag --{flag} not allowed for {' '.join(cmd)}")
        argv.append(f"--{flag}")
    if positionals:
        if not allow_pos:
            raise ValueError(f"positional arguments not allowed for {' '.join(cmd)}")
        for p in positionals:
            if not _KEY_TOKEN.fullmatch(p) or p.startswith("-"):
                raise ValueError("invalid positional argument")
            argv.append(p)
    if cmd != ("--version",):
        argv += ["--json", "--no-remote"]
    return argv


def redact_argv(argv: Sequence[str]) -> list[str]:
    """Audit form of argv: typed text replaced by its length; everything else scrubbed and truncated."""
    out: list[str] = []
    for a in argv[1:]:
        if a.startswith("--text="):
            out.append(f"--text=<{len(a) - 7} chars>")
        else:
            out.append(scrub(a)[:200])
    return out


# ----------------------------------------------------------------------------- parsed output
@dataclass(frozen=True)
class PeekabooResponse:
    success: bool
    data: dict[str, Any]
    error_code: str | None
    error_message: str | None
    outcome: dict[str, Any]
    raw: dict[str, Any]

    @property
    def mutation_dispatched(self) -> bool | None:
        v = self.outcome.get("mutation_dispatched")
        if v is None and isinstance(self.raw.get("error"), dict):
            v = self.raw["error"].get("mutation_dispatched")
        return v if isinstance(v, bool) else None

    @property
    def effect(self) -> str | None:
        v = self.outcome.get("effect", self.raw.get("effect"))
        return v if isinstance(v, str) else None

    @property
    def confirmed(self) -> bool:
        """True only when Peekaboo itself verified the change (``effect == confirmed``)."""
        return self.success and self.effect == "confirmed"


def parse_response(stdout: str) -> PeekabooResponse:
    """Parse ``--json`` output. Anything that is not a JSON object with a boolean ``success`` is refused."""
    text = stdout.strip()
    if not text:
        raise PeekabooError("peekaboo produced no JSON output", code="MALFORMED_OUTPUT")
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PeekabooError("peekaboo output is not valid JSON", code="MALFORMED_OUTPUT") from exc
    if not isinstance(obj, dict) or not isinstance(obj.get("success"), bool):
        raise PeekabooError("peekaboo JSON lacks a boolean 'success'", code="MALFORMED_OUTPUT")
    obj.pop("debug_logs", None)
    data = obj.get("data") if isinstance(obj.get("data"), dict) else {}
    err = obj.get("error") if isinstance(obj.get("error"), dict) else {}
    outcome = obj.get("outcome") if isinstance(obj.get("outcome"), dict) else {}
    code = (err or {}).get("code")
    msg = (err or {}).get("message")
    return PeekabooResponse(
        success=obj["success"], data=data or {}, error_code=code if isinstance(code, str) else None,
        error_message=msg if isinstance(msg, str) else None, outcome=outcome or {}, raw=obj,
    )


@dataclass(frozen=True)
class PermissionsStatus:
    screen_recording: bool | None
    accessibility: bool | None
    event_synthesizing: bool | None
    source: str

    @property
    def gui_ready(self) -> bool:
        return self.screen_recording is True and self.accessibility is True

    @property
    def missing(self) -> list[str]:
        out: list[str] = []
        if self.accessibility is not True:
            out.append("Accessibility")
        if self.screen_recording is not True:
            out.append("Screen Recording")
        return out


def parse_permissions(resp: PeekabooResponse) -> PermissionsStatus:
    vals: dict[str, bool] = {}
    for p in resp.data.get("permissions", []) or []:
        if isinstance(p, dict) and isinstance(p.get("name"), str) and isinstance(p.get("isGranted"), bool):
            vals[p["name"].lower()] = p["isGranted"]
    return PermissionsStatus(
        screen_recording=vals.get("screen recording"), accessibility=vals.get("accessibility"),
        event_synthesizing=vals.get("event synthesizing"), source=str(resp.data.get("source", "")),
    )


@dataclass(frozen=True)
class AppInfo:
    name: str
    bundle_id: str
    pid: int | None
    is_active: bool


def parse_app_list(resp: PeekabooResponse) -> list[AppInfo]:
    apps = resp.data.get("apps")
    if not isinstance(apps, list):
        raise PeekabooError("app list JSON lacks data.apps", code="MALFORMED_OUTPUT")
    out: list[AppInfo] = []
    for a in apps:
        if not isinstance(a, dict):
            continue
        pid = a.get("pid")
        out.append(AppInfo(name=str(a.get("name") or ""), bundle_id=str(a.get("bundle_id") or ""),
                           pid=pid if isinstance(pid, int) and not isinstance(pid, bool) else None,
                           is_active=a.get("is_active") is True))
    return out


@dataclass(frozen=True)
class WindowInfo:
    window_id: int | None
    title: str
    index: int | None
    is_on_screen: bool | None


def parse_window_list(resp: PeekabooResponse) -> list[WindowInfo]:
    wins = resp.data.get("windows")
    if not isinstance(wins, list):
        raise PeekabooError("window list JSON lacks data.windows", code="MALFORMED_OUTPUT")
    out: list[WindowInfo] = []
    for w in wins:
        if not isinstance(w, dict):
            continue
        wid, idx = w.get("window_id"), w.get("window_index")
        out.append(WindowInfo(window_id=wid if isinstance(wid, int) else None, title=str(w.get("window_title") or ""),
                              index=idx if isinstance(idx, int) else None,
                              is_on_screen=w.get("is_on_screen") if isinstance(w.get("is_on_screen"), bool) else None))
    return out


def _s(v: Any) -> str:
    return v if isinstance(v, str) else ""


@dataclass(frozen=True)
class UIElement:
    """One row of ``peekaboo see --json`` ``data.ui_elements[]`` (UIElementSummary in Peekaboo 4.9.7)."""

    id: str
    role: str
    ax_role: str = ""
    title: str = ""
    label: str = ""
    value: str = ""
    description: str = ""
    role_description: str = ""
    help: str = ""
    identifier: str = ""
    is_actionable: bool = False
    is_enabled: bool | None = None
    bounds: tuple[float, float, float, float] | None = None

    def to_policy(self) -> ElementInfo:
        return ElementInfo(role=self.role, ax_role=self.ax_role, label=self.label, title=self.title,
                           description=self.description, identifier=self.identifier, help=self.help,
                           role_description=self.role_description, value=self.value)

    def texts(self) -> list[str]:
        return [t for t in (self.label, self.title, self.description, self.identifier, self.value) if t]

    def matches_label(self, wanted: str) -> bool:
        w = normalize_text(wanted).lower()
        return any(normalize_text(t).lower() == w for t in (self.label, self.title, self.description, self.identifier)
                   if t)

    def same_identity(self, other: UIElement) -> bool:
        """Is ``other`` (from a newer snapshot) the same on-screen element? Ids are per-snapshot, so compare the
        stable accessibility attributes and position instead."""
        if (self.ax_role, self.role, self.identifier) != (other.ax_role, other.role, other.identifier):
            return False
        if self.bounds is None or other.bounds is None:
            return False
        return all(abs(a - b) <= 2.0 for a, b in zip(self.bounds, other.bounds, strict=True))


@dataclass
class Snapshot:
    app_key: str
    snapshot_id: str | None
    application_name: str
    window_title: str
    elements: list[UIElement]
    mutation_targeting_available: bool
    truncated: bool
    screenshot: str | None
    taken_at: float = field(default_factory=time.monotonic)
    consumed: bool = False
    requested_window_title: str | None = None

    def find_by_id(self, element_id: str) -> UIElement | None:
        return next((e for e in self.elements if e.id == element_id), None)

    def find_by_label(self, label: str, *, roles: Sequence[str] = ()) -> list[UIElement]:
        rl = {r.lower() for r in roles}
        return [e for e in self.elements if e.matches_label(label) and (not rl or e.role.lower() in rl
                                                                         or e.ax_role.lower() in rl)]

    def find_same(self, element: UIElement) -> UIElement | None:
        hits = [e for e in self.elements if element.same_identity(e)]
        return hits[0] if len(hits) == 1 else None

    def web_areas(self) -> list[UIElement]:
        return [e for e in self.elements if e.ax_role == "AXWebArea"]

    def static_texts(self) -> list[str]:
        out: list[str] = []
        for e in self.elements:
            if e.role.lower() in ("statictext", "text") or e.ax_role == "AXStaticText":
                out += [t for t in (e.value, e.label, e.title) if t]
        return out

    def surface_texts(self) -> tuple[str, ...]:
        """Visible, non-editable texts used for LIVE-mode detection (editable values are user content)."""
        texts = [self.window_title, *self.static_texts()]
        for e in self.elements:
            if e.ax_role in ("AXButton", "AXWindow", "AXHeading", "AXLink") or e.role == "button":
                texts += [t for t in (e.label, e.title, e.description) if t]
        return tuple(t for t in texts if t)


def parse_snapshot(resp: PeekabooResponse, *, app_key: str, redact_values: bool = False) -> Snapshot:
    d = resp.data
    rows = d.get("ui_elements")
    if not isinstance(rows, list):
        raise PeekabooError("see JSON lacks data.ui_elements", code="MALFORMED_OUTPUT")
    elements: list[UIElement] = []
    for r in rows:
        if not isinstance(r, dict) or not isinstance(r.get("id"), str) or not _ELEMENT_ID.fullmatch(r["id"]):
            continue  # unusable rows are dropped, never guessed
        b = r.get("bounds") if isinstance(r.get("bounds"), dict) else None
        bounds = None
        if b and all(isinstance(b.get(k), (int, float)) and not isinstance(b.get(k), bool)
                     for k in ("x", "y", "width", "height")):
            bounds = (float(b["x"]), float(b["y"]), float(b["width"]), float(b["height"]))
        value = _s(r.get("value"))
        label = _s(r.get("label"))
        is_text = any(k in (_s(r.get("role")) + _s(r.get("ax_role"))).lower() for k in ("textfield", "textarea"))
        elements.append(UIElement(
            id=r["id"], role=_s(r.get("role")), ax_role=_s(r.get("ax_role")), title=scrub(_s(r.get("title"))),
            # editable fields expose their content as label/value: never keep it for redacting apps (terminals)
            label="" if (redact_values and is_text) else scrub(label),
            value="" if redact_values else scrub(value),
            description=scrub(_s(r.get("description"))), role_description=_s(r.get("role_description")),
            help=scrub(_s(r.get("help"))), identifier=_s(r.get("identifier")),
            is_actionable=r.get("is_actionable") is True,
            is_enabled=r.get("is_enabled") if isinstance(r.get("is_enabled"), bool) else None, bounds=bounds,
        ))
    trunc = d.get("truncation") if isinstance(d.get("truncation"), dict) else None
    sid = d.get("snapshot_id")
    return Snapshot(
        app_key=app_key,
        snapshot_id=sid if isinstance(sid, str) and _SNAPSHOT_ID.fullmatch(sid) else None,
        application_name=_s(d.get("application_name")), window_title=_s(d.get("window_title")),
        elements=elements, mutation_targeting_available=d.get("mutation_targeting_available") is True,
        truncated=bool(trunc and any(bool(v) for k, v in trunc.items() if k != "warning")),
        screenshot=_s(d.get("screenshot_raw")) or None,
    )


@dataclass(frozen=True)
class ActionResult:
    action: str
    confirmed: bool
    mutation_dispatched: bool | None
    effect: str | None
    detail: dict[str, Any]


# ----------------------------------------------------------------------------- controller
@dataclass
class _BoundPage:
    url: str
    texts: tuple[str, ...]
    trading_mode_attr: str


class PeekabooController:
    """The GUI agent's only route to native macOS control."""

    def __init__(
        self,
        policy: GuiPolicy,
        *,
        binary: str | None = None,
        runner: Runner = subprocess_runner,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        env_source: Mapping[str, str] | None = None,
        audit: AuditLog | NullAuditLog | None = None,
        expected_version: str | None = PEEKABOO_PINNED_VERSION,
        clean_snapshots: bool = True,
    ) -> None:
        self.policy = policy
        self._binary = binary
        self._runner = runner
        self.timeout_s = timeout_s
        self._env_source = env_source
        self.audit = audit if audit is not None else policy.audit
        self.expected_version = expected_version
        self.clean_snapshots = clean_snapshots
        self._version_checked: str | None = None
        self._snapshots: dict[str, Snapshot] = {}
        self._browser_pages: dict[str, _BoundPage] = {}
        self._created_snapshot_ids: list[str] = []

    # -- plumbing ---------------------------------------------------------------------------
    @property
    def binary(self) -> str:
        b = self._binary or find_peekaboo()
        if not b:
            raise PeekabooUnavailable("peekaboo is not installed (brew install openclaw/tap/peekaboo)",
                                      code="NOT_INSTALLED")
        return b

    def child_env(self) -> dict[str, str]:
        return minimal_env(self._env_source)

    def _exec(self, command: Sequence[str], *, options: Sequence[tuple[str, str]] = (), flags: Sequence[str] = (),
              positionals: Sequence[str] = (), timeout_s: float | None = None) -> RunResult:
        argv = build_argv(self.binary, command, options=options, flags=flags, positionals=positionals)
        cmd = tuple(command)
        try:
            res = self._runner(argv, self.child_env(), timeout_s or self.timeout_s)
        except PeekabooError as exc:
            self.audit.record_event("peekaboo_exec", command=" ".join(cmd), argv=redact_argv(argv), ok=False,
                                    error_code=exc.code, mutating=cmd in _MUTATING)
            raise
        self.audit.record_event("peekaboo_exec", command=" ".join(cmd), argv=redact_argv(argv), ok=res.returncode == 0,
                                returncode=res.returncode, duration_ms=int(res.duration_s * 1000),
                                mutating=cmd in _MUTATING)
        return res

    def _json(self, command: Sequence[str], **kw: Any) -> PeekabooResponse:
        self.ensure_version()
        res = self._exec(command, **kw)
        resp = parse_response(res.stdout or res.stderr)
        self._remember_snapshot(resp)
        code = resp.error_code or ""
        if code.startswith(("PERMISSION_ERROR", "PERMISSION_DENIED")):
            missing = (["Accessibility"] if "ACCESSIBILITY" in code or "EVENT" in code
                       else ["Screen Recording"] if "SCREEN" in code else [])
            raise PeekabooPermissionError(resp.error_message or code, missing=missing, code=code)
        if not resp.success:
            raise PeekabooError(f"peekaboo {' '.join(command)} failed: {code or 'UNKNOWN'}: "
                                f"{scrub(resp.error_message or '')[:300]}", code=code or "UNKNOWN",
                                mutation_dispatched=resp.mutation_dispatched)
        if res.returncode != 0:
            raise PeekabooError("peekaboo exited non-zero but reported success", code="INCONSISTENT",
                                mutation_dispatched=resp.mutation_dispatched)
        return resp

    def _remember_snapshot(self, resp: PeekabooResponse) -> None:
        sid = resp.data.get("snapshot_id")
        if isinstance(sid, str) and _SNAPSHOT_ID.fullmatch(sid) and sid not in self._created_snapshot_ids:
            self._created_snapshot_ids.append(sid)

    def version(self) -> str:
        res = self._exec(("--version",), timeout_s=10)
        m = _VERSION_RE.search(res.stdout.strip())
        if res.returncode != 0 or not m:
            raise PeekabooUnavailable("could not determine peekaboo version", code="VERSION_UNKNOWN")
        return m.group(1)

    def ensure_version(self) -> str:
        if self._version_checked is None:
            v = self.version()
            if self.expected_version and v != self.expected_version:
                raise PeekabooUnavailable(f"peekaboo {v} installed; this agent is pinned to {self.expected_version}",
                                          code="VERSION_MISMATCH")
            self._version_checked = v
        return self._version_checked

    def permissions(self) -> PermissionsStatus:
        self.ensure_version()
        res = self._exec(("permissions", "status"), timeout_s=15)
        return parse_permissions(parse_response(res.stdout or res.stderr))

    def _rule(self, app: str, action: Action) -> AppRule:
        decision = self.policy.check_app(app, action)
        self.policy.enforce(decision)
        rule = self.policy.resolve_app(app)
        if rule is None:  # pragma: no cover - check_app already refused
            raise PolicyDenied(decision)
        return rule

    def close(self) -> int:
        """Remove the Peekaboo snapshot caches this controller created (element maps / raw captures under
        ``~/.peekaboo/snapshots``). Returns the number of snapshots cleaned."""
        cleaned = 0
        if not self.clean_snapshots:
            return 0
        while self._created_snapshot_ids:
            sid = self._created_snapshot_ids.pop()
            try:
                self._exec(("clean",), options=[("snapshot", sid)], timeout_s=15)
                cleaned += 1
            except PeekabooError:
                continue
        self._snapshots.clear()
        return cleaned

    def __enter__(self) -> PeekabooController:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- read-only --------------------------------------------------------------------------
    def list_apps(self) -> tuple[list[AppInfo], int]:
        """Running apps, filtered to the allowlist. Returns (allowlisted apps, number hidden)."""
        self.policy.record(True, Action.LIST, Reason.ALLOWED, "list running apps (filtered to the allowlist)")
        apps = parse_app_list(self._json(("app", "list")))
        visible = [a for a in apps if self.policy.is_allowlisted_app(a.name, a.bundle_id)]
        return visible, len(apps) - len(visible)

    def list_windows(self, app: str) -> list[WindowInfo]:
        rule = self._rule(app, Action.LIST)
        wins = parse_window_list(self._json(("window", "list"), options=[("app", rule.canonical_target)]))
        if rule.redact_values:
            return [WindowInfo(w.window_id, "" if self.policy.title_is_sensitive(w.title) else w.title, w.index,
                               w.is_on_screen) for w in wins]
        return wins

    def wait_for_window(self, app: str, *, timeout_s: float = 15.0, poll_s: float = 0.5) -> WindowInfo:
        """Poll ``window list`` until the app has an on-screen titled window (read-only)."""
        deadline = time.monotonic() + timeout_s
        while True:
            wins = [w for w in self.list_windows(app) if w.is_on_screen is not False and w.title]
            if wins:
                return wins[0]
            if time.monotonic() >= deadline:
                raise PeekabooError(f"{app} exposed no on-screen window within {timeout_s:.0f}s", code="NO_WINDOW")
            time.sleep(poll_s)

    def bind_browser_page(self, app: str, *, url: str, texts: Sequence[str] = (), trading_mode_attr: str = "") -> None:
        """Tell the controller which page our own Playwright process opened in the allowlisted browser.

        Without a bound page every browser window is an UNKNOWN surface (denied)."""
        rule = self._rule(app, Action.LIST)
        if not rule.is_browser:
            raise ValueError("bind_browser_page is only for allowlisted browser apps")
        self._browser_pages[rule.key] = _BoundPage(url, tuple(texts), trading_mode_attr)

    def surface_for(self, rule: AppRule, window_title: str, texts: Sequence[str] = ()) -> Surface:
        if rule.is_browser:
            bound = self._browser_pages.get(rule.key)
            if bound is None:
                return Surface(kind=SurfaceKind.UNKNOWN, app_name=rule.display_name, bundle_id=rule.canonical_target,
                               window_title=window_title)
            return self.policy.browser_surface(url=bound.url, title=window_title, texts=(*bound.texts, *texts),
                                               trading_mode_attr=bound.trading_mode_attr,
                                               app_name=rule.display_name, bundle_id=rule.canonical_target)
        return Surface(kind=SurfaceKind.NATIVE_APP, app_name=rule.display_name, bundle_id=rule.canonical_target,
                       window_title=window_title, texts=tuple(texts))

    def _snapshot_surface(self, rule: AppRule, snap: Snapshot) -> Surface:
        return self.surface_for(rule, snap.window_title, snap.surface_texts())

    def inspect(self, app: str, *, window_title: str | None = None, fresh: bool = False, depth: int | None = None,
                max_elements: int | None = None) -> Snapshot:
        """Accessibility element tree of an allowlisted app's window (tree only: no pixels are captured).

        The returned snapshot is the one later click/type/press calls are evaluated and dispatched against.
        Browsers default to a deep traversal (web content nests deeply; Peekaboo's default depth hides the
        text of nested page elements, which LIVE-mode detection and read-back need)."""
        rule = self._rule(app, Action.INSPECT)
        if window_title is not None and rule.redact_values and self.policy.title_is_sensitive(window_title):
            self.policy.enforce(self.policy.check_surface(self.surface_for(rule, window_title), Action.INSPECT))
        if rule.is_browser:
            depth = BROWSER_AX_DEPTH if depth is None else depth
            max_elements = BROWSER_AX_MAX_ELEMENTS if max_elements is None else max_elements
        options: list[tuple[str, str]] = [("app", rule.canonical_target)]
        if window_title:
            options.append(("window-title", window_title))
        if depth is not None:
            if not 1 <= depth <= 100:
                raise ValueError("depth must be 1..100")
            options.append(("depth", str(depth)))
        if max_elements is not None:
            if not 1 <= max_elements <= 5000:
                raise ValueError("max_elements must be 1..5000")
            options.append(("max-elements", str(max_elements)))
        flags = ["tree", "no-screenshot"] + (["fresh"] if fresh else [])
        self._drop_snapshot(rule.key)
        resp = self._json(("see",), options=options, flags=flags, timeout_s=max(self.timeout_s, 30))
        snap = parse_snapshot(resp, app_key=rule.key, redact_values=rule.redact_values)
        if snap.application_name and not any(
                normalize_text(snap.application_name).lower() == normalize_text(n).lower() for n in rule.names):
            raise PolicyDenied(self.policy.record(
                False, Action.INSPECT, Reason.APP_NOT_ALLOWLISTED, "inspected window belongs to a different app",
                {"app": rule.display_name, "observed_app": snap.application_name[:80]}))
        decision = self.policy.check_surface(self._snapshot_surface(rule, snap), Action.INSPECT)
        if not decision.allowed:
            raise PolicyDenied(decision)
        snap.requested_window_title = window_title
        self._snapshots[rule.key] = snap
        return snap

    def screenshot(self, app: str, name: str, *, window_title: str | None = None) -> Path:
        """Capture one allowlisted window into ``apps/mac-agent/artifacts/<name>-<ts>.png``."""
        rule = self._rule(app, Action.SCREENSHOT)
        path = self._artifact_path(name)
        self.policy.enforce(self.policy.check_artifact_path(path))
        if rule.screenshot_precheck or window_title is None:
            # Titles are checked BEFORE pixels are captured (IDE windows may show secrets).
            titles = [w.title for w in self.list_windows(app) if w.is_on_screen is not False and w.title]
            target_title = window_title or (titles[0] if titles else "")
            self.policy.enforce(self.policy.check_surface(self.surface_for(rule, target_title), Action.SCREENSHOT))
            window_title = target_title or None
        else:
            self.policy.enforce(self.policy.check_surface(self.surface_for(rule, window_title), Action.SCREENSHOT))
        options: list[tuple[str, str]] = [("app", rule.canonical_target), ("path", str(path)), ("format", "png")]
        if window_title:
            options.append(("window-title", window_title))
        resp = self._json(("see",), options=options, flags=["no-elements"])
        captured = resp.data.get("screenshot_raw") or resp.data.get("path")
        files = resp.data.get("files")
        if not captured and isinstance(files, list) and files and isinstance(files[0], dict):
            captured = files[0].get("path")
        if not isinstance(captured, str) or Path(captured).resolve() != path.resolve():
            self._discard(path)
            raise PeekabooError("screenshot output path does not match the requested artifact path",
                                code="UNEXPECTED_OUTPUT")
        observed_title = _s(resp.data.get("window_title")) or (window_title or "")
        decision = self.policy.check_surface(self.surface_for(rule, observed_title), Action.SCREENSHOT)
        if not decision.allowed:
            self._discard(path)
            raise PolicyDenied(decision)
        self.audit.record_event("artifact_written", kind="screenshot", path=str(path), app=rule.display_name)
        return path

    # -- mutating -----------------------------------------------------------------------------
    def launch(self, app: str) -> ActionResult:
        rule = self._rule(app, Action.LAUNCH)
        resp = self._json(("app", "launch"), options=[("bundle-id", rule.canonical_target)],
                          flags=["wait-ready", "foreground"], timeout_s=max(self.timeout_s, 45))
        if str(resp.data.get("bundle_id", "")).lower() != rule.canonical_target.lower():
            raise PeekabooError("launched bundle id does not match the allowlisted app", code="UNEXPECTED_OUTPUT",
                                mutation_dispatched=resp.mutation_dispatched)
        return ActionResult("launch", resp.confirmed, resp.mutation_dispatched, resp.effect,
                            {"pid": resp.data.get("pid"), "app_name": resp.data.get("app_name")})

    def focus(self, app: str) -> ActionResult:
        rule = self._rule(app, Action.FOCUS)
        resp = self._json(("app", "focus"), options=[("app", rule.canonical_target)], flags=["foreground"])
        return ActionResult("focus", resp.confirmed, resp.mutation_dispatched, resp.effect, {})

    def _drop_snapshot(self, key: str) -> None:
        self._snapshots.pop(key, None)

    def _current_snapshot(self, rule: AppRule, action: Action) -> Snapshot:
        snap = self._snapshots.get(rule.key)
        if snap is None or snap.consumed or time.monotonic() - snap.taken_at > SNAPSHOT_MAX_AGE_S:
            raise PolicyDenied(self.policy.record(False, action, Reason.ELEMENT_UNKNOWN,
                                                  "no fresh inspected snapshot for this app; inspect first",
                                                  {"app": rule.display_name}))
        if not snap.snapshot_id or not snap.mutation_targeting_available:
            raise PolicyDenied(self.policy.record(False, action, Reason.ELEMENT_UNKNOWN,
                                                  "snapshot is read-only (no reusable snapshot id)",
                                                  {"app": rule.display_name}))
        return snap

    def resolve_element(self, app: str, *, element_id: str | None = None, label: str | None = None,
                        roles: Sequence[str] = ()) -> UIElement | None:
        """Find an element in the current snapshot by opaque id or exact label. Ambiguous labels resolve to
        nothing (fail closed) unless exactly one candidate is actionable."""
        rule = self._rule(app, Action.INSPECT)
        snap = self._snapshots.get(rule.key)
        if snap is None:
            return None
        if element_id is not None:
            return snap.find_by_id(element_id)
        if label is not None:
            hits = snap.find_by_label(label, roles=roles)
            if len(hits) == 1:
                return hits[0]
            actionable = [h for h in hits if h.is_actionable]
            return actionable[0] if len(actionable) == 1 else None
        return None

    def _page_content_check(self, rule: AppRule, snap: Snapshot | None, element: UIElement | None,
                            action: Action) -> None:
        """Browsers: only page content may be driven — never the browser chrome (address bar, tabs, menus,
        site-information popups). The element must lie inside the single AXWebArea of the bound page."""
        if not rule.is_browser or element is None or snap is None:
            return
        areas = snap.web_areas()
        bound = self._browser_pages.get(rule.key)
        ok = False
        if len(areas) == 1 and element.bounds is not None and areas[0].bounds is not None and bound is not None:
            ax, ay, aw, ah = areas[0].bounds
            ex, ey, ew, eh = element.bounds
            inside = ex >= ax - 1 and ey >= ay - 1 and ex + ew <= ax + aw + 1 and ey + eh <= ay + ah + 1
            ok = inside and element.ax_role != "AXWebArea"
        if not ok:
            raise PolicyDenied(self.policy.record(
                False, action, Reason.OUTSIDE_PAGE_CONTENT,
                "browser chrome / element outside the bound page's web content is never driven",
                {"app": rule.display_name, "element": element.to_policy().summary()}))

    def check_click(self, app: str, element: UIElement | None) -> Decision:
        """Pure policy evaluation of a click on ``element`` in the current snapshot (audited, never dispatched)."""
        rule = self._rule(app, Action.CLICK)
        snap = self._snapshots.get(rule.key)
        surface = self._snapshot_surface(rule, snap) if snap else self.surface_for(rule, "")
        return self.policy.check_element(surface, element.to_policy() if element else None, Action.CLICK)

    def _dispatch(self, rule: AppRule, snap: Snapshot, command: tuple[str, ...], *,
                  options: Sequence[tuple[str, str]], flags: Sequence[str] = (),
                  positionals: Sequence[str] = ()) -> PeekabooResponse:
        """Run one mutating command against ``snap``. The snapshot is consumed whatever happens."""
        snap.consumed = True
        self._drop_snapshot(rule.key)
        return self._json(command, options=options, flags=flags, positionals=positionals)

    def click(self, app: str, *, element_id: str | None = None, label: str | None = None,
              roles: Sequence[str] = ()) -> ActionResult:
        """Click an element of the current snapshot, by opaque id or exact label. Never by free-text query or
        coordinates: the element we evaluated is exactly the element Peekaboo presses."""
        rule = self._rule(app, Action.CLICK)
        element = self.resolve_element(app, element_id=element_id, label=label, roles=roles)
        self.policy.enforce(self.check_click(app, element))
        assert element is not None
        snap = self._current_snapshot(rule, Action.CLICK)
        self._page_content_check(rule, snap, element, Action.CLICK)
        assert snap.snapshot_id is not None
        resp = self._dispatch(rule, snap, ("click",), options=[("on", element.id), ("snapshot", snap.snapshot_id)])
        return ActionResult("click", resp.confirmed, resp.mutation_dispatched, resp.effect,
                            {"element": element.to_policy().summary()})

    def type_text(self, app: str, text: str, *, element_id: str | None = None, label: str | None = None,
                  clear: bool = False, focus_click: bool = True, focus_app: bool = False,
                  foreground: bool = False) -> ActionResult:
        """Type into an allowlisted text element of the current snapshot.

        Flow: policy check (surface + element + text) -> policy-checked click to focus the element ->
        fresh inspection -> the SAME element (matched by role/identifier/bounds, never by per-snapshot id)
        is re-checked -> ``peekaboo type --text=<escaped> --snapshot=<fresh>``. The audit trail records only
        the text length. A dispatched-but-unconfirmed result is returned (never retried).

        ``focus_app`` first brings the (allowlisted) app to the foreground. ``foreground`` uses Peekaboo's
        foreground delivery (it focuses the snapshot's window, then types): Peekaboo 4.9.7 refuses background
        keystrokes for windows whose own keyboard focus it cannot pin (observed with Notes: ``INVALID_INPUT``).
        Foreground delivery is only used for native allowlisted apps, never for browsers (focus could move to
        the address bar), and the result is always confirmed by a fresh observation, never assumed."""
        rule = self._rule(app, Action.TYPE)
        element = self.resolve_element(app, element_id=element_id, label=label)
        snap0 = self._snapshots.get(rule.key)
        surface0 = self._snapshot_surface(rule, snap0) if snap0 else self.surface_for(rule, "")
        self.policy.enforce(self.policy.check_text(surface0, element.to_policy() if element else None, text, rule))
        assert element is not None
        self._page_content_check(rule, snap0, element, Action.TYPE)
        target = element
        if focus_app:
            self.focus(app)
        if focus_click:
            self.click(app, element_id=element.id)
            snap = self.inspect(app, window_title=snap0.requested_window_title if snap0 else None)
            refound = snap.find_same(element)
            if refound is None:
                raise PolicyDenied(self.policy.record(
                    False, Action.TYPE, Reason.ELEMENT_UNKNOWN,
                    "target element could not be re-identified after focusing; not typing",
                    {"app": rule.display_name}))
            target = refound
        fresh = self._current_snapshot(rule, Action.TYPE)
        self.policy.enforce(self.policy.check_text(self._snapshot_surface(rule, fresh), target.to_policy(), text, rule))
        self._page_content_check(rule, fresh, target, Action.TYPE)
        assert fresh.snapshot_id is not None
        try:
            if foreground and rule.is_browser:
                raise PolicyDenied(self.policy.record(False, Action.TYPE, Reason.CAPABILITY_NOT_ALLOWED,
                                                      "foreground typing is not allowed for browsers",
                                                      {"app": rule.display_name}))
            flags = (["clear"] if clear else []) + (["foreground"] if foreground else [])
            resp = self._dispatch(rule, fresh, ("type",),
                                  options=[("text", escape_type_text(text)), ("snapshot", fresh.snapshot_id)],
                                  flags=flags)
        except PeekabooError as exc:
            if exc.mutation_dispatched:
                # Dispatched but not confirmed: report, never retry blindly; the caller re-observes.
                return ActionResult("type", False, True, None, {"text_len": len(text), "error_code": exc.code})
            raise
        return ActionResult("type", resp.confirmed, resp.mutation_dispatched, resp.effect, {"text_len": len(text)})

    def press(self, app: str, keys: Sequence[str]) -> ActionResult:
        rule = self._rule(app, Action.PRESS)
        snap = self._current_snapshot(rule, Action.PRESS)
        self.policy.enforce(self.policy.check_keys(rule, keys, self._snapshot_surface(rule, snap)))
        assert snap.snapshot_id is not None
        resp = self._dispatch(rule, snap, ("press",), options=[("snapshot", snap.snapshot_id)],
                              positionals=[k.strip().lower() for k in keys])
        return ActionResult("press", resp.confirmed, resp.mutation_dispatched, resp.effect, {"keys": list(keys)})

    # -- helpers ------------------------------------------------------------------------------
    def _artifact_path(self, stem: str) -> Path:
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        safe = re.sub(r"[^A-Za-z0-9_.\-]", "-", stem)[:60]
        return self.policy.artifacts_dir / f"{safe}-{ts}.png"

    def _discard(self, path: Path | None) -> None:
        """Remove a capture WE just wrote under artifacts/ when the window turned out to be refused."""
        if path is not None and path.resolve().parent == self.policy.artifacts_dir:
            path.unlink(missing_ok=True)
