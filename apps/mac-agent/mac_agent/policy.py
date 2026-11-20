"""Allowlist / denylist policy for the GUI agent (spec §9, invariants 2, 3 and 7).

Design rules
------------
* **Default deny.** An app, URL, request, element or key chord is permitted only when an explicit
  allow rule matches *and* no deny rule matches. Unknown or unidentifiable state is denied.
* **Never a broker route.** Brokerage apps/sites, live-order controls and anything indicating LIVE
  mode are denied on every surface, fixtures included. Screen control is never an order path.
* **No secrets.** Password managers, Keychain, System Settings and messaging apps are denied;
  secret-looking text is never typed; terminals are never captured; sensitive windows are refused.
* **Audited.** Every decision is appended to ``apps/mac-agent/artifacts/audit.jsonl`` (see
  :mod:`mac_agent.audit`) without typed text, secrets or pixels.

All patterns live in ``rules.json`` next to this module; ``browser/src/rules.mjs`` loads the same
file so the Node/Playwright guard enforces the identical rule set (parity is tested against
``browser/test/policy_vectors.json`` from both languages).
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from functools import cached_property
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from .audit import AuditLog, NullAuditLog

RULES_PATH = Path(__file__).with_name("rules.json")
MAC_AGENT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = MAC_AGENT_ROOT.parents[1]
DEFAULT_FIXTURES_DIR = MAC_AGENT_ROOT / "fixtures"
DEFAULT_ARTIFACTS_DIR = MAC_AGENT_ROOT / "artifacts"

# Zero-width / bidi / BOM characters used to obfuscate labels ("P​lace order").
_FORMAT_CHARS = re.compile("[\u00ad\u200b-\u200f\u202a-\u202e\u2060-\u2064\u2066-\u2069\ufeff]")
_ARTIFACT_NAME = re.compile(r"[A-Za-z0-9_.\-]{1,96}\.(png|jpg)")


class Action(StrEnum):
    LIST = "list"
    INSPECT = "inspect"
    SCREENSHOT = "screenshot"
    CLICK = "click"
    TYPE = "type"
    PRESS = "press"
    LAUNCH = "launch"
    FOCUS = "focus"
    NAVIGATE = "navigate"
    REQUEST = "request"


class SurfaceKind(StrEnum):
    NATIVE_APP = "NATIVE_APP"
    BROWSER_FIXTURE = "BROWSER_FIXTURE"
    BROWSER_DASHBOARD = "BROWSER_DASHBOARD"
    UNKNOWN = "UNKNOWN"


class Reason(StrEnum):
    ALLOWED = "ALLOWED"
    APP_DENYLISTED = "APP_DENYLISTED"
    APP_NOT_ALLOWLISTED = "APP_NOT_ALLOWLISTED"
    CAPABILITY_NOT_ALLOWED = "CAPABILITY_NOT_ALLOWED"
    BROKERAGE_DENIED = "BROKERAGE_DENIED"
    URL_NOT_ALLOWLISTED = "URL_NOT_ALLOWLISTED"
    URL_MALFORMED = "URL_MALFORMED"
    LOOPBACK_PATH_DENIED = "LOOPBACK_PATH_DENIED"
    METHOD_NOT_ALLOWED = "METHOD_NOT_ALLOWED"
    FILE_OUTSIDE_FIXTURES = "FILE_OUTSIDE_FIXTURES"
    SURFACE_UNIDENTIFIED = "SURFACE_UNIDENTIFIED"
    LIVE_SURFACE = "LIVE_SURFACE"
    LIVE_ORDER_ELEMENT = "LIVE_ORDER_ELEMENT"
    ELEMENT_UNKNOWN = "ELEMENT_UNKNOWN"
    SENSITIVE_TARGET = "SENSITIVE_TARGET"
    SENSITIVE_WINDOW = "SENSITIVE_WINDOW"
    SECRET_TEXT = "SECRET_TEXT"  # noqa: S105 - policy denial code, not a credential
    TEXT_INVALID = "TEXT_INVALID"
    KEY_NOT_ALLOWED = "KEY_NOT_ALLOWED"
    PATH_OUTSIDE_ARTIFACTS = "PATH_OUTSIDE_ARTIFACTS"
    OUTSIDE_PAGE_CONTENT = "OUTSIDE_PAGE_CONTENT"


@dataclass(frozen=True)
class Decision:
    allowed: bool
    action: Action
    reason: Reason
    detail: str
    target: dict[str, Any] = field(default_factory=dict)
    rule: str | None = None

    def as_record(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "action": self.action.value,
            "reason": self.reason.value,
            "detail": self.detail,
            "rule": self.rule,
            "target": self.target,
        }


class PolicyDenied(PermissionError):
    """Raised by :meth:`GuiPolicy.enforce` for a denied decision. Carries the decision."""

    def __init__(self, decision: Decision) -> None:
        self.decision = decision
        super().__init__(f"{decision.action.value} denied [{decision.reason.value}]: {decision.detail}")


@dataclass(frozen=True)
class ElementInfo:
    """The policy-relevant view of one UI element (accessibility node or DOM element)."""

    role: str = ""
    label: str = ""
    title: str = ""
    description: str = ""
    identifier: str = ""
    help: str = ""
    role_description: str = ""
    value: str = ""
    placeholder: str = ""
    ax_role: str = ""
    subrole: str = ""

    @property
    def is_text_input(self) -> bool:
        return _is_text_input_role(self.role) or _is_text_input_role(self.ax_role)

    @property
    def is_secure(self) -> bool:
        roles = (self.role, self.ax_role, self.subrole, self.role_description)
        return any("secure" in r.lower() for r in roles if r)

    def action_texts(self) -> list[str]:
        """Texts describing what activating the element does (not user content)."""
        texts = [self.label, self.title, self.description, self.identifier, self.help, self.placeholder,
                 self.role_description]
        if not self.is_text_input:
            # Buttons occasionally carry their caption in AXValue; text fields carry user content there.
            texts.append(self.value)
        return [t for t in texts if t]

    def summary(self) -> dict[str, str]:
        return {
            "role": (self.role or self.ax_role)[:64],
            "label": (self.label or self.title or self.description or self.identifier)[:120],
        }


@dataclass(frozen=True)
class Surface:
    """Where an action would land: an app window, or a browser page."""

    kind: SurfaceKind
    app_name: str = ""
    bundle_id: str = ""
    window_title: str = ""
    url: str = ""
    texts: tuple[str, ...] = ()
    trading_mode_attr: str = ""

    def summary(self) -> dict[str, str]:
        out = {"kind": self.kind.value}
        if self.app_name:
            out["app"] = self.app_name
        if self.bundle_id:
            out["bundle_id"] = self.bundle_id
        if self.window_title:
            out["window_title"] = self.window_title[:160]
        if self.url:
            out["url"] = strip_query(self.url)
        return out


@dataclass(frozen=True)
class AppRule:
    key: str
    names: tuple[str, ...]
    bundle_ids: tuple[str, ...]
    capabilities: frozenset[str]
    keys: frozenset[str]
    type_charset: re.Pattern[str] | None
    is_browser: bool
    redact_values: bool = False
    screenshot_precheck: bool = False

    @property
    def canonical_target(self) -> str:
        """What we pass to ``peekaboo --app``: the first bundle id (exact, never user input)."""
        return self.bundle_ids[0]

    @property
    def display_name(self) -> str:
        return self.names[0]


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _FORMAT_CHARS.sub("", text)
    return " ".join(text.split())


def _is_text_input_role(role: str) -> bool:
    r = role.lower().replace(" ", "").replace("_", "")
    return any(k in r for k in ("textfield", "textarea", "textbox", "searchfield", "combobox", "searchbox",
                                "textinput", "editable"))


def strip_query(url: str) -> str:
    """URL without query/fragment (those can carry tokens) — the only URL form ever audited."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return "<malformed>"
    if parts.scheme == "file":
        return f"file://{parts.path}"
    host = parts.hostname or ""
    try:
        port = f":{parts.port}" if parts.port else ""
    except ValueError:
        port = ""
    if parts.scheme in ("http", "https"):
        return f"{parts.scheme}://{host}{port}{parts.path}"
    return f"{parts.scheme}:<opaque>"


def _compile(patterns: Iterable[str]) -> tuple[re.Pattern[str], ...]:
    # re.ASCII keeps \b / \w semantics identical to JavaScript RegExp without the 'u' flag.
    return tuple(re.compile(p, re.IGNORECASE | re.ASCII) for p in patterns)


def _first_match(patterns: Sequence[re.Pattern[str]], texts: Iterable[str]) -> tuple[str, str] | None:
    for text in texts:
        norm = normalize_text(text)
        if not norm:
            continue
        for pat in patterns:
            if pat.search(norm):
                return pat.pattern, norm
    return None


class Rules:
    """Parsed ``rules.json``."""

    def __init__(self, raw: dict[str, Any]) -> None:
        self.raw = raw
        self.version: str = raw["version"]
        self.apps: tuple[AppRule, ...] = tuple(
            self._app_rule(a, is_browser=False) for a in raw["native_apps"]
        ) + tuple(self._app_rule(a, is_browser=True) for a in raw["browser_apps"])
        self.allowed_surface_titles = _compile(raw["allowed_surface_titles"])
        self.fixture_surface_titles = _compile(raw["fixture_surface_titles"])
        self.deny_app = _compile(raw["deny_app_patterns"])
        self.deny_host = _compile(raw["deny_host_patterns"])
        self.loopback_hosts = frozenset(h.lower() for h in raw["loopback_hosts"])
        self.loopback_allowed_methods = frozenset(m.upper() for m in raw["loopback_allowed_methods"])
        self.loopback_path_deny = _compile(raw["loopback_path_deny_patterns"])
        self.fixture_dir_rel: str = raw["fixture_dir"]
        self.element_deny = _compile(raw["element_deny_patterns"])
        self.type_target_deny = _compile(raw["type_target_deny_patterns"])
        self.secure_roles = frozenset(r.lower() for r in raw["secure_roles"])
        self.live_surface = _compile(raw["live_surface_patterns"])
        self.sensitive_title = _compile(raw["sensitive_title_patterns"])
        self.secret_text = _compile(raw["secret_text_patterns"])
        self.max_text_length: int = int(raw["max_text_length"])

    @staticmethod
    def _app_rule(a: dict[str, Any], *, is_browser: bool) -> AppRule:
        charset = a.get("type_charset")
        return AppRule(
            key=a["key"],
            names=tuple(a["names"]),
            bundle_ids=tuple(a["bundle_ids"]),
            capabilities=frozenset(a["capabilities"]),
            keys=frozenset(k.lower() for k in a.get("keys", [])),
            type_charset=re.compile(charset) if charset else None,
            is_browser=is_browser,
            redact_values=bool(a.get("redact_values", False)),
            screenshot_precheck=bool(a.get("screenshot_precheck", False)),
        )

    @classmethod
    def load(cls, path: Path = RULES_PATH) -> Rules:
        return cls(json.loads(path.read_text(encoding="utf-8")))


class GuiPolicy:
    """Every GUI/browser action is checked here before any process is invoked."""

    def __init__(
        self,
        rules: Rules | None = None,
        *,
        audit: AuditLog | NullAuditLog | None = None,
        fixtures_dir: Path = DEFAULT_FIXTURES_DIR,
        artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR,
        allowed_loopback_ports: Iterable[int] | None = None,
        actor: str = "mac-agent",
    ) -> None:
        self.rules = rules or Rules.load()
        self.audit = audit if audit is not None else NullAuditLog()
        self.fixtures_dir = fixtures_dir.resolve()
        self.artifacts_dir = artifacts_dir.resolve()
        self._ports: set[int] | None = set(allowed_loopback_ports) if allowed_loopback_ports is not None else None
        self._fixture_ports: set[int] = set()
        self.actor = actor

    @property
    def version(self) -> str:
        return self.rules.version

    def allow_loopback_port(self, port: int) -> None:
        """Register a loopback port (e.g. the dashboard). No-op when all ports are allowed."""
        if self._ports is not None:
            self._ports.add(port)

    def register_fixture_port(self, port: int) -> None:
        """Register the port of a local fixture server we started (serves ``fixtures/`` only)."""
        self._fixture_ports.add(port)
        self.allow_loopback_port(port)

    # ------------------------------------------------------------------ helpers
    def _decide(
        self,
        allowed: bool,
        action: Action,
        reason: Reason,
        detail: str,
        target: dict[str, Any],
        rule: str | None = None,
    ) -> Decision:
        decision = Decision(allowed=allowed, action=action, reason=reason, detail=detail, target=target, rule=rule)
        self.audit.record_decision(decision, actor=self.actor, policy_version=self.version)
        return decision

    @staticmethod
    def enforce(decision: Decision) -> Decision:
        if not decision.allowed:
            raise PolicyDenied(decision)
        return decision

    def record(self, allowed: bool, action: Action, reason: Reason, detail: str,
               target: dict[str, Any] | None = None, rule: str | None = None) -> Decision:
        """Record an explicit decision taken by a caller (e.g. 'list running apps', 'no fresh snapshot')."""
        return self._decide(allowed, action, reason, detail, dict(target or {}), rule)

    def title_is_sensitive(self, title: str) -> bool:
        """Pure check: does a window title look like it shows secrets (.env, keychain, tokens...)?"""
        return _first_match(self.rules.sensitive_title, [title]) is not None

    @cached_property
    def _all_names(self) -> dict[str, AppRule]:
        out: dict[str, AppRule] = {}
        for rule in self.rules.apps:
            for n in (*rule.names, *rule.bundle_ids):
                out[n.lower()] = rule
        return out

    def resolve_app(self, app: str) -> AppRule | None:
        """Exact (case-insensitive) name or bundle-id match. No fuzzy matching."""
        return self._all_names.get(normalize_text(app).lower())

    def app_denylisted(self, *names: str) -> str | None:
        hit = _first_match(self.rules.deny_app, [n for n in names if n])
        return hit[0] if hit else None

    def is_allowlisted_app(self, name: str, bundle_id: str = "") -> bool:
        """Pure check (no audit) used to filter app/window listings."""
        if self.app_denylisted(name, bundle_id) is not None:
            return False
        rule = self.resolve_app(bundle_id) if bundle_id else None
        return rule is not None or (not bundle_id and self.resolve_app(name) is not None)

    # ------------------------------------------------------------------ apps
    def check_app(self, app: str, action: Action, *, bundle_id: str = "") -> Decision:
        target = {"app": app[:120], **({"bundle_id": bundle_id} if bundle_id else {})}
        denied_by = self.app_denylisted(app, bundle_id)
        if denied_by is not None:
            return self._decide(False, action, Reason.APP_DENYLISTED, "application is on the GUI denylist",
                                target, denied_by)
        rule = self.resolve_app(app) or (self.resolve_app(bundle_id) if bundle_id else None)
        if rule is None:
            return self._decide(False, action, Reason.APP_NOT_ALLOWLISTED,
                                "application is not on the GUI allowlist", target)
        if bundle_id and bundle_id.lower() not in {b.lower() for b in rule.bundle_ids}:
            return self._decide(False, action, Reason.APP_NOT_ALLOWLISTED,
                                "bundle id does not match the allowlisted application", target)
        if action.value not in rule.capabilities:
            return self._decide(False, action, Reason.CAPABILITY_NOT_ALLOWED,
                                f"{action.value} is not permitted for {rule.display_name}", target, rule.key)
        return self._decide(True, action, Reason.ALLOWED, f"{rule.display_name} allowlisted for {action.value}",
                            target, rule.key)

    # ------------------------------------------------------------------ URLs
    def classify_url(self, url: str) -> tuple[bool, Reason, str, str | None]:
        """Pure URL classification (no audit). Returns (allowed, reason, detail, rule)."""
        raw = url.strip()
        if raw == "about:blank":
            return True, Reason.ALLOWED, "blank page", None
        if any(ord(c) < 0x20 for c in raw):
            return False, Reason.URL_MALFORMED, "control characters in URL", None
        try:
            parts = urlsplit(raw)
        except ValueError:
            return False, Reason.URL_MALFORMED, "URL could not be parsed", None
        scheme = parts.scheme.lower()
        if scheme in ("http", "https", "ws", "wss"):
            if "\\" in parts.netloc:
                return False, Reason.URL_MALFORMED, "backslash in URL authority", None
            if "@" in parts.netloc:
                return False, Reason.URL_MALFORMED, "credentials in URL are refused", None
            try:
                host = (parts.hostname or "").lower()
            except ValueError:
                return False, Reason.URL_MALFORMED, "invalid host", None
            if not host:
                return False, Reason.URL_MALFORMED, "URL has no host", None
            hit = _first_match(self.rules.deny_host, [host])
            if hit is not None:
                return False, Reason.BROKERAGE_DENIED, f"host {host} matches the brokerage/finance denylist", hit[0]
            if host not in self.rules.loopback_hosts:
                return False, Reason.URL_NOT_ALLOWLISTED, f"host {host} is not loopback", None
            try:
                port = parts.port or (443 if scheme in ("https", "wss") else 80)
            except ValueError:
                return False, Reason.URL_MALFORMED, "invalid port", None
            if self._ports is not None and port not in self._ports:
                return False, Reason.URL_NOT_ALLOWLISTED, f"loopback port {port} is not allowlisted", None
            path = unquote(parts.path or "/")
            phit = _first_match(self.rules.loopback_path_deny, [path])
            if phit is not None:
                return False, Reason.LOOPBACK_PATH_DENIED, "operator/engine/control paths are denied", phit[0]
            if port in self._fixture_ports and "/.." in path:
                return False, Reason.URL_MALFORMED, "path traversal", None
            return True, Reason.ALLOWED, "loopback dashboard/fixture origin", None
        if scheme == "file":
            if parts.netloc not in ("", "localhost"):
                return False, Reason.FILE_OUTSIDE_FIXTURES, "remote file URL", None
            if self.file_url_in_fixtures(raw):
                return True, Reason.ALLOWED, "file fixture", None
            return False, Reason.FILE_OUTSIDE_FIXTURES, "file:// is limited to apps/mac-agent/fixtures", None
        return False, Reason.URL_NOT_ALLOWLISTED, f"scheme {scheme or '<none>'} is not allowlisted", None

    def file_url_in_fixtures(self, url: str) -> bool:
        try:
            parts = urlsplit(url)
            if parts.scheme != "file":
                return False
            resolved = Path(unquote(parts.path)).resolve(strict=False)
        except (ValueError, OSError, RuntimeError):
            return False
        return self.fixtures_dir in resolved.parents

    def check_url(self, url: str, action: Action = Action.NAVIGATE) -> Decision:
        allowed, reason, detail, rule = self.classify_url(url)
        return self._decide(allowed, action, reason, detail, {"url": strip_query(url)}, rule)

    def check_request(self, url: str, method: str = "GET", resource_type: str = "") -> Decision:
        """Browser sub-request / navigation check (route handler). Loopback writes are refused."""
        allowed, reason, detail, rule = self.classify_url(url)
        target = {"url": strip_query(url), "method": method.upper()[:12], "resource_type": resource_type[:24]}
        if allowed and urlsplit(url.strip()).scheme.lower() in ("http", "https", "ws", "wss") \
                and method.upper() not in self.rules.loopback_allowed_methods:
            return self._decide(False, Action.REQUEST, Reason.METHOD_NOT_ALLOWED,
                                f"{method.upper()} requests are refused for the GUI agent (read-only)", target)
        return self._decide(allowed, Action.REQUEST, reason, detail, target, rule)

    def is_fixture_url(self, url: str) -> bool:
        if self.file_url_in_fixtures(url):
            return True
        try:
            parts = urlsplit(url)
            return (parts.scheme in ("http", "https") and (parts.hostname or "") in self.rules.loopback_hosts
                    and parts.port in self._fixture_ports)
        except ValueError:
            return False

    # ------------------------------------------------------------------ surfaces
    def surface_is_live(self, surface: Surface) -> str | None:
        if surface.trading_mode_attr and normalize_text(surface.trading_mode_attr).lower() not in (
                "paper", "simulated", "paper_simulated", "replay", "research_only"):
            return f"trading-mode attribute={surface.trading_mode_attr[:40]!r}"
        hit = _first_match(self.rules.live_surface, [surface.window_title, *surface.texts])
        return hit[0] if hit else None

    def classify_browser_title(self, title: str) -> SurfaceKind:
        norm = normalize_text(title)
        if any(p.search(norm) for p in self.rules.fixture_surface_titles):
            return SurfaceKind.BROWSER_FIXTURE
        if any(p.search(norm) for p in self.rules.allowed_surface_titles):
            return SurfaceKind.BROWSER_DASHBOARD
        return SurfaceKind.UNKNOWN

    def browser_surface(self, *, url: str, title: str, texts: Sequence[str] = (), trading_mode_attr: str = "",
                        app_name: str = "", bundle_id: str = "") -> Surface:
        """Classify a browser page. A page is identified only if both its URL and its title are allowlisted;
        a fixture-looking title on a non-fixture origin is not trusted as a fixture."""
        kind = self.classify_browser_title(title)
        ok, *_ = self.classify_url(url)
        if not ok:
            kind = SurfaceKind.UNKNOWN
        elif kind == SurfaceKind.BROWSER_FIXTURE and not self.is_fixture_url(url):
            kind = SurfaceKind.BROWSER_DASHBOARD if any(
                p.search(normalize_text(title)) for p in self.rules.allowed_surface_titles) else SurfaceKind.UNKNOWN
        return Surface(kind=kind, app_name=app_name, bundle_id=bundle_id, window_title=title, url=url,
                       texts=tuple(texts), trading_mode_attr=trading_mode_attr)

    def check_surface(self, surface: Surface, action: Action) -> Decision:
        """App + window/page level check. Interaction on a LIVE-indicating surface is always denied."""
        target = surface.summary()
        if surface.kind == SurfaceKind.UNKNOWN:
            return self._decide(False, action, Reason.SURFACE_UNIDENTIFIED,
                                "surface could not be identified as an allowlisted app/page", target)
        if surface.app_name or surface.bundle_id:
            denied_by = self.app_denylisted(surface.app_name, surface.bundle_id)
            if denied_by is not None:
                return self._decide(False, action, Reason.APP_DENYLISTED, "application is on the GUI denylist",
                                    target, denied_by)
            if not self.is_allowlisted_app(surface.app_name, surface.bundle_id):
                return self._decide(False, action, Reason.APP_NOT_ALLOWLISTED,
                                    "surface belongs to an application that is not allowlisted", target)
        elif surface.kind == SurfaceKind.NATIVE_APP:
            return self._decide(False, action, Reason.SURFACE_UNIDENTIFIED, "native surface without an app identity",
                                target)
        if surface.url:
            ok, reason, detail, rule = self.classify_url(surface.url)
            if not ok:
                return self._decide(False, action, reason, detail, target, rule)
        live = self.surface_is_live(surface)
        if live is not None and action != Action.LIST:
            # Even inspection of a LIVE surface is refused: the agent has no business there.
            return self._decide(False, action, Reason.LIVE_SURFACE, "surface indicates LIVE trading mode",
                                target, live)
        if surface.window_title and action in (Action.SCREENSHOT, Action.INSPECT, Action.CLICK, Action.TYPE,
                                               Action.PRESS):
            hit = _first_match(self.rules.sensitive_title, [surface.window_title])
            if hit is not None:
                return self._decide(False, action, Reason.SENSITIVE_WINDOW, "window title looks sensitive",
                                    target, hit[0])
        return self._decide(True, action, Reason.ALLOWED, "surface allowlisted", target)

    # ------------------------------------------------------------------ elements
    def element_denial(self, element: ElementInfo, action: Action) -> tuple[Reason, str] | None:
        """Pure element check (no audit); shared by click and type."""
        hit = _first_match(self.rules.element_deny, element.action_texts())
        if hit is not None:
            return Reason.LIVE_ORDER_ELEMENT, hit[0]
        if action == Action.TYPE:
            roles = {r.lower() for r in (element.role, element.ax_role, element.subrole) if r}
            if element.is_secure or roles & self.rules.secure_roles:
                return Reason.SENSITIVE_TARGET, "secure text field"
            thit = _first_match(self.rules.type_target_deny, element.action_texts())
            if thit is not None:
                return Reason.SENSITIVE_TARGET, thit[0]
        return None

    def check_element(self, surface: Surface, element: ElementInfo | None, action: Action) -> Decision:
        surf = self.check_surface(surface, action)
        if not surf.allowed:
            return surf
        target = {**surface.summary(), "element": element.summary() if element else None}
        if element is None:
            return self._decide(False, action, Reason.ELEMENT_UNKNOWN,
                                "element is not in the current inspected snapshot", target)
        denial = self.element_denial(element, action)
        if denial is not None:
            reason, rule = denial
            detail = ("element matches a live-order / destructive-action pattern"
                      if reason == Reason.LIVE_ORDER_ELEMENT else "element is a sensitive input")
            return self._decide(False, action, reason, detail, target, rule)
        return self._decide(True, action, Reason.ALLOWED, "element allowed", target)

    # ------------------------------------------------------------------ text / keys / paths
    def text_problem(self, text: str, app_rule: AppRule | None = None) -> tuple[Reason, str] | None:
        if len(text) > self.rules.max_text_length:
            return Reason.TEXT_INVALID, f"text longer than {self.rules.max_text_length} chars"
        for ch in text:
            if ch in ("\n", "\t"):
                continue
            if unicodedata.category(ch).startswith("C"):
                return Reason.TEXT_INVALID, f"control/format character U+{ord(ch):04X} refused"
        hit = _first_match(self.rules.secret_text, [text])
        if hit is not None:
            return Reason.SECRET_TEXT, "text looks like a credential/secret and is never typed"
        if app_rule is not None and app_rule.type_charset is not None and not app_rule.type_charset.fullmatch(text):
            return Reason.TEXT_INVALID, f"text not permitted for {app_rule.display_name}"
        return None

    def check_text(self, surface: Surface, element: ElementInfo | None, text: str,
                   app_rule: AppRule | None = None) -> Decision:
        """Element + text check for typing. The audit record stores only the text length."""
        decision = self.check_element(surface, element, Action.TYPE)
        if not decision.allowed:
            return decision
        problem = self.text_problem(text, app_rule)
        target = {**surface.summary(), "text_len": len(text)}
        if problem is not None:
            return self._decide(False, Action.TYPE, problem[0], problem[1], target)
        return self._decide(True, Action.TYPE, Reason.ALLOWED, "text allowed", target)

    def check_keys(self, app_rule: AppRule, keys: Sequence[str], surface: Surface) -> Decision:
        surf = self.check_surface(surface, Action.PRESS)
        if not surf.allowed:
            return surf
        target = {**surface.summary(), "keys": [k[:24] for k in keys]}
        if not keys:
            return self._decide(False, Action.PRESS, Reason.KEY_NOT_ALLOWED, "no keys", target)
        for k in keys:
            if k.strip().lower() not in app_rule.keys:
                return self._decide(False, Action.PRESS, Reason.KEY_NOT_ALLOWED,
                                    f"key {k[:24]!r} is not allowlisted for {app_rule.display_name}", target)
        return self._decide(True, Action.PRESS, Reason.ALLOWED, "keys allowed", target)

    def check_artifact_path(self, path: Path, action: Action = Action.SCREENSHOT) -> Decision:
        resolved = path.resolve(strict=False)
        target = {"path": str(resolved)}
        if resolved.parent != self.artifacts_dir:
            return self._decide(False, action, Reason.PATH_OUTSIDE_ARTIFACTS,
                                "output files may only be written directly under apps/mac-agent/artifacts", target)
        if not _ARTIFACT_NAME.fullmatch(resolved.name):
            return self._decide(False, action, Reason.PATH_OUTSIDE_ARTIFACTS, "bad artifact file name", target)
        return self._decide(True, action, Reason.ALLOWED, "artifact path allowed", target)
