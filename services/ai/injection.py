# ruff: noqa: E501  (detection regexes are clearer unwrapped)
"""Prompt-injection defense for externally retrieved text (news, filings, web pages, UI text).

Pipeline for untrusted text (``sanitize_text``):
1. Unicode NFKC normalization; zero-width / bidi / "tag" characters (ASCII smuggling) are removed and
   the smuggled tag text is decoded so it can be scanned.
2. HTML is parsed (stdlib ``html.parser``): ``<script>/<style>/<template>`` bodies, comments and
   elements hidden by CSS/attributes are removed from the visible excerpt but still scanned.
3. Control characters are stripped, whitespace collapsed, credential-like strings redacted, long
   base64 blobs removed (and decoded for scanning), ``<``/``>`` escaped so the text can never forge
   the prompt's evidence delimiters, and the result truncated.
4. Pattern scan: instruction overrides, role/system impersonation, tool/function-call syntax,
   requests to place orders / change limits / reveal secrets, credential-like strings, encoded
   payloads and hidden text. Each finding has a severity; ``HIGH`` evidence is excluded from model
   context and makes the AI pipeline return ``NO_TRADE``.

Flags are stored on ``Evidence.injection_flags`` as ``"<SEVERITY>:<CODE>"`` strings.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import html
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from html.parser import HTMLParser

from packages.contracts import Evidence
from packages.contracts.enums import EvidenceSourceType, TrustLevel
from packages.core.ids import new_id


class Severity(IntEnum):
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3


@dataclass(frozen=True)
class InjectionFinding:
    code: str
    severity: Severity
    detail: str = ""

    @property
    def flag(self) -> str:
        return f"{self.severity.name}:{self.code}"


@dataclass(frozen=True)
class SanitizedText:
    text: str
    findings: tuple[InjectionFinding, ...]
    truncated: bool
    raw_sha256: str

    @property
    def max_severity(self) -> Severity:
        return max((f.severity for f in self.findings), default=Severity.NONE)

    @property
    def flags(self) -> tuple[str, ...]:
        return flags_from_findings(self.findings)


# ------------------------------------------------------------------------------------------------
# character classes
# ------------------------------------------------------------------------------------------------

_ZERO_WIDTH = {
    0x00AD,  # soft hyphen
    0x034F,  # combining grapheme joiner
    0x061C,  # arabic letter mark
    0x115F, 0x1160, 0x17B4, 0x17B5, 0x180E,
    0x200B, 0x200C, 0x200D, 0x200E, 0x200F,  # zero width space/joiners, LRM/RLM
    0x202A, 0x202B, 0x202C, 0x202D, 0x202E,  # bidi embeddings/overrides
    0x2060, 0x2061, 0x2062, 0x2063, 0x2064,  # word joiner, invisible operators
    0x2066, 0x2067, 0x2068, 0x2069,  # bidi isolates
    0x3164, 0xFEFF, 0xFFA0,
}
_TAG_START, _TAG_END = 0xE0000, 0xE007F  # Unicode "tag" block: invisible ASCII copies (ASCII smuggling)
_VARIATION_SELECTORS = set(range(0xFE00, 0xFE10)) | set(range(0xE0100, 0xE01F0))

_HIDDEN_STYLE_RE = re.compile(
    r"display\s*:\s*none|visibility\s*:\s*hidden|font-size\s*:\s*0(?![.\d]*[1-9])|opacity\s*:\s*0(?![.\d]*[1-9])"
    r"|color\s*:\s*(?:#fff(?:fff)?\b|white\b|transparent\b)|(?:height|width)\s*:\s*0(?:px)?\s*(?:;|$)"
    r"|position\s*:\s*absolute[^\"']*-\d{3,}px",
    re.IGNORECASE,
)
_SKIP_TAGS = frozenset({"script", "style", "template", "noscript", "iframe", "object", "embed", "svg", "math"})
_VOID_TAGS = frozenset({"area", "base", "br", "col", "hr", "img", "input", "link", "meta", "source", "track", "wbr"})
_BLOCK_TAGS = frozenset(
    {"p", "div", "br", "li", "ul", "ol", "tr", "td", "th", "table", "h1", "h2", "h3", "h4", "h5", "h6",
     "section", "article", "header", "footer", "blockquote", "pre"}
)
_LOOKS_LIKE_HTML_RE = re.compile(r"<\s*(?:[a-zA-Z][a-zA-Z0-9:-]*\b[^>]*|/\s*[a-zA-Z][a-zA-Z0-9:-]*\s*|!--)>?", re.DOTALL)


class _HtmlExtractor(HTMLParser):
    """Splits HTML into visible text, hidden text (CSS/attr-hidden, comments) and script bodies."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.visible: list[str] = []
        self.hidden: list[str] = []
        self.scripts: list[str] = []
        self._stack: list[tuple[str, str]] = []  # (tag, kind) kind in {"", "hidden", "skip"}
        self.tag_count = 0

    def _kind(self) -> str:
        kinds = {k for _, k in self._stack}
        if "skip" in kinds:
            return "skip"
        if "hidden" in kinds:
            return "hidden"
        return ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tag_count += 1
        tag = tag.lower()
        if tag in _BLOCK_TAGS:
            self.visible.append("\n")
        if tag in _VOID_TAGS:
            for name, value in attrs:  # alt/title text of images is a classic hiding spot
                if name in ("alt", "title", "aria-label") and value:
                    self.hidden.append(value)
            return
        kind = ""
        if tag in _SKIP_TAGS:
            kind = "skip"
        else:
            for name, value in attrs:
                lname = name.lower()
                v = value or ""
                hidden_attr = lname == "hidden" or (lname == "aria-hidden" and v.lower() == "true")
                if hidden_attr or (lname == "style" and _HIDDEN_STYLE_RE.search(v)):
                    kind = "hidden"
                elif lname in ("title", "alt", "data-prompt", "data-instructions") and v:
                    self.hidden.append(v)
        self._stack.append((tag, kind))

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tag_count += 1
        for name, value in attrs:
            if name in ("alt", "title", "aria-label") and value:
                self.hidden.append(value)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in _BLOCK_TAGS:
            self.visible.append("\n")
        for i in range(len(self._stack) - 1, -1, -1):
            if self._stack[i][0] == tag:
                del self._stack[i:]
                break

    def handle_data(self, data: str) -> None:
        kind = self._kind()
        if kind == "skip":
            self.scripts.append(data)
        elif kind == "hidden":
            self.hidden.append(data)
        else:
            self.visible.append(data)

    def handle_comment(self, data: str) -> None:
        self.hidden.append(data)


# ------------------------------------------------------------------------------------------------
# detection patterns
# ------------------------------------------------------------------------------------------------

_I = re.IGNORECASE
_ADDRESSEE = r"(?:you|assistant|ai|a\.i\.|model|agent|bot|llm|claude|gpt|sol|system)"
_PATTERNS: tuple[tuple[str, Severity, re.Pattern[str]], ...] = (
    # instruction overrides
    ("INSTRUCTION_OVERRIDE", Severity.HIGH, re.compile(
        r"\b(?:ignore|disregard|forget|override|overrule|bypass|skip|abandon)\b[^.\n]{0,60}?"
        r"\b(?:previous|prior|above|earlier|preceding|all|any|the|your|system|developer|original)\b[^.\n]{0,40}?"
        r"\b(?:instructions?|prompts?|rules?|guidelines?|directions?|directives?|messages?|context|constraints?|policy|policies)\b",
        _I)),
    ("INSTRUCTION_OVERRIDE", Severity.HIGH, re.compile(
        r"\b(?:new|updated|revised|real|actual|hidden|secret)\s+(?:system\s+)?(?:instructions?|directives?|prompt)\s*[:\-]", _I)),
    ("INSTRUCTION_OVERRIDE", Severity.HIGH, re.compile(
        r"\b(?:do\s+not|don't|never)\s+(?:follow|obey|listen\s+to)\b[^.\n]{0,40}\b(?:instructions?|rules?|system|operator|policy)\b", _I)),
    ("INSTRUCTION_OVERRIDE", Severity.HIGH, re.compile(rf"\bfrom\s+now\s+on\b[^.\n]{{0,40}}\b{_ADDRESSEE}\b", _I)),
    # role / system impersonation
    ("ROLE_IMPERSONATION", Severity.HIGH, re.compile(r"(?:^|\n)\s*(?:#+\s*)?(?:system|assistant|developer|operator)\s*(?:prompt\s*)?:", _I)),
    ("ROLE_IMPERSONATION", Severity.HIGH, re.compile(r"<\|?\s*(?:im_start|im_end|system|endoftext|eot_id|start_header_id)\s*\|?>", _I)),
    ("ROLE_IMPERSONATION", Severity.HIGH, re.compile(r"\[/?(?:INST|SYS|SYSTEM)\]|<<\s*/?SYS\s*>>", _I)),
    ("ROLE_IMPERSONATION", Severity.HIGH, re.compile(
        r"</?\s*(?:system|assistant|instructions?|tool_result|function_results?|untrusted_evidence|evidence|orchestrator_question)\b", _I)),
    ("ROLE_IMPERSONATION", Severity.HIGH, re.compile(
        r"&lt;/?\s*(?:system|assistant|instructions?|tool_result|function_results?|untrusted_evidence|evidence|orchestrator_question)\b", _I)),
    ("ROLE_IMPERSONATION", Severity.HIGH, re.compile(rf"\b{_ADDRESSEE}\s+(?:are|is)\s+now\b|\byou\s+must\s+now\s+act\b", _I)),
    ("ROLE_IMPERSONATION", Severity.HIGH, re.compile(
        r"\b(?:act|behave|respond)\s+as\s+(?:an?|the|my)\s+[^.\n]{0,30}\b(?:admin|administrator|developer|system|broker|root|operator)\b", _I)),
    ("ROLE_IMPERSONATION", Severity.HIGH, re.compile(r"\b(?:begin|end)\s+(?:system|developer)\s+(?:prompt|message)\b|\bjailbreak\b|\bDAN\s+mode\b", _I)),
    # tool / function-call syntax
    ("TOOL_CALL_SYNTAX", Severity.HIGH, re.compile(
        r"[\"'](?:name|tool|tool_name|function|recipient_name|recipient)[\"']\s*:\s*[\"'][^\"']{1,80}[\"']\s*,\s*"
        r"[\"'](?:arguments|parameters|input|args|params)[\"']\s*:", _I)),
    ("TOOL_CALL_SYNTAX", Severity.HIGH, re.compile(
        r"\b(?:function_call|tool_use|tool_calls?|tool_choice|functions\.\w+|to=functions)\b", _I)),
    ("TOOL_CALL_SYNTAX", Severity.HIGH, re.compile(
        r"\b(?:submit|place|cancel|modify|replace|create|send|execute)_?(?:live_)?(?:order|trade|limit)s?\s*\(", _I)),
    # requests to trade
    ("ORDER_REQUEST", Severity.HIGH, re.compile(
        rf"\b{_ADDRESSEE}\b[^.\n]{{0,40}}\b(?:must|should|need\s+to|has\s+to|have\s+to|will|shall|are\s+instructed\s+to)\b[^.\n]{{0,40}}"
        r"\b(?:buy|sell|short|purchase|liquidate|place|submit|execute|enter|route)\b", _I)),
    ("ORDER_REQUEST", Severity.HIGH, re.compile(
        r"\b(?:place|submit|execute|enter|send|route)\b[^.\n]{0,25}\b(?:market|limit|buy|sell|stop)?\s*orders?\b[^.\n]{0,40}"
        r"\b(?:now|immediately|right\s+away|asap|at\s+once|without\s+(?:review|approval|checks?))\b", _I)),
    ("ORDER_REQUEST", Severity.HIGH, re.compile(
        r"(?:^|[.!\n]\s*)(?:buy|sell|short|purchase)\s+(?:\d[\d,]*\s+(?:shares?|units?|contracts?)|all|everything|max(?:imum)?)\b", _I)),
    ("ORDER_REQUEST", Severity.MEDIUM, re.compile(
        r"\b(?:place|submit|execute)\s+(?:an?\s+|the\s+)?(?:market|limit|buy|sell)\s+orders?\b", _I)),
    # change limits / disable controls
    ("LIMIT_CHANGE", Severity.HIGH, re.compile(
        r"\b(?:increase|raise|change|modify|set|remove|disable|bypass|override|lift|ignore|turn\s+off|relax|double)\b[^.\n]{0,40}"
        r"\b(?:risk|trading|order|position|loss|exposure)\s+(?:limits?|polic(?:y|ies)|checks?|controls?|caps?|gates?)\b", _I)),
    ("LIMIT_CHANGE", Severity.HIGH, re.compile(
        r"\b(?:max_order_notional|max_position_notional|max_daily_loss|max_planned_loss_per_trade|max_spread_bps|"
        r"max_quote_age_seconds|min_account_equity|require_human_approval|kill[\s_-]?switch|entries[\s_-]?lock|"
        r"release[\s_-]?gate|bounded_live|live_approval)\b", _I)),
    # secret exfiltration
    ("SECRET_EXFILTRATION", Severity.HIGH, re.compile(
        r"\b(?:reveal|print|show|output|display|send|leak|exfiltrate|share|tell\s+me|repeat|disclose|dump|email|post)\b(?:[^.\n]|\.(?=\S)){0,50}?"
        r"(?:\b(?:api[\s_-]?keys?|secret(?:\s+keys?)?s?|credentials?|passwords?|(?:access|auth|bearer|api|session|refresh)[\s_-]?tokens?|system\s+prompt|"
        r"private\s+keys?|env(?:ironment)?\s+variables?|keychain|instructions\s+above)\b|\.env\b)", _I)),
)

_CREDENTIAL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("anthropic_key", re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}")),
    ("openai_key", re.compile(r"\bsk-(?:proj-|svcacct-|admin-)?[A-Za-z0-9_\-]{20,}")),
    ("alpaca_key", re.compile(r"\b(?:PK|AK|CK)[A-Z0-9]{16,}\b")),
    ("aws_key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("github_token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr|github_pat)_[A-Za-z0-9_]{20,}")),
    ("slack_token", re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}")),
    ("private_key", re.compile(r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----")),
    ("bearer_token", re.compile(r"\bBearer\s+[A-Za-z0-9._~+/\-]{20,}=*", re.IGNORECASE)),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}")),
)
_BASE64_RE = re.compile(r"(?<![A-Za-z0-9+/=])[A-Za-z0-9+/]{60,}={0,2}(?![A-Za-z0-9+/=])")
_HEX_BLOB_RE = re.compile(r"\b[0-9a-fA-F]{120,}\b")
_SPACED_LETTERS_RE = re.compile(r"\b(?:[A-Za-z][ .\-_]){5,}[A-Za-z]\b")


def flags_from_findings(findings: tuple[InjectionFinding, ...] | list[InjectionFinding]) -> tuple[str, ...]:
    return tuple(sorted({f.flag for f in findings}, key=lambda s: (-Severity[s.split(":", 1)[0]].value, s)))


def severity_of_flags(flags: tuple[str, ...] | list[str]) -> Severity:
    worst = Severity.NONE
    for flag in flags:
        name = flag.split(":", 1)[0]
        try:
            sev = Severity[name]
        except KeyError:  # unknown flag format: fail closed
            sev = Severity.HIGH
        worst = max(worst, sev)
    return worst


def evidence_severity(evidence: Evidence) -> Severity:
    return severity_of_flags(evidence.injection_flags)


def _squash_spaced_letters(text: str) -> str:
    """'i g n o r e  p r e v i o u s' -> 'ignore previous' so letter-spacing can't dodge the scan."""
    return _SPACED_LETTERS_RE.sub(lambda m: re.sub(r"[ .\-_]", "", m.group(0)), text)


# Letter-spaced text loses its word boundaries once whitespace is collapsed ("i g n o r e p r e v i o u s ..."), so
# spaced runs are also checked as a compact letters-only string against these phrase regexes.
_COMPACT_NEEDLES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("INSTRUCTION_OVERRIDE", re.compile(
        r"(?:ignore|disregard|forget|override|bypass)(?:all|the|any|your)*(?:previous|prior|above|earlier|preceding|system)"
        r"(?:instructions?|prompts?|rules|guidelines|messages)")),
    ("ORDER_REQUEST", re.compile(r"(?:submit|place|execute)(?:a|an|the)?(?:market|limit|live|buy|sell)?orders?")),
    ("SECRET_EXFILTRATION", re.compile(r"(?:reveal|print|show|leak|dump)(?:the|your|all)?(?:api|secret|private)keys?|systemprompt")),
    ("ROLE_IMPERSONATION", re.compile(r"(?:youarenow|actasthe(?:admin|broker|system|developer))")),
)


def _spaced_findings(text: str) -> list[InjectionFinding]:
    out: list[InjectionFinding] = []
    for m in _SPACED_LETTERS_RE.finditer(text):
        compact = re.sub(r"[^a-z]", "", m.group(0).lower())
        if len(compact) >= 12:
            out.append(InjectionFinding("SPACED_LETTERS", Severity.MEDIUM, m.group(0)[:60]))
        for code, pat in _COMPACT_NEEDLES:
            if pat.search(compact):
                out.append(InjectionFinding(code, Severity.HIGH, f"(letter-spaced) {compact[:80]}"))
    return out


def scan_text(text: str) -> list[InjectionFinding]:
    """Pattern scan of (already normalized) text. Pure; used for evidence, model questions and outputs."""
    findings: list[InjectionFinding] = []
    variants = [text]
    squashed = _squash_spaced_letters(text)
    if squashed != text:
        variants.append(squashed)
    seen: set[tuple[str, Severity]] = set()
    for variant in variants:
        for code, sev, pat in _PATTERNS:
            m = pat.search(variant)
            if m and (code, sev) not in seen:
                seen.add((code, sev))
                findings.append(InjectionFinding(code, sev, m.group(0)[:120]))
    for f in _spaced_findings(text):
        if (f.code, f.severity) not in seen:
            seen.add((f.code, f.severity))
            findings.append(f)
    for name, pat in _CREDENTIAL_PATTERNS:
        if pat.search(text) and ("CREDENTIAL_LIKE", Severity.HIGH) not in seen:
            seen.add(("CREDENTIAL_LIKE", Severity.HIGH))
            findings.append(InjectionFinding("CREDENTIAL_LIKE", Severity.HIGH, name))
    return findings


def redact_credentials(text: str) -> tuple[str, bool]:
    changed = False
    for _name, pat in _CREDENTIAL_PATTERNS:
        text, n = pat.subn("[REDACTED-CREDENTIAL]", text)
        changed = changed or n > 0
    return text, changed


def _decode_base64(blob: str) -> str | None:
    padded = blob + "=" * (-len(blob) % 4)
    try:
        raw = base64.b64decode(padded, validate=True)
    except (binascii.Error, ValueError):
        return None
    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    printable = sum(ch.isprintable() or ch in "\n\t" for ch in decoded)
    return decoded if decoded and printable / len(decoded) > 0.9 else None


def _strip_invisible(text: str) -> tuple[str, str, int, int]:
    """Returns (clean_text, smuggled_tag_text, n_zero_width_removed, n_control_removed)."""
    out: list[str] = []
    smuggled: list[str] = []
    zw = ctrl = 0
    for ch in text:
        cp = ord(ch)
        if _TAG_START <= cp <= _TAG_END:
            if 0xE0020 <= cp <= 0xE007E:
                smuggled.append(chr(cp - 0xE0000))
            zw += 1
            continue
        if cp in _ZERO_WIDTH or cp in _VARIATION_SELECTORS:
            zw += 1
            continue
        if ch in ("\n", "\t"):
            out.append(ch)
            continue
        if ch == "\r":
            out.append("\n")
            continue
        cat = unicodedata.category(ch)
        if cat in ("Cc", "Cf", "Cs", "Co", "Cn"):
            ctrl += 1
            continue
        if cat in ("Zl", "Zp"):
            out.append("\n")
            continue
        out.append(ch)
    return "".join(out), "".join(smuggled), zw, ctrl


def _collapse_whitespace(text: str) -> str:
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def sanitize_text(raw: str, *, max_chars: int = 4000) -> SanitizedText:
    """Normalize, de-weaponize and scan untrusted text. Never raises on hostile input."""
    raw_sha = hashlib.sha256(raw.encode("utf-8", errors="surrogatepass")).hexdigest()
    findings: list[InjectionFinding] = []
    scan_extra: list[str] = []

    text = unicodedata.normalize("NFKC", raw)
    text, smuggled, zw, ctrl = _strip_invisible(text)
    if smuggled:
        findings.append(InjectionFinding("HIDDEN_TEXT", Severity.MEDIUM, "unicode tag-character (ASCII smuggling) text"))
        scan_extra.append(smuggled)
    if zw:
        findings.append(InjectionFinding("HIDDEN_CHARS", Severity.MEDIUM, f"{zw} zero-width/bidi/invisible characters removed"))
    if ctrl:
        findings.append(InjectionFinding("CONTROL_CHARS", Severity.LOW, f"{ctrl} control/format characters removed"))

    if _LOOKS_LIKE_HTML_RE.search(text):
        # markup itself can forge prompt delimiters / roles (e.g. "</untrusted_evidence><system>"): scan it raw too,
        # because the parser strips tag names from the visible text
        scan_extra.append(text)
        parser = _HtmlExtractor()
        try:
            parser.feed(text)
            parser.close()
        except Exception:  # pragma: no cover - HTMLParser is lenient; fail closed if it ever is not
            findings.append(InjectionFinding("HTML_PARSE_ERROR", Severity.HIGH, "unparseable markup"))
            parser.visible = [re.sub(r"<[^>]*>", " ", text)]
        if parser.tag_count:
            findings.append(InjectionFinding("HTML_STRIPPED", Severity.LOW, f"{parser.tag_count} tags removed"))
        hidden = "".join(parser.hidden).strip()
        scripts = "".join(parser.scripts).strip()
        if hidden:
            findings.append(InjectionFinding("HIDDEN_TEXT", Severity.MEDIUM, "hidden HTML content/comments removed"))
            scan_extra.append(unicodedata.normalize("NFKC", html.unescape(hidden)))
        if scripts:
            findings.append(InjectionFinding("SCRIPT_CONTENT", Severity.MEDIUM, "script/style/embedded content removed"))
            scan_extra.append(scripts)
        text = "".join(parser.visible)
        # character references were decoded by the parser: invisible characters may have reappeared
        text, smuggled2, zw2, ctrl2 = _strip_invisible(unicodedata.normalize("NFKC", text))
        if smuggled2:
            scan_extra.append(smuggled2)
        if zw2 or ctrl2 or smuggled2:
            findings.append(InjectionFinding("HIDDEN_CHARS", Severity.MEDIUM, "invisible characters inside entities"))
    else:
        text = html.unescape(text)
        text, smuggled2, zw2, ctrl2 = _strip_invisible(unicodedata.normalize("NFKC", text))
        if smuggled2:
            scan_extra.append(smuggled2)
        if zw2 or ctrl2 or smuggled2:
            findings.append(InjectionFinding("HIDDEN_CHARS", Severity.MEDIUM, "invisible characters inside entities"))

    # encoded payloads: remove from the excerpt, decode for scanning
    def _b64(m: re.Match[str]) -> str:
        decoded = _decode_base64(m.group(0))
        if decoded:
            scan_extra.append(unicodedata.normalize("NFKC", decoded))
        return "[base64 blob removed]"

    text, n_b64 = _BASE64_RE.subn(_b64, text)
    if n_b64:
        findings.append(InjectionFinding("ENCODED_BLOB", Severity.MEDIUM, f"{n_b64} base64-like blob(s) removed"))
    text, n_hex = _HEX_BLOB_RE.subn("[hex blob removed]", text)
    if n_hex:
        findings.append(InjectionFinding("ENCODED_BLOB", Severity.LOW, f"{n_hex} hex blob(s) removed"))

    text = _collapse_whitespace(text)

    # scan visible text plus everything hidden/encoded that was removed from it
    findings.extend(scan_text(text))
    for extra in scan_extra:
        extra_clean, _, _, _ = _strip_invisible(unicodedata.normalize("NFKC", extra))
        for f in scan_text(_collapse_whitespace(extra_clean)):
            findings.append(InjectionFinding(f.code, f.severity, f"(hidden/encoded) {f.detail}"))
            if f.severity >= Severity.HIGH:
                findings.append(InjectionFinding("HIDDEN_INSTRUCTION", Severity.HIGH, f.detail))

    text, redacted = redact_credentials(text)
    if redacted:
        findings.append(InjectionFinding("CREDENTIAL_REDACTED", Severity.HIGH, "credential-like string redacted"))

    # neutralize delimiter forging: models see escaped angle brackets only
    text = text.replace("<", "&lt;").replace(">", "&gt;")

    truncated = len(text) > max_chars
    if truncated:
        text = text[: max(0, max_chars - 15)].rstrip() + " …[truncated]"
        findings.append(InjectionFinding("TRUNCATED", Severity.LOW, f"excerpt truncated to {max_chars} chars"))

    # de-duplicate by (code, severity) keeping the first detail
    unique: dict[tuple[str, Severity], InjectionFinding] = {}
    for f in findings:
        unique.setdefault((f.code, f.severity), f)
    return SanitizedText(text=text, findings=tuple(unique.values()), truncated=truncated, raw_sha256=raw_sha)


@dataclass(frozen=True)
class ExternalDocument:
    title: str
    text: str
    uri: str | None
    published_at: datetime | None
    retrieved_at: datetime
    source_type: EvidenceSourceType = EvidenceSourceType.NEWS
    source_id: str | None = None
    extra_flags: tuple[str, ...] = field(default=())


def sanitize_external(
    title: str,
    text: str,
    uri: str | None,
    published_at: datetime | None,
    retrieved_at: datetime,
    *,
    source_type: EvidenceSourceType = EvidenceSourceType.NEWS,
    source_id: str | None = None,
    available_at: datetime | None = None,
    max_chars: int = 4000,
    is_simulated: bool = False,
) -> Evidence:
    """Turn retrieved external text into ``UNTRUSTED_EXTERNAL`` evidence with injection flags.

    ``content_sha256`` is the SHA-256 of the *raw* retrieved text (provenance of what was fetched);
    ``excerpt`` is the sanitized text models may see. ``available_at`` defaults to ``retrieved_at``:
    the decision clock may not use a document before it was actually retrieved (point-in-time).
    """
    if source_type in (EvidenceSourceType.QUANT_SIGNAL, EvidenceSourceType.MARKET_DATA, EvidenceSourceType.RISK_POLICY):
        raise ValueError(f"{source_type} evidence is internal; use build_internal_evidence")
    body = sanitize_text(text, max_chars=max_chars)
    head = sanitize_text(title, max_chars=300)
    uri_findings: list[InjectionFinding] = []
    clean_uri: str | None = None
    if uri is not None:
        u = sanitize_text(uri, max_chars=2000)
        clean_uri = u.text.replace("&lt;", "").replace("&gt;", "")
        if not re.match(r"^https?://", clean_uri, re.IGNORECASE):
            uri_findings.append(InjectionFinding("NON_HTTP_URI", Severity.MEDIUM, clean_uri[:60]))
        uri_findings.extend(f for f in u.findings if f.severity >= Severity.MEDIUM)
    timing: list[InjectionFinding] = []
    if published_at is not None and published_at > retrieved_at:
        timing.append(InjectionFinding("PUBLISHED_AFTER_RETRIEVAL", Severity.MEDIUM, "published_at > retrieved_at"))
    avail = available_at or retrieved_at
    if avail < retrieved_at:
        timing.append(InjectionFinding("AVAILABLE_BEFORE_RETRIEVAL", Severity.MEDIUM, "available_at < retrieved_at"))
        avail = retrieved_at
    findings = tuple(body.findings) + tuple(f for f in head.findings if f.severity >= Severity.MEDIUM) + tuple(uri_findings) + tuple(timing)
    sid = source_id or f"ext:{hashlib.sha256((uri or title).encode()).hexdigest()[:16]}"
    return Evidence(
        evidence_id=new_id("ev"),
        source_type=source_type,
        source_id=sid[:256],
        trust=TrustLevel.UNTRUSTED_EXTERNAL,
        title=head.text or "(untitled)",
        uri=clean_uri,
        excerpt=body.text,
        content_sha256=body.raw_sha256,
        published_at=published_at,
        retrieved_at=retrieved_at,
        available_at=avail,
        injection_flags=flags_from_findings(list(findings)),
        is_simulated=is_simulated,
    )
