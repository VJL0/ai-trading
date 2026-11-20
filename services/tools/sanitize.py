"""Free-text hygiene for everything an agent sends through the tool interface.

Agent text (a trade thesis, counterarguments, a halt reason) is persisted in the audit trail, shown on the
dashboard and may later be read by other models. It is therefore *data*, and this module makes sure it stays
inert data:

* ``clean_text``: Unicode NFKC normalisation, removal of control / zero-width / bidi-override characters,
  whitespace collapsing. Never raises on content; the length cap is enforced by the Pydantic field.
* ``screen_text``: a deliberately simple, deterministic injection screen. Text that tries to *instruct* the
  system (override instructions, impersonate system/developer roles, ask to bypass risk, resume trading,
  change limits/mode, or exfiltrate credentials) is rejected (HTTP 422 at the tool boundary), never executed.
  ``neutralize_text`` is the non-refusing variant for halt reasons (a halt must never be blocked by the screen).
* ``sanitize_actor``: the agent name header becomes part of ``strategy_id``; restrict it to a safe charset.

Nothing here is a security boundary on its own: the deterministic risk engine, the signed-intent gateway and
the absence of any dangerous tool are. This screen only keeps instruction-shaped text out of the record.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Final

MAX_THESIS_CHARS: Final = 2000
MAX_COUNTERARGUMENT_CHARS: Final = 500
MAX_REASON_CHARS: Final = 500
MAX_ACTOR_CHARS: Final = 48

# C0/C1 controls except \t and \n, DEL, zero-width and bidi override/isolate characters, BOM.
_CONTROL_RE: Final = re.compile(
    "[\x00-\x08\x0b\x0c\x0d\x0e-\x1f\x7f-\x9f\u200b-\u200f\u202a-\u202e\u2060-\u2069\ufeff\ufff9-\ufffb]"
)
_WS_RE: Final = re.compile(r"[ \t]+")
_NL_RE: Final = re.compile(r"\n{3,}")

# Instruction-shaped phrases. Matched case-insensitively on the cleaned text with separators normalised.
_INJECTION_PATTERNS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = tuple(
    (name, re.compile(rx, re.IGNORECASE))
    for name, rx in (
        ("override_instructions",
         r"\b(ignore|disregard|forget|override)\b.{0,40}\b(previous|prior|above|earlier|all|any|system|safety)\b"
         r".{0,20}\b(instructions?|rules?|prompts?|guardrails?|constraints?)\b"),
        ("role_impersonation", r"(^|\n)\s*(system|developer|assistant|operator)\s*:"),
        ("role_tags", r"<\s*/?\s*(system|developer|instructions?|im_start|im_end)\b|<\|[a-z_]+\|>"),
        ("new_identity", r"\byou are now\b|\bact as (the )?(operator|admin|system)\b"),
        ("bypass_risk",
         r"\b(bypass|disable|circumvent|turn off|override|skip)\b.{0,30}"
         r"\b(risk (engine|checks?|limits?|gate|policy)|limits|halts?|approvals?|guardrails?|safety)\b"),
        ("resume_trading",
         r"\b(unhalt|unlock|re ?enable)\b.{0,12}\b(trading|entries|orders)\b"
         r"|\b(clear|lift|remove)\b.{0,12}\bhalts?\b|\bresume (new )?entries\b"),
        ("change_limits",
         r"\b(raise|increase|change|modify|lift|remove|loosen|relax)\b.{0,30}"
         r"\b(risk limits?|position limits?|loss limits?|risk policy|max order notional|max position notional"
         r"|max daily loss|max planned loss)\b"),
        ("change_mode", r"\b(switch|change|set|enable)\b.{0,20}\b(live|bounded live|live approval)\b( mode| trading)?"),
        ("secret_request", r"\b(api ?key|secret ?key|password|private key|operator ?token|agent ?token"
                           r"|engine ?token|keychain)\b|\.env\b"),
    )
)


class InjectionRejected(ValueError):
    """Text matched the injection screen (the message names the rule, never echoes the text)."""


def clean_text(value: object) -> object:
    """Normalise free text; non-strings pass through unchanged so Pydantic reports the type error."""
    if not isinstance(value, str):
        return value
    text = unicodedata.normalize("NFKC", value).replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_RE.sub("", text)
    lines = [_WS_RE.sub(" ", line).strip() for line in text.split("\n")]
    text = _NL_RE.sub("\n\n", "\n".join(lines)).strip()
    return text


def injection_flags(text: str) -> list[str]:
    probe = re.sub(r"\s+", " ", text.replace("_", " ").replace("-", " "))
    flags = [name for name, rx in _INJECTION_PATTERNS if rx.search(text) or rx.search(probe)]
    return sorted(set(flags))


def screen_text(text: str) -> str:
    flags = injection_flags(text)
    if flags:
        raise InjectionRejected(
            "text rejected by the injection screen (rules: " + ", ".join(flags) + "); state market reasoning only"
        )
    return text


def neutralize_text(text: str) -> str:
    """Like ``screen_text`` but never refuses: instruction-shaped text is replaced by an inert placeholder.

    Used where refusing would be unsafe — a halt request is the fail-safe direction and must always go
    through, but its free-text reason must not carry injected instructions into the audit trail / dashboard.
    """
    flags = injection_flags(text)
    if flags:
        return "[agent text withheld by the injection screen: " + ", ".join(flags) + "]"
    return text


_ACTOR_RE: Final = re.compile(r"[^A-Za-z0-9_.:-]")


def sanitize_actor(actor: str | None) -> str:
    """Restrict an agent name to ``[A-Za-z0-9_.:-]{1,48}``; empty becomes ``agent``."""
    cleaned = _ACTOR_RE.sub("_", (actor or "").strip())[:MAX_ACTOR_CHARS].strip("_")
    return cleaned or "agent"


def truncate(value: object, limit: int = 300) -> str | None:
    """Compact a runtime-provided string for token-economical outputs (``None`` stays ``None``)."""
    if value is None:
        return None
    text = str(value)
    text = _CONTROL_RE.sub("", text)
    return text if len(text) <= limit else text[: max(limit - 1, 0)] + "…"
