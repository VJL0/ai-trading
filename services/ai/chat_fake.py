"""Explicit FAKE "model" for the natural-language chat client (no network, deterministic, clearly labelled).

``ChatScript`` is a keyword-driven stand-in for GPT-6.1 Sol: given a Responses API request body it returns the
output items a model would (``reasoning`` + ``function_call`` items, or a final assistant ``message``). It only
ever calls tools that the request actually offered, builds arguments from the offered JSON schema (so it works
with whatever live catalog the running API serves), and answers from the tool outputs in the history.

It plugs into both fake transports:

* Sign in with ChatGPT: ``FakeChatGptServer.responder = ChatScript()`` (streaming SSE ``/v1/responses``);
* API keys: :class:`FakeChatOpenAIHandler` (``httpx2.MockTransport`` for the OpenAI SDK used by PydanticAI;
  inherits the request-shape validation of :class:`~services.ai.fake_transport.FakeOpenAIHandler`).

Answers produced here are scripted, not model output; the chat client labels every FAKE answer as such.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, cast

import httpx2

from .fake_transport import FakeOpenAIHandler, RecordedExchange

_DURATION_RE = re.compile(r"(\d+(?:\.\d+)?)\s*-?\s*(hours?|hrs?|h\b|minutes?|mins?|m\b)", re.IGNORECASE)
_SYMBOL_RE = re.compile(r"\b([A-Z]{2,5})\b")
_NOT_SYMBOLS = frozenset({"AI", "PNL", "OK", "NO", "ETF", "API", "GPT", "USD", "ET", "UTC"})


def offered_tools(body: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """``name -> parameters`` for every function tool in a request (top-level or inside a ``namespace``)."""
    out: dict[str, dict[str, Any]] = {}
    for t in cast(list[Any], body.get("tools") or []):
        if not isinstance(t, dict):
            continue
        td = cast(dict[str, Any], t)
        inner = cast(list[Any], td.get("tools") or []) if td.get("type") == "namespace" else [td]
        for x in inner:
            if isinstance(x, dict) and cast(dict[str, Any], x).get("type") == "function":
                xd = cast(dict[str, Any], x)
                params = xd.get("parameters")
                out[str(xd.get("name"))] = cast(dict[str, Any], params) if isinstance(params, dict) else {}
    return out


def _namespace(body: Mapping[str, Any]) -> str | None:
    for t in cast(list[Any], body.get("tools") or []):
        if isinstance(t, dict) and cast(dict[str, Any], t).get("type") == "namespace":
            return str(cast(dict[str, Any], t).get("name"))
    return None


def _user_text(item: Mapping[str, Any]) -> str:
    content = item.get("content")
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for c in cast(list[Any], content or []):
        if isinstance(c, dict) and cast(dict[str, Any], c).get("type") in ("input_text", "text"):
            parts.append(str(cast(dict[str, Any], c).get("text") or ""))
    return " ".join(parts)


def _turn(body: Mapping[str, Any]) -> tuple[str, dict[str, Any], set[str]]:
    """``(last user text, {tool name: parsed output} since that message, tools called since it)``."""
    items = [cast(dict[str, Any], i) for i in cast(list[Any], body.get("input") or []) if isinstance(i, dict)]
    last = -1
    for i, it in enumerate(items):
        if it.get("role") == "user" and it.get("type", "message") == "message":
            last = i
    text = _user_text(items[last]) if last >= 0 else ""
    names: dict[str, str] = {}
    outputs: dict[str, Any] = {}
    called: set[str] = set()
    for it in items[last + 1:]:
        if it.get("type") == "function_call":
            names[str(it.get("call_id"))] = str(it.get("name"))
            called.add(str(it.get("name")))
        elif it.get("type") == "function_call_output":
            name = names.get(str(it.get("call_id")), "?")
            try:
                outputs[name] = json.loads(str(it.get("output")))
            except json.JSONDecodeError:
                outputs[name] = {"error": "unparseable tool output"}
    return text, outputs, called


def duration_minutes(text: str) -> int | None:
    m = _DURATION_RE.search(text)
    if m is None:
        return 60 if re.search(r"\ban?\s+hour\b", text, re.IGNORECASE) else None
    value = float(m.group(1))
    unit = m.group(2).lower()
    return max(1, round(value * 60 if unit.startswith("h") else value))


def _nullable(schema: Mapping[str, Any]) -> bool:
    t = schema.get("type")
    if isinstance(t, list) and "null" in cast(list[Any], t):
        return True
    variants = cast(list[Any], schema.get("anyOf") or [])
    return any(isinstance(v, dict) and cast(dict[str, Any], v).get("type") == "null" for v in variants)


def _base_type(schema: Mapping[str, Any]) -> str | None:
    t = schema.get("type")
    if isinstance(t, str):
        return t
    if isinstance(t, list):
        return next((str(x) for x in cast(list[Any], t) if x != "null"), None)
    for v in cast(list[Any], schema.get("anyOf") or []):
        if isinstance(v, dict) and cast(dict[str, Any], v).get("type") not in (None, "null"):
            return str(cast(dict[str, Any], v)["type"])
    return None


def build_args(parameters: Mapping[str, Any], text: str) -> dict[str, Any]:
    """Arguments for a strict schema: duration-like fields from the request text, optional (nullable) fields
    left null (= server default), symbols named in the text, otherwise the type's neutral value."""
    props = cast(dict[str, Any], parameters.get("properties") or {})
    required = [str(r) for r in cast(list[Any], parameters.get("required") or list(props))]
    minutes = duration_minutes(text)
    symbols = [s for s in _SYMBOL_RE.findall(text) if s not in _NOT_SYMBOLS]
    args: dict[str, Any] = {}
    for name in required:
        schema = cast(dict[str, Any], props.get(name) or {})
        kind = _base_type(schema)
        lname = name.lower()
        duration_like = ("duration" in lname or lname in ("minutes", "hours", "length_minutes")) and \
            kind in ("integer", "number")
        if minutes is not None and duration_like and "hour" not in lname:
            args[name] = minutes
        elif minutes is not None and duration_like:
            args[name] = max(1, round(minutes / 60))
        elif "symbol" in lname and symbols:
            args[name] = symbols if kind == "array" else symbols[0]
        elif _nullable(schema):
            args[name] = None
        elif "default" in schema:
            args[name] = schema["default"]
        elif kind == "boolean":
            args[name] = False
        elif kind in ("integer", "number"):
            args[name] = schema.get("minimum", 1)
        elif kind == "array":
            args[name] = []
        elif kind == "object":
            args[name] = {}
        else:
            enum = cast(list[Any], schema.get("enum") or [])
            args[name] = enum[0] if enum else ""
    return args


def _walk(obj: Any) -> Iterable[dict[str, Any]]:
    if isinstance(obj, dict):
        d = cast(dict[str, Any], obj)
        yield d
        for v in d.values():
            yield from _walk(v)
    elif isinstance(obj, list):
        for v in cast(list[Any], obj):
            yield from _walk(v)


def no_trade_reasons(outputs: Mapping[str, Any]) -> list[str]:
    """Exact reasons found in tool outputs: NO_TRADE decision reasons first, then risk failures / lock reasons."""
    reasons: list[str] = []

    def add(value: Any) -> None:
        if isinstance(value, str) and value.strip() and value.strip() not in reasons:
            reasons.append(value.strip())

    for d in _walk(dict(outputs)):
        if d.get("action") == "NO_TRADE" or d.get("outcome") == "NO_TRADE" or d.get("status") == "NO_TRADE":
            add(d.get("reason") or d.get("no_trade_reason"))
    for d in _walk(dict(outputs)):
        add(d.get("no_trade_reason"))
        for key in ("risk_failures", "lock_reasons", "entries_locked_reasons", "reasons"):
            for v in cast(list[Any], d.get(key) or []) if isinstance(d.get(key), list) else []:
                add(v)
    return reasons[:4]


def _fmt_error(out: Any) -> str | None:
    if isinstance(out, dict):
        d = cast(dict[str, Any], out)
        if d.get("declined"):
            return "the owner declined"
        if d.get("error"):
            return str(d["error"])[:300]
    return None


def _first(offered: Mapping[str, Any], patterns: Iterable[str]) -> list[str]:
    names: list[str] = []
    for pat in patterns:
        rx = re.compile(pat)
        for n in offered:
            if rx.search(n) and n not in names:
                names.append(n)
                break
    return names


@dataclass
class ChatScript:
    """The scripted FAKE chat model. ``calls`` counts requests; ``seen_tools`` records the offered names."""

    calls: int = 0
    seen_tools: list[list[str]] = field(default_factory=list[list[str]])

    def __call__(self, body: dict[str, Any]) -> list[dict[str, Any]]:
        n = self.calls
        self.calls += 1
        offered = offered_tools(body)
        self.seen_tools.append(sorted(offered))
        text, outputs, called = _turn(body)
        low = text.lower()
        reasoning = {"type": "reasoning", "id": f"rs_chat_{n}", "summary": [
            {"type": "summary_text", "text": f"(fake chat reasoning step {n})"}], "encrypted_content": f"fake-chat-enc-{n}"}

        if "session" in low and re.search(r"\b(start|begin|run|launch)\b", low):
            plan = _first(offered, [r"^start\w*session"])
            if not plan:
                return [reasoning, self._message(n, "I can't start a session: the running app offers no session tool.")]
            return self._act(n, body, reasoning, plan[0], offered, text, outputs, called, self._started)
        if "session" in low and re.search(r"\b(stop|end|finish)\b", low):
            plan = _first(offered, [r"^stop\w*session"])
            if not plan:
                return [reasoning, self._message(n, "I can't stop a session: the running app offers no session tool.")]
            return self._act(n, body, reasoning, plan[0], offered, text, outputs, called, self._stopped)
        if re.search(r"\b(quant )?cycle\b|\bscan now\b", low) and not re.search(r"\bwhy\b", low):
            plan = _first(offered, [r"^run_quant_cycle$", r"cycle"])
            if not plan:
                return [reasoning, self._message(n, "I can't run a cycle: the running app offers no cycle tool.")]
            return self._act(n, body, reasoning, plan[0], offered, text, outputs, called, self._cycle)
        if re.search(r"\bwhy\b", low) and "trade" in low:
            plan = _first(offered, [r"decision", r"activity", r"session_report", r"^get_signals$", r"^get_risk_status$"])
            pending = [p for p in plan if p not in called]
            if pending and not outputs:
                return [reasoning, *[self._call(n, i, body, p, build_args(offered[p], text)) for i, p in enumerate(pending)]]
            reasons = no_trade_reasons(outputs)
            if reasons:
                answer = "I did not trade because: " + "; ".join(reasons) + "."
            else:
                answer = "I found no recorded NO_TRADE reason in the current activity, signals or risk status."
            return [reasoning, self._message(n, answer)]
        account = re.search(r"\b(account|equity|portfolio|positions|cash)\b", low)
        if account or re.search(r"performance|p&l|pnl|profit|how am i doing", low):
            plan = _first(offered, [r"^get_portfolio$"] if account else [r"performance", r"^get_portfolio$"])
            if plan and plan[0] not in called:
                return [reasoning, self._call(n, 0, body, plan[0], build_args(offered[plan[0]], text))]
            return [reasoning, self._message(n, self._performance(outputs))]
        plan = _first(offered, [r"^get_risk_status$"])
        if plan and plan[0] not in called and re.search(r"status|risk|halt|locked", low):
            return [reasoning, self._call(n, 0, body, plan[0], {})]
        names = ", ".join(sorted(offered)[:12]) or "none"
        return [reasoning, self._message(n, "I can start or stop paper sessions, explain decisions and show performance. "
                                            f"Tools available: {names}.")]

    # ------------------------------------------------------------------ pieces
    def _act(self, n: int, body: dict[str, Any], reasoning: dict[str, Any], tool: str,
             offered: Mapping[str, dict[str, Any]], text: str, outputs: Mapping[str, Any], called: set[str],
             summarize: Any) -> list[dict[str, Any]]:
        if tool not in called:
            return [reasoning, self._call(n, 0, body, tool, build_args(offered[tool], text))]
        out = outputs.get(tool)
        err = _fmt_error(out)
        if err is not None:
            return [reasoning, self._message(n, f"I did not run {tool}: {err}.")]
        return [reasoning, self._message(n, summarize(tool, out))]

    @staticmethod
    def _started(tool: str, out: Any) -> str:
        d = cast(dict[str, Any], out) if isinstance(out, dict) else {}
        sess = cast(dict[str, Any], d.get("session")) if isinstance(d.get("session"), dict) else d
        sid, state, ends = sess.get("session_id"), sess.get("state"), sess.get("ends_at")
        return f"Started paper session {sid} ({state}); it ends at {ends}. The trading service runs it on its own."

    @staticmethod
    def _cycle(tool: str, out: Any) -> str:
        d = cast(dict[str, Any], out) if isinstance(out, dict) else {}
        reason = f": {d.get('reason')}" if d.get("reason") else ""
        return (f"Ran one quant cycle ({d.get('execution_label')}): outcome {d.get('outcome')}{reason}; "
                f"candidate {d.get('candidate_symbol')}, {d.get('signals')} signals.")

    @staticmethod
    def _stopped(tool: str, out: Any) -> str:
        d = cast(dict[str, Any], out) if isinstance(out, dict) else {}
        sess = cast(dict[str, Any], d.get("session")) if isinstance(d.get("session"), dict) else d
        return f"Stop requested for session {sess.get('session_id')} (state {sess.get('state')})."

    @staticmethod
    def _performance(outputs: Mapping[str, Any]) -> str:
        for out in outputs.values():
            if isinstance(out, dict):
                d = cast(dict[str, Any], out)
                if "total_pnl" in d:
                    return (f"Performance ({d.get('mode_label')}, as of {d.get('asof')}): total P&L {d.get('total_pnl')}, "
                            f"daily P&L {d.get('daily_pnl')}, win rate {d.get('win_rate')}, "
                            f"max drawdown {d.get('max_drawdown')}.")
                account = d.get("account")
                if isinstance(account, dict):
                    a = cast(dict[str, Any], account)
                    return (f"Paper account equity {a.get('equity')}, cash {a.get('cash')}, daily P&L "
                            f"{d.get('daily_pnl')} (as of {d.get('asof')}).")
        return "I could not read performance data from the app."

    def _call(self, n: int, i: int, body: Mapping[str, Any], name: str, args: dict[str, Any]) -> dict[str, Any]:
        item: dict[str, Any] = {"type": "function_call", "id": f"fc_chat_{n}_{i}", "call_id": f"call_chat_{n}_{i}",
                                "name": name, "arguments": json.dumps(args, sort_keys=True), "status": "completed"}
        ns = _namespace(body)
        if ns is not None:
            item["namespace"] = ns
        return item

    @staticmethod
    def _message(n: int, text: str) -> dict[str, Any]:
        return {"type": "message", "id": f"msg_chat_{n}", "role": "assistant", "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}]}


@dataclass
class FakeChatOpenAIHandler(FakeOpenAIHandler):
    """OpenAI Responses API fake for the API-key chat path (validated like the pipeline's fake, scripted chat)."""

    script: ChatScript = field(default_factory=ChatScript)

    def reply(self, ex: RecordedExchange) -> httpx2.Response:
        return self._response(ex, self.script(ex.body))


__all__ = ["ChatScript", "FakeChatOpenAIHandler", "build_args", "duration_minutes", "no_trade_reasons", "offered_tools"]
