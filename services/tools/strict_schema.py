"""One canonical, provider-neutral JSON schema per tool input.

The same schema object is published to every consumer (HTTP catalog, MCP ``inputSchema``, OpenAI Responses
``function`` tools with ``strict: true``, PydanticAI tool definitions), so there is exactly one tool contract.
It is the intersection that every consumer accepts:

* root and every nested object: ``type: object``, ``additionalProperties: false``, every property listed in
  ``required`` (OpenAI strict mode); optional inputs are expressed as nullable (``null`` means "use default");
* ``$ref``/``$defs`` are inlined; ``title``/``default``/``examples`` and length keywords are removed (length
  limits move into the description and are still enforced by the Pydantic model on the server side);
* only conservative keywords survive: type, properties, required, additionalProperties, items, enum, const,
  anyOf, description, pattern, minimum, maximum, minItems, maxItems.

Server-side validation is always the Pydantic model (``extra=forbid``, Decimal-only money); this schema is the
advertised contract, never the enforcement point.
"""

from __future__ import annotations

import copy
from typing import Any, Final

from pydantic import BaseModel

_KEEP: Final[frozenset[str]] = frozenset(
    {
        "type", "properties", "required", "additionalProperties", "items", "enum", "const", "anyOf", "description",
        "pattern", "minimum", "maximum", "minItems", "maxItems",
    }
)
_LENGTH_HINTS: Final[tuple[tuple[str, str], ...]] = (("minLength", "min {} chars"), ("maxLength", "max {} chars"))


def _resolve(node: Any, defs: dict[str, Any], depth: int = 0) -> Any:
    if depth > 32:
        raise ValueError("schema nesting too deep / recursive $ref")
    if isinstance(node, dict):
        if "$ref" in node:
            ref = str(node["$ref"])
            if not ref.startswith("#/$defs/"):
                raise ValueError(f"unsupported $ref {ref}")
            target = copy.deepcopy(defs[ref.removeprefix("#/$defs/")])
            merged = {**target, **{k: v for k, v in node.items() if k != "$ref"}}
            return _resolve(merged, defs, depth + 1)
        return {k: _resolve(v, defs, depth + 1) for k, v in node.items() if k != "$defs"}
    if isinstance(node, list):
        return [_resolve(v, defs, depth + 1) for v in node]
    return node


def _nullable(node: dict[str, Any]) -> dict[str, Any]:
    if "anyOf" in node:
        if not any(isinstance(b, dict) and b.get("type") == "null" for b in node["anyOf"]):
            node["anyOf"] = [*node["anyOf"], {"type": "null"}]
        return node
    if "enum" in node or "const" in node:
        desc = node.pop("description", None)
        out: dict[str, Any] = {"anyOf": [node, {"type": "null"}]}
        if desc:
            out["description"] = desc
        return out
    t = node.get("type")
    if isinstance(t, str) and t != "null":
        node["type"] = [t, "null"]
    elif isinstance(t, list) and "null" not in t:
        node["type"] = [*t, "null"]
    return node


def _strictify(node: Any) -> Any:
    if isinstance(node, list):
        return [_strictify(v) for v in node]
    if not isinstance(node, dict):
        return node
    hints = [fmt.format(node[k]) for k, fmt in _LENGTH_HINTS if k in node]
    out: dict[str, Any] = {k: _strictify(v) for k, v in node.items() if k in _KEEP and k != "properties"}
    if hints:
        out["description"] = (str(out.get("description", "")) + f" ({', '.join(hints)})").strip()
    if node.get("type") == "object" or "properties" in node:
        props: dict[str, Any] = node.get("properties", {}) or {}
        originally_required = set(node.get("required", []) or [])
        new_props: dict[str, Any] = {}
        for name, sub in props.items():
            s = _strictify(sub)
            if name not in originally_required:
                s = _nullable(s)
            new_props[name] = s
        out["type"] = "object"
        out["properties"] = new_props
        out["required"] = list(new_props)
        out["additionalProperties"] = False
    return out


def strict_input_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Canonical strict JSON schema for a tool input model (see module docstring)."""
    raw = model.model_json_schema(mode="validation")
    defs: dict[str, Any] = raw.get("$defs", {})
    resolved = _resolve(raw, defs)
    schema = _strictify(resolved)
    schema.pop("description", None)  # the tool description lives on the tool, not on the schema root
    return schema


def output_schema(model: type[BaseModel]) -> dict[str, Any]:
    """JSON schema of a tool output in serialization mode (refs inlined, titles dropped)."""
    raw = model.model_json_schema(mode="serialization")
    defs: dict[str, Any] = raw.get("$defs", {})
    resolved = _resolve(raw, defs)
    return _drop_titles(resolved)


def _drop_titles(node: Any) -> Any:
    if isinstance(node, dict):
        return {k: _drop_titles(v) for k, v in node.items() if k not in ("title", "default", "examples")
                or (k == "title" and isinstance(v, dict))}
    if isinstance(node, list):
        return [_drop_titles(v) for v in node]
    return node
