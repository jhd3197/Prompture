"""Tool parameter schemas every provider accepts.

Tool schemas from outside sources (MCP servers, OpenAPI specs, hand-written
dicts) sometimes put ``oneOf`` / ``anyOf`` / ``allOf`` at the top level, or
leave out ``"type": "object"``. Anthropic rejects such a tool with
``input_schema does not support oneOf``, and OpenAI requires function
parameters to be a plain object. One bad tool fails the whole request, so
:func:`flatten_tool_schema` rewrites the top level into a single object
schema before it is sent. Combinators nested inside properties are valid
everywhere and are left alone.
"""

from __future__ import annotations

import copy
from typing import Any

_COMBINATORS = ("allOf", "oneOf", "anyOf")
_EMPTY: dict[str, Any] = {"type": "object", "properties": {}}


def _resolve(node: Any, defs: dict[str, Any]) -> dict[str, Any] | None:
    """Follow a local ``#/$defs/X`` or ``#/definitions/X`` reference."""
    if not isinstance(node, dict):
        return None
    ref = node.get("$ref")
    if isinstance(ref, str):
        for prefix in ("#/$defs/", "#/definitions/"):
            if ref.startswith(prefix):
                target = defs.get(ref[len(prefix) :])
                return target if isinstance(target, dict) else None
        return None
    return node


def _variant_label(branch: dict[str, Any], required: list[str]) -> str:
    title = branch.get("title")
    names = required or list((branch.get("properties") or {}).keys())
    fields = ", ".join(names) if names else "no arguments"
    return f"{title} ({fields})" if isinstance(title, str) and title else f"({fields})"


def _merge(schema: dict[str, Any], defs: dict[str, Any], depth: int = 0) -> tuple[dict, list[str], list[str]]:
    """Return ``(properties, required, variant labels)`` for one schema node."""
    properties: dict[str, Any] = dict(schema.get("properties") or {})
    required: list[str] = [r for r in schema.get("required") or [] if isinstance(r, str)]
    variants: list[str] = []
    if depth > 4:  # recursive definitions: stop rather than loop
        return properties, required, variants

    for key in _COMBINATORS:
        branches = schema.get(key)
        if not isinstance(branches, list):
            continue
        merged: list[tuple[dict, list[str]]] = []
        for raw in branches:
            branch = _resolve(raw, defs)
            if branch is None:
                continue
            props, req, nested = _merge(branch, defs, depth + 1)
            merged.append((props, req))
            variants.extend(nested)
            if key != "allOf" and (props or req):
                variants.append(_variant_label({**branch, "properties": props}, req))
        for props, _ in merged:
            for name, prop in props.items():
                properties.setdefault(name, prop)
        if key == "allOf":
            extra = [name for _, req in merged for name in req]
        elif merged:
            # Only what every alternative needs is required in the merged form.
            common = set(merged[0][1])
            for _, req in merged[1:]:
                common &= set(req)
            extra = [name for name in merged[0][1] if name in common]
        else:
            extra = []
        required.extend(name for name in extra if name not in required)
    return properties, required, variants


def flatten_tool_schema(schema: Any) -> dict[str, Any]:
    """Return *schema* as a top-level object schema every provider accepts.

    A schema that is already a plain object (``"type": "object"``, no top-level
    ``oneOf`` / ``anyOf`` / ``allOf``) is returned unchanged, so this is cheap to
    call on every request and safe to call twice. Otherwise the alternatives'
    properties are merged into one object: ``allOf`` keeps every branch's
    required fields, ``oneOf`` / ``anyOf`` keep only the fields all branches
    require, and the alternatives are listed in the schema's description so the
    model still knows which arguments go together.
    """
    if not isinstance(schema, dict):
        return copy.deepcopy(_EMPTY)
    has_combinator = any(key in schema for key in _COMBINATORS)
    if schema.get("type") == "object" and not has_combinator:
        return schema
    if not has_combinator and schema.get("type") not in (None, "object"):
        # A bare scalar or array schema has no named arguments to keep.
        return copy.deepcopy(_EMPTY)

    defs: dict[str, Any] = {}
    for key in ("definitions", "$defs"):
        if isinstance(schema.get(key), dict):
            defs.update(schema[key])

    properties, required, variants = _merge(schema, defs)
    out = {k: copy.deepcopy(v) for k, v in schema.items() if k not in _COMBINATORS and k != "type"}
    out = {"type": "object", **out, "properties": copy.deepcopy(properties)}
    if required:
        out["required"] = [name for name in required if name in properties]
        if not out["required"]:
            del out["required"]
    else:
        out.pop("required", None)
    if len(variants) > 1:
        note = "Use one of these argument sets: " + "; ".join(dict.fromkeys(variants)) + "."
        description = out.get("description")
        out["description"] = f"{description}\n\n{note}" if isinstance(description, str) and description else note
    return out


def flatten_tool_definitions(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply :func:`flatten_tool_schema` to OpenAI- or Anthropic-shaped tool dicts."""
    out: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            out.append(tool)
            continue
        fn = tool.get("function")
        if tool.get("type") == "function" and isinstance(fn, dict) and "parameters" in fn:
            params = flatten_tool_schema(fn["parameters"])
            out.append(tool if params is fn["parameters"] else {**tool, "function": {**fn, "parameters": params}})
        elif "input_schema" in tool:
            params = flatten_tool_schema(tool["input_schema"])
            out.append(tool if params is tool["input_schema"] else {**tool, "input_schema": params})
        else:
            out.append(tool)
    return out
