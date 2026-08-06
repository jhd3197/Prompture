"""Function calling / tool use support for Prompture.

Provides :class:`ToolDefinition` for describing callable tools,
:class:`ToolRegistry` for managing a collection of tools, and
:func:`tool_from_function` to auto-generate tool schemas from type hints.

Example::

    from prompture import ToolRegistry

    registry = ToolRegistry()

    @registry.tool
    def get_weather(city: str, units: str = "celsius") -> str:
        \"\"\"Get the current weather for a city.\"\"\"
        return f"Weather in {city}: 22 {units}"

    # Or register explicitly
    registry.register(get_weather)
"""

from __future__ import annotations

import inspect
import json
import logging
import re
import uuid
from collections.abc import Callable, Mapping
from dataclasses import MISSING, asdict, dataclass, field, is_dataclass
from dataclasses import fields as dataclass_fields
from datetime import date, datetime, time
from enum import Enum
from typing import Any, Literal, get_args, get_origin, get_type_hints, is_typeddict

from pydantic import BaseModel, TypeAdapter

from ..extraction.tools import _is_union_origin, convert_value
from ..infra.cost_mixin import prepare_strict_schema

try:  # same defensive pattern as prompture/extraction/validator.py
    import jsonschema
except Exception:  # pragma: no cover - jsonschema is a hard dependency
    jsonschema = None

logger = logging.getLogger("prompture.tools_schema")

#: Default cap on a tool description's length.  OpenAI rejects ``function``
#: descriptions longer than 1024 characters; other providers are more lenient.
#: Pass ``max_description_chars=None`` to opt out of truncation.
MAX_TOOL_DESCRIPTION_CHARS = 1024

# Mapping from Python types to JSON Schema types
_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    tuple: "array",
    dict: "object",
}


def _json_type_name(value: Any) -> str | None:
    """JSON Schema type name for a literal Python value (``None`` if unmappable)."""
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return None


def _enum_schema(values: list[Any]) -> dict[str, Any]:
    """Build an ``enum`` schema, adding ``type`` when all values share one."""
    schema: dict[str, Any] = {"enum": values}
    names = [_json_type_name(v) for v in values]
    if values and all(n is not None and n == names[0] for n in names):
        schema["type"] = names[0]
    return schema


def _is_structured_type(annotation: Any) -> bool:
    """True for nested structured types: pydantic models, dataclasses, TypedDicts."""
    if not isinstance(annotation, type):
        return False
    if is_typeddict(annotation):
        return True
    if is_dataclass(annotation):
        return True
    return issubclass(annotation, BaseModel)


def _structured_required_fields(annotation: Any, hints: dict[str, Any]) -> list[str] | None:
    """Required field names of a structured type, or ``None`` if undeterminable.

    ``None`` (rather than "all of them") keeps us from over-constraining a
    type whose optionality we cannot read.
    """
    if is_typeddict(annotation):
        required = getattr(annotation, "__required_keys__", None)
        if required is None:
            return None
        return [name for name in hints if name in required]
    if is_dataclass(annotation):
        return [
            f.name
            for f in dataclass_fields(annotation)
            if f.default is MISSING and f.default_factory is MISSING  # type: ignore[misc]
        ]
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return [name for name, f in annotation.model_fields.items() if f.is_required()]
    return None


def _structured_fallback_schema(annotation: Any) -> dict[str, Any] | None:
    """Object schema for a structured type :class:`TypeAdapter` cannot handle.

    pydantic refuses ``typing.TypedDict`` on Python < 3.12 (it requires the
    ``typing_extensions`` variant), which on two of the four supported Python
    versions left such a parameter advertised as a bare string — the exact
    silent degradation the richer type mapping exists to prevent.  Rebuild the
    object schema from resolved hints so the model still sees the real field
    names and types.

    Returns ``None`` when the type exposes nothing introspectable, leaving the
    caller's own fallback in charge.
    """
    try:
        hints = get_type_hints(annotation)
    except Exception:
        logger.debug("Could not resolve type hints for structured type %r", annotation, exc_info=True)
        return None
    hints.pop("return", None)
    if not hints:
        return None

    schema: dict[str, Any] = {
        "type": "object",
        "properties": {name: _python_type_to_json_schema(hint) for name, hint in hints.items()},
    }
    required = _structured_required_fields(annotation, hints)
    if required:
        schema["required"] = required
    return schema


def _python_type_to_json_schema(annotation: Any) -> dict[str, Any]:
    """Convert a Python type annotation to a JSON Schema snippet.

    Handles ``typing.Union`` and PEP-604 unions (``int | None``) as ``anyOf``
    (with ``{"type": "null"}`` for ``NoneType``), ``Literal`` and ``Enum`` as
    ``enum``, ``datetime``/``date``/``time``/``UUID`` as formatted strings,
    ``list``/``tuple``/``dict`` containers (``dict[str, X]`` gets
    ``additionalProperties``), nested pydantic models, dataclasses and
    ``TypedDict`` via :class:`pydantic.TypeAdapter` (falling back to
    :func:`_structured_fallback_schema` on the Python versions where pydantic
    refuses a given structured type), and ``Any`` as the empty schema. Unknown
    types fall back to ``{"type": "string"}``.

    A *missing* annotation yields the empty (unconstrained) schema rather than
    a string: an unannotated parameter means "type unknown", and since
    arguments are validated against this schema, claiming ``string`` would
    reject every correct non-string call (e.g. ``lambda x: x + 1``).
    """
    if annotation is inspect.Parameter.empty:
        return {}
    if annotation is Any or annotation is None:
        return {}

    origin = get_origin(annotation)
    args = get_args(annotation)

    # Union / Optional — both typing.Union[X, ...] and PEP-604 X | Y.
    if _is_union_origin(origin):
        return {
            "anyOf": [{"type": "null"} if a is type(None) else _python_type_to_json_schema(a) for a in args],
        }

    # Literal[...] values.
    if origin is Literal:
        return _enum_schema(list(args))

    # list[X]
    if origin is list:
        return {
            "type": "array",
            "items": _python_type_to_json_schema(args[0]) if args else {},
        }

    # tuple[X, ...] / tuple[A, B]
    if origin is tuple:
        if args and args[-1] is Ellipsis:
            return {"type": "array", "items": _python_type_to_json_schema(args[0])}
        if args:
            return {
                "type": "array",
                "prefixItems": [_python_type_to_json_schema(a) for a in args],
                "items": False,
            }
        return {"type": "array"}

    # dict[str, X]
    if origin is dict:
        schema: dict[str, Any] = {"type": "object"}
        if len(args) > 1 and args[1] is not Any:
            schema["additionalProperties"] = _python_type_to_json_schema(args[1])
        return schema

    # Enum subclasses.
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return _enum_schema([member.value for member in annotation])

    # Date/time-ish scalars.
    if annotation is datetime:
        return {"type": "string", "format": "date-time"}
    if annotation is date:
        return {"type": "string", "format": "date"}
    if annotation is time:
        return {"type": "string", "format": "time"}
    if annotation is uuid.UUID:
        return {"type": "string", "format": "uuid"}

    # Nested structured types: pydantic BaseModel, dataclass, TypedDict.
    if _is_structured_type(annotation):
        try:
            return TypeAdapter(annotation).json_schema()
        except Exception as exc:
            logger.debug("TypeAdapter schema generation failed for %r: %s", annotation, exc)
            manual = _structured_fallback_schema(annotation)
            if manual is not None:
                return manual

    # Simple types
    json_type = _TYPE_MAP.get(annotation, "string")
    return {"type": json_type}


def _make_nullable(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *schema* that also accepts ``null``."""
    schema = dict(schema)
    if isinstance(schema.get("type"), str):
        schema["type"] = [schema["type"], "null"]
    elif isinstance(schema.get("type"), list):
        if "null" not in schema["type"]:
            schema["type"] = [*schema["type"], "null"]
    elif isinstance(schema.get("anyOf"), list):
        if {"type": "null"} not in schema["anyOf"]:
            schema["anyOf"] = [*schema["anyOf"], {"type": "null"}]
    else:
        schema = {"anyOf": [schema, {"type": "null"}]}
    return schema


def _strict_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    """Normalise a tool-parameters schema for OpenAI strict tool use.

    Applies the same normalization as
    :func:`prompture.infra.cost_mixin.prepare_strict_schema`
    (``additionalProperties: false``, every property in ``required``) and
    additionally makes originally-optional parameters (those with a default,
    i.e. not in the source ``required`` list) nullable, since strict mode
    forces them into ``required``.
    """
    originally_required = set(parameters.get("required", []) or [])
    strict_schema = prepare_strict_schema(parameters)
    properties = strict_schema.get("properties")
    if isinstance(properties, dict):
        for key, prop in properties.items():
            if key not in originally_required and isinstance(prop, dict):
                properties[key] = _make_nullable(prop)
    return strict_schema


#: Valid tool names (OpenAI / Anthropic / Google all accept this shape).
_TOOL_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def _validate_tool_name(name: str) -> None:
    """Raise ``ValueError`` if *name* is not a valid tool name.

    Provider APIs reject tool names outside ``^[a-zA-Z0-9_-]{1,64}$``;
    validating at registration surfaces the problem at build time instead of
    as a mid-conversation API error (e.g. unchecked tukuy skill names flowing
    in via ``extraction/tukuy_bridge.py``).
    """
    if not _TOOL_NAME_RE.match(name):
        raise ValueError(
            f"Invalid tool name {name!r}: must match ^[a-zA-Z0-9_-]{{1,64}}$ "
            "(1-64 characters; letters, digits, underscore, hyphen)."
        )


@dataclass
class ToolDefinition:
    """Describes a single callable tool the LLM can invoke.

    Attributes:
        name: Unique tool identifier.
        description: Human-readable description shown to the LLM.
        parameters: JSON Schema describing the function parameters.
        function: The Python callable to execute.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    function: Callable[..., Any]

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_openai_format(self, strict: bool = False) -> dict[str, Any]:
        """Serialise to OpenAI ``tools`` array element format.

        With ``strict=True`` the parameters schema is normalised for OpenAI
        strict tool use — ``additionalProperties: false`` on every object,
        every property key listed in ``required``, and parameters that were
        optional (i.e. have a default) made nullable — and ``"strict": true``
        is set on the function object.
        """
        parameters = _strict_parameters(self.parameters) if strict else self.parameters
        function: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "parameters": parameters,
        }
        if strict:
            function["strict"] = True
        return {"type": "function", "function": function}

    def to_anthropic_format(self) -> dict[str, Any]:
        """Serialise to Anthropic ``tools`` array element format.

        .. note::
            Maintenance-only: every built-in driver serialises tools from the
            OpenAI shape (:meth:`to_openai_format`), so this converter is kept
            for external consumers and is a candidate for future deprecation.
        """
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }

    @property
    def security_metadata(self) -> dict[str, Any] | None:
        """Return tukuy security metadata if the tool wraps a tukuy skill.

        Returns ``None`` for native Prompture tools.  For tukuy-backed
        tools, returns a dict with ``name``, ``description``,
        ``side_effects``, ``requires_network``, ``is_tukuy_skill``,
        and UI metadata fields from tukuy >= 0.0.20.
        """
        skill_obj = getattr(self.function, "__skill__", None)
        if skill_obj is None:
            return None
        desc = skill_obj.descriptor
        meta: dict[str, Any] = {
            "name": desc.name,
            "description": desc.description,
            "side_effects": getattr(desc, "side_effects", False),
            "requires_network": getattr(desc, "requires_network", False),
            "is_tukuy_skill": True,
        }
        # UI metadata (tukuy >= 0.0.20)
        if hasattr(desc, "resolved_risk_level"):
            rl = desc.resolved_risk_level
            meta["risk_level"] = rl.value if hasattr(rl, "value") else str(rl)
        if hasattr(desc, "resolved_display_name"):
            meta["display_name"] = desc.resolved_display_name
        if getattr(desc, "icon", None) is not None:
            meta["icon"] = desc.icon
        if getattr(desc, "group", None) is not None:
            meta["group"] = desc.group
        if getattr(desc, "hidden", False):
            meta["hidden"] = True
        if getattr(desc, "deprecated", None) is not None:
            meta["deprecated"] = desc.deprecated
        config_params = getattr(desc, "config_params", None)
        if config_params:
            meta["config_params"] = [cp.to_dict() for cp in config_params]
        return meta

    def to_prompt_format(self) -> str:
        """Plain-text description suitable for prompt-based tool calling."""
        desc_lines = self.description.strip().split("\n")
        lines = [f"Tool: {self.name}", f"  Description: {desc_lines[0]}"]
        # Keep continuation lines under the Description label so the block stays readable.
        lines.extend(f"    {line.strip()}" if line.strip() else "" for line in desc_lines[1:])
        lines.append("  Parameters:")
        props = self.parameters.get("properties", {})
        required = set(self.parameters.get("required", []))
        if not props:
            lines.append("    (none)")
        else:
            for pname, pschema in props.items():
                ptype = pschema.get("type", "string")
                req_label = "required" if pname in required else "optional"
                desc = pschema.get("description", "")
                line = f"    - {pname} ({ptype}, {req_label})"
                if desc:
                    line += f": {desc}"
                lines.append(line)
        return "\n".join(lines)


#: Google-style section headers recognised when splitting a docstring.
_ARGS_HEADERS = ("args", "arguments", "parameters")
_RETURNS_HEADERS = ("returns", "return")
_YIELDS_HEADERS = ("yields", "yield")
_SECTION_HEADERS = frozenset(
    _ARGS_HEADERS
    + _RETURNS_HEADERS
    + _YIELDS_HEADERS
    + (
        "raises",
        "raise",
        "exceptions",
        "examples",
        "example",
        "notes",
        "note",
        "warning",
        "warnings",
        "attributes",
        "see also",
        "references",
        "todo",
    )
)


def _section_key(line: str) -> str | None:
    """Return the normalised section name if *line* is a docstring section header."""
    stripped = line.strip()
    if not stripped.endswith(":"):
        return None
    key = stripped[:-1].strip().lower()
    return key if key in _SECTION_HEADERS else None


def _split_docstring(docstring: str | None) -> tuple[str, dict[str, list[str]]]:
    """Split a Google-style docstring into its lead text and named sections.

    Returns ``(lead, sections)`` where *lead* is everything before the first
    recognised section header (summary line plus any extended description) and
    *sections* maps a lower-cased section name to its still-indented lines.
    """
    if not docstring:
        return "", {}

    lead: list[str] = []
    sections: dict[str, list[str]] = {}
    current: list[str] | None = None

    for line in docstring.split("\n"):
        key = _section_key(line)
        if key is not None:
            current = sections.setdefault(key, [])
            continue
        if current is None:
            lead.append(line)
        else:
            current.append(line)

    return "\n".join(lead).strip(), sections


def _first_section(sections: dict[str, list[str]], *keys: str) -> list[str]:
    """Return the first present section among *keys* (empty list if none)."""
    for key in keys:
        if key in sections:
            return sections[key]
    return []


def _collapse(lines: list[str]) -> str:
    """Flatten a docstring section into a single space-joined paragraph."""
    return " ".join(stripped for line in lines if (stripped := line.strip()))


def _strip_numpy_sections(text: str) -> str:
    """Truncate *text* at the first NumPy-style section header (``Parameters``,
    ``Returns``, … — a word-only line followed by a dash underline)."""
    lines = text.split("\n")
    for i in range(len(lines) - 1):
        if re.match(r"^[A-Za-z][A-Za-z _]*$", lines[i].strip()) and _is_numpy_underline(lines[i + 1]):
            return "\n".join(lines[:i])
    return text


def _strip_rest_fields(text: str) -> str:
    """Drop reST field lines (``:param x:``, ``:return:``, …) from *text*."""
    return "\n".join(line for line in text.split("\n") if not _REST_FIELD_RE.match(line.strip()))


def _docstring_description(docstring: str | None) -> str:
    """Build the tool description the model sees from *docstring*.

    Keeps the summary line **and** the extended description that follows it —
    that prose is usually the only place a tool's scope, caveats, and intended
    use are written down.  ``Args:`` is dropped (it is already encoded in the
    parameter schema), while ``Returns:``/``Yields:`` are appended as a single
    line so the model knows what it gets back.  NumPy-style sections and reST
    field lines are stripped from the lead as well.
    """
    lead, sections = _split_docstring(docstring)
    lead = _strip_rest_fields(_strip_numpy_sections(lead)).strip()
    parts: list[str] = [lead] if lead else []

    for label, keys in (("Returns", _RETURNS_HEADERS), ("Yields", _YIELDS_HEADERS)):
        body = _collapse(_first_section(sections, *keys))
        if body:
            parts.append(f"{label}: {body}")

    return "\n\n".join(parts).strip()


def _truncate_description(text: str, limit: int | None) -> str:
    """Trim *text* to *limit* characters on a paragraph, sentence, or word boundary."""
    if limit is None or len(text) <= limit:
        return text
    head = text[: limit - 1]
    for sep in ("\n\n", ". ", " "):
        idx = head.rfind(sep)
        if idx > limit // 2:
            head = head[:idx]
            break
    return head.rstrip(" \n.,;:") + "…"


def _parse_google_params(docstring: str | None) -> dict[str, str]:
    """Extract parameter descriptions from a Google-style docstring ``Args:`` section."""
    _, sections = _split_docstring(docstring)
    lines = _first_section(sections, *_ARGS_HEADERS)
    if not lines:
        return {}

    params: dict[str, str] = {}
    current_param: str | None = None
    current_desc_parts: list[str] = []
    args_indent: int | None = None

    for line in lines:
        stripped = line.strip()

        # Blank lines inside Args are just spacing
        if not stripped:
            continue

        # Determine indentation level of Args entries
        content_indent = len(line) - len(line.lstrip())
        if args_indent is None and stripped:
            args_indent = content_indent

        # New parameter line: "param_name: description" or "param_name (type): description"
        if content_indent == args_indent and ":" in stripped:
            # Save previous param
            if current_param is not None:
                params[current_param] = " ".join(current_desc_parts).strip()
            # Parse "param_name: desc" or "param_name (type): desc"
            colon_idx = stripped.index(":")
            param_part = stripped[:colon_idx].strip()
            # Remove type annotation in parens: "param_name (str)"
            if " (" in param_part:
                param_part = param_part[: param_part.index(" (")]
            current_param = param_part
            current_desc_parts = [stripped[colon_idx + 1 :].strip()]
        elif current_param is not None and content_indent > (args_indent or 0):
            # Continuation line for current parameter
            current_desc_parts.append(stripped)

    # Save last parameter
    if current_param is not None:
        params[current_param] = " ".join(current_desc_parts).strip()

    return params


def _is_numpy_underline(line: str) -> bool:
    """True for a NumPy-style section underline (``----------``)."""
    stripped = line.strip()
    return len(stripped) >= 3 and set(stripped) == {"-"}


#: NumPy-style parameter entry header: ``name : type`` (``*``/``**`` allowed).
_NUMPY_PARAM_RE = re.compile(r"^\*{0,2}(\w+)\s*:")

#: reST-style parameter field: ``:param name: desc`` or ``:param type name: desc``.
_REST_PARAM_RE = re.compile(r"^:param\s+(?:[\w.\[\], ]+\s+)?(\w+)\s*:\s*(.*)$")

#: reST field lines that should not leak into the tool description.
_REST_FIELD_RE = re.compile(r"^:(param|type|return|rtype|raises?|yield|yields)\b")


def _parse_numpy_params(docstring: str | None) -> dict[str, str]:
    """Extract parameter descriptions from a NumPy-style ``Parameters`` section.

    Expects the canonical shape::

        Parameters
        ----------
        x : int
            Description of x.
    """
    if not docstring:
        return {}
    lines = docstring.split("\n")

    # Locate the "Parameters" header (word-only line followed by a dash underline).
    start = None
    for i in range(len(lines) - 1):
        if lines[i].strip().lower() == "parameters" and _is_numpy_underline(lines[i + 1]):
            start = i + 2
            break
    if start is None:
        return {}

    params: dict[str, str] = {}
    current_param: str | None = None
    current_desc_parts: list[str] = []
    base_indent: int | None = None

    def _flush() -> None:
        if current_param is not None:
            params[current_param] = " ".join(current_desc_parts).strip()

    for j in range(start, len(lines)):
        line = lines[j]
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())

        # The section ends at the next underline-headed section (e.g. "Returns").
        if (
            j + 1 < len(lines)
            and stripped
            and re.match(r"^[A-Za-z][A-Za-z _]*$", stripped)
            and _is_numpy_underline(lines[j + 1])
        ):
            break

        match = _NUMPY_PARAM_RE.match(stripped)
        if match and (base_indent is None or indent <= base_indent):
            if base_indent is None:
                base_indent = indent
            if indent == base_indent:
                _flush()
                current_param = match.group(1)
                current_desc_parts = []
                continue
        if current_param is not None and stripped:
            current_desc_parts.append(stripped)

    _flush()
    return params


def _parse_rest_params(docstring: str | None) -> dict[str, str]:
    """Extract parameter descriptions from reST-style ``:param x:`` fields."""
    if not docstring:
        return {}
    params: dict[str, str] = {}
    for line in docstring.split("\n"):
        match = _REST_PARAM_RE.match(line.strip())
        if match:
            params[match.group(1)] = match.group(2).strip()
    return params


def _parse_docstring_params(docstring: str | None) -> dict[str, str]:
    """Extract parameter descriptions from a docstring.

    Supports Google-style (``Args:``), NumPy-style (``Parameters`` followed by
    a dash underline) and reST-style (``:param x:``) docstrings, tried in that
    order; the first style that yields any descriptions wins.
    """
    params = _parse_google_params(docstring)
    if params:
        return params
    params = _parse_numpy_params(docstring)
    if params:
        return params
    return _parse_rest_params(docstring)


def _coerce_argument(tool_name: str, param: str, value: Any, annotation: Any) -> Any:
    """Coerce one tool argument to *annotation* via ``extraction.tools.convert_value``.

    Raises ``ValueError`` with an LLM-friendly message (naming the argument,
    the expected type and the received value) so the model can self-correct.
    """
    if get_origin(annotation) is Literal:
        allowed = get_args(annotation)
        if value in allowed:
            return value
        raise ValueError(
            f"Invalid value for argument '{param}' of '{tool_name}': expected one of {list(allowed)!r}, got {value!r}."
        )
    try:
        return convert_value(value, annotation, field_name=param, use_defaults_on_failure=False)
    except Exception as exc:
        expected = getattr(annotation, "__name__", None) or str(annotation)
        raise ValueError(
            f"Invalid value for argument '{param}' of '{tool_name}': expected {expected}, got {value!r} ({exc})."
        ) from exc


def _json_compatible(value: Any) -> Any:
    """Convert a coerced argument value into a JSON-compatible value for
    schema validation (Enums → values, datetimes → ISO strings, …)."""
    if isinstance(value, Enum):
        return _json_compatible(value.value)
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, BaseModel):
        return _json_compatible(value.model_dump())
    if is_dataclass(value) and not isinstance(value, type):
        return _json_compatible(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _json_compatible(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_compatible(v) for v in value]
    return value


def tool_from_function(
    fn: Callable[..., Any],
    *,
    name: str | None = None,
    description: str | None = None,
    max_description_chars: int | None = MAX_TOOL_DESCRIPTION_CHARS,
) -> ToolDefinition:
    """Build a :class:`ToolDefinition` by inspecting *fn*'s signature and docstring.

    Parameters:
        fn: The callable to wrap.
        name: Override the tool name (defaults to ``fn.__name__``).
        description: Override the description.  By default the docstring's
            summary *and* extended description are used, with ``Returns:``
            appended; ``Args:`` is omitted because it already lives in the
            parameter schema.
        max_description_chars: Truncate the description to this many characters
            (see :data:`MAX_TOOL_DESCRIPTION_CHARS`).  ``None`` disables it.
    """
    tool_name = name or fn.__name__
    raw_doc = inspect.getdoc(fn) or ""
    tool_desc = description or _docstring_description(raw_doc) or f"Call {tool_name}"
    tool_desc = _truncate_description(tool_desc, max_description_chars)
    param_docs = _parse_docstring_params(raw_doc)

    sig = inspect.signature(fn)
    try:
        hints = get_type_hints(fn)
    except Exception:
        hints = {}

    if not param_docs and any(
        pname != "self" and hints.get(pname, p.annotation) is not inspect.Parameter.empty
        for pname, p in sig.parameters.items()
    ):
        logger.debug(
            "No parseable parameter docs found for %r (Google/NumPy/reST styles supported); "
            "falling back to parameter names for descriptions.",
            tool_name,
        )

    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue
        annotation = hints.get(param_name, param.annotation)
        prop = _python_type_to_json_schema(annotation)

        # Use docstring description if available, else fall back to parameter name
        doc_desc = param_docs.get(param_name)
        if doc_desc:
            prop.setdefault("description", doc_desc)
        else:
            prop.setdefault("description", f"Parameter: {param_name}")

        properties[param_name] = prop

        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    parameters: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        parameters["required"] = required
    if not properties:
        # Some providers reject empty properties with required=[].
        # Omit required entirely when there are no parameters.
        parameters.pop("required", None)

    return ToolDefinition(
        name=tool_name,
        description=tool_desc,
        parameters=parameters,
        function=fn,
    )


@dataclass
class ToolRegistry:
    """A collection of :class:`ToolDefinition` instances.

    Supports decorator-based and explicit registration::

        registry = ToolRegistry()

        @registry.tool
        def my_func(x: int) -> str:
            ...

        registry.register(another_func)
    """

    _tools: dict[str, ToolDefinition] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        fn: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        max_description_chars: int | None = MAX_TOOL_DESCRIPTION_CHARS,
    ) -> ToolDefinition:
        """Register *fn* as a tool and return the :class:`ToolDefinition`.

        Raises:
            ValueError: If the tool name is not valid (``^[a-zA-Z0-9_-]{1,64}$``).
        """
        td = tool_from_function(fn, name=name, description=description, max_description_chars=max_description_chars)
        _validate_tool_name(td.name)
        self._tools[td.name] = td
        return td

    def tool(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Decorator to register a function as a tool.

        Returns the original function unchanged so it remains callable.
        """
        self.register(fn)
        return fn

    def add(self, tool_def: ToolDefinition) -> None:
        """Add a pre-built :class:`ToolDefinition`.

        Raises:
            ValueError: If the tool name is not valid (``^[a-zA-Z0-9_-]{1,64}$``).
        """
        _validate_tool_name(tool_def.name)
        self._tools[tool_def.name] = tool_def

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def __bool__(self) -> bool:
        return bool(self._tools)

    @property
    def names(self) -> list[str]:
        return list(self._tools.keys())

    @property
    def definitions(self) -> list[ToolDefinition]:
        return list(self._tools.values())

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def subset(self, names: set[str] | list[str]) -> ToolRegistry:
        """Return a new registry containing only the named tools.

        Raises:
            KeyError: If any name is not registered.
        """
        names_set = set(names)
        unknown = names_set - set(self._tools)
        if unknown:
            raise KeyError(f"Unknown tools: {', '.join(sorted(unknown))}")
        new = ToolRegistry()
        for n in names_set:
            new.add(self._tools[n])
        return new

    def filter(self, predicate: Callable[[ToolDefinition], bool]) -> ToolRegistry:
        """Return a new registry with tools matching *predicate*.

        Args:
            predicate: A callable that receives a :class:`ToolDefinition`
                and returns ``True`` to include the tool.
        """
        new = ToolRegistry()
        for td in self._tools.values():
            if predicate(td):
                new.add(td)
        return new

    def exclude(self, names: set[str] | list[str]) -> ToolRegistry:
        """Return a new registry without the named tools.

        Missing names are silently ignored (no error).
        """
        names_set = set(names)
        new = ToolRegistry()
        for name, td in self._tools.items():
            if name not in names_set:
                new.add(td)
        return new

    # ------------------------------------------------------------------
    # Tukuy integration
    # ------------------------------------------------------------------

    @classmethod
    def from_tukuy_skills(
        cls,
        skills: list[Any],
        *,
        config: dict[str, Any] | None = None,
    ) -> ToolRegistry:
        """Create a :class:`ToolRegistry` from a list of tukuy skills.

        Convenience factory that converts each tukuy ``Skill`` or
        ``@skill``-decorated function into a :class:`ToolDefinition`
        and registers it.

        Args:
            skills: List of tukuy ``Skill`` instances or ``@skill``-decorated functions.
            config: Optional config dict passed to each skill.

        Returns:
            A populated :class:`ToolRegistry`.
        """
        registry = cls()
        registry.add_tukuy_skills(skills, config=config)
        return registry

    def add_tukuy_skill(
        self,
        skill_or_fn: Any,
        *,
        name: str | None = None,
        description: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> ToolDefinition:
        """Register a tukuy ``Skill`` or ``@skill``-decorated function as a tool.

        Args:
            skill_or_fn: A tukuy ``Skill`` instance or ``@skill``-decorated function.
            name: Override the tool name.
            description: Override the tool description.
            config: Optional config dict injected as a ``SkillContext`` into
                ``invoke()`` / ``ainvoke()`` calls.  Used to provide
                ``llm_backend`` for instructions.

        Returns:
            The registered :class:`ToolDefinition`.
        """
        from ..extraction.tukuy_bridge import skill_to_tool_definition

        td = skill_to_tool_definition(skill_or_fn, config=config)
        if name:
            td = ToolDefinition(name=name, description=td.description, parameters=td.parameters, function=td.function)
        if description:
            td = ToolDefinition(name=td.name, description=description, parameters=td.parameters, function=td.function)
        _validate_tool_name(td.name)
        self._tools[td.name] = td
        return td

    def add_tukuy_skills(
        self,
        skills: list[Any],
        *,
        config: dict[str, Any] | None = None,
    ) -> list[ToolDefinition]:
        """Register multiple tukuy skills at once.

        Args:
            skills: List of tukuy ``Skill`` instances or ``@skill``-decorated functions.
            config: Optional config dict passed to each skill.

        Returns:
            List of registered :class:`ToolDefinition` instances.
        """
        return [self.add_tukuy_skill(s, config=config) for s in skills]

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_openai_format(self, strict: bool = False) -> list[dict[str, Any]]:
        """Serialise all tools to the OpenAI ``tools`` array format.

        With ``strict=True`` each tool's parameters schema is normalised for
        OpenAI strict tool use (see :meth:`ToolDefinition.to_openai_format`).
        """
        return [td.to_openai_format(strict=strict) for td in self._tools.values()]

    def to_anthropic_format(self) -> list[dict[str, Any]]:
        return [td.to_anthropic_format() for td in self._tools.values()]

    def to_prompt_format(self) -> str:
        """Join all tool descriptions into a single plain-text block."""
        return "\n\n".join(td.to_prompt_format() for td in self._tools.values())

    # ------------------------------------------------------------------
    # Argument validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_arguments(td: ToolDefinition, arguments: dict[str, Any]) -> str | None:
        """Return an LLM-friendly error message if required arguments are missing, else ``None``."""
        schema = td.parameters
        if not schema or not isinstance(schema, dict):
            return None
        required = schema.get("required", [])
        properties = schema.get("properties", {})
        missing = [p for p in required if p not in arguments]
        if not missing:
            return None
        parts = []
        for p in missing:
            prop = properties.get(p, {})
            ptype = prop.get("type", "any")
            desc = prop.get("description", "")
            detail = f"  - {p} ({ptype})"
            if desc:
                detail += f": {desc}"
            parts.append(detail)
        return (
            f"Missing required argument(s) for '{td.name}'. "
            f"You must provide:\n" + "\n".join(parts) + "\n"
            f"You sent: {json.dumps(arguments) if arguments else '{} (empty)'}"
        )

    @staticmethod
    def _coerce_and_validate_arguments(
        td: ToolDefinition, arguments: dict[str, Any]
    ) -> tuple[dict[str, Any], str | None]:
        """Coerce *arguments* to the function's annotated types and validate them.

        Returns ``(coerced_arguments, error)``.  *error* is an LLM-friendly
        message (``None`` on success) describing exactly which argument is
        wrong and what was expected, so the model can self-correct instead of
        a bare ``TypeError`` escaping from the tool function.
        """
        schema = td.parameters if isinstance(td.parameters, dict) else {}
        properties = schema.get("properties", {}) or {}

        # Reject unknown extra keys (only when the schema declares properties).
        if properties:
            unknown = [k for k in arguments if k not in properties]
            if unknown:
                return arguments, (
                    f"Unknown argument(s) for '{td.name}': {', '.join(sorted(unknown))}. "
                    f"Valid arguments are: {', '.join(properties)}. "
                    f"You sent: {json.dumps(arguments, default=str)}"
                )

        # Coerce each argument towards the function's type annotation.
        try:
            hints = get_type_hints(td.function)
        except Exception:
            hints = {}
        coerced = dict(arguments)
        for key, value in arguments.items():
            annotation = hints.get(key)
            if annotation is None or annotation is Any:
                continue
            try:
                coerced[key] = _coerce_argument(td.name, key, value, annotation)
            except ValueError as exc:
                return arguments, str(exc)

        # Validate the coerced arguments against the tool's JSON Schema.
        if jsonschema is not None and schema:
            instance = {k: _json_compatible(v) for k, v in coerced.items()}
            errors = sorted(
                jsonschema.Draft7Validator(schema).iter_errors(instance),
                key=lambda e: list(e.path),
            )
            if errors:
                details = []
                for e in errors[:5]:
                    where = "/".join(str(p) for p in e.path) or "(root)"
                    details.append(f"  - {where}: {e.message}")
                return arguments, (
                    f"Invalid argument(s) for '{td.name}':\n"
                    + "\n".join(details)
                    + "\nExpected arguments matching schema: "
                    + json.dumps(schema, default=str)
                    + f"\nYou sent: {json.dumps(arguments, default=str)}"
                )
        return coerced, None

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(self, name: str, arguments: dict[str, Any]) -> Any:
        """Execute a registered tool by name with the given arguments.

        Arguments are coerced to the function's annotated types and validated
        against the tool's JSON Schema; on failure an LLM-friendly error
        string is returned (so the model can self-correct) instead of a bare
        ``TypeError`` escaping.

        Raises:
            KeyError: If no tool with *name* is registered.
        """
        td = self._tools.get(name)
        if td is None:
            raise KeyError(f"Tool not registered: {name!r}")
        error = self._validate_arguments(td, arguments)
        if error:
            return error
        coerced, error = self._coerce_and_validate_arguments(td, arguments)
        if error:
            return error
        return td.function(**coerced)

    async def aexecute(self, name: str, arguments: dict[str, Any]) -> Any:
        """Execute a registered tool, awaiting async tool functions.

        Like :meth:`execute` but properly handles coroutine functions
        and awaitables.  Prefers a dedicated ``_async_fn`` attached by
        the tukuy bridge (which uses ``Skill.ainvoke()`` for correct
        timing and error handling).  Falls back to awaiting the raw
        return value when it is an awaitable.

        Raises:
            KeyError: If no tool with *name* is registered.
        """
        td = self._tools.get(name)
        if td is None:
            raise KeyError(f"Tool not registered: {name!r}")
        error = self._validate_arguments(td, arguments)
        if error:
            return error
        coerced, error = self._coerce_and_validate_arguments(td, arguments)
        if error:
            return error
        # Prefer dedicated async wrapper (set by tukuy bridge)
        async_fn = getattr(td.function, "_async_fn", None)
        if async_fn is not None:
            return await async_fn(**coerced)
        result = td.function(**coerced)
        if inspect.isawaitable(result):
            return await result
        return result
