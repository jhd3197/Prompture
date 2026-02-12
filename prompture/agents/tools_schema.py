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
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, get_type_hints

logger = logging.getLogger("prompture.tools_schema")

# Mapping from Python types to JSON Schema types
_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _python_type_to_json_schema(annotation: Any) -> dict[str, Any]:
    """Convert a Python type annotation to a JSON Schema snippet."""
    if annotation is inspect.Parameter.empty or annotation is None:
        return {"type": "string"}

    # Handle Optional[X] (Union[X, None])
    origin = getattr(annotation, "__origin__", None)
    args = getattr(annotation, "__args__", ())

    if origin is type(None):
        return {"type": "string"}

    # Union types (Optional)
    if origin is not None and hasattr(origin, "__name__") and origin.__name__ == "Union":
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _python_type_to_json_schema(non_none[0])

    # list[X]
    if origin is list and args:
        return {"type": "array", "items": _python_type_to_json_schema(args[0])}

    # dict[str, X]
    if origin is dict:
        return {"type": "object"}

    # Simple types
    json_type = _TYPE_MAP.get(annotation, "string")
    return {"type": json_type}


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

    def to_openai_format(self) -> dict[str, Any]:
        """Serialise to OpenAI ``tools`` array element format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_anthropic_format(self) -> dict[str, Any]:
        """Serialise to Anthropic ``tools`` array element format."""
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
        lines = [f"Tool: {self.name}", f"  Description: {self.description}", "  Parameters:"]
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


def _parse_docstring_params(docstring: str | None) -> dict[str, str]:
    """Extract parameter descriptions from a Google-style docstring ``Args:`` section."""
    if not docstring:
        return {}
    lines = docstring.split("\n")
    params: dict[str, str] = {}
    in_args = False
    current_param: str | None = None
    current_desc_parts: list[str] = []
    args_indent: int | None = None

    for line in lines:
        stripped = line.strip()

        # Detect start of Args section
        if stripped in ("Args:", "Arguments:", "Parameters:"):
            in_args = True
            args_indent = None
            continue

        if not in_args:
            continue

        # Detect end of Args section (next section header like Returns:, Raises:, etc.)
        if stripped and not stripped.startswith("-") and stripped.endswith(":") and " " not in stripped:
            # Save last param
            if current_param is not None:
                params[current_param] = " ".join(current_desc_parts).strip()
            break

        # Empty line inside Args might end the section or just be spacing
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


def tool_from_function(
    fn: Callable[..., Any], *, name: str | None = None, description: str | None = None
) -> ToolDefinition:
    """Build a :class:`ToolDefinition` by inspecting *fn*'s signature and docstring.

    Parameters:
        fn: The callable to wrap.
        name: Override the tool name (defaults to ``fn.__name__``).
        description: Override the description (defaults to the first line of the docstring).
    """
    tool_name = name or fn.__name__
    raw_doc = inspect.getdoc(fn) or ""
    tool_desc = description or raw_doc.split("\n")[0] or f"Call {tool_name}"
    param_docs = _parse_docstring_params(raw_doc)

    sig = inspect.signature(fn)
    try:
        hints = get_type_hints(fn)
    except Exception:
        hints = {}

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
    ) -> ToolDefinition:
        """Register *fn* as a tool and return the :class:`ToolDefinition`."""
        td = tool_from_function(fn, name=name, description=description)
        self._tools[td.name] = td
        return td

    def tool(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Decorator to register a function as a tool.

        Returns the original function unchanged so it remains callable.
        """
        self.register(fn)
        return fn

    def add(self, tool_def: ToolDefinition) -> None:
        """Add a pre-built :class:`ToolDefinition`."""
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

    def add_tukuy_skill(
        self,
        skill_or_fn: Any,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> ToolDefinition:
        """Register a tukuy ``Skill`` or ``@skill``-decorated function as a tool.

        Args:
            skill_or_fn: A tukuy ``Skill`` instance or ``@skill``-decorated function.
            name: Override the tool name.
            description: Override the tool description.

        Returns:
            The registered :class:`ToolDefinition`.
        """
        from ..integrations.tukuy_bridge import skill_to_tool_definition

        td = skill_to_tool_definition(skill_or_fn)
        if name:
            td = ToolDefinition(name=name, description=td.description, parameters=td.parameters, function=td.function)
        if description:
            td = ToolDefinition(name=td.name, description=description, parameters=td.parameters, function=td.function)
        self._tools[td.name] = td
        return td

    def add_tukuy_skills(self, skills: list[Any]) -> list[ToolDefinition]:
        """Register multiple tukuy skills at once.

        Args:
            skills: List of tukuy ``Skill`` instances or ``@skill``-decorated functions.

        Returns:
            List of registered :class:`ToolDefinition` instances.
        """
        return [self.add_tukuy_skill(s) for s in skills]

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_openai_format(self) -> list[dict[str, Any]]:
        return [td.to_openai_format() for td in self._tools.values()]

    def to_anthropic_format(self) -> list[dict[str, Any]]:
        return [td.to_anthropic_format() for td in self._tools.values()]

    def to_prompt_format(self) -> str:
        """Join all tool descriptions into a single plain-text block."""
        return "\n\n".join(td.to_prompt_format() for td in self._tools.values())

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(self, name: str, arguments: dict[str, Any]) -> Any:
        """Execute a registered tool by name with the given arguments.

        Raises:
            KeyError: If no tool with *name* is registered.
        """
        td = self._tools.get(name)
        if td is None:
            raise KeyError(f"Tool not registered: {name!r}")
        return td.function(**arguments)

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
        # Prefer dedicated async wrapper (set by tukuy bridge)
        async_fn = getattr(td.function, "_async_fn", None)
        if async_fn is not None:
            return await async_fn(**arguments)
        result = td.function(**arguments)
        if inspect.isawaitable(result):
            return await result
        return result
