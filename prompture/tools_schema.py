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
    tool_desc = description or (inspect.getdoc(fn) or "").split("\n")[0] or f"Call {tool_name}"

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

        # Use parameter name as description fallback
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
