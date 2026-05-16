"""First-party Prompture tools ready to drop into a :class:`ToolRegistry`.

Each module here exports one or more tool builders that return a
:class:`prompture.ToolDefinition`.  Bring your own
:class:`~prompture.ToolRegistry` and add them with ``registry.add(tool)``.

Example::

    from prompture import ToolRegistry, DeepAgent
    from prompture.tools import PythonSandboxTool

    registry = ToolRegistry()
    registry.add(PythonSandboxTool().to_tool_definition())

    agent = DeepAgent(model="openai/gpt-4o", tools=registry)
"""

from .code_exec import PythonSandboxTool, python_execute_tool

__all__ = ["PythonSandboxTool", "python_execute_tool"]
