"""Doctor rows (category ``tools``) for CLI tools, registered on import.

One ``cli:<name>`` row per shipped and user-defined tool, plus ``cli:config``
when a ``.prompture/tools.*`` file exists. Checks are offline unless
``live=True`` (which may run a tool's ``live_check_args``, e.g. ``gh auth status``).
"""

from __future__ import annotations

from ...capabilities.health import HealthStatus, register_capability
from . import all_cli_tools
from .config import config_errors, config_paths


def _tool_check(name: str):
    def _check(live: bool) -> HealthStatus:
        tool = all_cli_tools().get(name)
        if tool is None:
            return HealthStatus(f"cli:{name}", "skipped", category="tools", message="definition removed")
        return tool.check(live)

    return _check


def _config_check(live: bool) -> HealthStatus:
    files = config_paths()
    if not files:
        return HealthStatus("cli:config", "skipped", category="tools", message="no .prompture/tools.yaml")
    tools = all_cli_tools()
    errors = config_errors()
    shown = ", ".join(str(f) for f in files)
    if errors:
        return HealthStatus(
            "cli:config",
            "error",
            category="tools",
            message=f"{len(errors)} problem(s) in {shown}: {errors[0][:160]}",
            fix_hint="Fix the listed definitions; invalid tools are skipped.",
            details={"files": [str(f) for f in files], "errors": errors, "tools": sorted(tools)},
        )
    return HealthStatus(
        "cli:config",
        "ok",
        category="tools",
        message=f"loaded {shown}",
        details={"files": [str(f) for f in files], "tools": sorted(tools)},
    )


def register_cli_capabilities() -> list[str]:
    """(Re)register a doctor row for every known CLI tool; returns the row names."""
    names: list[str] = []
    for name, tool in all_cli_tools().items():
        row = f"cli:{name}"
        register_capability(row, "tools", _tool_check(name), description=tool.description)
        names.append(row)
    if config_paths():
        register_capability("cli:config", "tools", _config_check, description="User CLI tool definitions")
        names.append("cli:config")
    return names


register_cli_capabilities()
