"""MCP hub health rows for ``prompture doctor``.

Registers one capability (category ``mcp``) that yields a row per registered
server. Offline checks validate the entry, confirm every ``${ENV}`` reference
is set, and for stdio servers probe the launcher (``npx --version``,
``uvx --version``, ...) or check the command exists. ``live=True`` also opens
a session, initializes it and lists tools (bounded by the server's timeout).
"""

from __future__ import annotations

import importlib.util
import os
import shutil

from ..capabilities.health import HealthStatus, register_capability
from ..capabilities.probe import cached_probe
from .hub import MCPConfigError, MCPHub, MCPServerConfig, check_server

__all__ = ["check_mcp_servers", "check_server_health"]

# Launchers that answer ``--version`` without side effects. Any other stdio
# command is only checked for existence: running an unknown server binary with
# ``--version`` might start it.
_PROBE_SAFE = {"npx", "uvx", "uv", "node", "python", "python3", "py", "docker", "deno", "bunx", "bun", "pipx", "pnpm"}


def _mcp_installed() -> bool:
    try:
        return importlib.util.find_spec("mcp") is not None
    except (ImportError, ValueError):
        return False


def _command_row(cfg: MCPServerConfig, row: HealthStatus) -> HealthStatus:
    command = cfg.command or ""
    base = os.path.basename(command).lower().removesuffix(".exe").removesuffix(".cmd")
    if base in _PROBE_SAFE:
        probe = cached_probe(command, timeout=15.0)
        row.details["launcher"] = {"command": command, "status": probe.status, "version": probe.version}
        if probe.status != "ok":
            row.status = probe.status
            row.message = f"launcher `{command}` is {probe.status}"
            row.fix_hint = probe.hint
        return row
    if os.path.isabs(command) or os.sep in command or "/" in command:
        found = os.path.exists(command)
    else:
        found = shutil.which(command) is not None
    if not found:
        row.status = "missing"
        row.message = f"command `{command}` not found"
        row.fix_hint = f"Install `{command}` or fix the command with `prompture mcp add {cfg.name} --force ...`."
    return row


def check_server_health(cfg: MCPServerConfig, *, live: bool = False, mcp_available: bool | None = None) -> HealthStatus:
    """Health row for one server."""
    name = f"mcp:{cfg.name}"
    details = {"transport": cfg.transport, "target": cfg.target, "scope": cfg.scope}
    if not cfg.enabled:
        return HealthStatus(name, "skipped", category="mcp", message="disabled", details=details)
    try:
        cfg.validate()
    except MCPConfigError as exc:
        return HealthStatus(
            name,
            "error",
            category="mcp",
            message=str(exc),
            fix_hint=f"Fix or re-add it: `prompture mcp add {cfg.name} --force ...`",
            details=details,
        )
    available = _mcp_installed() if mcp_available is None else mcp_available
    if not available:
        return HealthStatus(
            name,
            "missing",
            category="mcp",
            message="the 'mcp' package is not installed",
            fix_hint="pip install 'prompture[mcp]'",
            details=details,
        )
    missing = cfg.missing_env()
    if missing:
        return HealthStatus(
            name,
            "unconfigured",
            category="mcp",
            message=f"needs environment variable(s): {', '.join(missing)}",
            fix_hint=f"Set {', '.join(missing)} in your environment or .env.",
            details={**details, "missing_env": missing},
        )
    row = HealthStatus(
        name,
        "ok",
        category="mcp",
        active_backend=cfg.transport,
        message="configured" if cfg.transport == "http" else "launcher available",
        details=details,
    )
    if cfg.transport == "stdio":
        row = _command_row(cfg, row)
        if row.status != "ok":
            return row
    if not live:
        return row
    result = check_server(cfg)
    row.details["elapsed_ms"] = result.get("elapsed_ms")
    row.details["tools"] = result.get("tools", [])
    if result["status"] == "ok":
        row.message = f"live: {result['tool_count']} tool(s)"
        row.details["tool_count"] = result["tool_count"]
        return row
    row.status = result["status"]
    row.message = f"live check failed: {result.get('message', '')}"
    row.fix_hint = f"Run `prompture mcp test {cfg.name}` for details."
    return row


def check_mcp_servers(live: bool = False, *, hub: MCPHub | None = None) -> list[HealthStatus]:
    """One row per registered server (or a single hint row when none are registered)."""
    hub = hub or MCPHub()
    rows: list[HealthStatus] = [
        HealthStatus("mcp:registry", "error", category="mcp", message=err, fix_hint="Fix the JSON or remove the entry.")
        for err in hub.errors
    ]
    servers = hub.list()
    if not servers:
        rows.append(
            HealthStatus(
                "mcp",
                "skipped",
                category="mcp",
                message="no MCP servers registered",
                fix_hint="Optional: `prompture mcp add --preset exa` for keyless web search.",
            )
        )
        return rows
    available = _mcp_installed()
    rows.extend(check_server_health(cfg, live=live, mcp_available=available) for cfg in servers)
    return rows


register_capability(
    "mcp_servers",
    "mcp",
    check_mcp_servers,
    description="Registered MCP servers (config, launcher, env refs; --live lists tools)",
)
