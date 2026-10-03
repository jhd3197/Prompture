"""``prompture mcp`` — manage named MCP servers.

Subcommands: ``add``, ``list``, ``remove``, ``enable``, ``disable``, ``test``
and ``import``. Mutating commands accept ``--dry-run``; ``list`` and ``test``
accept ``--json``. Secrets are stored only as ``${ENV_VAR}`` references and
resolved values are never printed.

Exit codes: ``0`` success, ``1`` a server test failed / nothing to do,
``2`` bad input (unknown server, invalid entry, inline secret refused).
"""

from __future__ import annotations

import json
import sys
from typing import Any

import click

from ..mcp.hub import (
    PRESETS,
    InlineSecretError,
    MCPConfigError,
    MCPHub,
    MCPServerConfig,
    check_server,
    get_preset,
    list_presets,
)
from ..mcp.importers import EDITORS, editor_config_paths, import_from_editor

__all__ = ["COMMANDS", "mcp"]


def _hub(project_dir: str | None = None) -> MCPHub:
    return MCPHub(project_dir=project_dir)


def _parse_pairs(values: tuple[str, ...], label: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in values:
        key, sep, val = item.partition("=")
        if not sep or not key.strip():
            raise click.BadParameter(f"expected KEY=VALUE, got {item!r}", param_hint=label)
        out[key.strip()] = val
    return out


def _fail(message: str, code: int = 2) -> None:
    click.echo(f"Error: {message}", err=True)
    sys.exit(code)


def _server_row(cfg: MCPServerConfig) -> dict[str, Any]:
    return {
        "name": cfg.name,
        "scope": cfg.scope,
        "transport": cfg.transport,
        "target": cfg.target,
        "enabled": cfg.enabled,
        "env": cfg.referenced_env(),
        "missing_env": cfg.missing_env(),
        "timeout": float(cfg.timeout),
        "preset": cfg.preset,
        "description": cfg.description,
    }


@click.group()
def mcp() -> None:
    """Manage named MCP servers (mount with tools=["mcp:<name>"])."""


@mcp.command("add")
@click.argument("name", required=False)
@click.option("--preset", type=click.Choice(sorted(PRESETS)), help="Start from a built-in preset.")
@click.option("--url", help="Streamable HTTP endpoint (http transport).")
@click.option("--command", "command_", help="Executable to launch (stdio transport).")
@click.option("--arg", "args", multiple=True, help="Argument for --command (repeatable; for presets: appended).")
@click.option("--env", "env", multiple=True, help="KEY=VALUE for stdio servers; use ${VAR} for secrets (repeatable).")
@click.option("--header", "headers", multiple=True, help="KEY=VALUE HTTP header; use ${VAR} for secrets (repeatable).")
@click.option("--transport", type=click.Choice(["stdio", "http"]), help="Inferred from --url / --command.")
@click.option("--timeout", type=float, help="Seconds per call (default 60).")
@click.option("--cwd", help="Working directory for stdio servers.")
@click.option("--description", default="", help="Shown by `prompture mcp list`.")
@click.option("--disabled", is_flag=True, help="Register but keep disabled.")
@click.option("--project", is_flag=True, help="Write to ./.prompture/mcp.json instead of the user registry.")
@click.option("--force", is_flag=True, help="Overwrite an existing entry with the same name.")
@click.option(
    "--allow-inline-secret",
    is_flag=True,
    help="Store literal secret-looking values anyway (not recommended; prefer ${VAR}).",
)
@click.option("--dry-run", is_flag=True, help="Show what would be written without writing.")
def add_cmd(
    name: str | None,
    preset: str | None,
    url: str | None,
    command_: str | None,
    args: tuple[str, ...],
    env: tuple[str, ...],
    headers: tuple[str, ...],
    transport: str | None,
    timeout: float | None,
    cwd: str | None,
    description: str,
    disabled: bool,
    project: bool,
    force: bool,
    allow_inline_secret: bool,
    dry_run: bool,
) -> None:
    """Register an MCP server.

    \b
    Examples:
      prompture mcp add --preset exa
      prompture mcp add fs --preset filesystem --arg ./docs
      prompture mcp add search --url https://mcp.example.com/mcp --header "Authorization=Bearer ${SEARCH_TOKEN}"
      prompture mcp add local --command python --arg -m --arg my_server
    """
    if preset:
        cfg = get_preset(preset, name=name, extra_args=list(args) or None)
        if url:
            cfg.url = url
        if command_:
            cfg.command = command_
    else:
        if not name:
            _fail("give a NAME (or use --preset).")
        if not url and not command_:
            _fail("give --url (http) or --command (stdio), or use --preset.")
        cfg = MCPServerConfig(
            name=name or "",
            transport=transport or ("http" if url else "stdio"),
            command=command_,
            args=list(args),
            url=url,
        )
    if transport:
        cfg.transport = transport
    try:
        cfg.env.update(_parse_pairs(env, "--env"))
        cfg.headers.update(_parse_pairs(headers, "--header"))
    except click.BadParameter as exc:
        _fail(exc.format_message())
    if timeout is not None:
        cfg.timeout = timeout
    if cwd:
        cfg.cwd = cwd
    if description:
        cfg.description = description
    if disabled:
        cfg.enabled = False

    hub = _hub()
    scope = "project" if project else "user"
    try:
        path = hub.add(cfg, scope=scope, replace=force, allow_inline_secrets=allow_inline_secret, dry_run=dry_run)
    except InlineSecretError as exc:
        _fail(str(exc))
    except MCPConfigError as exc:
        _fail(str(exc))
    else:
        if allow_inline_secret:
            click.echo("Warning: literal secret values were stored in the registry file.", err=True)
        prefix = "[dry-run] would add" if dry_run else "Added"
        click.echo(f"{prefix} MCP server '{cfg.name}' ({cfg.transport}: {cfg.target}) to {path}")
        if dry_run:
            click.echo(json.dumps({cfg.name: cfg.to_dict()}, indent=2))
        spec = PRESETS.get(cfg.preset or "", {})
        if spec.get("requires"):
            click.echo(f"Needs `{spec['requires']}` on PATH.")
        missing = cfg.missing_env()
        if missing:
            click.echo(f"Set before use: {', '.join(missing)}")
        click.echo(f'Use it: Agent(..., tools=["mcp:{cfg.name}"])  |  check: prompture mcp test {cfg.name}')


@mcp.command("list")
@click.option("--json", "as_json", is_flag=True, help="Machine-readable output.")
@click.option("--presets", is_flag=True, help="List built-in presets instead of registered servers.")
def list_cmd(as_json: bool, presets: bool) -> None:
    """List registered servers (project entries override user ones)."""
    if presets:
        rows = list_presets()
        if as_json:
            click.echo(json.dumps(rows, indent=2))
            return
        for row in rows:
            req = f"  [needs {row['requires']}]" if row["requires"] else ""
            env = f"  [env: {', '.join(row['env'])}]" if row["env"] else ""
            click.echo(f"{row['name']:<20} {row['transport']:<6} {row['description']}{req}{env}")
        return
    hub = _hub()
    servers = [_server_row(c) for c in hub.list()]
    if as_json:
        click.echo(
            json.dumps(
                {
                    "user_registry": str(hub.user_path),
                    "project_registry": str(hub.project_path),
                    "servers": servers,
                    "errors": hub.errors,
                },
                indent=2,
            )
        )
        return
    for err in hub.errors:
        click.echo(f"Warning: {err}", err=True)
    if not servers:
        click.echo("No MCP servers registered. Try: prompture mcp add --preset exa  (see --presets)")
        return
    for row in servers:
        state = "" if row["enabled"] else " (disabled)"
        missing = f"  [missing env: {', '.join(row['missing_env'])}]" if row["missing_env"] else ""
        click.echo(f"{row['name']:<20} {row['scope']:<8} {row['transport']:<6} {row['target']}{state}{missing}")


@mcp.command("remove")
@click.argument("name")
@click.option("--project", "scope", flag_value="project", help="Remove from the project registry.")
@click.option("--user", "scope", flag_value="user", help="Remove from the user registry.")
@click.option("--dry-run", is_flag=True, help="Show what would be removed without writing.")
def remove_cmd(name: str, scope: str | None, dry_run: bool) -> None:
    """Remove a registered server."""
    hub = _hub()
    try:
        path = hub.remove(name, scope=scope, dry_run=dry_run)
    except MCPConfigError as exc:
        _fail(str(exc))
        return
    if path is None:
        _fail(f"no MCP server named {name!r}" + (f" in the {scope} registry" if scope else ""))
    prefix = "[dry-run] would remove" if dry_run else "Removed"
    click.echo(f"{prefix} MCP server '{name}' from {path}")


def _toggle(name: str, enabled: bool, dry_run: bool) -> None:
    hub = _hub()
    try:
        path = hub.set_enabled(name, enabled, dry_run=dry_run)
    except MCPConfigError as exc:
        _fail(str(exc))
        return
    word = "enable" if enabled else "disable"
    prefix = f"[dry-run] would {word}" if dry_run else f"{word.capitalize()}d"
    click.echo(f"{prefix} MCP server '{name}' in {path}")


@mcp.command("enable")
@click.argument("name")
@click.option("--dry-run", is_flag=True)
def enable_cmd(name: str, dry_run: bool) -> None:
    """Enable a registered server."""
    _toggle(name, True, dry_run)


@mcp.command("disable")
@click.argument("name")
@click.option("--dry-run", is_flag=True)
def disable_cmd(name: str, dry_run: bool) -> None:
    """Disable a registered server (kept in the registry)."""
    _toggle(name, False, dry_run)


@mcp.command("test")
@click.argument("names", nargs=-1)
@click.option("--json", "as_json", is_flag=True, help="Machine-readable output.")
@click.option("--timeout", type=float, help="Override the per-server timeout (seconds).")
def test_cmd(names: tuple[str, ...], as_json: bool, timeout: float | None) -> None:
    """Connect, initialize and list tools (all enabled servers when no NAME)."""
    hub = _hub()
    try:
        configs = [hub.require(n) for n in names] if names else hub.list(include_disabled=False)
    except MCPConfigError as exc:
        _fail(str(exc))
        return
    if not configs:
        if as_json:
            click.echo(json.dumps({"results": []}, indent=2))
        else:
            click.echo("No enabled MCP servers to test. Try: prompture mcp add --preset exa")
        sys.exit(1)
    results = [check_server(cfg, timeout=timeout) for cfg in configs]
    if as_json:
        click.echo(json.dumps({"results": results}, indent=2))
    else:
        for res in results:
            click.echo(
                f"{res['name']:<20} {res['status']:<12} {res.get('message', '')} ({res.get('elapsed_ms', 0)} ms)"
            )
            if res["status"] == "ok" and res["tools"]:
                shown = ", ".join(res["tools"][:20])
                more = f" ... (+{len(res['tools']) - 20})" if len(res["tools"]) > 20 else ""
                click.echo(f"  tools: {shown}{more}")
    if any(r["status"] != "ok" for r in results):
        sys.exit(1)


@mcp.command("import")
@click.option("--from", "editor", required=True, type=click.Choice(sorted(EDITORS)), help="Editor to import from.")
@click.option("--path", "path", type=click.Path(dir_okay=False), help="Explicit config file to read.")
@click.option("--yes", is_flag=True, help="Skip the confirmation prompt.")
@click.option("--project", is_flag=True, help="Write to ./.prompture/mcp.json instead of the user registry.")
@click.option("--force", is_flag=True, help="Overwrite servers that already exist.")
@click.option("--dry-run", is_flag=True, help="Show what would be imported without writing.")
def import_cmd(editor: str, path: str | None, yes: bool, project: bool, force: bool, dry_run: bool) -> None:
    """Import MCP servers from an editor's config (asks first; never automatic).

    Inline secrets are converted to ${ENV_VAR} references; you are told which
    variables to set. Secret values are never printed or stored.
    """
    candidates = [path] if path else [str(p) for p in editor_config_paths(editor)]
    label = EDITORS[editor]
    if not yes:
        click.echo(f"This reads {label}'s MCP configuration from:")
        for cand in candidates:
            click.echo(f"  {cand}")
        if not click.confirm("Read it now?", default=False):
            click.echo("Cancelled; nothing was read.")
            sys.exit(1)
    try:
        result = import_from_editor(editor, path=path)
    except MCPConfigError as exc:
        _fail(str(exc))
        return
    if result.source is None:
        _fail(f"no {label} MCP config found (looked in: {', '.join(candidates)})", code=1)
    click.echo(f"Read {result.source}")
    for skipped_name, reason in result.skipped:
        click.echo(f"  skip {skipped_name}: {reason}")
    if not result.servers:
        click.echo("No importable servers found.")
        sys.exit(1)

    hub = _hub()
    scope = "project" if project else "user"
    added, existing = [], []
    for cfg in result.servers:
        if hub.get(cfg.name) is not None and hub.scope_of(cfg.name) == scope and not force:
            existing.append(cfg.name)
            continue
        try:
            target = hub.add(cfg, scope=scope, replace=force, dry_run=dry_run)
        except MCPConfigError as exc:
            click.echo(f"  skip {cfg.name}: {exc}")
            continue
        added.append(cfg.name)
        prefix = "[dry-run] would import" if dry_run else "imported"
        click.echo(f"  {prefix} {cfg.name} ({cfg.transport}: {cfg.target}) -> {target}")
    for name in existing:
        click.echo(f"  exists {name} (use --force to overwrite)")
    if result.converted:
        click.echo("Inline secrets were replaced with ${VAR} references:")
        for item in result.converted:
            click.echo(f"  {item}")
    if result.env_to_set:
        click.echo(f"Set these environment variables before use: {', '.join(result.env_to_set)}")
    if not added and existing:
        sys.exit(1)


COMMANDS = [mcp]
