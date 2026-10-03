"""Opt-in import of MCP servers from an editor's own config file.

Nothing here runs implicitly: the CLI asks for confirmation before reading
another application's config (``prompture mcp import --from <editor>``), and
:func:`import_from_editor` only reads the file it is pointed at.

Inline secrets found in the imported ``env`` / ``headers`` / URL / args are
replaced with ``${ENV_VAR}`` references so they never land in Prompture's
registry; :class:`ImportResult.env_to_set` lists the variables the user must
export afterwards. Secret values are never returned or printed.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .hub import (
    _ENV_REF_RE,
    _SECRET_FLAG_RE,
    _SECRET_VALUE_RES,
    MCPConfigError,
    MCPServerConfig,
    looks_like_secret,
)

__all__ = ["EDITORS", "ImportResult", "convert_editor_entry", "editor_config_paths", "import_from_editor"]

#: Supported editors and a human label.
EDITORS: dict[str, str] = {
    "claude-desktop": "Claude Desktop",
    "claude-code": "Claude Code",
    "cursor": "Cursor",
    "vscode": "VS Code",
    "windsurf": "Windsurf",
}

_SECRET_PARAM_NAMES = {"token", "access_token", "api_key", "apikey", "api-key", "key", "secret", "password", "auth"}
_INPUT_REF_RE = re.compile(r"\$\{input:([^}]+)\}")
_VSCODE_ENV_REF_RE = re.compile(r"\$\{env:([A-Za-z_][A-Za-z0-9_]*)\}")


@dataclass
class ImportResult:
    """Outcome of reading one editor config."""

    editor: str
    source: Path | None
    servers: list[MCPServerConfig] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)
    env_to_set: list[str] = field(default_factory=list)
    converted: list[str] = field(default_factory=list)


def _appdata() -> Path:
    return Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")


def editor_config_paths(editor: str, project_dir: str | Path | None = None) -> list[Path]:
    """Candidate config files for *editor*, most specific first (existing or not)."""
    project = Path(project_dir or Path.cwd())
    home = Path.home()
    if editor == "claude-desktop":
        if sys.platform == "win32":
            return [_appdata() / "Claude" / "claude_desktop_config.json"]
        if sys.platform == "darwin":
            return [home / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"]
        return [home / ".config" / "Claude" / "claude_desktop_config.json"]
    if editor == "claude-code":
        return [project / ".mcp.json", home / ".claude.json"]
    if editor == "cursor":
        return [project / ".cursor" / "mcp.json", home / ".cursor" / "mcp.json"]
    if editor == "vscode":
        return [project / ".vscode" / "mcp.json"]
    if editor == "windsurf":
        return [home / ".codeium" / "windsurf" / "mcp_config.json"]
    raise MCPConfigError(f"Unknown editor {editor!r}. Known: {', '.join(EDITORS)}")


def _sanitize_server_name(name: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_-]+", "-", name.strip()).strip("-_")
    clean = re.sub(r"_{2,}", "_", clean)
    return (clean or "server")[:40]


def _env_name(*parts: str) -> str:
    raw = "_".join(p for p in parts if p)
    name = re.sub(r"[^A-Za-z0-9]+", "_", raw).strip("_").upper()
    if not name or name[0].isdigit():
        name = f"MCP_{name}"
    return name


def _normalize_refs(value: str) -> str:
    """Turn editor-specific placeholders into ``${VAR}`` references."""
    value = _VSCODE_ENV_REF_RE.sub(lambda m: "${" + m.group(1) + "}", value)
    return _INPUT_REF_RE.sub(lambda m: "${" + _env_name(m.group(1)) + "}", value)


class _Converter:
    def __init__(self, server: str) -> None:
        self.server = server
        self.env_to_set: list[str] = []
        self.converted: list[str] = []

    def _need(self, var: str) -> None:
        if var not in self.env_to_set:
            self.env_to_set.append(var)

    def env(self, env: dict[str, Any]) -> dict[str, str]:
        out: dict[str, str] = {}
        for key, val in env.items():
            sval = _normalize_refs(str(val))
            if looks_like_secret(key, sval):
                var = key if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key) else _env_name(self.server, key)
                out[key] = "${" + var + "}"
                self.converted.append(f"{self.server}: env.{key}")
                self._need(var)
            else:
                out[key] = sval
        return out

    def headers(self, headers: dict[str, Any]) -> dict[str, str]:
        out: dict[str, str] = {}
        for key, val in headers.items():
            sval = _normalize_refs(str(val))
            if looks_like_secret(key, sval):
                scheme, _, rest = sval.partition(" ")
                var = _env_name(self.server, "token" if key.lower() == "authorization" else key)
                if rest and scheme.lower() in ("bearer", "basic", "token"):
                    out[key] = f"{scheme} ${{{var}}}"
                else:
                    out[key] = "${" + var + "}"
                self.converted.append(f"{self.server}: headers.{key}")
                self._need(var)
            else:
                out[key] = sval
        return out

    def url(self, url: str) -> str:
        url = _normalize_refs(url)
        parts = urlsplit(url)
        changed = False
        query = []
        for k, v in parse_qsl(parts.query, keep_blank_values=True):
            if k.lower() in _SECRET_PARAM_NAMES and v and not _ENV_REF_RE.search(v):
                var = _env_name(self.server, k)
                query.append((k, "${" + var + "}"))
                self._need(var)
                self.converted.append(f"{self.server}: url query {k}")
                changed = True
            else:
                query.append((k, v))
        netloc = parts.netloc
        if "@" in netloc:
            netloc = netloc.rsplit("@", 1)[1]
            changed = True
            self.converted.append(f"{self.server}: url credentials dropped (set headers instead)")
        if not changed:
            return url
        encoded = urlencode(query, safe="${}")
        return urlunsplit((parts.scheme, netloc, parts.path, encoded, parts.fragment))

    def args(self, args: list[Any]) -> list[str]:
        out: list[str] = []
        prev = ""
        for i, raw in enumerate(args):
            arg = _normalize_refs(str(raw))
            if arg.startswith("-") and "=" in arg:
                flag, _, val = arg.partition("=")
                if _SECRET_FLAG_RE.match(flag) and val and not _ENV_REF_RE.search(val):
                    var = _env_name(self.server, flag.lstrip("-"))
                    arg = f"{flag}=${{{var}}}"
                    self._need(var)
                    self.converted.append(f"{self.server}: args[{i}]")
            elif (_SECRET_FLAG_RE.match(prev) and not _ENV_REF_RE.search(arg)) or (
                not _ENV_REF_RE.search(arg) and any(r.match(arg) for r in _SECRET_VALUE_RES)
            ):
                var = _env_name(self.server, prev.lstrip("-") if _SECRET_FLAG_RE.match(prev) else "secret")
                arg = "${" + var + "}"
                self._need(var)
                self.converted.append(f"{self.server}: args[{i}]")
            out.append(arg)
            prev = str(raw)
        return out


def convert_editor_entry(name: str, entry: Any) -> tuple[MCPServerConfig | None, str | None, list[str], list[str]]:
    """Convert one editor ``mcpServers`` entry.

    Returns ``(config | None, skip_reason | None, env_vars_to_set, converted_fields)``.
    """
    if not isinstance(entry, dict):
        return None, "entry is not an object", [], []
    server = _sanitize_server_name(name)
    conv = _Converter(server)
    kind = str(entry.get("type") or entry.get("transport") or "").lower()
    url = entry.get("url") or entry.get("serverUrl") or entry.get("httpUrl")
    if kind == "sse" or (not kind and url and str(url).rstrip("/").endswith("/sse")):
        return None, "SSE transport is not supported (use the server's streamable HTTP endpoint)", [], []
    if entry.get("disabled") is True and not url and not entry.get("command"):
        return None, "disabled and incomplete", [], []
    try:
        if url and kind in ("", "http", "streamable-http", "streamablehttp"):
            cfg = MCPServerConfig(
                name=server,
                transport="http",
                url=conv.url(str(url)),
                headers=conv.headers(dict(entry.get("headers") or {})),
            )
        elif entry.get("command"):
            cfg = MCPServerConfig(
                name=server,
                transport="stdio",
                command=_normalize_refs(str(entry["command"])),
                args=conv.args(list(entry.get("args") or [])),
                env=conv.env(dict(entry.get("env") or {})),
                cwd=_normalize_refs(str(entry["cwd"])) if entry.get("cwd") else None,
            )
        else:
            return None, f"unsupported entry (type={kind or 'unknown'})", [], []
        if entry.get("disabled") is True:
            cfg.enabled = False
        cfg.validate()
    except MCPConfigError as exc:
        return None, str(exc), [], []
    for var in cfg.referenced_env():
        if var not in conv.env_to_set and not os.environ.get(var):
            conv._need(var)
    return cfg, None, conv.env_to_set, conv.converted


def _entries_from(editor: str, data: Any, project_dir: Path) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    if editor == "vscode":
        servers = data.get("servers")
        return servers if isinstance(servers, dict) else {}
    entries: dict[str, Any] = {}
    top = data.get("mcpServers")
    if isinstance(top, dict):
        entries.update(top)
    if editor == "claude-code":
        projects = data.get("projects")
        if isinstance(projects, dict):
            for key in (str(project_dir), str(project_dir).replace("\\", "/")):
                proj = projects.get(key)
                if isinstance(proj, dict) and isinstance(proj.get("mcpServers"), dict):
                    entries.update(proj["mcpServers"])
    return entries


def import_from_editor(
    editor: str,
    *,
    path: str | Path | None = None,
    project_dir: str | Path | None = None,
) -> ImportResult:
    """Read *editor*'s MCP config and convert it. Call only after the user agreed.

    Args:
        editor: One of :data:`EDITORS`.
        path: Explicit config file (default: the editor's first existing candidate).
        project_dir: Project directory for project-level configs (default: cwd).
    """
    if editor not in EDITORS:
        raise MCPConfigError(f"Unknown editor {editor!r}. Known: {', '.join(EDITORS)}")
    project = Path(project_dir or Path.cwd())
    if path is not None:
        source: Path | None = Path(path)
    else:
        source = next((p for p in editor_config_paths(editor, project) if p.is_file()), None)
    result = ImportResult(editor=editor, source=source)
    if source is None or not source.is_file():
        return result
    try:
        data = json.loads(source.read_text(encoding="utf-8") or "{}")
    except (OSError, json.JSONDecodeError) as exc:
        raise MCPConfigError(f"Cannot read {source}: {exc}") from exc
    seen: set[str] = set()
    for name, entry in _entries_from(editor, data, project).items():
        cfg, reason, env_vars, converted = convert_editor_entry(str(name), entry)
        if cfg is None:
            result.skipped.append((str(name), reason or "unsupported"))
            continue
        if cfg.name in seen:
            result.skipped.append((str(name), f"duplicate name after sanitizing ({cfg.name})"))
            continue
        seen.add(cfg.name)
        cfg.description = cfg.description or f"imported from {EDITORS[editor]}"
        result.servers.append(cfg)
        result.converted.extend(converted)
        for var in env_vars:
            if var not in result.env_to_set:
                result.env_to_set.append(var)
    return result
