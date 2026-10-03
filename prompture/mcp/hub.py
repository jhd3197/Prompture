"""MCP hub — named, health-checked MCP servers mountable by name.

Servers live in a small JSON registry:

* user file ``~/.prompture/mcp.json`` (override with ``PROMPTURE_MCP_CONFIG``)
* project file ``.prompture/mcp.json`` in the working directory, which
  overrides user entries with the same name.

Each entry names a transport (``stdio`` or ``http``), a command or URL, and
optional env / headers. Secrets are written as ``${ENV_VAR}`` references and
resolved only when a session is opened; the registry never stores them and
nothing here prints a resolved value.

Agents mount servers by name::

    Agent("openai/gpt-4o-mini", tools=["mcp:search"])
    Agent("openai/gpt-4o-mini", tools=["mcp:github/search_issues,get_issue"])

Sessions are opened lazily and pooled: each server gets one long-lived session
on a background event-loop thread, so both sync ``Agent.run`` and async agents
can call its tools. Tool names are prefixed ``<server>__<tool>`` to avoid
collisions between servers. Every call records a usage event (server, tool,
duration, ok/error) through the global usage tracker.
"""

from __future__ import annotations

import asyncio
import atexit
import concurrent.futures
import contextlib
import hashlib
import json
import logging
import os
import re
import tempfile
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterable, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from ..agents.tools_schema import ToolDefinition, ToolRegistry
from ..integrations.mcp_bridge import _mcp_field, _serialize_call_result
from ..security.redaction import scrub_secrets

logger = logging.getLogger("prompture.mcp.hub")

__all__ = [
    "DEFAULT_TIMEOUT",
    "PRESETS",
    "USER_CONFIG_ENV",
    "InlineSecretError",
    "MCPConfigError",
    "MCPConnectionError",
    "MCPHub",
    "MCPHubError",
    "MCPServerConfig",
    "MCPSessionPool",
    "MCPTimeoutError",
    "MissingEnvError",
    "acheck_server",
    "check_server",
    "env_refs",
    "find_inline_secrets",
    "get_pool",
    "get_preset",
    "list_presets",
    "load_mcp_registry_by_name",
    "load_mcp_registry_by_name_sync",
    "looks_like_secret",
    "open_session",
    "prefixed_tool_name",
    "project_registry_path",
    "resolve_env_refs",
    "resolve_mcp_tools",
    "set_pool",
    "user_registry_path",
]

TRANSPORTS = ("stdio", "http")
USER_CONFIG_ENV = "PROMPTURE_MCP_CONFIG"
REGISTRY_VERSION = 1
DEFAULT_TIMEOUT = 60.0
# First ``npx -y`` / ``uvx`` launch may download the server package.
STDIO_STARTUP_TIMEOUT = 120.0
TOOL_PREFIX_SEP = "__"

_SERVER_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,39}$")
_TOOL_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_INVALID_TOOL_CHARS = re.compile(r"[^a-zA-Z0-9_-]")
_ENV_REF_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Keys whose literal values are treated as secrets.
_SECRET_KEY_RE = re.compile(
    r"(api[_-]?key|apikey|token|secret|passw(or)?d|pwd|auth|bearer|credential|cookie|session|private[_-]?key|access[_-]?key)",
    re.IGNORECASE,
)
# Value shapes that look like credentials even under an innocent key.
_SECRET_VALUE_RES = (
    re.compile(r"^(Bearer|Basic|Token)\s+\S{8,}$", re.IGNORECASE),
    re.compile(r"^(sk|pk|rk)-[A-Za-z0-9_-]{16,}$"),
    re.compile(r"^(ghp|gho|ghu|ghs|ghr|github_pat|glpat|xox[abprs])[-_][A-Za-z0-9_-]{10,}$"),
    re.compile(r"^AKIA[0-9A-Z]{16}$"),
    re.compile(r"^AIza[0-9A-Za-z_-]{30,}$"),
    re.compile(r"^eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]*$"),
)
_SECRET_FLAG_RE = re.compile(
    r"^--?(api[-_]?key|token|access[-_]?token|secret|password|auth(orization)?)$", re.IGNORECASE
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class MCPHubError(Exception):
    """Base error for the MCP hub. Messages are always scrubbed."""

    def __init__(self, message: str) -> None:
        super().__init__(scrub_secrets(message))


class MCPConfigError(MCPHubError, ValueError):
    """The registry or a server entry is invalid."""


class InlineSecretError(MCPConfigError):
    """A literal secret was about to be written to the registry."""

    def __init__(self, message: str, *, fields: list[str]) -> None:
        super().__init__(message)
        self.fields = fields


class MissingEnvError(MCPConfigError):
    """A ``${ENV}`` reference has no value in the environment."""

    def __init__(self, message: str, *, missing: list[str]) -> None:
        super().__init__(message)
        self.missing = missing


class MCPConnectionError(MCPHubError):
    """A server could not be started / reached, or its session died."""


class MCPTimeoutError(MCPHubError, TimeoutError):
    """A server did not start or answer within its timeout."""


# ---------------------------------------------------------------------------
# ${ENV} references and secret detection
# ---------------------------------------------------------------------------


def env_refs(value: Any) -> list[str]:
    """Names of the ``${VAR}`` references in *value* (strings, lists, dicts), in order."""
    found: list[str] = []

    def _walk(v: Any) -> None:
        if isinstance(v, str):
            for m in _ENV_REF_RE.finditer(v):
                if m.group(1) not in found:
                    found.append(m.group(1))
        elif isinstance(v, Mapping):
            for item in v.values():
                _walk(item)
        elif isinstance(v, (list, tuple)):
            for item in v:
                _walk(item)

    _walk(value)
    return found


def resolve_env_refs(value: str, environ: Mapping[str, str] | None = None) -> str:
    """Replace ``${VAR}`` / ``${VAR:-default}`` in *value* with environment values.

    Raises:
        MissingEnvError: A referenced variable is unset (and has no default).
    """
    env = os.environ if environ is None else environ
    missing: list[str] = []

    def _sub(m: re.Match[str]) -> str:
        name, default = m.group(1), m.group(2)
        val = env.get(name)
        if val:
            return val
        if default is not None:
            return default
        missing.append(name)
        return ""

    out = _ENV_REF_RE.sub(_sub, value)
    if missing:
        names = ", ".join(dict.fromkeys(missing))
        raise MissingEnvError(f"Environment variable(s) not set: {names}", missing=list(dict.fromkeys(missing)))
    return out


def _has_ref(value: str) -> bool:
    return bool(_ENV_REF_RE.search(value))


def looks_like_secret(key: str, value: str) -> bool:
    """True when *value* is a literal (no ``${...}``) that looks like a credential.

    A value counts as secret when its *key* names a credential (``API_KEY``,
    ``Authorization``, ``token``, ...) or the value itself has a well-known
    credential shape (``sk-...``, ``ghp_...``, ``Bearer ...``, a JWT).
    """
    if not isinstance(value, str) or not value.strip():
        return False
    stripped = _ENV_REF_RE.sub("", value).strip()
    if not stripped:
        return False
    if any(r.match(stripped) for r in _SECRET_VALUE_RES) or scrub_secrets(stripped) != stripped:
        return True
    if _has_ref(value):
        # "Bearer ${TOKEN}" — the literal remainder is just a scheme word.
        return False
    if not key or not _SECRET_KEY_RE.search(key) or len(stripped) < 8:
        return False
    # Paths, URLs, sentences and short plain words under a credential-ish key
    # ("SESSION_DIR=/tmp/x", "AUTH_MODE=oauth") are configuration, not secrets.
    return not (any(c in stripped for c in ("/", "\\", " ")) or (stripped.isalpha() and len(stripped) < 16))


def find_inline_secrets(config: MCPServerConfig) -> list[str]:
    """Return the fields of *config* that hold literal secret-looking values."""
    hits: list[str] = []
    for key, val in config.env.items():
        if looks_like_secret(key, val):
            hits.append(f"env.{key}")
    for key, val in config.headers.items():
        if looks_like_secret(key, val):
            hits.append(f"headers.{key}")
    if config.url:
        literal = _ENV_REF_RE.sub("", config.url)
        if scrub_secrets(literal) != literal:
            hits.append("url")
    prev = ""
    for i, arg in enumerate(config.args):
        if "=" in arg and arg.startswith("-"):
            flag, _, val = arg.partition("=")
            if _SECRET_FLAG_RE.match(flag) and val and not _has_ref(val):
                hits.append(f"args[{i}]")
        elif not _has_ref(arg) and (_SECRET_FLAG_RE.match(prev) or any(r.match(arg) for r in _SECRET_VALUE_RES)):
            hits.append(f"args[{i}]")
        prev = arg
    return hits


# ---------------------------------------------------------------------------
# Server config
# ---------------------------------------------------------------------------


@dataclass
class MCPServerConfig:
    """One named MCP server.

    Attributes:
        name: Registry name (``^[A-Za-z0-9][A-Za-z0-9_-]{0,39}$``); also the tool prefix.
        transport: ``"stdio"`` (spawn *command*) or ``"http"`` (streamable HTTP at *url*).
        command: Executable for stdio servers (``npx``, ``uvx``, ``python``, ...).
        args: Arguments for *command*; may contain ``${VAR}`` references.
        url: Endpoint for http servers; may contain ``${VAR}`` references.
        env: Extra environment for stdio servers (``{"TOKEN": "${MY_TOKEN}"}``).
        headers: Extra HTTP headers (``{"Authorization": "Bearer ${MY_TOKEN}"}``).
        timeout: Seconds per tool call / list (and HTTP connect).
        enabled: Disabled servers are skipped by ``mcp:all`` and health checks.
        description: Free text shown by ``prompture mcp list``.
        preset: Preset the entry was created from, if any.
        cwd: Working directory for stdio servers.
        scope: Where the entry came from (``user`` / ``project`` / ``preset``); not stored.
    """

    name: str
    transport: str = "stdio"
    command: str | None = None
    args: list[str] = field(default_factory=list)
    url: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)
    timeout: float = DEFAULT_TIMEOUT
    enabled: bool = True
    description: str = ""
    preset: str | None = None
    cwd: str | None = None
    scope: str = field(default="", compare=False)

    # -- validation ---------------------------------------------------------

    def validate(self) -> None:
        """Raise :class:`MCPConfigError` when the entry cannot work."""
        if not isinstance(self.name, str) or not _SERVER_NAME_RE.match(self.name):
            raise MCPConfigError(
                f"Invalid MCP server name {self.name!r}: use 1-40 letters, digits, '_' or '-' (start with a letter/digit)."
            )
        if TOOL_PREFIX_SEP in self.name:
            raise MCPConfigError(f"MCP server name {self.name!r} must not contain {TOOL_PREFIX_SEP!r}.")
        if self.transport not in TRANSPORTS:
            raise MCPConfigError(
                f"MCP server {self.name!r}: transport must be one of {TRANSPORTS}, got {self.transport!r}."
            )
        if self.transport == "stdio":
            if not self.command or not isinstance(self.command, str):
                raise MCPConfigError(f"MCP server {self.name!r}: stdio transport needs a command.")
        else:
            if not self.url or not isinstance(self.url, str):
                raise MCPConfigError(f"MCP server {self.name!r}: http transport needs a url.")
            probe_url = _ENV_REF_RE.sub("x", self.url)
            parts = urlsplit(probe_url)
            if parts.scheme not in ("http", "https") or not parts.netloc:
                raise MCPConfigError(
                    f"MCP server {self.name!r}: url must be an http(s) URL, got {scrub_secrets(self.url)!r}."
                )
        if not isinstance(self.args, list) or not all(isinstance(a, str) for a in self.args):
            raise MCPConfigError(f"MCP server {self.name!r}: args must be a list of strings.")
        for label, mapping in (("env", self.env), ("headers", self.headers)):
            if not isinstance(mapping, dict) or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in mapping.items()
            ):
                raise MCPConfigError(f"MCP server {self.name!r}: {label} must map strings to strings.")
        for key in self.env:
            if not _ENV_NAME_RE.match(key):
                raise MCPConfigError(f"MCP server {self.name!r}: invalid env var name {key!r}.")
        try:
            timeout = float(self.timeout)
        except (TypeError, ValueError):
            raise MCPConfigError(f"MCP server {self.name!r}: timeout must be a number.") from None
        if timeout <= 0:
            raise MCPConfigError(f"MCP server {self.name!r}: timeout must be positive.")

    # -- serialization ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Registry JSON for this entry (without the name, which is the key)."""
        out: dict[str, Any] = {"transport": self.transport}
        if self.transport == "stdio":
            out["command"] = self.command
            if self.args:
                out["args"] = list(self.args)
            if self.env:
                out["env"] = dict(self.env)
            if self.cwd:
                out["cwd"] = self.cwd
        else:
            out["url"] = self.url
            if self.headers:
                out["headers"] = dict(self.headers)
        if float(self.timeout) != DEFAULT_TIMEOUT:
            out["timeout"] = float(self.timeout)
        if not self.enabled:
            out["enabled"] = False
        if self.description:
            out["description"] = self.description
        if self.preset:
            out["preset"] = self.preset
        return out

    @classmethod
    def from_dict(cls, name: str, data: Mapping[str, Any], *, scope: str = "") -> MCPServerConfig:
        """Build an entry from registry JSON. Accepts ``type`` as an alias of ``transport``."""
        if not isinstance(data, Mapping):
            raise MCPConfigError(f"MCP server {name!r}: entry must be an object.")
        transport = data.get("transport") or data.get("type")
        if not transport:
            transport = "http" if data.get("url") else "stdio"
        if transport == "streamable-http":
            transport = "http"
        timeout = data.get("timeout", DEFAULT_TIMEOUT)
        cfg = cls(
            name=name,
            transport=str(transport),
            command=data.get("command"),
            args=list(data.get("args") or []),
            url=data.get("url"),
            env=dict(data.get("env") or {}),
            headers=dict(data.get("headers") or {}),
            timeout=timeout if timeout is not None else DEFAULT_TIMEOUT,
            enabled=bool(data.get("enabled", True)),
            description=str(data.get("description") or ""),
            preset=data.get("preset"),
            cwd=data.get("cwd"),
            scope=scope,
        )
        return cfg

    # -- derived ------------------------------------------------------------

    @property
    def startup_timeout(self) -> float:
        """Seconds allowed to start the server and initialize the session."""
        if self.transport == "stdio":
            return max(float(self.timeout), STDIO_STARTUP_TIMEOUT)
        return float(self.timeout)

    @property
    def target(self) -> str:
        """Display string (command line or URL) with secrets scrubbed."""
        if self.transport == "stdio":
            return scrub_secrets(" ".join([self.command or "", *self.args]).strip())
        return scrub_secrets(self.url or "")

    def referenced_env(self) -> list[str]:
        """Every ``${VAR}`` this entry needs at connect time."""
        return env_refs([self.command or "", self.args, self.url or "", self.env, self.headers, self.cwd or ""])

    def missing_env(self, environ: Mapping[str, str] | None = None) -> list[str]:
        """Referenced variables that are unset (ignoring ones with a ``:-default``)."""
        env = os.environ if environ is None else environ
        missing: list[str] = []
        for text in _iter_strings(
            [self.command or "", self.args, self.url or "", self.env, self.headers, self.cwd or ""]
        ):
            for m in _ENV_REF_RE.finditer(text):
                if m.group(2) is None and not env.get(m.group(1)) and m.group(1) not in missing:
                    missing.append(m.group(1))
        return missing

    def resolved(self, environ: Mapping[str, str] | None = None) -> MCPServerConfig:
        """A copy with every ``${VAR}`` substituted. Never log the result."""
        missing = self.missing_env(environ)
        if missing:
            raise MissingEnvError(
                f"MCP server {self.name!r} needs environment variable(s): {', '.join(missing)}",
                missing=missing,
            )

        def r(v: str | None) -> str | None:
            return resolve_env_refs(v, environ) if v else v

        return MCPServerConfig(
            name=self.name,
            transport=self.transport,
            command=r(self.command),
            args=[resolve_env_refs(a, environ) for a in self.args],
            url=r(self.url),
            env={k: resolve_env_refs(v, environ) for k, v in self.env.items()},
            headers={k: resolve_env_refs(v, environ) for k, v in self.headers.items()},
            timeout=self.timeout,
            enabled=self.enabled,
            description=self.description,
            preset=self.preset,
            cwd=r(self.cwd),
            scope=self.scope,
        )

    def fingerprint(self) -> str:
        """Stable hash of the stored config, used to detect edits under a pooled session."""
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:16]


def _iter_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for v in value.values():
            yield from _iter_strings(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _iter_strings(v)


# ---------------------------------------------------------------------------
# Presets (opt-in)
# ---------------------------------------------------------------------------

#: Built-in server presets. ``requires`` names the launcher binary (if any).
PRESETS: dict[str, dict[str, Any]] = {
    "exa": {
        "transport": "http",
        "url": "https://mcp.exa.ai/mcp",
        "description": "Exa web search (hosted, no key needed)",
        "requires": None,
    },
    "deepwiki": {
        "transport": "http",
        "url": "https://mcp.deepwiki.com/mcp",
        "description": "DeepWiki: ask questions about public GitHub repositories (hosted, no key)",
        "requires": None,
    },
    "github": {
        "transport": "http",
        "url": "https://api.githubcopilot.com/mcp/",
        "headers": {"Authorization": "Bearer ${GITHUB_TOKEN}"},
        "description": "GitHub's hosted MCP server (needs GITHUB_TOKEN)",
        "requires": None,
    },
    "fetch": {
        "transport": "stdio",
        "command": "uvx",
        "args": ["mcp-server-fetch"],
        "description": "Reference server: fetch a URL and return it as markdown",
        "requires": "uvx",
    },
    "time": {
        "transport": "stdio",
        "command": "uvx",
        "args": ["mcp-server-time"],
        "description": "Reference server: current time and timezone conversion",
        "requires": "uvx",
    },
    "git": {
        "transport": "stdio",
        "command": "uvx",
        "args": ["mcp-server-git"],
        "description": "Reference server: read and search a local git repository",
        "requires": "uvx",
    },
    "memory": {
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-memory"],
        "description": "Reference server: knowledge-graph memory",
        "requires": "npx",
    },
    "filesystem": {
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem"],
        # Allowed directories; replaced by any --arg given on the command line.
        "default_extra_args": ["."],
        "description": "Reference server: file access limited to the given directories",
        "requires": "npx",
    },
    "sequential-thinking": {
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"],
        "description": "Reference server: structured step-by-step reasoning tool",
        "requires": "npx",
    },
}


def list_presets() -> list[dict[str, Any]]:
    """Preset summaries: name, transport, target, requires, description, env vars needed."""
    rows = []
    for name, spec in PRESETS.items():
        cfg = get_preset(name)
        rows.append(
            {
                "name": name,
                "transport": cfg.transport,
                "target": cfg.target,
                "requires": spec.get("requires"),
                "env": cfg.referenced_env(),
                "description": spec.get("description", ""),
            }
        )
    return rows


def get_preset(preset: str, *, name: str | None = None, extra_args: list[str] | None = None) -> MCPServerConfig:
    """Build a server config from a preset.

    Args:
        preset: Preset name (see :data:`PRESETS`).
        name: Registry name (defaults to the preset name).
        extra_args: Arguments appended to the preset command. For presets with
            ``default_extra_args`` (``filesystem``'s directory list) they
            replace the defaults.
    """
    spec = PRESETS.get(preset)
    if spec is None:
        raise MCPConfigError(f"Unknown MCP preset {preset!r}. Known: {', '.join(PRESETS)}")
    args = list(spec.get("args", []))
    args.extend(extra_args if extra_args else spec.get("default_extra_args", []))
    return MCPServerConfig(
        name=name or preset,
        transport=spec["transport"],
        command=spec.get("command"),
        args=args,
        url=spec.get("url"),
        env=dict(spec.get("env", {})),
        headers=dict(spec.get("headers", {})),
        description=spec.get("description", ""),
        preset=preset,
        scope="preset",
    )


# ---------------------------------------------------------------------------
# Registry files
# ---------------------------------------------------------------------------


def user_registry_path() -> Path:
    """``$PROMPTURE_MCP_CONFIG`` or ``~/.prompture/mcp.json``."""
    override = os.environ.get(USER_CONFIG_ENV)
    if override:
        return Path(override).expanduser()
    return Path.home() / ".prompture" / "mcp.json"


def project_registry_path(project_dir: str | Path | None = None) -> Path:
    """``<project_dir or cwd>/.prompture/mcp.json``."""
    return Path(project_dir or Path.cwd()) / ".prompture" / "mcp.json"


def _read_registry(path: Path, scope: str) -> tuple[dict[str, MCPServerConfig], list[str]]:
    """Parse a registry file. Returns ``(servers, errors)``; a missing file is empty."""
    if not path.is_file():
        return {}, []
    try:
        data = json.loads(path.read_text(encoding="utf-8") or "{}")
    except (OSError, json.JSONDecodeError) as exc:
        return {}, [f"{path}: cannot read MCP registry ({exc})"]
    if not isinstance(data, dict):
        return {}, [f"{path}: MCP registry must be a JSON object"]
    raw = data.get("servers", data.get("mcpServers", {}))
    if not isinstance(raw, dict):
        return {}, [f"{path}: 'servers' must be an object"]
    servers: dict[str, MCPServerConfig] = {}
    errors: list[str] = []
    for name, entry in raw.items():
        try:
            servers[name] = MCPServerConfig.from_dict(name, entry, scope=scope)
        except Exception as exc:
            errors.append(f"{path}: server {name!r}: {exc}")
    return servers, errors


def _atomic_write_json(path: Path, data: Any) -> None:
    """Write JSON atomically (temp file beside *path* + ``os.replace``), owner-only."""
    parent = path.parent
    if not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(OSError):
            os.chmod(parent, 0o700)
    payload = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
        with contextlib.suppress(OSError):
            os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(tmp)
        raise


class MCPHub:
    """The merged user + project MCP server registry.

    Args:
        user_path: User registry file (default :func:`user_registry_path`).
        project_dir: Directory whose ``.prompture/mcp.json`` is the project
            registry (default: the working directory).
        project_path: Explicit project registry file (overrides *project_dir*).
        include_project: Set ``False`` to ignore the project registry.
    """

    def __init__(
        self,
        user_path: str | Path | None = None,
        *,
        project_dir: str | Path | None = None,
        project_path: str | Path | None = None,
        include_project: bool = True,
    ) -> None:
        self.user_path = Path(user_path) if user_path else user_registry_path()
        self.project_path: Path | None = (
            Path(project_path) if project_path else project_registry_path(project_dir) if include_project else None
        )
        self._user: dict[str, MCPServerConfig] = {}
        self._project: dict[str, MCPServerConfig] = {}
        self.errors: list[str] = []
        self.load()

    # -- reading ------------------------------------------------------------

    def load(self) -> MCPHub:
        """(Re)read both registry files."""
        self._user, user_errors = _read_registry(self.user_path, "user")
        self.errors = list(user_errors)
        self._project = {}
        if self.project_path is not None and self.project_path.resolve() != self.user_path.resolve():
            self._project, project_errors = _read_registry(self.project_path, "project")
            self.errors.extend(project_errors)
        return self

    def list(self, *, include_disabled: bool = True) -> list[MCPServerConfig]:
        """Merged servers (project overrides user by name), sorted by name."""
        merged = {**self._user, **self._project}
        return [merged[n] for n in sorted(merged) if include_disabled or merged[n].enabled]

    def names(self) -> list[str]:
        return [s.name for s in self.list()]

    def get(self, name: str, *, presets: bool = False) -> MCPServerConfig | None:
        """Registered server *name*; with ``presets=True`` fall back to a same-named preset."""
        found = self._project.get(name) or self._user.get(name)
        if found is None and presets and name in PRESETS:
            return get_preset(name)
        return found

    def require(self, name: str, *, presets: bool = True) -> MCPServerConfig:
        """Like :meth:`get` but raise a helpful :class:`MCPConfigError` when absent."""
        cfg = self.get(name, presets=presets)
        if cfg is None:
            known = ", ".join(self.names()) or "none registered"
            raise MCPConfigError(
                f"Unknown MCP server {name!r} (registered: {known}). "
                f"Add it with `prompture mcp add {name} --url ...` / `--command ...`, "
                f"or use a preset: {', '.join(PRESETS)}."
            )
        return cfg

    def scope_of(self, name: str) -> str | None:
        if name in self._project:
            return "project"
        if name in self._user:
            return "user"
        return None

    # -- writing ------------------------------------------------------------

    def path_for(self, scope: str) -> Path:
        if scope == "user":
            return self.user_path
        if scope == "project":
            if self.project_path is None:
                raise MCPConfigError("This hub was created without a project registry.")
            return self.project_path
        raise MCPConfigError(f"scope must be 'user' or 'project', got {scope!r}")

    def _servers_for(self, scope: str) -> dict[str, MCPServerConfig]:
        return self._user if scope == "user" else self._project

    def add(
        self,
        config: MCPServerConfig,
        *,
        scope: str = "user",
        replace: bool = False,
        allow_inline_secrets: bool = False,
        dry_run: bool = False,
    ) -> Path:
        """Validate and store *config* in the *scope* registry. Returns the file path.

        Raises:
            MCPConfigError: Invalid entry, or the name exists and ``replace`` is False.
            InlineSecretError: A literal secret-looking value would be stored
                (use ``${ENV}`` references instead).
        """
        config.validate()
        secrets = find_inline_secrets(config)
        if secrets and not allow_inline_secrets:
            raise InlineSecretError(
                f"MCP server {config.name!r} has literal secret-looking values in: {', '.join(secrets)}. "
                "Store a ${ENV_VAR} reference instead (e.g. --header 'Authorization=Bearer ${MY_TOKEN}') "
                "and set the variable in your environment.",
                fields=secrets,
            )
        path = self.path_for(scope)
        servers = self._servers_for(scope)
        if config.name in servers and not replace:
            raise MCPConfigError(
                f"MCP server {config.name!r} already exists in {path}; use replace/--force to overwrite."
            )
        if dry_run:
            return path
        # Re-read the file right before writing so concurrent edits aren't lost.
        current, _ = _read_registry(path, scope)
        config.scope = scope
        current[config.name] = config
        self._write(path, current)
        servers.clear()
        servers.update(current)
        return path

    def remove(self, name: str, *, scope: str | None = None, dry_run: bool = False) -> Path | None:
        """Remove *name* from *scope* (default: wherever it is defined). Returns the file changed."""
        scope = scope or self.scope_of(name)
        if scope is None:
            return None
        path = self.path_for(scope)
        current, _ = _read_registry(path, scope)
        if name not in current:
            return None
        if dry_run:
            return path
        del current[name]
        self._write(path, current)
        servers = self._servers_for(scope)
        servers.clear()
        servers.update(current)
        return path

    def set_enabled(self, name: str, enabled: bool, *, dry_run: bool = False) -> Path:
        """Toggle a registered server on/off in the file that defines it."""
        scope = self.scope_of(name)
        if scope is None:
            raise MCPConfigError(f"Unknown MCP server {name!r}.")
        cfg = self._servers_for(scope)[name]
        cfg.enabled = enabled
        return self.add(cfg, scope=scope, replace=True, allow_inline_secrets=True, dry_run=dry_run)

    def save(self, scope: str = "user") -> Path:
        """Write the in-memory *scope* registry to disk."""
        path = self.path_for(scope)
        self._write(path, self._servers_for(scope))
        return path

    @staticmethod
    def _write(path: Path, servers: Mapping[str, MCPServerConfig]) -> None:
        data = {"version": REGISTRY_VERSION, "servers": {n: servers[n].to_dict() for n in sorted(servers)}}
        _atomic_write_json(path, data)


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

SessionFactory = Callable[[MCPServerConfig], AbstractAsyncContextManager[Any]]


def _stdio_env(env: dict[str, str]) -> dict[str, str] | None:
    """Merge *env* over the SDK's safe default environment (keeps ``PATH`` etc.)."""
    if not env:
        return None
    try:
        from mcp.client.stdio import get_default_environment

        base = dict(get_default_environment())
    except Exception:  # pragma: no cover - SDK layout differences
        base = {
            k: v
            for k, v in os.environ.items()
            if k.upper() in ("PATH", "HOME", "USERPROFILE", "SYSTEMROOT", "TEMP", "TMP", "APPDATA", "LOCALAPPDATA")
        }
    base.update(env)
    return base


@asynccontextmanager
async def open_session(config: MCPServerConfig) -> AsyncIterator[Any]:
    """Open an initialized MCP session for *config*, resolving ``${ENV}`` refs now.

    Raises:
        MissingEnvError: A referenced variable is unset.
    """
    from .client import http_session, stdio_session

    resolved = config.resolved()
    if resolved.transport == "http":
        async with http_session(
            resolved.url or "", resolved.headers or None, timeout=float(resolved.timeout)
        ) as session:
            yield session
        return
    if resolved.cwd:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(
            command=resolved.command or "", args=resolved.args, env=_stdio_env(resolved.env), cwd=resolved.cwd
        )
        async with stdio_client(params) as streams, ClientSession(streams[0], streams[1]) as session:
            await session.initialize()
            yield session
        return
    async with stdio_session(resolved.command or "", resolved.args, _stdio_env(resolved.env)) as session:
        yield session


def _leaf_exception(exc: BaseException) -> BaseException:
    """Unwrap ``ExceptionGroup``s raised by task groups to the first real cause."""
    while True:
        inner = getattr(exc, "exceptions", None)  # BaseExceptionGroup (3.11+ / exceptiongroup backport)
        if not isinstance(inner, (list, tuple)) or not inner or not isinstance(inner[0], BaseException):
            return exc
        exc = inner[0]


def _describe(exc: BaseException) -> str:
    leaf = _leaf_exception(exc)
    text = str(leaf) or type(leaf).__name__
    return scrub_secrets(f"{type(leaf).__name__}: {text}" if str(leaf) else text)


async def _list_all_tools(session: Any, max_pages: int = 20) -> list[Any]:
    """List every tool on *session*, following pagination cursors when supported."""
    listing = await session.list_tools()
    tools = list(getattr(listing, "tools", listing) or [])
    cursor = _mcp_field(listing, "nextCursor", "next_cursor")
    pages = 1
    while cursor and pages < max_pages:
        try:
            listing = await session.list_tools(cursor=cursor)
        except TypeError:
            try:
                from mcp import types as mcp_types

                listing = await session.list_tools(params=mcp_types.PaginatedRequestParams(cursor=cursor))
            except Exception:
                break
        tools.extend(getattr(listing, "tools", listing) or [])
        cursor = _mcp_field(listing, "nextCursor", "next_cursor")
        pages += 1
    return tools


class _LoopThread:
    """A daemon thread running one asyncio loop that owns every pooled session."""

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def loop(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._loop is None or self._thread is None or not self._thread.is_alive():
                loop = asyncio.new_event_loop()
                started = threading.Event()

                def _run() -> None:
                    asyncio.set_event_loop(loop)
                    loop.call_soon(started.set)
                    loop.run_forever()

                thread = threading.Thread(target=_run, name="prompture-mcp-pool", daemon=True)
                thread.start()
                started.wait(5)
                self._loop, self._thread = loop, thread
            return self._loop

    def submit(self, coro: Any) -> concurrent.futures.Future[Any]:
        return asyncio.run_coroutine_threadsafe(coro, self.loop())

    def call_soon(self, fn: Callable[[], Any]) -> None:
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(fn)

    def is_current(self) -> bool:
        return self._thread is not None and threading.current_thread() is self._thread

    def stop(self, timeout: float = 5.0) -> None:
        with self._lock:
            loop, thread = self._loop, self._thread
            self._loop = self._thread = None
        if loop is None or thread is None:
            return
        if loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout)
        if not thread.is_alive():
            with contextlib.suppress(Exception):
                loop.close()


class _Connection:
    """One long-lived session for one server, held open by a task on the pool loop."""

    def __init__(self, config: MCPServerConfig, pool: MCPSessionPool) -> None:
        self.config = config
        self.fingerprint = config.fingerprint()
        self._pool = pool
        self._lock = threading.Lock()
        self._session: Any = None
        self._holder: concurrent.futures.Future[Any] | None = None
        self._stop: asyncio.Event | None = None
        self.tools: list[Any] | None = None

    @property
    def alive(self) -> bool:
        return self._session is not None and self._holder is not None and not self._holder.done()

    async def _hold(self, ready: concurrent.futures.Future[Any]) -> None:
        self._stop = stop = asyncio.Event()
        try:
            async with self._pool.session_factory(self.config) as session:
                if not ready.done():
                    ready.set_result(session)
                await stop.wait()
        except asyncio.CancelledError:
            if not ready.done():
                ready.set_exception(MCPTimeoutError(f"MCP server {self.config.name!r} start was cancelled"))
            raise
        except BaseException as exc:
            if not ready.done():
                ready.set_exception(exc)
            else:
                logger.debug("MCP session %s ended: %s", self.config.name, _describe(exc))

    def ensure(self) -> Any:
        """Return the live session, starting the server if needed (blocking)."""
        if self._pool._loop.is_current():
            raise RuntimeError("MCP pool sessions cannot be opened from the pool's own event loop thread.")
        with self._lock:
            if self.alive:
                return self._session
            self._reset_locked()
            ready: concurrent.futures.Future[Any] = concurrent.futures.Future()
            holder = self._pool._loop.submit(self._hold(ready))
            self._holder = holder
            timeout = self.config.startup_timeout
            try:
                session = ready.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                holder.cancel()
                self._holder = None
                raise MCPTimeoutError(
                    f"MCP server {self.config.name!r} did not start within {timeout:g}s ({self.config.target})"
                ) from None
            except MissingEnvError:
                self._holder = None
                raise
            except BaseException as exc:
                self._holder = None
                raise MCPConnectionError(
                    f"Could not connect to MCP server {self.config.name!r} ({self.config.target}): {_describe(exc)}"
                ) from _leaf_exception(exc)
            self._session = session
            return session

    def _reset_locked(self, wait: float = 5.0) -> None:
        holder, stop = self._holder, self._stop
        self._session = None
        self._holder = None
        self._stop = None
        self.tools = None
        if holder is None or holder.done():
            return
        if stop is not None:
            self._pool._loop.call_soon(stop.set)
        else:
            holder.cancel()
        try:
            holder.result(timeout=wait)
        except BaseException:
            holder.cancel()

    def close(self, wait: float = 5.0) -> None:
        with self._lock:
            self._reset_locked(wait)

    def _after_error(self) -> None:
        """Drop a session whose holder task has ended so the next call reconnects."""
        holder = self._holder
        if holder is not None and holder.done():
            self.close(wait=0)

    def run(self, make_coro: Callable[[Any], Any], timeout: float) -> Any:
        session = self.ensure()
        fut = self._pool._loop.submit(make_coro(session))
        try:
            return fut.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            fut.cancel()
            raise MCPTimeoutError(f"MCP server {self.config.name!r} did not answer within {timeout:g}s") from None
        except BaseException:
            self._after_error()
            raise

    async def arun(self, make_coro: Callable[[Any], Any], timeout: float) -> Any:
        session = self._session if self.alive else await asyncio.to_thread(self.ensure)
        fut = self._pool._loop.submit(make_coro(session))
        try:
            return await asyncio.wait_for(asyncio.wrap_future(fut), timeout)
        except asyncio.TimeoutError:
            fut.cancel()
            raise MCPTimeoutError(f"MCP server {self.config.name!r} did not answer within {timeout:g}s") from None
        except BaseException:
            self._after_error()
            raise


class MCPSessionPool:
    """Pooled MCP sessions: one long-lived session per server on a background loop.

    Args:
        session_factory: ``config -> async context manager yielding an initialized
            session`` (default :func:`open_session`). Tests pass a fake.
    """

    def __init__(self, session_factory: SessionFactory | None = None) -> None:
        self.session_factory: SessionFactory = session_factory or open_session
        self._loop = _LoopThread()
        self._conns: dict[str, _Connection] = {}
        self._lock = threading.Lock()

    def connection(self, config: MCPServerConfig) -> _Connection:
        """The pooled connection for *config* (replaced when the stored config changed)."""
        fp = config.fingerprint()
        with self._lock:
            conn = self._conns.get(config.name)
            if conn is not None and conn.fingerprint == fp:
                return conn
            stale = conn
            conn = _Connection(config, self)
            self._conns[config.name] = conn
        if stale is not None:
            stale.close()
        return conn

    def is_connected(self, name: str) -> bool:
        conn = self._conns.get(name)
        return bool(conn and conn.alive)

    def list_tools(self, config: MCPServerConfig, *, refresh: bool = False) -> list[Any]:
        """Tool descriptors of *config*'s server (cached per session)."""
        conn = self.connection(config)
        if conn.tools is None or refresh or not conn.alive:
            conn.tools = conn.run(_list_all_tools, float(config.timeout))
        return list(conn.tools)

    async def alist_tools(self, config: MCPServerConfig, *, refresh: bool = False) -> list[Any]:
        conn = self.connection(config)
        if conn.tools is None or refresh or not conn.alive:
            conn.tools = await conn.arun(_list_all_tools, float(config.timeout))
        return list(conn.tools)

    def call_tool(self, config: MCPServerConfig, tool: str, arguments: dict[str, Any]) -> Any:
        """Call *tool* on the pooled session (blocking; safe from any non-pool thread)."""
        conn = self.connection(config)
        return conn.run(lambda s: s.call_tool(tool, arguments), float(config.timeout))

    async def acall_tool(self, config: MCPServerConfig, tool: str, arguments: dict[str, Any]) -> Any:
        conn = self.connection(config)
        return await conn.arun(lambda s: s.call_tool(tool, arguments), float(config.timeout))

    def run_coroutine(self, coro: Any, timeout: float) -> Any:
        """Run an arbitrary coroutine on the pool loop and wait for it."""
        fut = self._loop.submit(coro)
        try:
            return fut.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            fut.cancel()
            raise

    def close(self, name: str | None = None) -> None:
        """Close one server's session (or all of them)."""
        with self._lock:
            if name is None:
                conns = list(self._conns.values())
                self._conns.clear()
            else:
                found = self._conns.pop(name, None)
                conns = [found] if found else []
        for conn in conns:
            try:
                conn.close()
            except Exception:  # pragma: no cover - best effort
                logger.debug("closing MCP session %s failed", conn.config.name, exc_info=True)

    def shutdown(self) -> None:
        """Close every session and stop the background loop."""
        self.close()
        self._loop.stop()


_pool: MCPSessionPool | None = None
_pool_lock = threading.Lock()


def get_pool() -> MCPSessionPool:
    """The process-wide session pool (created on first use, closed at exit)."""
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = MCPSessionPool()
        return _pool


def set_pool(pool: MCPSessionPool | None) -> MCPSessionPool | None:
    """Replace the process-wide pool (the old one is shut down). Returns the old pool."""
    global _pool
    with _pool_lock:
        old, _pool = _pool, pool
    if old is not None and old is not pool:
        old.shutdown()
    return old


def _shutdown_pool() -> None:
    pool = _pool
    if pool is not None:
        with contextlib.suppress(Exception):
            pool.shutdown()


# Close sessions before the interpreter stops thread pools: HTTP session
# teardown resolves the host through the loop's default executor. Fall back
# to a plain atexit hook where the threading hook is unavailable.
try:
    threading._register_atexit(_shutdown_pool)  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - other interpreters / late import
    atexit.register(_shutdown_pool)


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------


def prefixed_tool_name(server: str, tool: str, taken: set[str] | None = None) -> str:
    """``<server>__<tool>`` sanitized to ``^[a-zA-Z0-9_-]{1,64}$`` and unique within *taken*."""
    raw = f"{server}{TOOL_PREFIX_SEP}{tool}"
    name = _INVALID_TOOL_CHARS.sub("_", raw)
    if len(name) > 64 or (taken is not None and name in taken):
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:6]
        name = f"{name[:57]}_{digest}"
    if taken is not None:
        n = 2
        base = name
        while name in taken:
            suffix = f"_{n}"
            name = f"{base[: 64 - len(suffix)]}{suffix}"
            n += 1
        taken.add(name)
    if not _TOOL_NAME_RE.match(name):  # pragma: no cover - defensive
        raise MCPConfigError(f"Could not build a valid tool name for {server!r}/{tool!r}")
    return name


def _record_tool_call(
    server: str,
    tool: str,
    exposed: str,
    elapsed_ms: float,
    *,
    ok: bool,
    error: str | None = None,
    error_type: str | None = None,
) -> None:
    """Record one MCP tool call through the global usage tracker. Never raises."""
    try:
        from ..infra.tracker import UsageEvent, get_tracker

        tracker = get_tracker()
        if not getattr(tracker, "_enabled", False):
            return
        tracker.record(
            UsageEvent(
                model_name=f"mcp/{server}",
                provider="mcp",
                tool_name=exposed,
                operation="mcp_tool_call",
                elapsed_ms=elapsed_ms,
                status="success" if ok else "error",
                error_type=error_type if not ok else None,
                error_message=scrub_secrets(error)[:500] if (error and not ok) else None,
                metadata={"modality": "mcp", "mcp_server": server, "mcp_tool": tool},
            )
        )
    except Exception:
        logger.debug("recording MCP tool call failed", exc_info=True)


def _finish_result(server: str, tool: str, exposed: str, result: Any, start: float) -> str:
    elapsed = (time.monotonic() - start) * 1000
    text = _serialize_call_result(result)
    if _mcp_field(result, "isError", "is_error", default=False):
        _record_tool_call(server, tool, exposed, elapsed, ok=False, error=text, error_type="MCPToolError")
        return f"Error from MCP tool '{exposed}': {text}"
    _record_tool_call(server, tool, exposed, elapsed, ok=True)
    return text


def _fail_result(server: str, tool: str, exposed: str, exc: BaseException, start: float) -> str:
    elapsed = (time.monotonic() - start) * 1000
    message = _describe(exc)
    _record_tool_call(
        server, tool, exposed, elapsed, ok=False, error=message, error_type=type(_leaf_exception(exc)).__name__
    )
    return f"Error calling MCP tool '{exposed}' on server '{server}': {message}"


def _make_tool_definition(config: MCPServerConfig, tool: Any, exposed: str, pool: MCPSessionPool) -> ToolDefinition:
    raw = str(_mcp_field(tool, "name"))
    description = str(_mcp_field(tool, "description") or f"MCP tool {raw}")
    parameters = _mcp_field(tool, "inputSchema", "input_schema")
    if not isinstance(parameters, dict):
        parameters = {"type": "object", "properties": {}}
    server = config.name

    def _call(**arguments: Any) -> str:
        start = time.monotonic()
        try:
            result = pool.call_tool(config, raw, arguments)
        except Exception as exc:
            return _fail_result(server, raw, exposed, exc, start)
        return _finish_result(server, raw, exposed, result, start)

    async def _acall(**arguments: Any) -> str:
        start = time.monotonic()
        try:
            result = await pool.acall_tool(config, raw, arguments)
        except Exception as exc:
            return _fail_result(server, raw, exposed, exc, start)
        return _finish_result(server, raw, exposed, result, start)

    _call.__name__ = exposed
    _call.__doc__ = description
    _call._async_fn = _acall  # type: ignore[attr-defined]
    return ToolDefinition(
        name=exposed,
        description=f"[{server}] {description}",
        parameters=parameters,
        function=_call,
        metadata={"source": "mcp", "mcp_server": server, "mcp_tool": raw},
    )


def _build_definitions(
    config: MCPServerConfig,
    tools: list[Any],
    pool: MCPSessionPool,
    only: list[str] | None,
    taken: set[str],
) -> list[ToolDefinition]:
    defs: list[ToolDefinition] = []
    available: list[str] = []
    for tool in tools:
        raw = _mcp_field(tool, "name")
        if not raw:
            continue
        raw = str(raw)
        available.append(raw)
        if only is not None:
            prefixed = prefixed_tool_name(config.name, raw)
            if raw not in only and prefixed not in only:
                continue
        defs.append(_make_tool_definition(config, tool, prefixed_tool_name(config.name, raw, taken), pool))
    if only is not None:
        found = {d.metadata["mcp_tool"] for d in defs}
        missing = [t for t in only if t not in found and not any(d.name == t for d in defs)]
        if missing:
            raise MCPConfigError(
                f"MCP server {config.name!r} has no tool(s) {', '.join(missing)}. Available: {', '.join(available)}"
            )
    return defs


def _parse_spec(spec: str, hub: MCPHub) -> list[tuple[MCPServerConfig, list[str] | None]]:
    """``all`` | ``server`` | ``server/tool1,tool2`` → ``[(config, tool filter)]``."""
    spec = spec.strip()
    if not spec:
        raise MCPConfigError("Empty MCP tool spec; use 'mcp:<server>' or 'mcp:all'.")
    server, _, tools_part = spec.partition("/")
    only: list[str] | None = [t.strip() for t in tools_part.split(",") if t.strip()] or None
    if server in ("all", "*"):
        configs = hub.list(include_disabled=False)
        if not configs:
            raise MCPConfigError("No enabled MCP servers are registered; add one with `prompture mcp add`.")
        return [(c, only) for c in configs]
    config = hub.require(server)
    if not config.enabled:
        raise MCPConfigError(f"MCP server {server!r} is disabled; enable it with `prompture mcp enable {server}`.")
    return [(config, only)]


def resolve_mcp_tools(
    name: str,
    *,
    hub: MCPHub | None = None,
    pool: MCPSessionPool | None = None,
) -> list[ToolDefinition]:
    """Resolve ``tools=["mcp:<name>"]`` into tool definitions.

    *name* is ``<server>``, ``<server>/<tool>[,<tool>...]`` or ``all``. A
    registered server wins; otherwise a preset of the same name is used (so
    ``mcp:exa`` works without registering anything). Connects lazily on first
    use and keeps the session pooled for later calls.
    """
    hub = hub or MCPHub()
    pool = pool or get_pool()
    taken: set[str] = set()
    defs: list[ToolDefinition] = []
    for config, only in _parse_spec(name, hub):
        defs.extend(_build_definitions(config, pool.list_tools(config), pool, only, taken))
    return defs


def load_mcp_registry_by_name_sync(
    name: str,
    registry: ToolRegistry | None = None,
    *,
    hub: MCPHub | None = None,
    pool: MCPSessionPool | None = None,
) -> ToolRegistry:
    """Blocking variant of :func:`load_mcp_registry_by_name`."""
    registry = registry if registry is not None else ToolRegistry()
    for td in resolve_mcp_tools(name, hub=hub, pool=pool):
        registry.add(td)
    return registry


async def load_mcp_registry_by_name(
    name: str,
    registry: ToolRegistry | None = None,
    *,
    hub: MCPHub | None = None,
    pool: MCPSessionPool | None = None,
) -> ToolRegistry:
    """Load the tools of registered server *name* (or ``all``) into a :class:`ToolRegistry`.

    The session is pooled on a background loop, so the returned tools work from
    both sync (``registry.execute``) and async (``registry.aexecute``) callers.
    """
    hub = hub or MCPHub()
    pool = pool or get_pool()
    registry = registry if registry is not None else ToolRegistry()
    taken: set[str] = set()
    for config, only in _parse_spec(name, hub):
        tools = await pool.alist_tools(config)
        for td in _build_definitions(config, tools, pool, only, taken):
            registry.add(td)
    return registry


# ---------------------------------------------------------------------------
# One-shot check (initialize + list tools)
# ---------------------------------------------------------------------------


async def acheck_server(
    config: MCPServerConfig,
    *,
    timeout: float | None = None,
    session_factory: SessionFactory | None = None,
) -> dict[str, Any]:
    """Open a fresh session, list tools and close it.

    Returns ``{"name", "status", "tools", "tool_count", "elapsed_ms", "message"}``
    where status is ``ok | unconfigured | timeout | error``. Never raises for
    server-side problems.
    """
    factory = session_factory or open_session
    limit = float(timeout) if timeout else config.startup_timeout
    start = time.monotonic()
    out: dict[str, Any] = {"name": config.name, "transport": config.transport, "target": config.target}
    try:
        config.validate()
        missing = config.missing_env()
        if missing:
            out.update(
                status="unconfigured",
                message=f"set environment variable(s): {', '.join(missing)}",
                tools=[],
                tool_count=0,
            )
            return out

        async def _go() -> list[Any]:
            async with factory(config) as session:
                return await _list_all_tools(session)

        tools = await asyncio.wait_for(_go(), limit)
    except asyncio.TimeoutError:
        out.update(status="timeout", message=f"no answer within {limit:g}s", tools=[], tool_count=0)
    except MCPConfigError as exc:
        out.update(status="error", message=str(exc), tools=[], tool_count=0)
    except Exception as exc:
        out.update(status="error", message=_describe(exc), tools=[], tool_count=0)
    else:
        names = [str(_mcp_field(t, "name")) for t in tools if _mcp_field(t, "name")]
        out.update(status="ok", message=f"{len(names)} tool(s)", tools=names, tool_count=len(names))
    out["elapsed_ms"] = int((time.monotonic() - start) * 1000)
    return out


def check_server(
    config: MCPServerConfig,
    *,
    timeout: float | None = None,
    session_factory: SessionFactory | None = None,
) -> dict[str, Any]:
    """Blocking :func:`acheck_server`; safe to call with or without a running loop."""
    coro = acheck_server(config, timeout=timeout, session_factory=session_factory)
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(asyncio.run, coro).result()
