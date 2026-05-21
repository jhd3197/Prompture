"""Discovery module for auto-detecting available models."""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any, Literal, overload

from ..drivers.provider_descriptors import PROVIDER_DESCRIPTOR_MAP, ProviderDescriptor, _resolve_cls
from .cache import MemoryCacheBackend
from .coding_agent_specs import CODING_AGENT_SPECS, get_spec
from .provider_env import ProviderEnvironment
from .settings import settings

logger = logging.getLogger(__name__)

# Default TTL for discovery cache (seconds)
_DISCOVERY_CACHE_TTL = 300  # 5 minutes

# Module-level cache for discovery results.  Thread-safe via MemoryCacheBackend's
# internal lock.  Keyed by a string describing the call parameters.
_discovery_cache = MemoryCacheBackend(maxsize=32)


def _cfg_value(
    env: ProviderEnvironment | None,
    attr: str,
    env_var: str | None = None,
) -> str | None:
    """Resolve a config value: env -> settings -> os.getenv.

    When *env* is provided, checks ``env.<attr>`` first, then ``settings.<attr>``.
    When *env* is ``None``, checks ``settings.<attr>`` then ``os.getenv(env_var)``.
    """
    if env is not None:
        return env.resolve(attr)
    return getattr(settings, attr, None) or (os.getenv(env_var) if env_var else None)


def _is_provider_configured(provider_name: str, env: ProviderEnvironment | None = None) -> bool:
    """Determine if a provider is configured, using its ProviderDescriptor metadata."""
    desc = PROVIDER_DESCRIPTOR_MAP.get(provider_name)
    if desc is None:
        return False

    # Skip aliases — only check canonical providers.
    if desc.alias_for is not None:
        return False

    # Always-available providers (ollama, lmstudio, airllm).
    if desc.always_available:
        # local_http is special: only available if LOCAL_HTTP_ENDPOINT is set.
        return not (provider_name == "local_http" and not os.getenv("LOCAL_HTTP_ENDPOINT"))

    # Complex check (e.g. Azure).
    if desc.is_configured_fn is not None:
        return desc.is_configured_fn(env=env)

    # Simple check: a settings attribute must be truthy.
    if desc.is_configured_check:
        env_var = desc.is_configured_check.upper()
        return bool(_cfg_value(env, desc.is_configured_check, env_var))

    return False


def _get_list_models_kwargs(
    provider: str,
    env: ProviderEnvironment | None = None,
) -> dict[str, Any]:
    """Return keyword arguments for ``driver_cls.list_models()`` based on provider config."""
    desc = PROVIDER_DESCRIPTOR_MAP.get(provider)
    if desc is None:
        return {}

    kw: dict[str, Any] = {}
    for ctor_kwarg, settings_attr, env_var in desc.list_models_kwargs:
        kw[ctor_kwarg] = _cfg_value(env, settings_attr, env_var)
    return kw


_SOURCE_LABELS: dict[str, str] = {
    "api": "live",
    "static": "known",
    "catalog": "catalog",
}


@dataclasses.dataclass(frozen=True)
class CodingAgentInfo:
    """Availability metadata for an installed coding-agent CLI."""

    id: str
    name: str
    available: bool
    binary: str
    custom_path: bool = False
    source: str | None = None
    verified: bool = False
    healthy: bool | None = None
    error: str | None = None


@dataclasses.dataclass(frozen=True)
class CodingAgentExecutable:
    """A concrete way to invoke a coding-agent CLI."""

    binary: str
    argv: tuple[str, ...]
    display: str
    source: str = "PATH"


_WINDOWS_CLI_EXTENSIONS = (".cmd", ".ps1", ".bat")
_WINDOWS_CMD_EXTENSIONS = (".cmd", ".bat")
_CODING_AGENT_HEALTH_TIMEOUT = 5
_NODE_CLI_ENTRYPOINT_FALLBACKS = (
    Path("dist") / "index.js",
    Path("bin") / "cli.js",
    Path("cli.js"),
    Path("index.js"),
)


def resolve_coding_agent_binary(binary: str) -> str | None:
    """Resolve a coding-agent CLI binary from PATH.

    On Windows, npm-installed CLIs often ship an extensionless shim beside a
    real ``.cmd`` wrapper. Prefer the wrapper extensions before falling back to
    the bare command, matching CachiBot's runtime behavior.
    """
    if not binary:
        return None

    expanded = os.path.expanduser(os.path.expandvars(binary))
    candidates: list[str] = []
    if platform.system() == "Windows":
        root, ext = os.path.splitext(expanded)
        if not ext:
            candidates.extend(f"{root}{suffix}" for suffix in _WINDOWS_CLI_EXTENSIONS)
    candidates.append(expanded)

    for candidate in candidates:
        if path := shutil.which(candidate):
            return path
    return None


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _looks_like_path(value: str) -> bool:
    return (
        os.path.sep in value
        or (os.path.altsep is not None and os.path.altsep in value)
        or (len(value) > 2 and value[1] == ":" and value[2] in "\\/")
    )


def _manual_windows_to_wsl_path(value: str) -> str:
    if len(value) > 2 and value[1] == ":" and value[2] in "\\/":
        drive = value[0].lower()
        rest = value[3:].replace("\\", "/")
        return f"/mnt/{drive}/{rest}"
    return value


def _manual_wsl_to_windows_path(value: Path) -> str:
    text = str(value)
    if text.startswith("/mnt/") and len(text) > 6 and text[6] == "/":
        drive = text[5].upper()
        rest = text[7:].replace("/", "\\")
        return f"{drive}:\\{rest}"
    return text


def _path_for_windows_executable(path: Path) -> str:
    """Convert WSL /mnt paths for Windows executables when possible."""
    if shutil.which("wslpath"):
        try:
            completed = subprocess.run(
                ["wslpath", "-w", str(path)],
                capture_output=True,
                text=True,
                timeout=1,
                check=False,
            )
            if completed.returncode == 0 and completed.stdout.strip():
                return completed.stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            pass
    return _manual_wsl_to_windows_path(path)


def _path_for_windows_node(path: Path) -> str:
    return _path_for_windows_executable(path)


def _quote_windows_cmd_path(value: str) -> str:
    if any(char.isspace() for char in value):
        return f'"{value}"'
    return value


def _command_name_variants(binary: str) -> list[str]:
    if _looks_like_path(binary):
        return [os.path.expanduser(os.path.expandvars(binary))]

    variants = [binary]
    root, ext = os.path.splitext(binary)
    if not ext:
        variants.extend(f"{root}{suffix}" for suffix in _WINDOWS_CLI_EXTENSIONS)
    return _dedupe(variants)


def _existing_path(value: str) -> str | None:
    expanded = _manual_windows_to_wsl_path(os.path.expanduser(os.path.expandvars(value)))
    path = Path(expanded)
    if path.is_file():
        return str(path)
    return None


def _resolved_paths_for_binary(binary: str) -> list[str]:
    paths: list[str] = []
    for variant in _command_name_variants(binary):
        if existing := _existing_path(variant):
            paths.append(existing)
        if found := shutil.which(variant):
            paths.append(found)
    return _dedupe(paths)


@functools.lru_cache(maxsize=1)
def _npm_global_prefixes_cached() -> tuple[Path, ...]:
    """Shell out to ``npm prefix -g`` at most once per process.

    Discovery is invoked many times across a process lifetime (CLI command,
    health-check loop, server endpoint), and the npm global prefix is stable
    inside a single process. Cache the result so we don't re-spawn npm for
    every coding-agent lookup. Callers can :meth:`cache_clear` if they ever
    need to force a refresh.
    """
    prefixes: list[Path] = []
    for npm_binary in _resolved_paths_for_binary("npm"):
        try:
            completed = subprocess.run(
                [npm_binary, "prefix", "-g"],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if completed.returncode != 0:
            continue
        prefix = completed.stdout.strip()
        if prefix:
            prefixes.append(Path(_manual_windows_to_wsl_path(prefix)))
    return tuple(dict.fromkeys(prefixes))


def _npm_global_prefixes() -> list[Path]:
    return list(_npm_global_prefixes_cached())


def _package_path(package_name: str) -> Path:
    return Path(*package_name.split("/"))


def _package_command_name(package_name: str) -> str:
    return package_name.rsplit("/", 1)[-1]


def _npm_package_dir_candidates(base_dir: Path, package_name: str) -> list[Path]:
    package_path = _package_path(package_name)
    bases = [base_dir]
    try:
        resolved = base_dir.resolve()
    except OSError:
        resolved = base_dir
    bases.append(resolved)

    candidates: list[Path] = []
    for base in dict.fromkeys(bases):
        candidates.extend(
            [
                base / "node_modules" / package_path,
                base / "lib" / "node_modules" / package_path,
                base.parent / "node_modules" / package_path,
                base.parent / "lib" / "node_modules" / package_path,
            ]
        )
    return [path for path in dict.fromkeys(candidates) if path.is_dir()]


def _npm_package_entrypoints(
    package_dir: Path,
    *,
    command_name: str,
    package_name: str,
) -> list[Path]:
    rel_paths: list[Path] = []
    package_json = package_dir / "package.json"
    if package_json.is_file():
        try:
            metadata = json.loads(package_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metadata = {}
        bin_spec = metadata.get("bin") if isinstance(metadata, dict) else None
        if isinstance(bin_spec, str):
            rel_paths.append(Path(bin_spec))
        elif isinstance(bin_spec, dict):
            preferred_names = [
                command_name,
                package_name,
                _package_command_name(package_name),
            ]
            for name in preferred_names:
                entry = bin_spec.get(name)
                if isinstance(entry, str):
                    rel_paths.append(Path(entry))
            rel_paths.extend(Path(entry) for entry in bin_spec.values() if isinstance(entry, str))

    rel_paths.extend(_NODE_CLI_ENTRYPOINT_FALLBACKS)

    candidates = [path if path.is_absolute() else package_dir / path for path in dict.fromkeys(rel_paths)]
    return [path for path in candidates if path.is_file()]


def _node_binary_candidates_for_dir(base_dir: Path) -> list[str]:
    node_paths = [
        str(base_dir / "node"),
        str(base_dir / "node.exe"),
        str(base_dir / "bin" / "node"),
        str(base_dir / "bin" / "node.exe"),
        *_resolved_paths_for_binary("node"),
        *_resolved_paths_for_binary("node.exe"),
    ]
    try:
        resolved = base_dir.resolve()
    except OSError:
        resolved = base_dir
    node_paths.extend(
        [
            str(resolved / "node"),
            str(resolved / "node.exe"),
            str(resolved / "bin" / "node"),
            str(resolved / "bin" / "node.exe"),
        ]
    )

    return [node for node in _dedupe(node_paths) if Path(node).is_file() or shutil.which(node) is not None]


def _node_package_candidates_from_dir(
    agent_id: str,
    base_dir: Path,
) -> list[CodingAgentExecutable]:
    executables: list[CodingAgentExecutable] = []
    spec = get_spec(agent_id)
    packages = spec.npm_packages if spec else ()
    if not packages:
        return executables

    node_paths = _node_binary_candidates_for_dir(base_dir)
    if not node_paths:
        return executables

    command_name = spec.default_binary if spec else agent_id
    for package_name in packages:
        package_dirs = _npm_package_dir_candidates(base_dir, package_name)
        if not package_dirs:
            continue
        for package_dir in package_dirs:
            entrypoints = _npm_package_entrypoints(
                package_dir,
                command_name=command_name,
                package_name=package_name,
            )
            for node in node_paths:
                is_windows_node = node.lower().endswith(".exe")
                for entrypoint in entrypoints:
                    entry_arg = _path_for_windows_node(entrypoint) if is_windows_node else str(entrypoint)
                    argv = (node, "--no-warnings=DEP0040", entry_arg)
                    executables.append(
                        CodingAgentExecutable(
                            binary=node,
                            argv=argv,
                            display=" ".join(argv),
                            source=f"npm:{package_name}",
                        )
                    )
    return executables


def _node_package_candidates_from_shim(
    agent_id: str,
    shim_path: str,
) -> list[CodingAgentExecutable]:
    spec = get_spec(agent_id)
    if spec is None or not spec.npm_packages:
        return []

    path = Path(shim_path)
    bases = [path.parent]
    with suppress(OSError):
        bases.append(path.resolve().parent)

    candidates: list[CodingAgentExecutable] = []
    for base in dict.fromkeys(bases):
        candidates.extend(_node_package_candidates_from_dir(agent_id, base))
    return candidates


def _windows_cmd_wrapper_candidates(path: str) -> list[CodingAgentExecutable]:
    """Return cmd.exe-based candidates for Windows npm .cmd/.bat wrappers."""
    if not path.lower().endswith(_WINDOWS_CMD_EXTENSIONS):
        return []

    cmd_paths = _resolved_paths_for_binary("cmd.exe")
    if not cmd_paths and (cmd := shutil.which("cmd.exe")):
        cmd_paths = [cmd]

    windows_path = _quote_windows_cmd_path(_path_for_windows_executable(Path(path)))
    return [
        CodingAgentExecutable(
            binary=cmd,
            argv=(cmd, "/d", "/s", "/c", windows_path),
            display=f"{cmd} /d /s /c {windows_path}",
            source="windows-cmd",
        )
        for cmd in cmd_paths
    ]


def _is_windows_wrapper_path(path: str) -> bool:
    return path.lower().endswith(_WINDOWS_CLI_EXTENSIONS)


def _is_windows_npm_shim_path(path: str, all_paths: list[str]) -> bool:
    lower = path.lower()
    if not lower.startswith("/mnt/"):
        return False
    return any(other.lower() in {f"{lower}.cmd", f"{lower}.bat"} for other in all_paths)


def _extra_node_package_candidates(
    agent_id: str,
    binary: str,
) -> list[CodingAgentExecutable]:
    # Skip the entire npm-prefix probe for agents that don't ship via npm
    # (aider via pip, cursor-agent from the Cursor installer, crush from
    # brew/scoop, etc.). Avoids an `npm prefix -g` subprocess per resolution.
    spec = get_spec(agent_id)
    if spec is None or not spec.npm_packages:
        return []

    candidates: list[CodingAgentExecutable] = []
    if _looks_like_path(binary):
        path = Path(_manual_windows_to_wsl_path(os.path.expanduser(os.path.expandvars(binary))))
        bases = [path.parent if path.is_file() or path.suffix else path]
        for base in bases:
            candidates.extend(_node_package_candidates_from_dir(agent_id, base))
        return candidates

    for prefix in _npm_global_prefixes():
        candidates.extend(_node_package_candidates_from_dir(agent_id, prefix))
        candidates.extend(_node_package_candidates_from_dir(agent_id, prefix / "bin"))
    return candidates


def _dedupe_executables(candidates: list[CodingAgentExecutable]) -> list[CodingAgentExecutable]:
    seen: set[tuple[str, ...]] = set()
    unique: list[CodingAgentExecutable] = []
    for candidate in candidates:
        if candidate.argv not in seen:
            seen.add(candidate.argv)
            unique.append(candidate)
    return unique


# Backwards-compatible internal helpers kept for tests and downstream users
# that reached into this private module during the first implementation.
def _gemini_node_candidates_from_dir(base_dir: Path) -> list[CodingAgentExecutable]:
    return _node_package_candidates_from_dir("gemini", base_dir)


def _gemini_node_candidates_from_shim(shim_path: str) -> list[CodingAgentExecutable]:
    return _node_package_candidates_from_shim("gemini", shim_path)


def get_coding_agent_executable_candidates(
    agent_id: str,
    binary: str,
) -> list[CodingAgentExecutable]:
    """Return candidate executable forms for a coding-agent CLI."""
    paths = _resolved_paths_for_binary(binary)
    candidates: list[CodingAgentExecutable] = []
    late_direct_candidates: list[CodingAgentExecutable] = []

    for path in paths:
        direct = CodingAgentExecutable(binary=path, argv=(path,), display=path)
        if _is_windows_wrapper_path(path) or _is_windows_npm_shim_path(path, paths):
            late_direct_candidates.append(direct)
        else:
            candidates.append(direct)

    for path in paths:
        candidates.extend(_windows_cmd_wrapper_candidates(path))

    candidates.extend(late_direct_candidates)

    for path in paths:
        candidates.extend(_node_package_candidates_from_shim(agent_id, path))
    candidates.extend(_extra_node_package_candidates(agent_id, binary))

    return _dedupe_executables(candidates)


def _summarize_health_error(output: str) -> str:
    """Return a compact single-line health-check failure reason."""
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        return "health check failed with no output"
    return lines[0][:240]


def verify_coding_agent_binary(
    agent_id: str,
    binary: str,
    *,
    timeout: int = _CODING_AGENT_HEALTH_TIMEOUT,
) -> tuple[bool, str | None]:
    """Run a lightweight health check for a resolved coding-agent binary."""
    try:
        completed = subprocess.run(
            [binary, "--version"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, f"{agent_id} health check timed out after {timeout}s"
    except OSError as exc:
        return False, f"{agent_id} could not start: {exc}"

    if completed.returncode == 0:
        return True, None

    output = "\n".join(part for part in (completed.stderr, completed.stdout) if part)
    return False, _summarize_health_error(output)


def verify_coding_agent_executable(
    agent_id: str,
    executable: CodingAgentExecutable,
    *,
    timeout: int = _CODING_AGENT_HEALTH_TIMEOUT,
) -> tuple[bool, str | None]:
    """Run a lightweight health check for a candidate executable."""
    try:
        completed = subprocess.run(
            [*executable.argv, "--version"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, f"{agent_id} health check timed out after {timeout}s"
    except OSError as exc:
        return False, f"{agent_id} could not start: {exc}"

    if completed.returncode == 0:
        return True, None

    output = "\n".join(part for part in (completed.stderr, completed.stdout) if part)
    return False, _summarize_health_error(output)


def resolve_coding_agent_executable(
    agent_id: str,
    binary: str,
    *,
    verify: bool = False,
    verify_timeout: int = _CODING_AGENT_HEALTH_TIMEOUT,
) -> tuple[CodingAgentExecutable | None, bool | None, str | None]:
    """Resolve a coding-agent executable, optionally picking the first healthy candidate."""
    candidates = get_coding_agent_executable_candidates(agent_id, binary)
    if not candidates:
        return None, False if verify else None, f"'{binary}' is not installed or not on PATH"

    if not verify:
        return candidates[0], None, None

    last_error: str | None = None
    for candidate in candidates:
        healthy, error = verify_coding_agent_executable(agent_id, candidate, timeout=verify_timeout)
        if healthy:
            return candidate, True, None
        last_error = error

    return candidates[0], False, last_error


def get_available_coding_agents(
    *,
    agent_paths: Mapping[str, str] | None = None,
    include_unavailable: bool = True,
    verify: bool = False,
    verify_timeout: int = _CODING_AGENT_HEALTH_TIMEOUT,
) -> list[CodingAgentInfo]:
    """Auto-detect installed coding-agent CLIs.

    Args:
        agent_paths: Optional explicit binary paths by agent id. Any id
            registered in :data:`CODING_AGENT_SPECS` is accepted (Claude,
            Codex, Gemini, Qwen, Aider, OpenCode, Cursor Agent, Crush today,
            plus anything callers have added). Explicit paths override PATH
            discovery for that agent.
        include_unavailable: When ``True`` (default), return all known agents
            with ``available=False`` for missing CLIs. When ``False``, return
            only discovered agents.
        verify: When ``True``, run ``--version`` for resolved binaries and
            mark CLIs unavailable when the executable shim fails.
        verify_timeout: Timeout in seconds for each health check.

    Returns:
        A list of :class:`CodingAgentInfo` entries suitable for app settings,
        command autocomplete, or local capability checks.
    """
    configured_paths = agent_paths or {}
    agents: list[CodingAgentInfo] = []

    for agent_id, spec in CODING_AGENT_SPECS.items():
        display_name = spec.display_name
        default_binary = spec.default_binary
        custom_binary = configured_paths.get(agent_id, "")
        binary_name = custom_binary or default_binary
        executable, healthy, error = resolve_coding_agent_executable(
            agent_id,
            binary_name,
            verify=verify,
            verify_timeout=verify_timeout,
        )

        available = executable is not None and (healthy is not False)

        if not available and not include_unavailable:
            continue

        binary_display = executable.display if executable is not None else binary_name
        source = executable.source if executable is not None else None
        agents.append(
            CodingAgentInfo(
                id=agent_id,
                name=display_name,
                available=available,
                binary=binary_display,
                custom_path=bool(custom_binary),
                source=source,
                verified=verify,
                healthy=healthy,
                error=error,
            )
        )

    return agents


def pick_best_coding_agent(
    *,
    prefer: Sequence[str] | None = None,
    require_tool_use: bool = False,
    require_structured_output: bool = False,
    require_questions: bool = False,
    require_session_resume: bool = False,
    agent_paths: Mapping[str, str] | None = None,
    verify: bool = True,
    verify_timeout: int = _CODING_AGENT_HEALTH_TIMEOUT,
) -> CodingAgentInfo | None:
    """Pick the most preferred installed coding-agent CLI matching capability requirements.

    Combines :func:`get_available_coding_agents` (filtered to healthy
    entries) with the capability flags on :class:`CodingAgentSpec` so
    callers can ask for "a healthy CLI that supports session resume"
    without writing the loop themselves.

    Args:
        prefer: Ordered list of agent ids to try first.  When any of
            these is healthy and meets the requirements, it wins.  When
            None or empty, the spec registry's default ordering is used.
        require_tool_use: Reject CLIs whose spec sets
            ``supports_tool_use=False``.
        require_structured_output: Reject CLIs without a JSON event
            parser (``spec.supports_structured_output``).
        require_questions: Reject CLIs that don't emit clarifying
            ``question`` events.
        require_session_resume: Reject CLIs that don't support
            ``run_coding_agent(..., session_id=...)``.
        agent_paths: Forwarded to :func:`get_available_coding_agents`.
        verify: Run health checks (default True so we never pick a
            broken shim).
        verify_timeout: Health-check timeout.

    Returns:
        The matching :class:`CodingAgentInfo`, or ``None`` when nothing
        on the box satisfies the requirements.
    """
    from .coding_agent_specs import get_spec

    candidates = get_available_coding_agents(
        agent_paths=agent_paths,
        include_unavailable=False,
        verify=verify,
        verify_timeout=verify_timeout,
    )
    healthy = [info for info in candidates if info.available and info.healthy is not False]

    def _meets_requirements(info: CodingAgentInfo) -> bool:
        spec = get_spec(info.id)
        if spec is None:
            return False
        if require_tool_use and not spec.supports_tool_use:
            return False
        if require_structured_output and not spec.supports_structured_output:
            return False
        if require_questions and not spec.supports_questions:
            return False
        if require_session_resume and not spec.supports_session_resume:
            return False
        return True

    by_id = {info.id: info for info in healthy}
    if prefer:
        for agent_id in prefer:
            info = by_id.get(agent_id)
            if info is not None and _meets_requirements(info):
                return info

    for info in healthy:
        if _meets_requirements(info):
            return info
    return None


def _config_via(desc: ProviderDescriptor) -> str:
    """Derive the env var or label that provides access for this provider."""
    if desc.always_available:
        return "local"
    if desc.is_configured_check:
        return desc.is_configured_check.upper()
    # Complex providers (Azure) — use name-based convention.
    return f"{desc.name.upper()}_API_KEY"


def _get_driver_class(provider_name: str) -> type | None:
    """Resolve the sync LLM driver class for a provider from its descriptor."""
    desc = PROVIDER_DESCRIPTOR_MAP.get(provider_name)
    if desc is None or desc.llm_sync is None:
        return None
    return _resolve_cls(desc.llm_sync.cls_path)


@overload
def get_available_models(
    *,
    env: ProviderEnvironment | None = None,
    include_capabilities: Literal[False] = False,
    include_source: Literal[False] = False,
    grouped: Literal[False] = False,
    verified_only: bool = False,
    force_refresh: bool = False,
    cache_ttl: int | None = None,
) -> list[str]: ...


@overload
def get_available_models(  # type: ignore[overload-overlap]
    *,
    env: ProviderEnvironment | None = None,
    include_capabilities: Literal[True],
    include_source: bool = False,
    grouped: Literal[False] = False,
    verified_only: bool = False,
    force_refresh: bool = False,
    cache_ttl: int | None = None,
) -> list[dict[str, Any]]: ...


@overload
def get_available_models(
    *,
    env: ProviderEnvironment | None = None,
    include_capabilities: Literal[False] = False,
    include_source: Literal[True] = ...,
    grouped: Literal[False] = False,
    verified_only: bool = False,
    force_refresh: bool = False,
    cache_ttl: int | None = None,
) -> list[dict[str, str]]: ...


@overload
def get_available_models(
    *,
    env: ProviderEnvironment | None = None,
    include_capabilities: bool = False,
    include_source: bool = False,
    grouped: Literal[True] = ...,
    verified_only: bool = False,
    force_refresh: bool = False,
    cache_ttl: int | None = None,
) -> dict[str, dict[str, Any]]: ...


def get_available_models(
    *,
    env: ProviderEnvironment | None = None,
    include_capabilities: bool = False,
    include_source: bool = False,
    grouped: bool = False,
    verified_only: bool = False,
    force_refresh: bool = False,
    cache_ttl: int | None = None,
) -> list[str] | list[dict[str, Any]] | list[dict[str, str]] | dict[str, dict[str, Any]]:
    """Auto-detect available models based on configured drivers and environment variables.

    Iterates through supported providers and checks if they are configured
    (e.g. API key present).  For static drivers, returns models from their
    ``MODEL_PRICING`` keys.  For dynamic drivers (like Ollama), attempts to
    fetch available models from the endpoint.

    Results are cached in memory with a configurable TTL to avoid redundant
    provider queries on repeated calls.

    Args:
        env: Optional per-consumer environment for isolated API keys.
            When ``None``, uses the global settings singleton (current behavior).
        include_capabilities: When ``True``, return enriched dicts with
            ``model``, ``provider``, ``model_id``, and ``capabilities``
            fields instead of plain ``"provider/model_id"`` strings.
        include_source: When ``True`` (and ``grouped=False`` and
            ``include_capabilities=False``), return ``list[dict]`` with
            ``{"model": ..., "source": "live"|"known"|"catalog"}`` instead
            of plain strings.
        grouped: When ``True``, return a ``dict`` keyed by provider name
            with ``display_name``, ``source``, ``config_via``, and
            ``models`` fields.
        verified_only: When ``True``, only return models that have been
            successfully used (as recorded by the usage ledger).
        force_refresh: When ``True``, bypass the cache and re-query all
            providers.
        cache_ttl: Cache time-to-live in seconds.  Defaults to
            ``_DISCOVERY_CACHE_TTL`` (300 s / 5 min).

    Returns:
        A sorted list of unique model strings (default), enriched dicts,
        source-annotated dicts, or a grouped dict by provider.
    """
    ttl = cache_ttl if cache_ttl is not None else _DISCOVERY_CACHE_TTL

    # Build cache key — include a hash of env-provided fields to avoid cross-bot pollution
    mode = "grouped" if grouped else ("source" if include_source else str(include_capabilities))
    if env is not None:
        env_sig = tuple(sorted(k for k, v in vars(env).items() if v is not None))
        env_hash = hashlib.md5(str(env_sig).encode(), usedforsecurity=False).hexdigest()[:8]
        cache_key = f"models:{mode}:{verified_only}:env={env_hash}"
    else:
        cache_key = f"models:{mode}:{verified_only}"

    if not force_refresh:
        cached = _discovery_cache.get(cache_key)
        if cached is not None:
            logger.debug("Discovery cache hit for key=%s", cache_key)
            return cached  # type: ignore[no-any-return]

    logger.debug("Discovery cache miss for key=%s — querying providers", cache_key)
    # Maps model string -> source ("api", "static", "catalog"). First source wins.
    model_sources: dict[str, str] = {}
    configured_providers: set[str] = set()
    # Providers where list_models() returned an authoritative model list.
    # For these, we trust the API response and skip MODEL_PRICING / models.dev additions.
    api_authoritative_providers: set[str] = set()

    # Iterate canonical providers from descriptors (skip aliases).
    for provider_name, desc in PROVIDER_DESCRIPTOR_MAP.items():
        if desc.alias_for is not None:
            continue

        try:
            if not _is_provider_configured(provider_name, env):
                continue

            configured_providers.add(provider_name)

            driver_cls = _get_driver_class(provider_name)
            if driver_cls is None:
                continue

            # API-based Detection: call driver's list_models() classmethod.
            # When this succeeds, the API response is authoritative — it reflects
            # exactly which models the API key has access to.
            api_kwargs = _get_list_models_kwargs(provider_name, env=env)
            api_succeeded = False
            try:
                api_models = driver_cls.list_models(**api_kwargs)  # type: ignore[attr-defined]
                if api_models is not None:
                    api_succeeded = True
                    api_authoritative_providers.add(provider_name)
                    for model_id in api_models:
                        model_str = f"{provider_name}/{model_id}"
                        if model_str not in model_sources:
                            model_sources[model_str] = "api"
            except Exception as e:
                logger.warning("list_models() failed for %s: %s", provider_name, e)

            # Static Detection: Get models from the capabilities knowledge base
            # (JSON rate files) when the API didn't return an authoritative list
            # (failed, returned None, or the driver doesn't implement list_models).
            if not api_succeeded:
                from .model_rates import get_kb_models_for_provider

                for model_id in get_kb_models_for_provider(provider_name):
                    model_str = f"{provider_name}/{model_id}"
                    if model_str not in model_sources:
                        model_sources[model_str] = "static"

        except Exception as e:
            logger.warning(f"Error detecting models for provider {provider_name}: {e}")
            continue

    # Enrich with live model list from models.dev cache — but only for providers
    # where the API didn't return an authoritative list.  When list_models()
    # succeeded, the API already told us exactly which models the key can use;
    # adding the full models.dev catalog would surface models the user can't
    # actually access (e.g. GPT-5 when the key only has GPT-4o access).
    from .model_rates import PROVIDER_MAP, get_all_provider_models

    for prompture_name, api_name in PROVIDER_MAP.items():
        if prompture_name in configured_providers and prompture_name not in api_authoritative_providers:
            for model_id in get_all_provider_models(api_name):
                model_str = f"{prompture_name}/{model_id}"
                if model_str not in model_sources:
                    model_sources[model_str] = "catalog"

    sorted_models = sorted(model_sources.keys())

    # --- verified_only filtering ---
    verified_set: set[str] | None = None
    if verified_only or include_capabilities:
        try:
            from .ledger import _get_ledger

            ledger = _get_ledger()
            verified_set = ledger.get_verified_models()
        except Exception:
            logger.debug("Could not load ledger for verified models", exc_info=True)
            verified_set = set()

    if verified_only and verified_set is not None:
        sorted_models = [m for m in sorted_models if m in verified_set]

    # ── grouped mode ────────────────────────────────────────────────────
    if grouped:
        groups: dict[str, dict[str, Any]] = {}
        for model_str in sorted_models:
            provider = model_str.split("/", 1)[0]
            model_id = model_str.split("/", 1)[1] if "/" in model_str else model_str
            if provider not in groups:
                prov_desc = PROVIDER_DESCRIPTOR_MAP.get(provider)
                source_raw = model_sources.get(model_str, "static")
                groups[provider] = {
                    "display_name": prov_desc.display_name if prov_desc else provider,
                    "source": _SOURCE_LABELS.get(source_raw, source_raw),
                    "config_via": _config_via(prov_desc) if prov_desc else "unknown",
                    "models": [],
                }
            groups[provider]["models"].append(model_id)
        _discovery_cache.set(cache_key, groups, ttl=ttl)
        return groups

    # ── include_source mode ───────────────────────────────────────────
    if include_source and not include_capabilities:
        source_list: list[dict[str, str]] = [
            {
                "model": m,
                "source": _SOURCE_LABELS.get(model_sources.get(m, "static"), model_sources.get(m, "static")),
            }
            for m in sorted_models
        ]
        _discovery_cache.set(cache_key, source_list, ttl=ttl)
        return source_list

    if not include_capabilities:
        _discovery_cache.set(cache_key, sorted_models, ttl=ttl)
        return sorted_models

    # Build enriched dicts with capabilities and lifecycle from models.dev
    from .model_rates import get_model_capabilities, get_model_lifecycle

    # Fetch all ledger stats for annotation (keyed by model_name)
    ledger_stats: dict[str, dict[str, Any]] = {}
    try:
        from .ledger import _get_ledger

        for row in _get_ledger().get_all_stats():
            name = row["model_name"]
            if name not in ledger_stats:
                ledger_stats[name] = row
            else:
                # Aggregate across API key hashes
                existing = ledger_stats[name]
                existing["use_count"] += row["use_count"]
                existing["total_tokens"] += row["total_tokens"]
                existing["total_cost"] += row["total_cost"]
                if row["last_used"] > existing["last_used"]:
                    existing["last_used"] = row["last_used"]
    except Exception:
        logger.debug("Could not load ledger stats for enrichment", exc_info=True)

    enriched: list[dict[str, Any]] = []
    for model_str in sorted_models:
        parts = model_str.split("/", 1)
        provider = parts[0]
        model_id = parts[1] if len(parts) > 1 else parts[0]

        caps = get_model_capabilities(provider, model_id)
        caps_dict = dataclasses.asdict(caps) if caps is not None else None

        lifecycle = get_model_lifecycle(provider, model_id)

        entry: dict[str, Any] = {
            "model": model_str,
            "provider": provider,
            "model_id": model_id,
            "capabilities": caps_dict,
            "source": model_sources.get(model_str, "unknown"),
            "verified": verified_set is not None and model_str in verified_set,
            "status": lifecycle.get("status") if lifecycle else None,
            "family": lifecycle.get("family") if lifecycle else None,
            "release_date": lifecycle.get("release_date") if lifecycle else None,
            "superseded_by": lifecycle.get("superseded_by") if lifecycle else None,
            "end_of_support": lifecycle.get("end_of_support") if lifecycle else None,
        }

        stats = ledger_stats.get(model_str)
        if stats:
            entry["last_used"] = stats["last_used"]
            entry["use_count"] = stats["use_count"]
        else:
            entry["last_used"] = None
            entry["use_count"] = 0

        enriched.append(entry)

    _discovery_cache.set(cache_key, enriched, ttl=ttl)
    return enriched


def display_available_models(
    *,
    env: ProviderEnvironment | None = None,
    force_refresh: bool = False,
    return_string: bool = False,
) -> str | None:
    """Print a human-friendly grouped view of available models.

    Args:
        env: Optional per-consumer environment for isolated API keys.
        force_refresh: Bypass cache and re-query providers.
        return_string: When ``True``, return the formatted string instead
            of printing it.

    Returns:
        The formatted string when *return_string* is ``True``, else ``None``.
    """
    grouped = get_available_models(grouped=True, env=env, force_refresh=force_refresh)

    lines: list[str] = []
    lines.append("Available Models")
    lines.append("=" * 60)
    lines.append("")

    for provider, info in sorted(grouped.items()):
        display = info.get("display_name") or provider
        config = info.get("config_via", "unknown")
        source = info.get("source", "unknown")
        models: list[str] = info.get("models", [])
        count = len(models)

        if config == "local":
            header = f"{display} (local)  [{source} - {count} model{'s' if count != 1 else ''}]"
        else:
            header = f"{display} (via {config})  [{source} - {count} model{'s' if count != 1 else ''}]"
        lines.append(header)

        # 3-column layout
        col_width = 24
        for i in range(0, len(models), 3):
            row = models[i : i + 3]
            lines.append("  " + "".join(m.ljust(col_width) for m in row).rstrip())

        lines.append("")

    lines.append("Legend: live = confirmed by API | known = from rate files | catalog = from models.dev")

    output = "\n".join(lines)

    if return_string:
        return output

    print(output, file=sys.stdout)
    return None


def clear_discovery_cache() -> None:
    """Clear the in-memory discovery cache, forcing the next call to re-query providers."""
    _discovery_cache.clear()


def get_available_audio_models(
    *,
    env: ProviderEnvironment | None = None,
    modality: str | None = None,
) -> list[str]:
    """Auto-detect available audio models (STT and TTS) based on configured API keys.

    Checks which audio providers are configured and returns their supported models.

    Args:
        env: Optional per-consumer environment for isolated API keys.
            When ``None``, uses the global settings singleton (current behavior).
        modality: Filter by ``"stt"`` or ``"tts"``. Returns both when ``None``.

    Returns:
        A sorted list of unique model strings (e.g. ``"openai/whisper-1"``).
    """
    from ..drivers.openai_stt_driver import OpenAISTTDriver
    from ..drivers.openai_tts_driver import OpenAITTSDriver

    available: set[str] = set()

    # OpenAI audio models (requires same openai_api_key as LLM)
    if _cfg_value(env, "openai_api_key", "OPENAI_API_KEY"):
        if modality is None or modality == "stt":
            for model_id in OpenAISTTDriver.AUDIO_PRICING:
                available.add(f"openai/{model_id}")
        if modality is None or modality == "tts":
            for model_id in OpenAITTSDriver.AUDIO_PRICING:
                available.add(f"openai/{model_id}")

    # ElevenLabs audio models
    elevenlabs_key = _cfg_value(env, "elevenlabs_api_key", "ELEVENLABS_API_KEY")
    elevenlabs_endpoint = getattr(settings, "elevenlabs_endpoint", None) or "https://api.elevenlabs.io/v1"
    if elevenlabs_key:
        from ..drivers.elevenlabs_stt_driver import ElevenLabsSTTDriver
        from ..drivers.elevenlabs_tts_driver import ElevenLabsTTSDriver

        if modality is None or modality == "stt":
            # STT: static list (no API listing endpoint for STT models)
            stt_models = ElevenLabsSTTDriver.list_models(api_key=elevenlabs_key, endpoint=elevenlabs_endpoint)
            if stt_models:
                for model_id in stt_models:
                    available.add(f"elevenlabs/{model_id}")

        if modality is None or modality == "tts":
            # TTS: dynamic discovery via GET /v1/models, fallback to AUDIO_PRICING
            tts_models = ElevenLabsTTSDriver.list_models(api_key=elevenlabs_key, endpoint=elevenlabs_endpoint)
            if tts_models is not None:
                for model_id in tts_models:
                    available.add(f"elevenlabs/{model_id}")
            else:
                for model_id in ElevenLabsTTSDriver.AUDIO_PRICING:
                    available.add(f"elevenlabs/{model_id}")

    # Cartesia TTS models
    if _cfg_value(env, "cartesia_api_key", "CARTESIA_API_KEY") and (modality is None or modality == "tts"):
        for model_id in (
            "sonic-2",
            "sonic-2-2025-03-07",
            "sonic-turbo",
            "sonic-english",
            "sonic-multilingual",
        ):
            available.add(f"cartesia/{model_id}")

    # Deepgram STT + TTS models
    if _cfg_value(env, "deepgram_api_key", "DEEPGRAM_API_KEY"):
        if modality is None or modality == "stt":
            for model_id in (
                "nova-3",
                "nova-3-medical",
                "nova-2",
                "nova-2-meeting",
                "nova-2-phonecall",
                "enhanced",
                "base",
            ):
                available.add(f"deepgram/{model_id}")
        if modality is None or modality == "tts":
            for model_id in (
                "aura-2-thalia-en",
                "aura-2-arcas-en",
                "aura-2-perseus-en",
                "aura-asteria-en",
                "aura-luna-en",
            ):
                available.add(f"deepgram/{model_id}")

    # AssemblyAI STT models
    if _cfg_value(env, "assemblyai_api_key", "ASSEMBLYAI_API_KEY") and (modality is None or modality == "stt"):
        for model_id in ("universal", "nano", "best", "slam-1"):
            available.add(f"assemblyai/{model_id}")

    # Runway TTS + sound-effect models (no STT)
    runway_key = _cfg_value(env, "runway_api_key", "RUNWAY_API_KEY") or _cfg_value(
        env, "runwayml_api_secret", "RUNWAYML_API_SECRET"
    )
    if runway_key and (modality is None or modality == "tts"):
        from ..drivers.runway_tts_driver import RunwayTTSDriver

        for model_id in RunwayTTSDriver.KNOWN_MODELS:
            available.add(f"runway/{model_id}")

    # Dynamic discovery: check modalities_output from models.dev capabilities
    # for any models that the pricing dicts don't know about yet.
    from .model_rates import get_model_capabilities

    for model_str in get_available_models(env=env):
        parts = model_str.split("/", 1)
        if len(parts) != 2:
            continue
        provider, model_id = parts
        caps = get_model_capabilities(provider, model_id)
        if caps and "audio" in caps.modalities_output and (modality is None or modality == "tts"):
            available.add(model_str)

    return sorted(available)


def get_available_image_gen_models(
    *,
    env: ProviderEnvironment | None = None,
) -> list[str]:
    """Auto-detect available image generation models based on configured API keys.

    Checks which image gen providers are configured and returns their supported models.

    Args:
        env: Optional per-consumer environment for isolated API keys.
            When ``None``, uses the global settings singleton (current behavior).

    Returns:
        A sorted list of unique model strings (e.g. ``"openai/dall-e-3"``).
    """
    from ..drivers.google_img_gen_driver import GoogleImageGenDriver
    from ..drivers.grok_img_gen_driver import GrokImageGenDriver
    from ..drivers.openai_img_gen_driver import OpenAIImageGenDriver
    from ..drivers.stability_img_gen_driver import StabilityImageGenDriver

    available: set[str] = set()

    # OpenAI image gen models (requires openai_api_key)
    if _cfg_value(env, "openai_api_key", "OPENAI_API_KEY"):
        for model_id in OpenAIImageGenDriver.IMAGE_PRICING:
            available.add(f"openai/{model_id}")

    # Google image gen models (requires google_api_key)
    if _cfg_value(env, "google_api_key", "GOOGLE_API_KEY"):
        for model_id in GoogleImageGenDriver.IMAGE_PRICING:
            available.add(f"google/{model_id}")

    # Stability AI image gen models
    stability_key = _cfg_value(env, "stability_api_key", "STABILITY_API_KEY")
    if stability_key:
        for model_id in StabilityImageGenDriver.IMAGE_PRICING:
            available.add(f"stability/{model_id}")

    # Grok/xAI image gen models (requires grok_api_key)
    if _cfg_value(env, "grok_api_key", "GROK_API_KEY"):
        for model_id in GrokImageGenDriver.IMAGE_PRICING:
            available.add(f"grok/{model_id}")

    # Runway image gen models (requires runway_api_key or RUNWAYML_API_SECRET)
    runway_key = _cfg_value(env, "runway_api_key", "RUNWAY_API_KEY") or _cfg_value(
        env, "runwayml_api_secret", "RUNWAYML_API_SECRET"
    )
    if runway_key:
        from ..drivers.runway_img_gen_driver import RunwayImageGenDriver

        for model_id in RunwayImageGenDriver.KNOWN_MODELS:
            available.add(f"runway/{model_id}")

    # Ideogram image gen models
    if _cfg_value(env, "ideogram_api_key", "IDEOGRAM_API_KEY"):
        from ..drivers.ideogram_img_gen_driver import IdeogramImageGenDriver

        for model_id in IdeogramImageGenDriver.KNOWN_MODELS:
            available.add(f"ideogram/{model_id}")

    # Black Forest Labs (BFL) — direct Flux image gen models
    if _cfg_value(env, "bfl_api_key", "BFL_API_KEY"):
        from ..drivers.bfl_img_gen_driver import BFLImageGenDriver

        for model_id in BFLImageGenDriver.KNOWN_MODELS:
            available.add(f"bfl/{model_id}")

    # Dynamic discovery: check modalities_output from models.dev capabilities
    # for any models that the pricing dicts don't know about yet.
    from .model_rates import get_model_capabilities

    for model_str in get_available_models(env=env):
        parts = model_str.split("/", 1)
        if len(parts) != 2:
            continue
        provider, model_id = parts
        caps = get_model_capabilities(provider, model_id)
        if caps and "image" in caps.modalities_output:
            available.add(model_str)

    return sorted(available)


def get_available_video_gen_models(
    *,
    env: ProviderEnvironment | None = None,
) -> list[str]:
    """Auto-detect available video generation models based on configured API keys.

    Args:
        env: Optional per-consumer environment for isolated API keys.
            When ``None``, uses the global settings singleton (current behavior).

    Returns:
        A sorted list of unique model strings (e.g. ``"grok/grok-imagine-video"``).
    """
    from ..drivers.grok_video_gen_driver import GrokVideoGenDriver

    available: set[str] = set()

    grok_video_key = _cfg_value(env, "grok_api_key", "GROK_API_KEY") or _cfg_value(env, "xai_api_key", "XAI_API_KEY")
    if grok_video_key:
        for model_id in GrokVideoGenDriver.KNOWN_MODELS:
            available.add(f"grok/{model_id}")

    # Runway video gen models (text/image/video → video)
    runway_key = _cfg_value(env, "runway_api_key", "RUNWAY_API_KEY") or _cfg_value(
        env, "runwayml_api_secret", "RUNWAYML_API_SECRET"
    )
    if runway_key:
        from ..drivers.runway_video_gen_driver import RunwayVideoGenDriver

        for model_id in RunwayVideoGenDriver.KNOWN_MODELS:
            available.add(f"runway/{model_id}")

    # Luma AI (Dream Machine) video gen models
    if _cfg_value(env, "luma_api_key", "LUMA_API_KEY"):
        from ..drivers.luma_video_gen_driver import LumaVideoGenDriver

        for model_id in LumaVideoGenDriver.KNOWN_MODELS:
            available.add(f"luma/{model_id}")

    # Pika Labs video gen models
    if _cfg_value(env, "pika_api_key", "PIKA_API_KEY"):
        from ..drivers.pika_video_gen_driver import PikaVideoGenDriver

        for model_id in PikaVideoGenDriver.KNOWN_MODELS:
            available.add(f"pika/{model_id}")

    # Dynamic discovery: check modalities_output from models.dev capabilities
    # for any models that the built-in video drivers don't know about yet.
    from .model_rates import get_model_capabilities

    for model_str in get_available_models(env=env):
        parts = model_str.split("/", 1)
        if len(parts) != 2:
            continue
        provider, model_id = parts
        caps = get_model_capabilities(provider, model_id)
        if caps and "video" in caps.modalities_output:
            available.add(model_str)

    return sorted(available)


def get_available_rerank_models(
    *,
    env: ProviderEnvironment | None = None,
) -> list[str]:
    """Auto-detect available rerank models based on configured API keys.

    Checks which rerank providers are configured (Cohere, Voyage, Jina) and
    returns their supported models.

    Args:
        env: Optional per-consumer environment for isolated API keys.
            When ``None``, uses the global settings singleton.

    Returns:
        A sorted list of unique model strings (e.g. ``"cohere/rerank-v3.5"``).
    """
    available: set[str] = set()

    # Cohere rerank models
    if _cfg_value(env, "cohere_api_key", "COHERE_API_KEY"):
        for model_id in ("rerank-v3.5", "rerank-english-v3.0", "rerank-multilingual-v3.0"):
            available.add(f"cohere/{model_id}")

    # Voyage AI rerank models
    if _cfg_value(env, "voyage_api_key", "VOYAGE_API_KEY"):
        for model_id in ("rerank-2.5", "rerank-2.5-lite", "rerank-2"):
            available.add(f"voyage/{model_id}")

    # Jina AI rerank models
    if _cfg_value(env, "jina_api_key", "JINA_API_KEY"):
        for model_id in (
            "jina-reranker-v2-base-multilingual",
            "jina-reranker-v1-base-en",
            "jina-colbert-v2",
        ):
            available.add(f"jina/{model_id}")

    # Mixedbread rerank models
    if _cfg_value(env, "mixedbread_api_key", "MIXEDBREAD_API_KEY"):
        for model_id in (
            "mxbai-rerank-large-v1",
            "mxbai-rerank-base-v1",
            "mxbai-rerank-xsmall-v1",
        ):
            available.add(f"mixedbread/{model_id}")

    return sorted(available)


def get_available_moderation_models(
    *,
    env: ProviderEnvironment | None = None,
) -> list[str]:
    """Auto-detect available moderation models based on configured API keys.

    Checks which moderation providers are configured (OpenAI, Mistral) and
    returns their supported models.

    Args:
        env: Optional per-consumer environment for isolated API keys.
            When ``None``, uses the global settings singleton.

    Returns:
        A sorted list of unique model strings (e.g. ``"openai/omni-moderation-latest"``).
    """
    available: set[str] = set()

    # OpenAI moderation models
    if _cfg_value(env, "openai_api_key", "OPENAI_API_KEY"):
        for model_id in (
            "omni-moderation-latest",
            "omni-moderation-2024-09-26",
            "text-moderation-latest",
            "text-moderation-stable",
        ):
            available.add(f"openai/{model_id}")

    # Mistral moderation models
    if _cfg_value(env, "mistral_api_key", "MISTRAL_API_KEY"):
        for model_id in (
            "mistral-moderation-latest",
            "mistral-moderation-2411",
        ):
            available.add(f"mistral/{model_id}")

    return sorted(available)


def get_available_embedding_models(
    *,
    env: ProviderEnvironment | None = None,
) -> list[str]:
    """Auto-detect available embedding models based on configured API keys.

    Checks which embedding providers are configured and returns their supported models.

    Args:
        env: Optional per-consumer environment for isolated API keys.
            When ``None``, uses the global settings singleton (current behavior).

    Returns:
        A sorted list of unique model strings (e.g. ``"openai/text-embedding-3-small"``).
    """
    from ..drivers.cohere_embedding_driver import CohereEmbeddingDriver
    from ..drivers.jina_embedding_driver import JinaEmbeddingDriver
    from ..drivers.openai_embedding_driver import OpenAIEmbeddingDriver
    from ..drivers.voyage_embedding_driver import VoyageEmbeddingDriver

    available: set[str] = set()

    # OpenAI embedding models (requires openai_api_key)
    if _cfg_value(env, "openai_api_key", "OPENAI_API_KEY"):
        for model_id in OpenAIEmbeddingDriver.EMBEDDING_PRICING:
            available.add(f"openai/{model_id}")

    # Cohere embedding models (requires cohere_api_key)
    if _cfg_value(env, "cohere_api_key", "COHERE_API_KEY"):
        for model_id in CohereEmbeddingDriver.EMBEDDING_PRICING:
            available.add(f"cohere/{model_id}")

    # Voyage AI embedding models (requires voyage_api_key)
    if _cfg_value(env, "voyage_api_key", "VOYAGE_API_KEY"):
        for model_id in VoyageEmbeddingDriver.EMBEDDING_PRICING:
            available.add(f"voyage/{model_id}")

    # Jina AI embedding models (requires jina_api_key)
    if _cfg_value(env, "jina_api_key", "JINA_API_KEY"):
        for model_id in JinaEmbeddingDriver.EMBEDDING_PRICING:
            available.add(f"jina/{model_id}")

    # Nomic embedding models (requires nomic_api_key)
    if _cfg_value(env, "nomic_api_key", "NOMIC_API_KEY"):
        from ..drivers.nomic_embedding_driver import NomicEmbeddingDriver

        for model_id in NomicEmbeddingDriver.EMBEDDING_PRICING:
            available.add(f"nomic/{model_id}")

    # Mixedbread embedding models (requires mixedbread_api_key)
    if _cfg_value(env, "mixedbread_api_key", "MIXEDBREAD_API_KEY"):
        from ..drivers.mxbai_embedding_driver import MxbaiEmbeddingDriver

        for model_id in MxbaiEmbeddingDriver.EMBEDDING_PRICING:
            available.add(f"mixedbread/{model_id}")

    # Ollama embedding models (always available — local)
    available.add("ollama/nomic-embed-text")
    available.add("ollama/all-minilm")
    available.add("ollama/mxbai-embed-large")
    available.add("ollama/snowflake-arctic-embed")
    available.add("ollama/bge-m3")

    return sorted(available)
