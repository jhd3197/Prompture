"""User-defined CLI tools from ``.prompture/tools.yaml`` (or ``tools.json``).

Search order: ``./.prompture/`` (project) then ``~/.prompture/`` (user); a
project definition wins over a user one with the same name, and both win over
the shipped ``gh`` / ``yt-dlp`` definitions. YAML needs PyYAML; JSON always
works. Example::

    tools:
      - name: kubectl
        command: kubectl
        description: Read-only Kubernetes queries
        version_args: [version, --client]
        env:
          KUBECONFIG: ${MY_KUBECONFIG}     # read from the environment at run time
        commands:
          - name: get_pods
            subcommand: [get, pods]
            description: List pods in a namespace
            fixed_args: [-o, json]
            output: json
            args:
              - name: namespace
                flag: --namespace
                pattern: "^[a-z0-9-]{1,63}$"

The ``commands`` list *is* the allowlist — declare read-only commands only.
A top-level mapping ``{name: {...}}`` is accepted in place of the ``tools``
list. Invalid definitions are skipped and reported by :func:`config_errors`
(and the ``cli:config`` doctor row).
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .adapter import CLITool, CLIToolError

logger = logging.getLogger("prompture.tools.cli")

CONFIG_DIRNAME = ".prompture"
CONFIG_BASENAMES = ("tools.yaml", "tools.yml", "tools.json")

_errors: list[str] = []
_errors_lock = threading.Lock()


def config_paths(cwd: str | os.PathLike[str] | None = None, *, include_home: bool = True) -> list[Path]:
    """Existing config files, project first (at most one per directory)."""
    dirs = [Path(cwd or os.getcwd()) / CONFIG_DIRNAME]
    if include_home:
        home = Path.home() / CONFIG_DIRNAME
        if home.resolve() != dirs[0].resolve():
            dirs.append(home)
    found: list[Path] = []
    for d in dirs:
        for base in CONFIG_BASENAMES:
            candidate = d / base
            if candidate.is_file():
                found.append(candidate)
                break
    return found


def _read(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        raise CLIToolError(
            f"{path}: PyYAML is not installed — `pip install pyyaml` or use {path.with_suffix('.json').name}"
        ) from None
    return yaml.safe_load(text)


def _entries(data: Any, path: Path) -> list[tuple[str | None, Mapping[str, Any]]]:
    if data is None:
        return []
    if isinstance(data, Mapping) and "tools" in data:
        data = data["tools"]
    if isinstance(data, Mapping):
        out = []
        for name, spec in data.items():
            if not isinstance(spec, Mapping):
                raise CLIToolError(f"{path}: tool {name!r} must be a mapping")
            out.append((str(name), spec))
        return out
    if isinstance(data, list):
        out = []
        for spec in data:
            if not isinstance(spec, Mapping):
                raise CLIToolError(f"{path}: every entry under 'tools' must be a mapping")
            out.append((None, spec))
        return out
    raise CLIToolError(f"{path}: expected a 'tools' list or mapping")


def parse_cli_tools(data: Any, *, source: str = "<config>") -> tuple[list[CLITool], list[str]]:
    """Build tools from already-loaded config *data*; returns ``(tools, errors)``."""
    tools: list[CLITool] = []
    errors: list[str] = []
    try:
        entries = _entries(data, Path(source))
    except CLIToolError as exc:
        return [], [str(exc)]
    for name, spec in entries:
        try:
            tools.append(CLITool.from_dict(spec, name=name))
        except (CLIToolError, TypeError, ValueError) as exc:
            errors.append(f"{source}: {exc}")
    return tools, errors


def load_cli_tools(
    paths: str | os.PathLike[str] | Iterable[str | os.PathLike[str]] | None = None,
    *,
    cwd: str | os.PathLike[str] | None = None,
    include_home: bool = True,
) -> list[CLITool]:
    """Load user-defined CLI tools.

    Args:
        paths: Explicit config file(s). Default: :func:`config_paths`.
        cwd: Project directory for the default search (default: current dir).
        include_home: Also read ``~/.prompture/`` in the default search.

    Returns:
        Valid tools, earlier files winning on name collisions. Problems are
        logged and kept for :func:`config_errors`; this never raises. Tools
        found by the default search outside ``~/.prompture`` (i.e. in a
        project) are marked untrusted, so health checks never execute them.
    """
    if paths is None:
        files = config_paths(cwd, include_home=include_home)
    elif isinstance(paths, (str, os.PathLike)):
        files = [Path(paths)]
    else:
        files = [Path(p) for p in paths]

    tools: dict[str, CLITool] = {}
    errors: list[str] = []
    for path in files:
        try:
            data = _read(path)
        except (OSError, ValueError, CLIToolError) as exc:
            errors.append(f"{path}: {exc}" if str(path) not in str(exc) else str(exc))
            continue
        except Exception as exc:  # YAML parser errors and friends
            errors.append(f"{path}: {type(exc).__name__}: {exc}")
            continue
        parsed, errs = parse_cli_tools(data, source=str(path))
        errors.extend(errs)
        trusted = paths is not None or _is_user_config(path)
        for tool in parsed:
            tool.trusted = trusted
            tools.setdefault(tool.name, tool)
    for err in errors:
        logger.warning("CLI tool config: %s", err)
    with _errors_lock:
        _errors[:] = errors
    return list(tools.values())


def _is_user_config(path: Path) -> bool:
    """True for files under the user's own ``~/.prompture`` directory."""
    try:
        user_dir = (Path.home() / ".prompture").resolve()
        return path.resolve().is_relative_to(user_dir)
    except (OSError, ValueError):
        return False


def config_errors() -> list[str]:
    """Problems found by the last :func:`load_cli_tools` call."""
    with _errors_lock:
        return list(_errors)
