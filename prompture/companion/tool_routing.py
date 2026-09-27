"""Point coding CLIs at the companion's router, and back; install their live-state hooks.

- **Claude Code**: ``env.ANTHROPIC_BASE_URL`` in ``~/.claude/settings.json``
  becomes ``<companion>/tools/claude-code``. Its login (API key or
  subscription) is untouched; the router forwards it.
- **Codex**: ``~/.codex/config.toml`` gets ``model_provider = "prompture"``
  and a ``[model_providers.prompture]`` table pointing at
  ``<companion>/tools/codex/v1``, with ``requires_openai_auth`` so its ChatGPT
  or API-key login still applies.

Only those entries are written, each marked as Prompture's, and whatever they
replaced is kept in the companion prefs and put back when routing is turned
off. A config file that doesn't parse is left alone. Routing is re-applied
when the companion starts (its port may have changed) and taken back when it
stops, so a CLI never points at a companion that isn't running.

Hooks (Claude Code only) are separate and opt-in: a few ``hooks`` entries in
``settings.json`` that run :mod:`.hook` so a permission prompt shows as
"waiting" right away. They do nothing while no companion runs.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from ..infra.coding_agent_usage import env_path, home
from .router import TOOLS

PREFS_FILE = Path.home() / ".prompture" / "companion-prefs.json"
MARK = "prompture-router"
CLAUDE_KEY = "ANTHROPIC_BASE_URL"
HOOK_MODULE = "prompture.companion.hook"
#: Claude Code events the hooks report. Tool events only on completion, so a
#: tool call never waits on the hook before it runs.
HOOK_EVENTS = ("UserPromptSubmit", "PostToolUse", "Notification", "Stop", "SessionEnd")
_TOOL_EVENTS = {"PreToolUse", "PostToolUse"}

_OURS_CLAUDE = re.compile(r"^http://(127\.0\.0\.1|localhost):\d+/tools/claude-code/?$")
_TABLE = re.compile(r"^\s*\[")
_PROVIDER = re.compile(r"^\s*model_provider\s*=")


class RoutingError(Exception):
    """A config file couldn't be read or written; ``str()`` says which and why."""


def _obj(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8") or "{}")
    except (OSError, ValueError) as exc:
        raise RoutingError(f"{path} isn't valid JSON ({exc}); fix it and try again.") from exc
    if not isinstance(data, dict):
        raise RoutingError(f"{path} isn't a JSON object.")
    return data


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".prompture.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


class ToolRouting:
    """Routing and hook switches for each CLI; see the module docstring."""

    def __init__(
        self,
        prefs_file: str | Path | None = PREFS_FILE,
        *,
        claude_root: str | Path | None = None,
        codex_root: str | Path | None = None,
        python: str | None = None,
    ) -> None:
        self.prefs_file = Path(prefs_file) if prefs_file else None
        self.claude_root = Path(claude_root) if claude_root else env_path("CLAUDE_CONFIG_DIR", home() / ".claude")
        self.codex_root = Path(codex_root) if codex_root else env_path("CODEX_HOME", home() / ".codex")
        self.python = (python or sys.executable).replace("\\", "/")
        self._memory: dict[str, Any] = {}

    # -- prefs ----------------------------------------------------------------

    def _prefs(self) -> dict[str, Any]:
        if self.prefs_file is None:
            return self._memory
        try:
            data = json.loads(self.prefs_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def _save_prefs(self, prefs: dict[str, Any]) -> None:
        if self.prefs_file is None:
            self._memory = prefs
            return
        _write(self.prefs_file, json.dumps(prefs, indent=2))

    def enabled(self) -> list[str]:
        return [t for t in self._prefs().get("routed_tools", []) if t in TOOLS]

    def _backup(self, tool: str) -> dict[str, Any]:
        backups = self._prefs().get("routing_backup")
        entry = backups.get(tool) if isinstance(backups, dict) else None
        return entry if isinstance(entry, dict) else {}

    def _set_backup(self, tool: str, entry: dict[str, Any] | None) -> None:
        prefs = self._prefs()
        backups = _obj(prefs.get("routing_backup"))
        if entry is None:
            backups.pop(tool, None)
        else:
            backups[tool] = entry
        prefs["routing_backup"] = backups
        self._save_prefs(prefs)

    def upstream(self, tool: str) -> str | None:
        """The gateway a tool pointed at before routing, which passed-through calls keep going to."""
        if tool == "claude-code":
            previous = self._backup(tool).get(CLAUDE_KEY)
            return previous if isinstance(previous, str) and previous else None
        return None

    # -- paths ----------------------------------------------------------------

    @property
    def claude_settings(self) -> Path:
        return self.claude_root / "settings.json"

    @property
    def codex_config(self) -> Path:
        return self.codex_root / "config.toml"

    def installed(self, tool: str) -> bool:
        return (self.claude_root if tool == "claude-code" else self.codex_root).is_dir()

    @staticmethod
    def tool_url(base_url: str, tool: str) -> str:
        url = f"{base_url.rstrip('/')}/tools/{tool}"
        return f"{url}/v1" if TOOLS[tool].dialect == "openai" else url

    # -- state ----------------------------------------------------------------

    def routed(self, tool: str) -> bool:
        """Whether the tool's config points at a Prompture router right now."""
        try:
            if tool == "claude-code":
                env = _load_json(self.claude_settings).get("env")
                return isinstance(env, dict) and bool(_OURS_CLAUDE.match(str(env.get(CLAUDE_KEY) or "")))
            text = self.codex_config.read_text(encoding="utf-8") if self.codex_config.exists() else ""
            return f"# {MARK}" in text
        except (OSError, RoutingError):
            return False

    def status(self, base_url: str) -> dict[str, Any]:
        on = set(self.enabled())
        return {
            "url": base_url,
            "tools": [
                {
                    "id": tool.id,
                    "name": tool.name,
                    "installed": self.installed(tool.id),
                    "enabled": tool.id in on,
                    "routed": self.routed(tool.id),
                    "url": self.tool_url(base_url, tool.id),
                    "config": str(self.claude_settings if tool.id == "claude-code" else self.codex_config),
                }
                for tool in TOOLS.values()
            ],
            "hooks": {"claude": self.hooks_installed()},
        }

    # -- switching ------------------------------------------------------------

    def set_enabled(self, tool: str, enabled: bool, base_url: str) -> None:
        """Turn routing on or off for *tool*, and remember the choice."""
        if tool not in TOOLS:
            raise RoutingError(f"Unknown tool {tool!r}.")
        if enabled:
            self.apply(tool, base_url)
        else:
            self.restore(tool)
        prefs = self._prefs()
        tools = [t for t in prefs.get("routed_tools", []) if t != tool]
        prefs["routed_tools"] = [*tools, tool] if enabled else tools
        self._save_prefs(prefs)

    def apply_enabled(self, base_url: str) -> list[str]:
        """Point every enabled tool at *base_url* (companion start); returns the problems."""
        problems = []
        for tool in self.enabled():
            try:
                self.apply(tool, base_url)
            except RoutingError as exc:
                problems.append(str(exc))
        return problems

    def restore_all(self) -> None:
        """Take routing back from every tool (companion stop); the choices stay on."""
        for tool in TOOLS:
            try:
                if self.routed(tool):
                    self.restore(tool)
            except RoutingError:
                continue

    def apply(self, tool: str, base_url: str) -> None:
        if tool == "claude-code":
            self._apply_claude(self.tool_url(base_url, tool))
        else:
            self._apply_codex(self.tool_url(base_url, tool))

    def restore(self, tool: str) -> None:
        if tool == "claude-code":
            self._restore_claude()
        else:
            self._restore_codex()

    # -- Claude Code ----------------------------------------------------------

    def _apply_claude(self, url: str) -> None:
        data = _load_json(self.claude_settings)
        env = _obj(data.get("env"))
        current = env.get(CLAUDE_KEY)
        if not _OURS_CLAUDE.match(str(current or "")):
            self._set_backup("claude-code", {CLAUDE_KEY: current})
        env[CLAUDE_KEY] = url
        data["env"] = env
        _write(self.claude_settings, json.dumps(data, indent=2) + "\n")

    def _restore_claude(self) -> None:
        data = _load_json(self.claude_settings)
        env = _obj(data.get("env"))
        if _OURS_CLAUDE.match(str(env.get(CLAUDE_KEY) or "")):
            previous = self._backup("claude-code").get(CLAUDE_KEY)
            if previous:
                env[CLAUDE_KEY] = previous
            else:
                env.pop(CLAUDE_KEY, None)
            if env:
                data["env"] = env
            else:
                data.pop("env", None)
            _write(self.claude_settings, json.dumps(data, indent=2) + "\n")
        self._set_backup("claude-code", None)

    # -- Codex ----------------------------------------------------------------

    @staticmethod
    def _strip_codex(lines: list[str]) -> list[str]:
        out: list[str] = []
        inside = False
        for line in lines:
            if line.strip() == f"# {MARK}: begin":
                inside = True
                continue
            if inside:
                if line.strip() == f"# {MARK}: end":
                    inside = False
                continue
            if line.rstrip().endswith(f"# {MARK}"):
                continue
            out.append(line)
        while out and not out[-1].strip():
            out.pop()
        return out

    def _read_codex(self) -> list[str]:
        try:
            return self.codex_config.read_text(encoding="utf-8").splitlines() if self.codex_config.exists() else []
        except OSError as exc:
            raise RoutingError(f"Can't read {self.codex_config}: {exc}") from exc

    def _apply_codex(self, url: str) -> None:
        lines = self._strip_codex(self._read_codex())
        first_table = next((i for i, line in enumerate(lines) if _TABLE.match(line)), len(lines))
        found = next((i for i in range(first_table) if _PROVIDER.match(lines[i])), None)
        backup = self._backup("codex")
        if found is not None:
            self._set_backup("codex", {"model_provider": lines.pop(found)})
        elif "model_provider" not in backup:
            self._set_backup("codex", {"model_provider": None})
        lines.insert(0, f'model_provider = "prompture"  # {MARK}')
        lines += [
            "",
            f"# {MARK}: begin",
            "[model_providers.prompture]",
            'name = "Prompture"',
            f'base_url = "{url}"',
            'wire_api = "responses"',
            "requires_openai_auth = true",
            f"# {MARK}: end",
        ]
        _write(self.codex_config, "\n".join(lines) + "\n")

    def _restore_codex(self) -> None:
        lines = self._read_codex()
        stripped = self._strip_codex(lines)
        previous = self._backup("codex").get("model_provider")
        if isinstance(previous, str) and previous.strip():
            stripped.insert(0, previous)
        if stripped != lines:
            _write(self.codex_config, "\n".join(stripped) + ("\n" if stripped else ""))
        self._set_backup("codex", None)

    # -- hooks ----------------------------------------------------------------

    @property
    def hook_command(self) -> str:
        return f'"{self.python}" -m {HOOK_MODULE} claude'

    def hooks_installed(self) -> bool:
        try:
            hooks = _load_json(self.claude_settings).get("hooks")
        except RoutingError:
            return False
        return isinstance(hooks, dict) and any(
            HOOK_MODULE in str(h.get("command", ""))
            for groups in hooks.values()
            if isinstance(groups, list)
            for group in groups
            if isinstance(group, dict)
            for h in group.get("hooks") or []
            if isinstance(h, dict)
        )

    @staticmethod
    def _without_ours(groups: Any) -> list[Any]:
        kept = []
        for group in groups if isinstance(groups, list) else []:
            if not isinstance(group, dict):
                kept.append(group)
                continue
            hooks = [h for h in group.get("hooks") or [] if HOOK_MODULE not in str(h.get("command", ""))]
            if hooks:
                kept.append({**group, "hooks": hooks})
        return kept

    def set_hooks(self, enabled: bool) -> None:
        """Install or remove Claude Code's live-state hooks (only Prompture's entries)."""
        data = _load_json(self.claude_settings)
        hooks = _obj(data.get("hooks"))
        for event in set(hooks) | set(HOOK_EVENTS):
            groups = self._without_ours(hooks.get(event))
            if enabled and event in HOOK_EVENTS:
                entry: dict[str, Any] = {"hooks": [{"type": "command", "command": self.hook_command, "timeout": 5}]}
                if event in _TOOL_EVENTS:
                    entry = {"matcher": "*", **entry}
                groups.append(entry)
            if groups:
                hooks[event] = groups
            else:
                hooks.pop(event, None)
        if hooks:
            data["hooks"] = hooks
        else:
            data.pop("hooks", None)
        _write(self.claude_settings, json.dumps(data, indent=2) + "\n")
