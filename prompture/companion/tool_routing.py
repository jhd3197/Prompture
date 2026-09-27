"""Point coding CLIs at the companion's router, and back; install their live-state hooks.

- **Claude Code**: ``env.ANTHROPIC_BASE_URL`` in ``~/.claude/settings.json``
  becomes ``<companion>/tools/claude-code``. Its login (API key or
  subscription) is untouched; the router forwards it.
- **Codex**: ``~/.codex/config.toml`` gets ``model_provider = "prompture"``
  and a ``[model_providers.prompture]`` table pointing at
  ``<companion>/tools/codex/v1``, with ``requires_openai_auth`` so its ChatGPT
  or API-key login still applies.
- **Gemini CLI** (signed in with Google): ``~/.gemini/.env`` gets
  ``CODE_ASSIST_ENDPOINT=<companion>/tools/gemini-cli`` between marker
  comments. Gemini CLI reads that file only when the project has no ``.env``
  of its own (or ``.gemini/.env``), so those projects go direct; API-key
  sign-ins aren't routed (Gemini CLI has no base-URL setting for them).

Only those entries are written, each marked as Prompture's, and whatever they
replaced is kept in the companion prefs and put back when routing is turned
off. A config file that doesn't parse is left alone. Routing is re-applied
when the companion starts (its port may have changed) and taken back when it
stops, so a CLI never points at a companion that isn't running. A companion
killed outright can't take it back; the next one to start fixes the configs,
and ``prompture companion --restore`` puts them back without starting one.

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
#: Run as a script, not with ``-m``: importing the prompture package costs ~2 s,
#: and Claude Code waits for the hook after every tool call.
HOOK_FILE = Path(__file__).with_name("hook.py")
#: Claude Code events the hooks report. Tool events only on completion, so a
#: tool call never waits on the hook before it runs.
HOOK_EVENTS = ("UserPromptSubmit", "PostToolUse", "Notification", "Stop", "SessionEnd")
_TOOL_EVENTS = {"PreToolUse", "PostToolUse"}

_OURS_CLAUDE = re.compile(r"^http://(127\.0\.0\.1|localhost):\d+/tools/claude-code/?$")
_TABLE = re.compile(r"^\s*\[")
_PROVIDER = re.compile(r"^\s*model_provider\s*=")
_CODEX_LINE = f'model_provider = "prompture"  # {MARK}'
_CODEX_BLOCK = re.compile(rf"\n?# {MARK}: begin\n.*?# {MARK}: end\n?", re.S)
GEMINI_KEY = "CODE_ASSIST_ENDPOINT"
_GEMINI_LINE = re.compile(rf"^\s*(export\s+)?{GEMINI_KEY}\s*=")


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
    """Replace *path* atomically, keeping its line ends (CRLF stays CRLF, LF stays LF)."""
    try:
        crlf = b"\r\n" in path.read_bytes()
    except OSError:
        crlf = False
    if crlf and "\r\n" not in text:
        text = text.replace("\n", "\r\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".prompture.tmp")
    tmp.write_bytes(text.encode("utf-8"))
    os.replace(tmp, path)


class ToolRouting:
    """Routing and hook switches for each CLI; see the module docstring."""

    def __init__(
        self,
        prefs_file: str | Path | None = PREFS_FILE,
        *,
        claude_root: str | Path | None = None,
        codex_root: str | Path | None = None,
        gemini_root: str | Path | None = None,
        python: str | None = None,
    ) -> None:
        self.prefs_file = Path(prefs_file) if prefs_file else None
        self.claude_root = Path(claude_root) if claude_root else env_path("CLAUDE_CONFIG_DIR", home() / ".claude")
        self.codex_root = Path(codex_root) if codex_root else env_path("CODEX_HOME", home() / ".codex")
        self.gemini_root = Path(gemini_root) if gemini_root else home() / ".gemini"
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

    @property
    def gemini_env(self) -> Path:
        return self.gemini_root / ".env"

    def config_path(self, tool: str) -> Path:
        return {"claude-code": self.claude_settings, "codex": self.codex_config}.get(tool, self.gemini_env)

    def installed(self, tool: str) -> bool:
        return {"claude-code": self.claude_root, "codex": self.codex_root}.get(tool, self.gemini_root).is_dir()

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
            path = self.codex_config if tool == "codex" else self.gemini_env
            text = path.read_text(encoding="utf-8") if path.exists() else ""
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
                    "config": str(self.config_path(tool.id)),
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
        """Point every enabled tool at *base_url* (companion start); returns the problems.

        A tool still pointing at a router it wasn't enabled for (left behind
        by a companion that was killed before it could restore it) is put back.
        """
        problems = []
        on = self.enabled()
        for tool in TOOLS:
            try:
                if tool in on:
                    self.apply(tool, base_url)
                elif self.routed(tool):
                    self.restore(tool)
            except RoutingError as exc:
                problems.append(str(exc))
        return problems

    def restore_all(self) -> list[str]:
        """Take routing back from every tool (companion stop); the choices stay on.

        Returns the tools that were put back.
        """
        restored = []
        for tool in TOOLS:
            try:
                if self.routed(tool):
                    self.restore(tool)
                    restored.append(tool)
            except RoutingError:
                continue
        return restored

    def apply(self, tool: str, base_url: str) -> None:
        if tool == "claude-code":
            self._apply_claude(self.tool_url(base_url, tool))
        elif tool == "codex":
            self._apply_codex(self.tool_url(base_url, tool))
        else:
            self._apply_gemini(self.tool_url(base_url, tool))

    def restore(self, tool: str) -> None:
        if tool == "claude-code":
            self._restore_claude()
        elif tool == "codex":
            self._restore_codex()
        else:
            self._restore_gemini()

    # -- Claude Code ----------------------------------------------------------

    def _apply_claude(self, url: str) -> None:
        data = _load_json(self.claude_settings)
        env = _obj(data.get("env"))
        current = env.get(CLAUDE_KEY)
        if not _OURS_CLAUDE.match(str(current or "")):
            self._set_backup("claude-code", {CLAUDE_KEY: current})
        env[CLAUDE_KEY] = url
        data["env"] = env
        _write(self.claude_settings, json.dumps(data, indent=2, ensure_ascii=False) + "\n")

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
            _write(self.claude_settings, json.dumps(data, indent=2, ensure_ascii=False) + "\n")
        self._set_backup("claude-code", None)

    # -- Codex ----------------------------------------------------------------

    def _codex_text(self) -> tuple[str, str]:
        """The config with ``\\n`` line ends, and the line end the file uses."""
        try:
            raw = self.codex_config.read_bytes().decode("utf-8") if self.codex_config.exists() else ""
        except (OSError, UnicodeDecodeError) as exc:
            raise RoutingError(f"Can't read {self.codex_config}: {exc}") from exc
        return raw.replace("\r\n", "\n"), "\r\n" if "\r\n" in raw else "\n"

    @staticmethod
    def _strip_codex(text: str) -> str:
        """*text* without what :meth:`_apply_codex` added."""
        text = _CODEX_BLOCK.sub("", text)
        return "".join(line for line in text.splitlines(keepends=True) if line.rstrip("\n") != _CODEX_LINE)

    def _apply_codex(self, url: str) -> None:
        text, newline = self._codex_text()
        body = self._strip_codex(text)
        lines = body.split("\n")
        first_table = next((i for i, line in enumerate(lines) if _TABLE.match(line)), len(lines))
        found = next((i for i in range(first_table) if _PROVIDER.match(lines[i])), None)
        if f"# {MARK}" not in text or found is not None:
            # What gets put back: the provider line it replaces, where it was, and the file's last newline.
            self._set_backup(
                "codex",
                {
                    "model_provider": lines[found] if found is not None else None,
                    "line": found,
                    "final_newline": body.endswith("\n") or not body,
                },
            )
        if found is not None:
            lines.pop(found)
        body = "\n".join(lines)
        if body and not body.endswith("\n"):
            body += "\n"
        block = (
            f'\n# {MARK}: begin\n[model_providers.prompture]\nname = "Prompture"\nbase_url = "{url}"\n'
            f'wire_api = "responses"\nrequires_openai_auth = true\n# {MARK}: end\n'
        )
        _write(self.codex_config, f"{_CODEX_LINE}\n{body}{block}".replace("\n", newline))

    def _restore_codex(self) -> None:
        text, newline = self._codex_text()
        if f"# {MARK}" in text:
            body = self._strip_codex(text)
            backup = self._backup("codex")
            previous, at = backup.get("model_provider"), backup.get("line")
            if isinstance(previous, str):
                lines = body.split("\n")
                lines.insert(at if isinstance(at, int) and 0 <= at <= len(lines) else 0, previous)
                body = "\n".join(lines)
            if backup.get("final_newline") is False and body.endswith("\n"):
                body = body[:-1]
            _write(self.codex_config, body.replace("\n", newline))
        self._set_backup("codex", None)

    # -- Gemini CLI -----------------------------------------------------------

    def _gemini_text(self) -> tuple[str | None, str]:
        """``~/.gemini/.env`` with ``\\n`` line ends (``None`` when there is none), and its line end."""
        path = self.gemini_env
        if not path.exists():
            return None, "\n"
        try:
            raw = path.read_bytes().decode("utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise RoutingError(f"Can't read {path}: {exc}") from exc
        return raw.replace("\r\n", "\n"), "\r\n" if "\r\n" in raw else "\n"

    def _apply_gemini(self, url: str) -> None:
        text, newline = self._gemini_text()
        body = _CODEX_BLOCK.sub("", text or "")
        lines = body.split("\n")
        found = next((i for i, line in enumerate(lines) if _GEMINI_LINE.match(line)), None)
        if text is None or f"# {MARK}" not in text or found is not None:
            self._set_backup(
                "gemini-cli",
                {
                    "existed": text is not None,
                    "line": lines[found] if found is not None else None,
                    "at": found,
                    "final_newline": body.endswith("\n") or not body,
                },
            )
        if found is not None:
            lines.pop(found)  # the endpoint it pointed at is kept, and put back later
        body = "\n".join(lines)
        if body and not body.endswith("\n"):
            body += "\n"
        block = f"\n# {MARK}: begin\n{GEMINI_KEY}={url}\n# {MARK}: end\n"
        _write(self.gemini_env, (body + block).replace("\n", newline))

    def _restore_gemini(self) -> None:
        text, newline = self._gemini_text()
        if text is not None and f"# {MARK}" in text:
            backup = self._backup("gemini-cli")
            body = _CODEX_BLOCK.sub("", text)
            if isinstance(backup.get("line"), str):
                lines = body.split("\n")
                at = backup.get("at")
                lines.insert(at if isinstance(at, int) and 0 <= at <= len(lines) else len(lines), backup["line"])
                body = "\n".join(lines)
            if backup.get("final_newline") is False and body.endswith("\n"):
                body = body[:-1]
            if backup.get("existed") is False and not body.strip():
                self.gemini_env.unlink(missing_ok=True)  # it wasn't there before routing
            else:
                _write(self.gemini_env, body.replace("\n", newline))
        self._set_backup("gemini-cli", None)

    # -- hooks ----------------------------------------------------------------

    @property
    def hook_command(self) -> str:
        return f'"{self.python}" "{HOOK_FILE.as_posix()}" claude'

    @staticmethod
    def _ours(command: Any) -> bool:
        """A hook command Prompture installed (this form, or the older ``-m`` one)."""
        text = str(command or "").replace("\\", "/")
        return HOOK_MODULE in text or "/companion/hook.py" in text

    def refresh_hooks(self) -> bool:
        """Re-install installed hooks whose command is out of date (Python moved, older form)."""
        try:
            hooks = _load_json(self.claude_settings).get("hooks")
        except RoutingError:
            return False
        commands = {
            str(h.get("command"))
            for groups in (hooks.values() if isinstance(hooks, dict) else [])
            if isinstance(groups, list)
            for group in groups
            if isinstance(group, dict)
            for h in group.get("hooks") or []
            if isinstance(h, dict) and self._ours(h.get("command"))
        }
        if not commands or commands == {self.hook_command}:
            return False
        self.set_hooks(True)
        return True

    def hooks_installed(self) -> bool:
        try:
            hooks = _load_json(self.claude_settings).get("hooks")
        except RoutingError:
            return False
        return isinstance(hooks, dict) and any(
            self._ours(h.get("command"))
            for groups in hooks.values()
            if isinstance(groups, list)
            for group in groups
            if isinstance(group, dict)
            for h in group.get("hooks") or []
            if isinstance(h, dict)
        )

    @classmethod
    def _without_ours(cls, groups: Any) -> list[Any]:
        kept = []
        for group in groups if isinstance(groups, list) else []:
            if not isinstance(group, dict):
                kept.append(group)
                continue
            hooks = [h for h in group.get("hooks") or [] if not cls._ours(h.get("command"))]
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
        _write(self.claude_settings, json.dumps(data, indent=2, ensure_ascii=False) + "\n")
