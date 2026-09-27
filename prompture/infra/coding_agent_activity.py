"""Which coding agents are mid-turn right now, read from the end of their session logs.

:mod:`.coding_agent_usage` counts calls once they are logged. This module
answers the live question: is Claude Code or Codex working at this moment?

- **Claude Code** (``~/.claude/projects/*/<session>.jsonl``): the newest
  conversation entry decides. An ``assistant`` entry that stopped for
  ``tool_use`` (or is still streaming), or a ``user`` entry (a prompt or a tool
  result), means the turn is running. ``end_turn``, a ``turn_duration`` /
  ``stop_hook_summary`` system entry, or "[Request interrupted by user]" means
  it ended. Subagent (sidechain) entries are skipped: while one runs, the main
  agent's last entry is the ``tool_use`` that started it.
- **Codex** (``~/.codex/sessions/**/rollout-*.jsonl``): the newest of
  ``task_started`` against ``task_complete`` / ``turn_aborted``.

Only files written in the last :data:`HOT_SECONDS` are read, from the end.
A turn whose log went quiet for that long counts as over (a terminal closed
mid-turn). :func:`running_agents` narrows it further: when the agent's
process is known not to be running, its sessions aren't either.

Only entry types, stop reasons, model names, times and folder names are read,
never prompts, replies or tool output.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .coding_agent_usage import env_path, home, parse_ts, project_name, recent_files

#: A mid-turn log untouched this long is taken as abandoned. Long tool runs
#: (builds, test suites) write nothing until they finish, so this is generous.
HOT_SECONDS = 10 * 60
#: How far back from a log's end a state is looked for.
TAIL_BYTES = 1 << 20
#: Seconds a process listing is reused.
PROCESS_INTERVAL = 5.0

_CLAUDE_DONE_STOPS = {"end_turn", "stop_sequence", "max_tokens", "refusal"}
_CLAUDE_DONE_SYSTEM = {"turn_duration", "stop_hook_summary"}
_INTERRUPTED = "[Request interrupted by user"


@dataclass(frozen=True)
class ActiveTurn:
    """A coding agent session in the middle of a turn."""

    agent: str
    session: str
    model: str | None
    project: str | None
    since: datetime

    @property
    def request_id(self) -> str:
        return f"agent:{self.agent}:{self.session}"


def tail_entries(path: Path, max_bytes: int = TAIL_BYTES) -> Iterator[dict[str, Any]]:
    """JSON entries from the end of a JSONL file, newest first."""
    try:
        with path.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            start = max(0, size - max_bytes)
            fh.seek(start)
            chunk = fh.read(size - start)
    except OSError:
        return
    lines = chunk.split(b"\n")
    if start > 0:
        lines = lines[1:]  # the first line was cut
    for raw in reversed(lines):
        if not raw.strip():
            continue
        try:
            entry = json.loads(raw)
        except ValueError:
            continue  # a line still being written
        if isinstance(entry, dict):
            yield entry


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    return ""


def _is_prompt(message: dict[str, Any]) -> bool:
    content = message.get("content")
    if isinstance(content, list):
        return not any(isinstance(b, dict) and b.get("type") == "tool_result" for b in content)
    return isinstance(content, str)


def claude_turn(path: Path) -> ActiveTurn | None:
    """The running turn in a Claude Code transcript, or ``None`` if it's between turns."""
    working = False
    model: str | None = None
    cwd: str | None = None
    since: datetime | None = None
    for entry in tail_entries(path):
        kind = entry.get("type")
        if kind not in ("user", "assistant", "system") or entry.get("isSidechain"):
            continue
        message = entry.get("message") if isinstance(entry.get("message"), dict) else {}
        if not working:
            # The newest conversation entry decides.
            if kind == "system":
                if entry.get("subtype") in _CLAUDE_DONE_SYSTEM:
                    return None
                continue
            if kind == "assistant" and message.get("stop_reason") in _CLAUDE_DONE_STOPS:
                return None
            if kind == "user" and _text_of(message.get("content")).startswith(_INTERRUPTED):
                return None
            working = True
        cwd = cwd or entry.get("cwd")
        if kind == "assistant" and not model:
            name = message.get("model")
            if isinstance(name, str) and not name.startswith("<"):
                model = name
        if kind == "user" and _is_prompt(message):
            since = parse_ts(entry.get("timestamp"))  # the prompt that started this turn
            break
        since = parse_ts(entry.get("timestamp")) or since
    if not working:
        return None
    return ActiveTurn(
        agent="claude",
        session=path.stem,
        model=f"claude/{model}" if model else None,
        project=project_name(cwd),
        since=since or datetime.now(timezone.utc),
    )


def codex_turn(path: Path) -> ActiveTurn | None:
    """The running turn in a Codex rollout log, or ``None`` if it's between turns."""
    since: datetime | None = None
    model: str | None = None
    cwd: str | None = None
    for entry in tail_entries(path):
        payload = entry.get("payload") if isinstance(entry.get("payload"), dict) else {}
        if entry.get("type") == "event_msg" and since is None:
            if payload.get("type") in ("task_complete", "turn_aborted"):
                return None
            if payload.get("type") == "task_started":
                since = parse_ts(entry.get("timestamp")) or datetime.now(timezone.utc)
        elif entry.get("type") == "turn_context" and not model:
            model = payload.get("model") if isinstance(payload.get("model"), str) else None
            cwd = payload.get("cwd") if isinstance(payload.get("cwd"), str) else None
        if since is not None and model:
            break
    if since is None:
        return None
    # rollout-<timestamp>-<uuid>.jsonl: the uuid is the session
    session = path.stem.rsplit("-", 5)[-5:]
    return ActiveTurn(
        agent="codex",
        session="-".join(session) if len(session) == 5 else path.stem,
        model=f"openai/{model}" if model else None,
        project=project_name(cwd),
        since=since,
    )


# ------------------------------------------------------------------ processes

_PROCESS_NAMES = {"claude": "claude", "codex": "codex"}
_PACKAGES = {"@anthropic-ai/claude-code": "claude", "@openai/codex": "codex"}
#: Runtimes an npm-installed agent runs under; seen only by name, they can't rule an agent out.
_RUNTIMES = {"node", "bun"}


def _basename(token: str) -> str:
    name = token.replace("\\", "/").rsplit("/", 1)[-1].lower()
    for ext in (".exe", ".cmd", ".js", ".mjs"):
        if name.endswith(ext):
            return name[: -len(ext)]
    return name


def agents_in(commands: list[list[str]]) -> set[str]:
    """Agents a list of process command lines (argv lists) shows running."""
    found: set[str] = set()
    for argv in commands:
        for token in argv[:3]:  # the program, or a runtime and its script
            agent = _PROCESS_NAMES.get(_basename(token))
            if agent:
                found.add(agent)
        joined = " ".join(argv).replace("\\", "/")
        found.update(agent for pkg, agent in _PACKAGES.items() if pkg in joined)
    return found


def _windows_image_names() -> list[str]:
    """Executable names of every process, from a Toolhelp snapshot (milliseconds; ``tasklist`` takes seconds)."""
    import ctypes
    from ctypes import wintypes

    class ProcessEntry(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.c_size_t),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", ctypes.c_long),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", ctypes.c_wchar * 260),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel32.Process32FirstW.argtypes = kernel32.Process32NextW.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ProcessEntry),
    ]
    snapshot = kernel32.CreateToolhelp32Snapshot(0x2, 0)  # TH32CS_SNAPPROCESS
    if snapshot in (None, wintypes.HANDLE(-1).value):
        raise OSError("process snapshot failed")
    names: list[str] = []
    try:
        entry = ProcessEntry()
        entry.dwSize = ctypes.sizeof(ProcessEntry)
        ok = kernel32.Process32FirstW(snapshot, ctypes.byref(entry))
        while ok:
            names.append(entry.szExeFile)
            ok = kernel32.Process32NextW(snapshot, ctypes.byref(entry))
    finally:
        kernel32.CloseHandle(snapshot)
    return names


def _process_commands() -> tuple[list[list[str]], bool] | None:
    """``(argv lists, complete)``; ``complete`` is False when only image names were seen."""
    if sys.platform == "win32":
        return [[n] for n in _windows_image_names()], False
    proc = Path("/proc")
    if proc.is_dir():
        commands = []
        for entry in proc.iterdir():
            if not entry.name.isdigit():
                continue
            try:
                raw = (entry / "cmdline").read_bytes()
            except OSError:
                continue
            if raw:
                commands.append([a.decode("utf-8", "replace") for a in raw.split(b"\0") if a])
        return commands, True
    out = subprocess.run(["ps", "-axo", "args="], capture_output=True, text=True, timeout=5).stdout
    return [line.split() for line in out.splitlines() if line.strip()], True


def running_agents() -> set[str] | None:
    """Agents whose process may be running, or ``None`` when that can't be told.

    Where only image names are visible (Windows), a running ``node`` or
    ``bun`` could be an npm-installed agent, so both count as maybe running.
    """
    try:
        listed = _process_commands()
    except (OSError, subprocess.SubprocessError):
        return None
    if listed is None:
        return None
    commands, complete = listed
    found = agents_in(commands)
    if not complete and any(_basename(argv[0]) in _RUNTIMES for argv in commands if argv):
        found |= set(_PROCESS_NAMES.values())
    return found


# ------------------------------------------------------------------ scanner


class AgentActivity:
    """Running turns across Claude Code and Codex sessions.

    ``claude_root`` / ``codex_root`` default to the agents' own folders;
    ``processes`` returns :func:`running_agents`-style answers (tests pass a stub).
    """

    def __init__(
        self,
        claude_root: str | Path | None = None,
        codex_root: str | Path | None = None,
        *,
        hot_seconds: float = HOT_SECONDS,
        processes: Any = running_agents,
    ) -> None:
        self.claude_root = Path(claude_root) if claude_root else env_path("CLAUDE_CONFIG_DIR", home() / ".claude")
        self.codex_root = Path(codex_root) if codex_root else env_path("CODEX_HOME", home() / ".codex")
        self.hot = timedelta(seconds=hot_seconds)
        self.processes = processes
        self._procs: tuple[float, set[str] | None] | None = None

    def _running(self) -> set[str] | None:
        if self.processes is None:
            return None
        if self._procs is None or time.monotonic() - self._procs[0] > PROCESS_INTERVAL:
            self._procs = (time.monotonic(), self.processes())
        return self._procs[1]

    def scan(self, now: datetime | None = None) -> list[ActiveTurn]:
        since = (now or datetime.now(timezone.utc)) - self.hot
        found: list[ActiveTurn] = []
        for path in recent_files(self.claude_root / "projects", "*/*.jsonl", since):
            turn = claude_turn(path)
            if turn:
                found.append(turn)
        for path in recent_files(self.codex_root / "sessions", "**/rollout-*.jsonl", since):
            turn = codex_turn(path)
            if turn:
                found.append(turn)
        if found:
            running = self._running()
            if running is not None:
                found = [t for t in found if t.agent in running]
        return found
