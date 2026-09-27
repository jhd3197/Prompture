"""Shared project memory for coding agents, served by the companion.

Claude Code and Codex working in the same project get the same notes: the
decisions, conventions, commands and verified fixes in
:class:`~prompture.session_memory.project.ProjectMemory`. Each session gets
them **once, at its start**, never again per request:

- **Claude Code**, through its ``UserPromptSubmit`` hook (see :mod:`.hook`):
  on a session's first prompt the companion answers with the relevant notes
  and the hook hands them to Claude Code as extra context.
- **Codex** (or Claude Code without the hook), through the router: the
  session's first request gets the notes as a leading message, and every
  later request of that session gets the *same* text in the same place, so
  the conversation reads as if it had always been there and the prompt
  cache keeps working.

What a session received is logged (``/v1/memory/injections``) so it can be
inspected. Only notes relevant to the first prompt are chosen (see
:meth:`~prompture.session_memory.project.ProjectMemory.select`), within a
token budget. Agents add notes with ``prompture memory add`` (the block
tells them how), people add, edit and delete them from a companion app.

Each finished task's tool sequence also feeds a per-project
:class:`~prompture.agents.skill_miner.SkillMiner`, whose proposals for
reusable workflows can be saved as a ``SKILL.md`` in the project.

Only the prompt is read to choose notes, and it isn't stored.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any

from ..session_memory.project import DEFAULT_BUDGET_TOKENS, ProjectMemory, estimate_tokens, project_name

logger = logging.getLogger("prompture.companion.memory")

MEMORY_DIR = Path.home() / ".prompture" / "memory"
DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "budget_tokens": DEFAULT_BUDGET_TOKENS,
    "verified_only": True,
    "teach": True,
}
#: Tasks kept per project for skill mining.
MAX_TASKS = 500
#: How often a procedure must recur before it's proposed as a skill.
SKILL_OCCURRENCES = 3


def save_hint(python: str | None = None) -> str:
    """The line that tells an agent how to leave a verified note for the next one."""
    exe = (python or sys.executable).replace("\\", "/")
    return (
        "To save something you verified for later sessions (a fix, a command that works, a convention): "
        f'run `"{exe}" -m prompture memory add --kind fix|command|convention|decision --verified '
        '--source <file:line> "<what and why>"` from the project folder.'
    )


# ------------------------------------------------------------------ reading requests

_CWD_TAG = re.compile(r"<cwd>\s*([^<]+?)\s*</cwd>")
_CWD_LINE = re.compile(r"(?im)^\s*(?:primary\s+)?working directory:\s*(.+?)\s*$")
_GEMINI_CWD = re.compile(r"I'm currently working in the directory:\s*(.+?)\s*$", re.M)
_NOT_A_PROMPT = (
    "<environment_context>",
    "<user_instructions>",
    "# AGENTS.md instructions",
    "<system-reminder>",
    "This is the Gemini CLI. We are setting up the context",
    "<project-memory",
)


def _items(dialect: str, body: dict[str, Any]) -> list[Any]:
    """A request's conversation: Anthropic ``messages``, Responses ``input``, Gemini ``contents``."""
    if dialect == "anthropic":
        return list(body.get("messages") or [])
    if dialect == "gemini":
        _nested = body.get("request")
        inner: dict[str, Any] = _nested if isinstance(_nested, dict) else body
        return [
            {
                "role": "assistant" if c.get("role") == "model" else "user",
                "content": c.get("parts") or [],
                "gemini": True,
            }
            for c in inner.get("contents") or []
            if isinstance(c, dict)
        ]
    return list(body.get("input") or [])


def _texts(content: Any) -> list[str]:
    if isinstance(content, str):
        return [content]
    if isinstance(content, list):
        return [str(b.get("text")) for b in content if isinstance(b, dict) and isinstance(b.get("text"), str)]
    return []


def cwd_of(dialect: str, body: dict[str, Any]) -> str | None:
    """The working directory a CLI states in its request (Claude Code's system prompt, Codex's environment context)."""
    if dialect == "anthropic":
        for text in _texts(body.get("system")):
            if m := (_CWD_TAG.search(text) or _CWD_LINE.search(text)):
                return m.group(1).strip()
        return None
    if dialect == "gemini":
        for item in _items(dialect, body)[:2]:
            for text in _texts(item["content"]):
                if m := _GEMINI_CWD.search(text):
                    return m.group(1).strip().rstrip(".")
        return None
    for item in body.get("input") or []:
        if isinstance(item, dict) and item.get("type") == "message":
            for text in _texts(item.get("content")):
                if m := _CWD_TAG.search(text):
                    return m.group(1).strip()
    return None


_WRAPPED = re.compile(r"^\s*<([A-Za-z_][\w\- ]*)>.*</\1>\s*$", re.S)


def _injected(text: str) -> bool:
    """Context a CLI adds on the user's behalf: known preambles, or text wrapped whole in one tag."""
    return text.lstrip().startswith(_NOT_A_PROMPT) or bool(_WRAPPED.match(text))


def _is_prompt(message: Any, dialect: str) -> bool:
    if not isinstance(message, dict) or message.get("role") != "user":
        return False
    if dialect == "openai" and message.get("type", "message") != "message":
        return False
    content = message.get("content")
    if isinstance(content, list) and any(
        isinstance(b, dict) and (b.get("type") == "tool_result" or "functionResponse" in b) for b in content
    ):
        return False
    texts = [t for t in _texts(content) if t.strip()]
    return bool(texts) and not all(_injected(t) for t in texts)


def prompts_of(dialect: str, body: dict[str, Any]) -> list[int]:
    """Indexes of the user's own prompts in a request's history (not tool results or injected context)."""
    return [i for i, m in enumerate(_items(dialect, body)) if _is_prompt(m, dialect)]


def prompt_text(dialect: str, body: dict[str, Any], index: int) -> str:
    texts = [t for t in _texts(_items(dialect, body)[index].get("content")) if not _injected(t)]
    return "\n".join(texts)


def _command_head(arguments: Any) -> str | None:
    """``npm test`` from a shell tool's arguments: the first two words of its last command."""
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError:
            return None
    if not isinstance(arguments, dict):
        return None
    cmd = arguments.get("command") or arguments.get("cmd")
    if isinstance(cmd, list):
        cmd = cmd[-1] if cmd else ""
    if not isinstance(cmd, str) or not cmd.strip():
        return None
    last = re.split(r"&&|;|\|\|", cmd)[-1].strip()
    words = [w for w in last.split() if not w.startswith("-")][:2]
    return " ".join(words) or None


def previous_task(dialect: str, body: dict[str, Any]) -> dict[str, Any] | None:
    """The task before the newest prompt: its prompt, tool steps and final text (for skill mining)."""
    prompts = prompts_of(dialect, body)
    if len(prompts) < 2:
        return None
    start, end = prompts[-2], prompts[-1]
    items = _items(dialect, body)
    steps: list[str] = []
    output = ""
    for item in items[start + 1 : end]:
        if not isinstance(item, dict):
            continue
        if dialect == "gemini":
            if item.get("role") != "assistant":
                continue
            for part in item["content"]:
                call = part.get("functionCall") if isinstance(part, dict) else None
                if isinstance(call, dict):
                    name = str(call.get("name") or "tool")
                    head = _command_head(call.get("args")) if "shell" in name else None
                    steps.append(f"{name}({head})" if head else name)
            output = " ".join(_texts(item["content"])) or output
        elif dialect == "anthropic":
            if item.get("role") != "assistant":
                continue
            for block in item.get("content") or [] if isinstance(item.get("content"), list) else []:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    name = str(block.get("name") or "tool")
                    head = (
                        _command_head(block.get("input")) if name.lower() in ("bash", "shell", "powershell") else None
                    )
                    steps.append(f"{name}({head})" if head else name)
            output = " ".join(_texts(item.get("content"))) or output
        else:
            kind = item.get("type")
            if kind in ("function_call", "custom_tool_call", "local_shell_call"):
                name = str(item.get("name") or kind)
                head = _command_head(item.get("arguments") or item.get("action") or item.get("input"))
                steps.append(f"{name}({head})" if head else name)
            elif kind == "message" and item.get("role") == "assistant":
                output = " ".join(_texts(item.get("content"))) or output
    return {"prompt": prompt_text(dialect, body, start)[:400], "steps": steps, "output": output[:600]}


# ------------------------------------------------------------------ the service


class MemoryService:
    """Project notes, what each session received, and mined skill proposals. Thread-safe."""

    def __init__(
        self,
        memory: ProjectMemory | None = None,
        *,
        directory: str | Path | None = MEMORY_DIR,
        python: str | None = None,
    ) -> None:
        self.dir = Path(directory) if directory else None
        self.memory = memory or ProjectMemory(db_path=(self.dir / "projects.db") if self.dir else None)
        self.python = python
        self._lock = threading.Lock()
        self._settings: dict[str, Any] | None = None
        self._injections: deque[dict[str, Any]] = deque(maxlen=2000)
        self._by_session: dict[tuple[str, str], dict[str, Any]] = {}
        self._cwd: dict[str, str] = {}  # project -> folder
        self._miners: dict[str, Any] = {}
        self._seen_tasks: set[tuple[str, int]] = set()
        self._loaded = False

    # -- files ----------------------------------------------------------------

    def _path(self, name: str) -> Path | None:
        return self.dir / name if self.dir else None

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        for record in self._read_jsonl("injections.jsonl"):
            self._injections.append(record)
            self._by_session[(record.get("agent", ""), record.get("session", ""))] = record
        for task in self._read_jsonl("tasks.jsonl"):
            if task.get("project") and task.get("cwd"):
                self._cwd.setdefault(task["project"], task["cwd"])
            self._mine(task, persist=False)

    def _read_jsonl(self, name: str) -> list[dict[str, Any]]:
        path = self._path(name)
        if path is None or not path.exists():
            return []
        out = []
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if isinstance(record, dict):
                    out.append(record)
        except OSError:
            logger.debug("could not read %s", path, exc_info=True)
        return out

    def _append(self, name: str, record: dict[str, Any]) -> None:
        path = self._path(name)
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            logger.debug("could not write %s", path, exc_info=True)

    def warm(self) -> None:
        """Load everything a session's first prompt needs, ahead of it."""
        try:
            self.settings()
            with self._lock:
                self._load()
            self.memory.projects()
        except Exception:
            logger.debug("memory warm-up failed", exc_info=True)

    # -- settings -------------------------------------------------------------

    def settings(self) -> dict[str, Any]:
        with self._lock:
            if self._settings is None:
                loaded: dict[str, Any] = {}
                path = self._path("settings.json")
                if path is not None and path.exists():
                    try:
                        raw = json.loads(path.read_text(encoding="utf-8"))
                        loaded = raw if isinstance(raw, dict) else {}
                    except (OSError, ValueError):
                        loaded = {}
                self._settings = {**DEFAULTS, **{k: v for k, v in loaded.items() if k in DEFAULTS}}
            return dict(self._settings)

    def save_settings(self, changes: dict[str, Any]) -> dict[str, Any]:
        current = self.settings()
        for key in ("enabled", "verified_only", "teach"):
            if isinstance(changes.get(key), bool):
                current[key] = changes[key]
        budget = changes.get("budget_tokens")
        if isinstance(budget, int) and not isinstance(budget, bool):
            current["budget_tokens"] = max(50, min(8000, budget))
        with self._lock:
            self._settings = current
        path = self._path("settings.json")
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(current, indent=2), encoding="utf-8")
        return dict(current)

    # -- injection ------------------------------------------------------------

    def remember_folder(self, project: str | None, cwd: str | None) -> None:
        if project and cwd:
            with self._lock:
                self._cwd[project] = cwd

    def folder(self, project: str) -> str | None:
        with self._lock:
            self._load()
            return self._cwd.get(project)

    def received(self, agent: str, session: str) -> dict[str, Any] | None:
        """What a session was given, if anything."""
        with self._lock:
            self._load()
            return self._by_session.get((agent, session))

    def inject(self, agent: str, session: str, project: str | None, query: str, *, via: str) -> str | None:
        """The notes a session starts with, chosen once; ``None`` when there are none to give.

        ``via`` is ``"hook"`` or ``"router"``. A session that already got its
        notes gets nothing new from the hook; through the router it gets the
        same text again, since every request must carry it.
        """
        if not session or not project:
            return None
        existing = self.received(agent, session)
        if existing is not None:
            return existing["text"] if via == "router" and existing.get("via") == "router" else None
        conf = self.settings()
        if not conf["enabled"]:
            return None
        facts = self.memory.select(
            project, query, budget_tokens=int(conf["budget_tokens"]), verified_only=bool(conf["verified_only"])
        )
        text = self.memory.render(project, facts, save_hint=save_hint(self.python) if conf["teach"] else None)
        if not text:
            return None
        record = {
            "id": uuid.uuid4().hex[:12],
            "ts": time.time(),
            "agent": agent,
            "session": session,
            "project": project,
            "via": via,
            "fact_ids": [f.id for f in facts],
            "tokens": estimate_tokens(text),
            "text": text,
        }
        with self._lock:
            if (agent, session) in self._by_session:  # another thread got there first
                return None if via == "hook" else self._by_session[(agent, session)]["text"]
            self._by_session[(agent, session)] = record
            self._injections.append(record)
        self._append("injections.jsonl", record)
        return text

    def injections(
        self, project: str | None = None, session: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        with self._lock:
            self._load()
            records = list(self._injections)
        records = [
            r
            for r in records
            if (project is None or r.get("project") == project) and (session is None or r.get("session") == session)
        ]
        return list(reversed(records[-limit:]))

    # -- skills ---------------------------------------------------------------

    def observe_task(
        self, agent: str, session: str, index: int, project: str | None, cwd: str | None, task: dict[str, Any]
    ) -> None:
        """A task finished (the next prompt started): feed its tool steps to the project's skill miner."""
        if not project or not task.get("steps"):
            return
        with self._lock:
            self._load()
            key = (f"{agent}:{session}", index)
            if key in self._seen_tasks:
                return
            self._seen_tasks.add(key)
            if cwd:
                self._cwd[project] = cwd
        record = {"ts": time.time(), "agent": agent, "project": project, "cwd": cwd, **task}
        self._mine(record, persist=True)

    def _miner(self, project: str) -> Any:
        miner = self._miners.get(project)
        if miner is None:
            from ..agents.skill_miner import SkillMiner

            miner = self._miners[project] = SkillMiner(
                judge=False, auto_register=False, min_occurrences=SKILL_OCCURRENCES, min_tools=2, min_steps=3
            )
        return miner

    def _mine(self, task: dict[str, Any], *, persist: bool) -> None:
        project = task.get("project")
        steps = [str(s) for s in task.get("steps") or []]
        if not project or len(steps) < 3:
            return
        from ..agents.types import AgentResult

        result = AgentResult(
            output=task.get("output") or "",
            output_text=task.get("output") or "(task finished)",
            messages=[{"role": "user", "content": task.get("prompt") or ""}],
            usage={},
            all_tool_calls=[{"name": s} for s in steps],
        )
        try:
            self._miner(project).observe(result)
        except Exception:  # mining is a suggestion; never a failure
            logger.debug("skill mining failed", exc_info=True)
        if persist:
            self._append("tasks.jsonl", task)

    def skills(self, project: str) -> list[dict[str, Any]]:
        with self._lock:
            self._load()
            miner = self._miners.get(project)
        folder = self.folder(project)
        out = []
        for p in miner.proposals if miner is not None else []:
            saved = folder is not None and (Path(folder) / ".claude" / "skills" / p.name / "SKILL.md").exists()
            out.append(
                {
                    "name": p.name,
                    "description": p.description,
                    "when_to_use": p.when_to_use,
                    "steps": list(p.tool_sequence),
                    "occurrences": p.occurrences,
                    "markdown": p.to_markdown(),
                    "saved": saved,
                }
            )
        return out

    def save_skill(self, project: str, name: str) -> str:
        """Write a proposal as ``<project>/.claude/skills/<name>/SKILL.md``; returns the path."""
        miner = self._miners.get(project)
        proposal = next((p for p in (miner.proposals if miner is not None else []) if p.name == name), None)
        if proposal is None or miner is None:
            raise KeyError(f"No skill proposal named {name!r} for {project}.")
        folder = self.folder(project)
        if not folder or not Path(folder).is_dir():
            raise FileNotFoundError(f"Don't know where {project} is on this PC yet.")
        return str(miner.save(proposal, skills_dir=Path(folder) / ".claude" / "skills", overwrite=True))

    # -- projects -------------------------------------------------------------

    def projects(self) -> list[dict[str, Any]]:
        """Projects with notes, plus ones agents worked in (so a first note can be added)."""
        known = {p["project"]: p for p in self.memory.projects()}
        with self._lock:
            self._load()
            folders = dict(self._cwd)
            seen = {r.get("project") for r in self._injections if r.get("project")}
        for name in seen | set(folders):
            known.setdefault(name, {"project": name, "facts": 0, "verified": 0})
        for name, entry in known.items():
            entry["folder"] = folders.get(name)
        return sorted(known.values(), key=lambda p: (-p["facts"], p["project"]))


__all__ = [
    "MemoryService",
    "cwd_of",
    "previous_task",
    "project_name",
    "prompts_of",
    "save_hint",
]
