"""Automations: coding-agent steps queued to run one after another.

A queue is a project folder, a coding agent (Claude Code or Codex) and a list
of steps such as ``/gsd:execute-phase 3``. The companion runs each step as soon
as the one before it finishes: like pre-moves in chess, the next hour of work
is decided before this one ends. Steps run unattended, so the agent's
permission prompts are skipped (``approval_mode="auto"``).

A queue pauses by itself when a step fails, when the agent ends a step on a
question (answering continues that step's session), when the agent's plan
window is nearly used up (it resumes when the window resets), or when the run
passes a cost cap. Steps that haven't started can be added, removed and
reordered while it runs. A step either continues the previous step's session
(``"same"``) or starts a fresh one (``"new"``).

One queue runs at a time. Runs and each step's log are kept under
``~/.prompture/automations/``.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import logging
import os
import re
import threading
import time
import uuid
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

from ..infra.coding_agent_events import CodingAgentEvent
from .live import LiveBus

logger = logging.getLogger("prompture.companion.automations")

ROOT = Path.home() / ".prompture" / "automations"

#: Agents a queue can drive: their stream carries events, costs and session ids.
AGENTS = {"claude": "Claude Code", "codex": "Codex"}
#: Where each agent's plan windows are in the companion's rate limits.
PLAN_TARGETS = {"claude": "claude/claude-code", "codex": "openai/codex"}
#: A plan window with this share (percent) or less left counts as nearly used up.
NEAR_LIMIT = 10.0
#: When a step hits a usage limit and no reset time is known, try again after this long.
LIMIT_RETRY_SECONDS = 15 * 60
HISTORY_MAX = 50
LOG_MAX = 5000
ENDED = ("finished", "failed", "stopped")

#: Appended to Claude Code's system prompt for every step.
UNATTENDED = (
    "You are running unattended, as one step of a queue: nobody is watching this session. "
    "Finish the task without asking follow-up questions or offering next steps. "
    "Only end with a question if you cannot continue without an answer."
)

Runner = Callable[..., AsyncIterator[CodingAgentEvent]]
Limits = Callable[[], dict[str, dict[str, Any]]]


class AutomationError(Exception):
    """A request the current state can't take; ``status`` is the HTTP status to answer with."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


@dataclasses.dataclass
class Step:
    id: str
    text: str
    session: str = "same"  # "same": continue the previous step's session; "new": start fresh
    status: str = "waiting"  # waiting | running | done | failed | skipped | asked | stopped
    started_at: float | None = None
    ended_at: float | None = None
    duration_s: float = 0.0
    cost_usd: float | None = None
    tokens: int = 0
    action: str | None = None
    #: When the attempt running now started (a step can run again after an answer or a retry).
    running_since: float | None = None
    session_id: str | None = None
    error: str | None = None
    #: What to send next instead of ``text``: an answer, or "continue" after a usage limit.
    reply: str | None = None


@dataclasses.dataclass
class Run:
    id: str
    cwd: str
    agent: str
    model: str | None
    steps: list[Step]
    stop: dict[str, Any]
    created_at: float
    status: str = "running"  # running | paused | finished | failed | stopped
    reason: str | None = None  # why it's paused: ask | limit | fail | cost | manual
    current: int = 0
    pausing: bool = False  # pause once the running step finishes
    question: str | None = None
    resumes_at: float | None = None
    ended_at: float | None = None
    note: str | None = None
    #: Skip the plan-limit check for the next step ("Resume now").
    ignore_limit: bool = False

    @property
    def project(self) -> str:
        return Path(self.cwd).name or self.cwd

    @property
    def cost_usd(self) -> float:
        return round(sum(s.cost_usd or 0.0 for s in self.steps), 6)

    def to_dict(self) -> dict[str, Any]:
        steps = [{k: v for k, v in dataclasses.asdict(s).items() if k != "reply"} for s in self.steps]
        return {
            "id": self.id,
            "cwd": self.cwd,
            "project": self.project,
            "agent": self.agent,
            "agent_name": AGENTS.get(self.agent, self.agent),
            "model": self.model,
            "status": self.status,
            "reason": self.reason,
            "current": self.current,
            "pausing": self.pausing,
            "question": self.question,
            "resumes_at": self.resumes_at,
            "created_at": self.created_at,
            "ended_at": self.ended_at,
            "note": self.note,
            "stop": dict(self.stop),
            "cost_usd": self.cost_usd,
            "duration_s": round(sum(s.duration_s for s in self.steps), 1),
            "steps": steps,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Run:
        fields = {f.name for f in dataclasses.fields(Step)}
        steps = [Step(**{k: v for k, v in s.items() if k in fields}) for s in data.get("steps", [])]
        keep = {f.name for f in dataclasses.fields(cls)} - {"steps"}
        return cls(steps=steps, **{k: v for k, v in data.items() if k in keep})


def roadmap_steps(cwd: str | os.PathLike[str]) -> list[str]:
    """``/gsd:execute-phase N`` for each unchecked phase in ``.planning/ROADMAP.md``, in order."""
    path = Path(cwd) / ".planning" / "ROADMAP.md"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    phases = re.findall(r"^\s*[-*]\s*\[ \]\s*\**\s*Phase\s+(\d+(?:\.\d+)?)\b", text, re.MULTILINE | re.IGNORECASE)
    return [f"/gsd:execute-phase {n}" for n in dict.fromkeys(phases)]


def _ends_on_question(text: str | None) -> bool:
    lines = [line.strip() for line in (text or "").strip().splitlines() if line.strip()]
    return bool(lines) and lines[-1].rstrip("*_ )").endswith("?")


_LIMIT_RE = re.compile(r"usage limit|limit reached|rate limit|quota|hit your limit", re.IGNORECASE)
_LIMIT_EPOCH_RE = re.compile(r"limit reached\|(\d{10})")


def _short(value: Any, limit: int = 120) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def describe(event: CodingAgentEvent, cwd: str) -> tuple[str, str]:
    """``(action, log line)`` for a tool call: "Editing src/app.py…" and "Edit src/app.py"."""
    name = event.tool_name or "tool"
    args = event.tool_input or {}
    target = args.get("file_path") or args.get("path") or args.get("notebook_path")
    if isinstance(target, str):
        with contextlib.suppress(ValueError):
            target = str(Path(target).resolve().relative_to(Path(cwd).resolve()))
        target = target.replace("\\", "/")
    lower = name.lower()
    if lower in ("read", "notebookread"):
        return f"Reading {_short(target)}…", f"Read {target}"
    if lower in ("edit", "multiedit", "write", "notebookedit", "apply_patch"):
        return f"Editing {_short(target)}…", f"Edit {target}"
    if lower in ("bash", "exec", "powershell", "shell"):
        command = args.get("command")
        command = " ".join(command) if isinstance(command, list) else command
        return f"Running {_short(command, 80)}…", f"$ {_short(command, 400)}"
    if lower in ("grep", "glob"):
        return f"Searching {_short(args.get('pattern'), 60)}…", f"{name} {_short(args.get('pattern'), 200)}"
    if lower in ("task", "agent"):
        return "Running a subagent…", f"Subagent: {_short(args.get('description') or args.get('prompt'), 200)}"
    if lower in ("webfetch", "websearch"):
        return "Browsing the web…", f"{name} {_short(args.get('url') or args.get('query'), 200)}"
    if lower == "todowrite":
        return "Updating its to-do list…", "Updated its to-do list"
    if lower == "skill":
        return (
            f"Using {_short(args.get('skill') or args.get('command'), 60)}…",
            f"Skill {_short(args.get('skill'), 200)}",
        )
    return f"Using {name}…", name


class Automations:
    """The queue runner behind ``/v1/automations``; see the module docstring."""

    def __init__(
        self,
        root: Path = ROOT,
        *,
        bus: LiveBus | None = None,
        limits: Limits | None = None,
        runner: Runner | None = None,
        installed: Callable[[str], bool] | None = None,
    ) -> None:
        self.root = root
        self.bus = bus
        self.limits = limits
        self.runner = runner
        self.installed = installed or _installed
        self.lock = threading.RLock()
        self.wake = threading.Condition(self.lock)
        self.run: Run | None = None
        self.history: list[dict[str, Any]] = []
        self._active: tuple[asyncio.AbstractEventLoop, asyncio.Task[Any]] | None = None
        self._cancel: str | None = None  # "skip" | "stop" for the running step
        self._agents: tuple[float, list[dict[str, Any]]] | None = None
        self._load()

    # ------------------------------------------------------------ storage

    def _load(self) -> None:
        with contextlib.suppress(OSError, ValueError):
            data = json.loads((self.root / "history.json").read_text(encoding="utf-8"))
            if isinstance(data, list):
                self.history = [r for r in data if isinstance(r, dict)][:HISTORY_MAX]
        # A run the companion was driving when it stopped can't pick up where it was.
        with contextlib.suppress(OSError, ValueError, TypeError):
            path = self.root / "current.json"
            run = Run.from_dict(json.loads(path.read_text(encoding="utf-8")))
            path.unlink()
            if run.status not in ENDED:
                for step in run.steps:
                    if step.status == "running":
                        step.status = "stopped"
                run.status, run.reason, run.ended_at = "stopped", None, run.ended_at or time.time()
                run.note = "Prompture stopped"
            self.history = [r for r in self.history if r.get("id") != run.id]
            self.history.insert(0, run.to_dict())
            self._write("history.json", self.history[:HISTORY_MAX])

    def _write(self, name: str, data: Any) -> None:
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            tmp = self.root / f"{name}.tmp"
            tmp.write_text(json.dumps(data), encoding="utf-8")
            os.replace(tmp, self.root / name)
        except OSError:
            logger.debug("could not save %s", name, exc_info=True)

    def _save(self, run: Run) -> None:
        """Persist ``run`` and tell clients it changed."""
        body = run.to_dict()
        if run.status in ENDED:
            self.history = [r for r in self.history if r.get("id") != run.id]
            self.history.insert(0, body)
            del self.history[HISTORY_MAX:]
            self._write("history.json", self.history)
            with contextlib.suppress(OSError):
                (self.root / "current.json").unlink()
        else:
            self._write("current.json", body)
        if self.bus is not None:
            self.bus.publish("automation.updated", {"automation": body})

    def _log(self, run: Run, step: Step, text: str, kind: str = "info") -> None:
        started = step.started_at or time.time()
        line = {"t": round(max(0.0, time.time() - started), 1), "text": text, "kind": kind}
        try:
            folder = self.root / "logs" / run.id
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{step.id}.jsonl"
            if not path.exists() or path.stat().st_size < LOG_MAX * 600:
                with path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(line) + "\n")
        except OSError:
            logger.debug("could not write a log line", exc_info=True)

    # ------------------------------------------------------------ reading

    def agents(self) -> list[dict[str, Any]]:
        """The agents a queue can use, and whether each is installed (cached for a minute)."""
        if self._agents is None or time.monotonic() - self._agents[0] > 60:
            out = []
            for agent_id, name in AGENTS.items():
                try:
                    ok = self.installed(agent_id)
                except Exception:
                    ok = False
                out.append({"id": agent_id, "name": name, "installed": ok})
            self._agents = (time.monotonic(), out)
        return self._agents[1]

    def state(self) -> dict[str, Any]:
        with self.lock:
            return {
                "current": self.run.to_dict() if self.run else None,
                "history": [_summary(r) for r in self.history],
                "agents": self.agents(),
            }

    def get_run(self, run_id: str) -> dict[str, Any]:
        with self.lock:
            if self.run and self.run.id == run_id:
                return self.run.to_dict()
            for r in self.history:
                if r.get("id") == run_id:
                    return r
        raise AutomationError(404, "No such run.")

    def log(self, run_id: str, step_id: str) -> list[dict[str, Any]]:
        if not re.fullmatch(r"[\w-]+", run_id) or not re.fullmatch(r"[\w-]+", step_id):
            raise AutomationError(404, "No such step.")
        path = self.root / "logs" / run_id / f"{step_id}.jsonl"
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        out = []
        for raw in lines[-LOG_MAX:]:
            with contextlib.suppress(ValueError):
                out.append(json.loads(raw))
        return out

    # ------------------------------------------------------------ changing

    def start(self, body: dict[str, Any]) -> dict[str, Any]:
        cwd = str(body.get("cwd") or "").strip()
        agent = str(body.get("agent") or "claude")
        if not cwd or not Path(cwd).is_dir():
            raise AutomationError(422, "Pick a project folder that exists.")
        if agent not in AGENTS:
            raise AutomationError(422, f"agent must be one of {', '.join(AGENTS)}.")
        steps = _parse_steps(body.get("steps"))
        if not steps:
            raise AutomationError(422, "Add at least one step.")
        stop = body.get("stop") if isinstance(body.get("stop"), dict) else {}
        cost = stop.get("cost_usd")
        model = str(body.get("model") or "").strip()
        with self.lock:
            if self.run and self.run.status not in ENDED:
                raise AutomationError(409, "A queue is already running.")
            run = Run(
                id=uuid.uuid4().hex[:12],
                cwd=str(Path(cwd).resolve()),
                agent=agent,
                model=None if model in ("", "default") else model,
                steps=steps,
                stop={
                    "fail": stop.get("fail", True) is not False,
                    "ask": stop.get("ask", True) is not False,
                    "limit": stop.get("limit", True) is not False,
                    "cost_usd": float(cost) if isinstance(cost, (int, float)) and cost > 0 else None,
                },
                created_at=time.time(),
            )
            self.run = run
            self._save(run)
            threading.Thread(target=self._drive, args=(run,), name="prompture-automation", daemon=True).start()
            return run.to_dict()

    def set_steps(self, items: Any) -> dict[str, Any]:
        """Replace the steps that haven't started, keeping the ids of those that stay."""
        run = self._live_run()
        with self.lock:
            fixed = run.steps[: run.current + 1] if run.current < len(run.steps) else list(run.steps)
            waiting = {s.id: s for s in run.steps[len(fixed) :]}
            new = []
            for item in items if isinstance(items, list) else []:
                if not isinstance(item, dict) or not str(item.get("text") or "").strip():
                    continue
                old = waiting.get(str(item.get("id") or ""))
                step = old or Step(id=uuid.uuid4().hex[:8], text="")
                step.text = str(item["text"]).strip()
                step.session = "new" if item.get("session") == "new" else "same"
                new.append(step)
            run.steps = fixed + new
            self._save(run)
            self.wake.notify_all()
            return run.to_dict()

    def pause(self) -> dict[str, Any]:
        run = self._live_run()
        with self.lock:
            if run.status == "running":
                run.pausing = True
                self._save(run)
            return run.to_dict()

    def resume(self) -> dict[str, Any]:
        """Carry on. After a failure that re-runs the step; after a question it moves on."""
        run = self._live_run()
        with self.lock:
            if run.status == "running":
                run.pausing = False
            elif run.status == "paused":
                step = run.steps[run.current] if run.current < len(run.steps) else None
                if run.reason == "cost":
                    run.stop["cost_usd"] = None
                elif run.reason == "limit":
                    run.ignore_limit = True
                elif run.reason == "fail" and step is not None:
                    step.status, step.error = "waiting", None
                elif run.reason == "ask" and step is not None:
                    step.status = "done"
                    run.current += 1
                self._carry_on(run)
            self._save(run)
            self.wake.notify_all()
            return run.to_dict()

    def answer(self, text: str) -> dict[str, Any]:
        run = self._live_run()
        text = (text or "").strip()
        if not text:
            raise AutomationError(422, "Type an answer.")
        with self.lock:
            if run.status != "paused" or run.reason != "ask":
                raise AutomationError(409, "The queue isn't waiting for an answer.")
            step = run.steps[run.current]
            step.status, step.reply = "waiting", text
            self._log(run, step, f"You: {text}", "you")
            self._carry_on(run)
            self._save(run)
            self.wake.notify_all()
            return run.to_dict()

    def skip(self) -> dict[str, Any]:
        run = self._live_run()
        with self.lock:
            step = run.steps[run.current] if run.current < len(run.steps) else None
            if step is None:
                return run.to_dict()
            if step.status == "running":
                self._cancel_step("skip")
            else:
                step.status = "skipped"
                run.current += 1
                if run.status == "paused":
                    self._carry_on(run)
                self._save(run)
                self.wake.notify_all()
            return run.to_dict()

    def stop(self, note: str | None = None) -> dict[str, Any]:
        run = self._live_run()
        with self.lock:
            status = "failed" if run.status == "paused" and run.reason == "fail" else "stopped"
            if run.current < len(run.steps) and run.steps[run.current].status == "running":
                self._cancel_step("stop")
            self._end(run, status, note)
            self.wake.notify_all()
            return run.to_dict()

    def shutdown(self) -> None:
        """Stop the running queue (the companion is going away and its agent with it)."""
        with self.lock:
            if self.run and self.run.status not in ENDED:
                self.stop("Prompture stopped")

    # ------------------------------------------------------------ internals

    def _live_run(self) -> Run:
        with self.lock:
            if self.run is None or self.run.status in ENDED:
                raise AutomationError(409, "No queue is running.")
            return self.run

    def _carry_on(self, run: Run) -> None:
        run.status, run.reason, run.question, run.resumes_at = "running", None, None, None

    def _pause(self, run: Run, reason: str, resumes_at: float | None = None) -> None:
        run.status, run.reason, run.resumes_at, run.pausing = "paused", reason, resumes_at, False
        self._save(run)

    def _end(self, run: Run, status: str, note: str | None = None) -> None:
        run.status, run.ended_at, run.pausing = status, time.time(), False
        if status != "failed":
            run.reason = None
        if note:
            run.note = note
        for step in run.steps:
            if step.status == "running":
                step.status = "stopped"
                step.action = step.running_since = None
        self._save(run)

    def _cancel_step(self, how: str) -> None:
        self._cancel = how
        if self._active is not None:
            loop, task = self._active
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(task.cancel)

    def _session_before(self, run: Run, index: int) -> str | None:
        for step in reversed(run.steps[:index]):
            if step.session_id:
                return step.session_id
        return None

    def _plan_near(self, run: Run) -> tuple[bool, float | None]:
        """Whether the agent's plan is nearly used up, and when the tightest window resets."""
        if self.limits is None or not run.stop.get("limit"):
            return False, None
        try:
            snap = self.limits().get(PLAN_TARGETS.get(run.agent, ""))
        except Exception:
            logger.debug("plan limits failed", exc_info=True)
            return False, None
        now = time.time()
        resets: list[float] = []
        for window in (snap or {}).get("windows", {}).values():
            limit, left, reset = window.get("limit"), window.get("remaining"), window.get("resets_at")
            if not limit or left is None or (reset and reset <= now):
                continue
            if left / limit * 100 <= NEAR_LIMIT:
                resets.append(float(reset) if reset else now + LIMIT_RETRY_SECONDS)
        return (True, max(resets)) if resets else (False, None)

    def _next_job(self, run: Run, step: Step) -> tuple[str, str | None] | None:
        """What to send for ``step`` and on which session, or ``None`` after pausing the run."""
        if run.pausing:
            self._pause(run, "manual")
            return None
        cap = run.stop.get("cost_usd")
        if cap and run.cost_usd >= cap:
            self._pause(run, "cost")
            return None
        if run.ignore_limit:
            run.ignore_limit = False
        else:
            near, resets = self._plan_near(run)
            if near:
                self._pause(run, "limit", resets)
                return None
        if step.reply:
            task, step.reply = step.reply, None
            return task, step.session_id
        inherit = step.session == "same" and run.current > 0
        return step.text, self._session_before(run, run.current) if inherit else None

    def _drive(self, run: Run) -> None:
        while True:
            with self.lock:
                if run is not self.run or run.status in ENDED:
                    return
                if run.status == "paused":
                    if run.reason == "limit" and run.resumes_at and time.time() >= run.resumes_at:
                        run.ignore_limit = False
                        self._carry_on(run)
                        self._save(run)
                    else:
                        self.wake.wait(timeout=30)
                        continue
                if run.current >= len(run.steps):
                    self._end(run, "finished")
                    return
                step = run.steps[run.current]
                if step.status in ("done", "skipped", "failed"):
                    run.current += 1
                    continue
                job = self._next_job(run, step)
                if job is None:
                    continue
                task, session_id = job
                step.status, step.error, step.action = "running", None, None
                step.started_at = step.started_at or time.time()
                step.running_since = time.time()
                started = time.time()
                self._cancel = None
                if task == step.text:
                    how = "continues the previous session" if session_id else "new session"
                    self._log(run, step, f"{AGENTS[run.agent]} · {task} · {how}", "cmd")
                self._save(run)
            outcome = self._execute(run, step, task, session_id)
            with self.lock:
                step.ended_at = time.time()
                step.duration_s = round(step.duration_s + step.ended_at - started, 1)
                step.action = step.running_since = None
                self._settle(run, step, outcome)

    def _settle(self, run: Run, step: Step, outcome: dict[str, Any]) -> None:
        cancelled, error = outcome.get("cancelled"), outcome.get("error")
        if run.status in ENDED:  # stopped while it ran
            self._save(run)
            return
        if cancelled == "skip":
            step.status = "skipped"
            self._log(run, step, "Skipped", "info")
            run.current += 1
        elif error:
            step.error = _short(error, 300)
            self._log(run, step, _short(error, 600), "err")
            if _LIMIT_RE.search(error):
                step.status = "waiting"
                if step.session_id:
                    step.reply = "Continue where you left off."
                match = _LIMIT_EPOCH_RE.search(error)
                resets = float(match.group(1)) if match else self._plan_near(run)[1]
                self._pause(run, "limit", resets or time.time() + LIMIT_RETRY_SECONDS)
                return
            step.status = "failed"
            if run.stop.get("fail"):
                self._pause(run, "fail")
                return
            run.current += 1
        elif run.stop.get("ask") and _ends_on_question(outcome.get("text")):
            step.status = "asked"
            run.question = _short(_last_paragraph(outcome.get("text")), 600)
            self._pause(run, "ask")
            return
        else:
            step.status = "done"
            self._log(run, step, "Done", "ok")
            run.current += 1
        self._save(run)

    def _execute(self, run: Run, step: Step, task: str, session_id: str | None) -> dict[str, Any]:
        outcome: dict[str, Any] = {"error": None, "text": None, "cancelled": None}

        async def go() -> None:
            runner = self.runner
            if runner is None:
                from ..infra.coding_agents import astream_coding_agent

                runner = astream_coding_agent
            extra = ["--append-system-prompt", UNATTENDED] if run.agent == "claude" else []
            stream = runner(
                run.agent,
                task,
                cwd=run.cwd,
                approval_mode="auto",
                model=run.model,
                session_id=session_id,
                extra_args=extra,
            )
            async with contextlib.aclosing(stream):  # type: ignore[type-var]
                async for event in stream:
                    self._on_event(run, step, event, outcome)

        loop = asyncio.new_event_loop()
        job = loop.create_task(go())
        with self.lock:
            self._active = (loop, job)
            if self._cancel:  # skipped or stopped before it got going
                job.cancel()
        try:
            loop.run_until_complete(job)
        except asyncio.CancelledError:
            outcome["cancelled"] = self._cancel or "stop"
        except Exception as exc:
            logger.debug("automation step failed", exc_info=True)
            outcome["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            with self.lock:
                self._active = None
            with contextlib.suppress(Exception):
                loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()
        return outcome

    def _on_event(self, run: Run, step: Step, event: CodingAgentEvent, outcome: dict[str, Any]) -> None:
        with self.lock:
            if event.session_id:
                step.session_id = event.session_id
            if event.type == "tool_call":
                step.action, line = describe(event, run.cwd)
                self._log(run, step, line, "tool")
            elif event.type == "message" and event.text:
                outcome["text"] = event.text
                self._log(run, step, _short(event.text, 1200), "msg")
            elif event.type == "done":
                if event.cost_usd is not None:
                    step.cost_usd = round((step.cost_usd or 0.0) + event.cost_usd, 6)
                step.tokens += (event.input_tokens or 0) + (event.output_tokens or 0)
                if event.text:
                    outcome["text"] = event.text
                if event.error:
                    outcome["error"] = outcome["error"] or event.error
            elif event.type == "error" and event.error:
                # The first error says why; a non-zero exit that follows only repeats it.
                outcome["error"] = outcome["error"] or event.error


def _last_paragraph(text: str | None) -> str:
    parts = [p.strip() for p in re.split(r"\n\s*\n", (text or "").strip()) if p.strip()]
    return parts[-1] if parts else ""


def _parse_steps(items: Any) -> list[Step]:
    steps = []
    for item in items if isinstance(items, list) else []:
        text = str((item.get("text") if isinstance(item, dict) else item) or "").strip()
        if not text:
            continue
        session = item.get("session") if isinstance(item, dict) else None
        steps.append(Step(id=uuid.uuid4().hex[:8], text=text, session="new" if session == "new" else "same"))
    return steps


def _summary(run: dict[str, Any]) -> dict[str, Any]:
    """A history row: everything but the per-step detail."""
    steps = run.get("steps") or []
    touched = [s for s in steps if s.get("status") != "waiting"]
    last = (touched or steps or [{}])[-1]
    return {
        **{k: v for k, v in run.items() if k != "steps"},
        "steps": [{"status": s.get("status"), "text": s.get("text")} for s in steps],
        "last": last.get("text"),
        "last_error": last.get("error"),
    }


def _installed(agent_id: str) -> bool:
    from ..infra.discovery import resolve_coding_agent_executable

    executable, _healthy, _error = resolve_coding_agent_executable(agent_id, agent_id)
    return executable is not None
