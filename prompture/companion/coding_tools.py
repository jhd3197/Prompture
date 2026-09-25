"""Usage from coding agents that don't run through Prompture, for the companion.

The reading happens in the engine — :mod:`prompture.infra.coding_agent_usage`
and one reader per agent in :mod:`prompture.infra.coding_agent_readers`
(Claude Code, Codex, Kimi Code, Gemini CLI, Qwen Code, OpenCode, Cline,
Roo Code, Continue; Cursor and Antigravity are detected only). This module
adapts it to the companion API:

- calls join the ledger's rows in ``/v1/spend`` and stream as
  ``request.finished`` events tagged with the agent (``key_name``, ``tool``);
- plan windows (Codex from its logs; Claude Code only when opted in) join
  ``/v1/limits`` as ``source: "plan"`` targets;
- ``/v1/tools`` lists every agent: installed, runnable by Prompture, and its
  usage for the period.

Costs are what the same tokens would cost on the API; subscriptions don't
bill per token. Only token counts, model names, times and folder names are
read — never prompts, replies or tool output.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ..infra.coding_agent_usage import AgentCall, CodingAgentUsage, UsageReader, coding_agents_overview
from .live import LiveBus
from .summary import UsageRow, window_start

logger = logging.getLogger("prompture.companion")

#: Seconds an installed-agents answer is reused (it looks up executables on PATH).
OVERVIEW_TTL = 60.0
#: How far back calls are kept: a year of activity, plus a week of slack.
RETENTION = timedelta(days=372)
#: Companion choices that outlive a restart (today: whether Claude plan windows are fetched).
PREFS_FILE = Path.home() / ".prompture" / "companion-prefs.json"


def _load_prefs(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def call_event(call: AgentCall, name: str) -> dict[str, Any]:
    """A ``request.finished`` payload for an agent call, shaped like the ledger's."""
    return {
        "request_id": call.id,
        "ts": call.ts.isoformat(),
        "key_id": None,
        "key_name": name,
        "model": call.model,
        "served_by": None,
        "project": call.project,
        "status": "ok",
        "error": None,
        "prompt_tokens": call.input_tokens,
        "completion_tokens": call.output_tokens,
        "cost_usd": call.cost_usd,
        "latency_ms": 0,
        "attempts": 1,
        "fallback": False,
        "deprioritized": [],
        "tool": call.agent,
    }


class CodingToolSource:
    """Every local coding agent's usage, as the companion needs it.

    With no arguments every registered reader runs. ``claude_dir`` /
    ``codex_dir`` (tests, custom locations) restrict it to Claude Code and
    Codex at those folders; ``readers`` sets the list explicitly.
    """

    def __init__(
        self,
        claude_dir: str | Path | None = None,
        codex_dir: str | Path | None = None,
        *,
        plan_usage: bool | None = None,
        cache_dir: str | Path | None = None,
        readers: list[UsageReader] | None = None,
        prefs_file: str | Path | None = PREFS_FILE,
    ) -> None:
        from ..infra.coding_agent_readers import ClaudeCodeReader, CodexReader
        from ..infra.coding_agent_usage import USAGE_READERS

        self.prefs_file = Path(prefs_file) if prefs_file else None
        explicit_dirs = bool(claude_dir or codex_dir)
        if plan_usage is None and not explicit_dirs and "PROMPTURE_CLAUDE_PLAN_USAGE" not in os.environ:
            # The environment wins; otherwise the choice made in a companion app.
            plan_usage = bool(_load_prefs(self.prefs_file).get("claude_plan_usage")) if self.prefs_file else None
        if readers is None:
            claude = ClaudeCodeReader(claude_dir, plan_usage=plan_usage, cache_dir=cache_dir)
            codex = CodexReader(codex_dir)
            if explicit_dirs:
                readers = [claude, codex]
            else:
                others = [cls() for agent, cls in USAGE_READERS.items() if agent not in ("claude", "codex")]
                readers = [claude, codex, *others]
        self.usage = CodingAgentUsage(readers, retention=RETENTION)
        self.names = {r.agent: r.display_name for r in readers}
        self._overview: tuple[float, list[dict[str, Any]]] | None = None

    @property
    def plan_usage(self) -> bool:
        """Whether Claude Code's plan windows are fetched (opt-in; see ``ClaudeCodeReader``)."""
        return any(getattr(r, "plan_usage", False) for r in self.usage.readers)

    def set_claude_plan_usage(self, enabled: bool) -> None:
        """Turn Claude Code's plan windows on or off, and remember the choice."""
        for reader in self.usage.readers:
            if hasattr(reader, "plan_usage"):
                reader.plan_usage = enabled
                if not enabled:
                    reader._plan = None  # forget the last answer too
        if self.prefs_file:
            prefs = _load_prefs(self.prefs_file)
            prefs["claude_plan_usage"] = enabled
            try:
                self.prefs_file.parent.mkdir(parents=True, exist_ok=True)
                self.prefs_file.write_text(json.dumps(prefs), encoding="utf-8")
            except OSError:
                logger.debug("could not save companion prefs", exc_info=True)

    def refresh(self, *, force: bool = False) -> list[AgentCall]:
        """Read what the logs gained since the last scan; returns the new calls."""
        return self.usage.refresh(force=force)

    def calls(self, period: str = "day", now: datetime | None = None, offset_minutes: int = 0) -> list[AgentCall]:
        return self.usage.calls(window_start(period, now, offset_minutes))

    def rows(self, period: str = "day", now: datetime | None = None, offset_minutes: int = 0) -> list[UsageRow]:
        """Calls in the current ``period`` window, as usage rows."""
        return [
            UsageRow(model=c.model, cost_usd=c.cost_usd, tokens=c.tokens, project=c.project)
            for c in self.calls(period, now, offset_minutes)
        ]

    def rate_limits(self) -> dict[str, dict[str, Any]]:
        """Plan windows keyed like provider targets: ``openai/codex``, ``claude/claude-code``."""
        self.usage.refresh()
        return self.usage.plan_limits()

    def event(self, call: AgentCall) -> dict[str, Any]:
        return call_event(call, self.names.get(call.agent, call.agent))

    def tools(self, period: str = "day", now: datetime | None = None, offset_minutes: int = 0) -> dict[str, Any]:
        """``/v1/tools``: each agent's usage for the period, plus what is installed."""
        if self._overview is None or time.monotonic() - self._overview[0] > OVERVIEW_TTL:
            try:
                self._overview = (time.monotonic(), coding_agents_overview())
            except Exception:
                logger.debug("coding agent overview failed", exc_info=True)
                self._overview = (time.monotonic(), [])
        start = window_start(period, now, offset_minutes)
        return {
            "period": period,
            "start": start.isoformat(),
            "agents": self.usage.summary(start),
            "installed": [a for a in self._overview[1] if a["installed"]],
            "claude_plan_usage": self.plan_usage,
        }

    def events_since(self, since: datetime, limit: int = 500) -> list[dict[str, Any]]:
        """``request.finished`` payloads for calls at or after ``since``, newest last."""
        return [self.event(c) for c in self.usage.calls(since)[-limit:]]

    def tail(self, bus: LiveBus, stop: threading.Event, interval: float = 3.0) -> None:
        """Publish a ``request.finished`` event for every call logged from now on."""
        self.refresh(force=True)
        while not stop.wait(interval):
            try:
                for call in sorted(self.refresh(force=True), key=lambda c: c.ts):
                    if datetime.now(timezone.utc) - call.ts < timedelta(minutes=30):
                        bus.publish("request.finished", self.event(call))
            except Exception:  # a bad log line must never stop the companion
                logger.debug("coding tool scan failed", exc_info=True)
