"""Usage that local coding agents record on disk.

Coding agents (Claude Code, Codex, Kimi Code, Gemini CLI, Qwen Code, OpenCode,
Cline, …) keep their own session logs. Each :class:`UsageReader` knows where
one agent writes and turns what it finds into :class:`AgentCall` rows — one
per model call, with the model, token counts and, when the agent records it,
the cost. :class:`CodingAgentUsage` runs every available reader incrementally
and keeps a deduplicated, rolling window of calls.

Agent ids match :data:`~prompture.infra.coding_agent_specs.CODING_AGENT_SPECS`
where the agent can also be *run* by Prompture (``claude``, ``codex``,
``gemini``, ``qwen``, ``opencode``), so one id covers detecting an agent,
reading its usage and driving it. :func:`coding_agents_overview` puts both
sides together.

Readers only read local files, never another app's credentials, and only
token counts, model names, times and folder names — never prompts, replies or
tool output. (Claude Code's plan windows are the one exception, and are off
unless explicitly enabled; see
:class:`~prompture.infra.coding_agent_readers.ClaudeCodeReader`.)

Costs: when an agent doesn't record one, it is estimated from Prompture's
model rates (``cost_source="estimated"``) — for subscription tools that is the
pay-as-you-go API equivalent, not what the plan bills. Unknown models get
``cost_source="unknown"`` and a cost of 0.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import sys
import threading
import time
from collections.abc import Iterable, Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePath
from typing import Any, ClassVar, Literal

logger = logging.getLogger("prompture.coding_agents")

CostSource = Literal["reported", "estimated", "unknown"]

#: How long :class:`CodingAgentUsage` keeps calls by default (covers a month window).
DEFAULT_RETENTION = timedelta(days=35)
#: Minimum seconds between two scans of the agents' folders.
SCAN_INTERVAL = 2.0


@dataclasses.dataclass(frozen=True)
class AgentCall:
    """One model call a coding agent logged."""

    id: str
    agent: str
    ts: datetime
    model: str  # "provider/model"
    input_tokens: int  # the whole prompt, cache reads and writes included
    output_tokens: int  # reasoning included
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0.0
    cost_source: CostSource = "unknown"
    project: str | None = None
    session: str | None = None

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class UsageReader:
    """Reads one agent's local logs. Subclasses set ``agent`` and ``display_name``.

    :meth:`read` is incremental: each call yields only what the logs gained
    since the previous call (a reader may re-yield a call whose numbers grew;
    :class:`CodingAgentUsage` keeps the larger). ``since`` bounds how far back
    a first read goes.
    """

    agent: ClassVar[str]
    display_name: ClassVar[str]
    #: False for agents that are detected but keep usage elsewhere (e.g. their servers).
    has_local_usage: ClassVar[bool] = True

    def paths(self) -> list[Path]:
        """Folders or files where this agent keeps its data."""
        return []

    def available(self) -> bool:
        return any(p.exists() for p in self.paths())

    def read(self, since: datetime) -> Iterator[AgentCall]:
        return iter(())

    def plan_limits(self) -> dict[str, dict[str, Any]]:
        """Subscription plan windows, keyed like rate-limit targets. Most agents have none."""
        return {}


#: Reader classes by agent id. Add one with :func:`register_usage_reader`.
USAGE_READERS: dict[str, type[UsageReader]] = {}


def register_usage_reader(cls: type[UsageReader]) -> type[UsageReader]:
    """Register a :class:`UsageReader` subclass (usable as a decorator)."""
    USAGE_READERS[cls.agent] = cls
    return cls


# ------------------------------------------------------------------ helpers for readers


def home() -> Path:
    return Path.home()


def env_path(var: str, default: Path) -> Path:
    value = os.environ.get(var)
    return Path(value) if value else default


def editor_storage_roots() -> list[Path]:
    """Per-user data folders of VS Code and its forks (Cursor, Windsurf, Antigravity, …)."""
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA") or home() / "AppData" / "Roaming")
    elif sys.platform == "darwin":
        base = home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or home() / ".config")
    names = ("Code", "Code - Insiders", "VSCodium", "Cursor", "Windsurf", "Antigravity", "Trae", "Kiro", "Void")
    return [base / n for n in names]


def parse_ts(value: Any) -> datetime | None:
    """ISO-8601 strings, or epoch seconds / milliseconds, as an aware UTC datetime."""
    if isinstance(value, (int, float)) and value > 0:
        return datetime.fromtimestamp(value / 1000 if value > 1e11 else value, tz=timezone.utc)
    if not isinstance(value, str) or not value:
        return None
    try:
        ts = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def as_int(value: Any) -> int:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


def project_name(cwd: Any) -> str | None:
    """The last folder of a working directory, as the project name."""
    if not isinstance(cwd, str) or not cwd:
        return None
    return PurePath(cwd.replace("\\", "/").rstrip("/")).name or None


def estimate_cost(
    provider: str, model: str, *, fresh_in: int, cache_read: int = 0, cache_write: int = 0, output: int = 0
) -> tuple[float, CostSource]:
    """What these tokens cost at the model's API rates, from Prompture's rate tables."""
    try:
        from .model_rates import get_model_rates

        rates = get_model_rates(provider, model)
    except Exception:
        rates = None
    if not rates or not (rates.get("input") or rates.get("output")):
        return 0.0, "unknown"
    rate_in = rates.get("input") or 0.0
    cost = (
        fresh_in * rate_in
        + cache_read * (rates.get("cache_read") or rate_in)
        + cache_write * (rates.get("cache_write") or rate_in)
        + output * (rates.get("output") or 0.0)
    )
    return round(cost / 1_000_000, 6), "estimated"


class JsonlCursor:
    """Byte offsets per file, so an append-only JSONL log is read once."""

    def __init__(self) -> None:
        self.offsets: dict[Path, int] = {}

    def new_lines(self, path: Path) -> Iterator[tuple[int, str]]:
        """``(offset, line)`` for each complete line added since the last call."""
        start = self.offsets.get(path, 0)
        try:
            size = path.stat().st_size
            if size < start:  # rewritten or truncated
                start = 0
            if size == start:
                return
            with path.open("rb") as fh:
                fh.seek(start)
                chunk = fh.read(size - start)
        except OSError:
            return
        end = chunk.rfind(b"\n")
        if end < 0:  # a line still being written
            return
        self.offsets[path] = start + end + 1
        pos = start
        for raw in chunk[: end + 1].split(b"\n")[:-1]:
            yield pos, raw.decode("utf-8", errors="replace")
            pos += len(raw) + 1


class ChangedFiles:
    """Remembers (size, mtime) per file, for logs that are rewritten whole."""

    def __init__(self) -> None:
        self.seen: dict[Path, tuple[int, float]] = {}

    def changed(self, path: Path) -> bool:
        try:
            st = path.stat()
        except OSError:
            return False
        key = (st.st_size, st.st_mtime)
        if self.seen.get(path) == key:
            return False
        self.seen[path] = key
        return True


def recent_files(root: Path, pattern: str, since: datetime) -> Iterator[Path]:
    """Files under ``root`` matching ``pattern`` that changed since ``since``."""
    if not root.is_dir():
        return
    cutoff = since.timestamp()
    for path in root.glob(pattern):
        try:
            if path.is_file() and path.stat().st_mtime >= cutoff:
                yield path
        except OSError:
            continue


# ------------------------------------------------------------------ aggregation


class CodingAgentUsage:
    """Every available agent's calls, deduplicated, over a rolling window.

    Thread-safe. :meth:`refresh` scans at most every ``scan_interval`` seconds
    unless forced, reading only what the logs gained since the last scan.
    """

    def __init__(
        self,
        readers: Iterable[UsageReader] | None = None,
        *,
        retention: timedelta = DEFAULT_RETENTION,
        scan_interval: float = SCAN_INTERVAL,
    ) -> None:
        if readers is None:
            from . import coding_agent_readers  # noqa: F401  (registers the built-in readers)

            readers = [cls() for cls in USAGE_READERS.values()]
        self.readers = list(readers)
        self.retention = retention
        self.scan_interval = scan_interval
        self._lock = threading.Lock()
        self._calls: dict[str, AgentCall] = {}
        self._scanned_at = 0.0

    def refresh(self, *, force: bool = False) -> list[AgentCall]:
        """Scan the logs; returns calls seen for the first time."""
        with self._lock:
            if not force and time.monotonic() - self._scanned_at < self.scan_interval:
                return []
            self._scanned_at = time.monotonic()
            since = datetime.now(timezone.utc) - self.retention
            new: list[AgentCall] = []
            for reader in self.readers:
                if not reader.has_local_usage:
                    continue
                try:
                    for call in reader.read(since):
                        if call.ts < since:
                            continue
                        old = self._calls.get(call.id)
                        if old is None:
                            new.append(call)
                        elif call.tokens <= old.tokens:
                            continue  # a copy (resumed session) or a partial count
                        self._calls[call.id] = call
                except Exception:  # one agent's odd log must never hide the others
                    logger.debug("reading %s usage failed", reader.agent, exc_info=True)
            self._calls = {k: c for k, c in self._calls.items() if c.ts >= since}
            return new

    def calls(self, since: datetime | None = None) -> list[AgentCall]:
        """Calls at or after ``since`` (all retained calls by default), oldest first."""
        self.refresh()
        with self._lock:
            calls = list(self._calls.values())
        if since is not None:
            calls = [c for c in calls if c.ts >= since]
        return sorted(calls, key=lambda c: c.ts)

    def summary(self, since: datetime | None = None) -> list[dict[str, Any]]:
        """Per-agent totals (with per-model and per-project breakdowns) for calls since ``since``."""
        by_agent: dict[str, dict[str, Any]] = {}
        names = {r.agent: r.display_name for r in self.readers}
        for c in self.calls(since):
            a = by_agent.setdefault(
                c.agent,
                {
                    "agent": c.agent,
                    "name": names.get(c.agent, c.agent),
                    "requests": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cache_read_tokens": 0,
                    "cache_write_tokens": 0,
                    "reasoning_tokens": 0,
                    "tokens": 0,
                    "cost_usd": 0.0,
                    "cost_sources": set(),
                    "last_used": None,
                    "models": {},
                    "projects": {},
                },
            )
            a["requests"] += 1
            for key in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "reasoning_tokens"):
                a[key] += getattr(c, key)
            a["tokens"] += c.tokens
            a["cost_usd"] += c.cost_usd
            a["cost_sources"].add(c.cost_source)
            a["last_used"] = c.ts.isoformat()
            for group, key in (("models", c.model), ("projects", c.project or "")):
                g = a[group].setdefault(key, {"requests": 0, "tokens": 0, "cost_usd": 0.0})
                g["requests"] += 1
                g["tokens"] += c.tokens
                g["cost_usd"] += c.cost_usd
        out = []
        for a in sorted(by_agent.values(), key=lambda x: -x["tokens"]):
            sources = a.pop("cost_sources")
            a["cost_usd"] = round(a["cost_usd"], 6)
            a["cost_source"] = next(
                (s for s in ("reported", "estimated") if sources == {s}), "mixed" if len(sources) > 1 else "unknown"
            )
            for group, label in (("models", "model"), ("projects", "project")):
                a[group] = [
                    {label: k or None, **{**v, "cost_usd": round(v["cost_usd"], 6)}}
                    for k, v in sorted(a[group].items(), key=lambda kv: -kv[1]["tokens"])
                ]
            out.append(a)
        return out

    def plan_limits(self) -> dict[str, dict[str, Any]]:
        """Plan windows from every agent that has them."""
        out: dict[str, dict[str, Any]] = {}
        for reader in self.readers:
            try:
                out.update(reader.plan_limits())
            except Exception:
                logger.debug("%s plan limits failed", reader.agent, exc_info=True)
        return out


def coding_agents_overview(*, verify: bool = False) -> list[dict[str, Any]]:
    """Every known coding agent: whether it is installed, runnable by Prompture, and has local usage.

    ``runnable`` means Prompture can drive it (:func:`~prompture.infra.coding_agents.run_coding_agent`);
    ``usage`` means its calls can be read from this machine.
    """
    from . import coding_agent_readers  # noqa: F401
    from .coding_agent_specs import CODING_AGENT_SPECS
    from .discovery import resolve_coding_agent_executable

    ids = list(dict.fromkeys([*USAGE_READERS, *CODING_AGENT_SPECS]))
    out = []
    for agent_id in ids:
        spec = CODING_AGENT_SPECS.get(agent_id)
        reader_cls = USAGE_READERS.get(agent_id)
        reader = reader_cls() if reader_cls else None
        runnable = False
        if spec is not None:
            try:
                exe, healthy, _ = resolve_coding_agent_executable(agent_id, spec.default_binary, verify=verify)
                runnable = exe is not None and healthy is not False
            except Exception:
                runnable = False
        has_data = bool(reader and reader.available())
        out.append(
            {
                "id": agent_id,
                "name": (reader_cls.display_name if reader_cls else spec.display_name if spec else agent_id),
                "installed": runnable or has_data,
                "runnable": runnable,
                "usage": bool(has_data and reader_cls and reader_cls.has_local_usage),
            }
        )
    return out
