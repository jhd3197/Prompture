"""What each call through the router cost, where it went, and what it saved.

The router (:mod:`.router`) writes one :class:`RoutedCall` per model request
Claude Code or Codex sends through it, passed through or routed, to
``~/.prompture/router/calls.jsonl``. Each record answers:

- **where it went**: the model the CLI asked for, the model that answered,
  and the rule that decided it, with the reason in words;
- **what it used**: input (with cache reads and writes), output, latency,
  time to first token, errors, and every attempt a fallback made;
- **what it cost**: the API cost of the destination (``cost_usd``), and an
  estimate of what the original path would have cost (``baseline_usd``).

Subscription traffic is kept apart. A call the CLI makes on its plan login
costs nothing extra on the original path (``baseline_usd`` is 0; the API
price is kept as ``plan_equivalent_usd`` for context), so routing it to a paid
API shows up as *new spend*, never as savings.

Usage of passed-through calls is read from the vendor's reply as it streams
(:class:`UsageSniffer`); routed calls take it from the driver. Prices come
from Prompture's rate tables (:func:`~prompture.infra.coding_agent_usage.estimate_cost`),
so every dollar figure here is an estimate unless a driver reported it.

Only metadata is stored: models, token counts, timings, rule names, folder
names. Never prompts, replies or tool output.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
import os
import threading
from collections import deque
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("prompture.companion.calls")

CALLS_FILE = Path.home() / ".prompture" / "router" / "calls.jsonl"
#: Calls older than this are dropped when the log is compacted.
RETENTION = timedelta(days=35)
#: Records kept in memory for queries (a busy month fits).
MAX_IN_MEMORY = 50_000

#: Where a request's cost is billed: the CLI's plan login, a pay-per-token API
#: key, a model on this machine, or unknown (a custom gateway).
BILLING = ("subscription", "api", "local", "unknown")
#: Providers that run on this machine: no bill.
LOCAL_PROVIDERS = {"ollama", "lmstudio", "llamacpp", "local_http", "huggingface", "airllm", "vllm", "mlx"}


# ------------------------------------------------------------------ usage


@dataclasses.dataclass
class Usage:
    """Token counts of one reply. ``input_tokens`` is the whole prompt, cache reads and writes included."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    model: str | None = None  # what the vendor says answered

    @property
    def fresh_input(self) -> int:
        return max(0, self.input_tokens - self.cache_read_tokens - self.cache_write_tokens)

    def merge_anthropic(self, usage: dict[str, Any]) -> None:
        """Anthropic ``usage``: ``input_tokens`` excludes the cached part, so add it back."""
        fresh = _int(usage.get("input_tokens"))
        read = _int(usage.get("cache_read_input_tokens"))
        write = _int(usage.get("cache_creation_input_tokens"))
        if fresh or read or write:
            self.cache_read_tokens = max(self.cache_read_tokens, read)
            self.cache_write_tokens = max(self.cache_write_tokens, write)
            self.input_tokens = max(self.input_tokens, fresh + read + write)
        self.output_tokens = max(self.output_tokens, _int(usage.get("output_tokens")))

    def merge_gemini(self, usage: dict[str, Any]) -> None:
        """Gemini ``usageMetadata``: ``promptTokenCount`` includes the cached part; thoughts are output."""
        self.input_tokens = max(self.input_tokens, _int(usage.get("promptTokenCount")))
        self.cache_read_tokens = max(self.cache_read_tokens, _int(usage.get("cachedContentTokenCount")))
        out = _int(usage.get("candidatesTokenCount")) + _int(usage.get("thoughtsTokenCount"))
        self.output_tokens = max(self.output_tokens, out)

    def merge_openai(self, usage: dict[str, Any]) -> None:
        """Responses / chat ``usage``: ``input_tokens`` already includes ``cached_tokens``."""
        details = usage.get("input_tokens_details") or usage.get("prompt_tokens_details") or {}
        self.input_tokens = max(self.input_tokens, _int(usage.get("input_tokens") or usage.get("prompt_tokens")))
        self.output_tokens = max(self.output_tokens, _int(usage.get("output_tokens") or usage.get("completion_tokens")))
        if isinstance(details, dict):
            self.cache_read_tokens = max(self.cache_read_tokens, _int(details.get("cached_tokens")))

    @classmethod
    def from_meta(cls, meta: dict[str, Any] | None, model: str | None = None) -> Usage:
        """A Prompture driver's ``meta`` (``prompt_tokens`` is the whole prompt)."""
        meta = meta or {}
        return cls(
            input_tokens=_int(meta.get("prompt_tokens")),
            output_tokens=_int(meta.get("completion_tokens")),
            cache_read_tokens=_int(meta.get("cached_prompt_tokens")),
            cache_write_tokens=_int(meta.get("cache_creation_tokens")),
            model=str(meta.get("model_name") or model or "") or None,
        )


def _int(value: Any) -> int:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


class UsageSniffer:
    """Reads token usage out of a vendor reply as it passes through, without changing it.

    Feed it the raw bytes (SSE or JSON); only events that carry ``usage`` or a
    model name are parsed, so the text deltas cost a substring check each.
    """

    def __init__(self, dialect: str) -> None:
        self.dialect = dialect
        self.usage = Usage()
        self._buffer = b""
        self._json = bytearray()
        self.streaming = False

    def feed(self, chunk: bytes, *, stream: bool) -> None:
        chunk = chunk.replace(b"\r\n", b"\n")
        if not stream:
            if len(self._json) < 8 << 20:  # a non-streamed reply; parsed in finish()
                self._json.extend(chunk)
            return
        self.streaming = True
        self._buffer += chunk
        while b"\n\n" in self._buffer:
            event, self._buffer = self._buffer.split(b"\n\n", 1)
            self._event(event)

    def finish(self) -> Usage:
        if (
            self.streaming
            and self._buffer.strip()
            and not self._buffer.lstrip().startswith((b"data:", b"event:", b":"))
        ):
            # A reply that streamed without SSE framing: one JSON document.
            self._json.extend(self._buffer)
            self._buffer = b""
        if self._json:
            with contextlib.suppress(ValueError):
                self._take(json.loads(bytes(self._json)))
            self._json.clear()
        if self._buffer.strip():
            self._event(self._buffer)
            self._buffer = b""
        return self.usage

    def _event(self, raw: bytes) -> None:
        if b"usage" not in raw and b"message_start" not in raw and b'"model"' not in raw and b"modelVersion" not in raw:
            return
        for line in raw.split(b"\n"):
            if line.startswith(b"data:"):
                try:
                    data = json.loads(line[5:].strip() or b"null")
                except ValueError:
                    continue
                if isinstance(data, dict):
                    self._take(data)

    def _take(self, data: dict[str, Any]) -> None:
        kind = data.get("type")
        if self.dialect == "gemini":
            _nested = data.get("response")
            inner: dict[str, Any] = _nested if isinstance(_nested, dict) else data
            if isinstance(inner.get("modelVersion"), str):
                self.usage.model = inner["modelVersion"]
            if isinstance(inner.get("usageMetadata"), dict):
                self.usage.merge_gemini(inner["usageMetadata"])
            return
        if self.dialect == "anthropic":
            message = data.get("message") if kind == "message_start" else data
            if isinstance(message, dict):
                if isinstance(message.get("model"), str):
                    self.usage.model = message["model"]
                if isinstance(message.get("usage"), dict):
                    self.usage.merge_anthropic(message["usage"])
            if kind == "message_delta" and isinstance(data.get("usage"), dict):
                self.usage.merge_anthropic(data["usage"])
            return
        nested = data.get("response")
        response: dict[str, Any] = nested if isinstance(nested, dict) else data
        if isinstance(response.get("model"), str):
            self.usage.model = response["model"]
        if isinstance(response.get("usage"), dict):
            self.usage.merge_openai(response["usage"])


# ------------------------------------------------------------------ pricing


def split_model(model: str, default_provider: str) -> tuple[str, str]:
    """``"claude/claude-sonnet-5"`` → ``("claude", "claude-sonnet-5")``; bare ids take ``default_provider``."""
    if "/" in model:
        provider, rest = model.split("/", 1)
        return provider, rest
    return default_provider, model


def price(model: str, usage: Usage, default_provider: str) -> tuple[float, str]:
    """API cost of *usage* on *model*: ``(usd, "estimated" | "unknown" | "local")``."""
    provider, model_id = split_model(model, default_provider)
    if provider in LOCAL_PROVIDERS:
        return 0.0, "local"
    from ..infra.coding_agent_usage import estimate_cost

    return estimate_cost(
        provider,
        model_id,
        fresh_in=usage.fresh_input,
        cache_read=usage.cache_read_tokens,
        cache_write=usage.cache_write_tokens,
        output=usage.output_tokens,
    )


def billing_of(model: str) -> str:
    """How a Prompture model is billed: ``local`` for this machine's servers, else ``api``."""
    return "local" if model.split("/", 1)[0] in LOCAL_PROVIDERS else "api"


# ------------------------------------------------------------------ records


@dataclasses.dataclass
class RoutedCall:
    """One model request a coding CLI sent through the router. See the module docstring."""

    id: str
    ts: str  # ISO start time, UTC
    tool: str  # "claude-code" | "codex"
    kind: str  # main, tool_result, title, probe, compaction
    endpoint: str
    requested: str  # provider/model the CLI asked for
    served: str  # provider/model that answered (or was last tried)
    route: str  # "passthrough" | "native" (same vendor, other model) | "routed"
    rule: dict[str, Any]  # {"source", "match", "target", "reason"}
    billing: str  # of the destination
    original_billing: str  # of the path the CLI would have taken
    session: str | None = None
    project: str | None = None
    status: str = "ok"  # "ok" | "error"
    error: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    latency_ms: int = 0
    ttft_ms: int | None = None
    attempts: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    cost_usd: float = 0.0  # new API spend, every attempt included
    cost_source: str = "unknown"
    baseline_usd: float = 0.0  # the original path, estimated; 0 when it was a plan
    baseline_source: str = "unknown"
    plan_equivalent_usd: float = 0.0  # API price of plan traffic, for context
    savings_usd: float = 0.0  # baseline - cost; negative when routing cost more
    switched: bool = False  # the session's destination changed with this call (prompt cache lost)
    tool_failed: bool | None = None  # the tool result this request reports failed (None: not a tool result)
    escalated: bool = False  # this request moved its task to a stronger model

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RoutedCall:
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names})

    @property
    def when(self) -> datetime:
        try:
            return datetime.fromisoformat(self.ts)
        except ValueError:
            return datetime.fromtimestamp(0, tz=timezone.utc)

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def new_spend_usd(self) -> float:
        """API spend on a call whose original path was a plan: a bill routing created."""
        return self.cost_usd if self.original_billing == "subscription" else 0.0

    def event(self) -> dict[str, Any]:
        """What the live stream says about the call when it ends."""
        return {
            "call_id": self.id,
            "requested": self.requested,
            "served": self.served,
            "route": self.route,
            "rule": self.rule.get("reason"),
            "billing": self.billing,
            "cost_usd": round(self.cost_usd, 6),
            "savings_usd": round(self.savings_usd, 6),
            "attempts": len(self.attempts),
        }


def settle(call: RoutedCall, usage: Usage, *, vendor: str, cache_ratio: float | None = None) -> RoutedCall:
    """Fill in *call*'s tokens and dollars from the reply's *usage*.

    ``vendor`` is the Prompture provider of the CLI's own vendor ("claude" or
    "openai"). ``cache_ratio`` is the share of the prompt the original path
    read from cache earlier in the session; when a call leaves that path it
    is assumed to have kept that share, so savings aren't inflated by a cache
    the original model would have had.
    """
    call.input_tokens = usage.input_tokens
    call.output_tokens = usage.output_tokens
    call.cache_read_tokens = usage.cache_read_tokens
    call.cache_write_tokens = usage.cache_write_tokens
    if call.route == "routed":
        cost_known = call.cost_source not in ("unknown", "")
        if not cost_known:
            call.cost_usd, call.cost_source = price(call.served, usage, vendor)
    else:
        spent, source = price(call.served, usage, vendor)
        if call.billing == "subscription":
            call.plan_equivalent_usd, call.cost_usd, call.cost_source = spent, 0.0, "plan"
        else:
            call.cost_usd, call.cost_source = spent, source
    call.cost_usd += sum(float(a.get("cost_usd") or 0) for a in call.attempts[:-1])

    if call.original_billing == "subscription":
        call.baseline_usd, call.baseline_source = 0.0, "plan"
        if call.route != "passthrough":
            call.plan_equivalent_usd, _ = price(call.requested, _as_original(usage, cache_ratio), vendor)
    elif call.route == "passthrough":
        call.baseline_usd, call.baseline_source = call.cost_usd, call.cost_source
    else:
        call.baseline_usd, call.baseline_source = price(call.requested, _as_original(usage, cache_ratio), vendor)
    call.savings_usd = round(call.baseline_usd - call.cost_usd, 6) if call.baseline_source != "unknown" else 0.0
    call.cost_usd = round(call.cost_usd, 6)
    call.baseline_usd = round(call.baseline_usd, 6)
    call.plan_equivalent_usd = round(call.plan_equivalent_usd, 6)
    return call


def _as_original(usage: Usage, cache_ratio: float | None) -> Usage:
    """*usage* as the original path would have seen it: its cache, not the destination's."""
    if cache_ratio is None:
        return dataclasses.replace(usage, cache_write_tokens=0)
    read = int(usage.input_tokens * max(0.0, min(1.0, cache_ratio)))
    return dataclasses.replace(usage, cache_read_tokens=read, cache_write_tokens=0)


# ------------------------------------------------------------------ the log


class CallLog:
    """Routed-call records: appended to a JSONL file, queried from memory. Thread-safe."""

    def __init__(self, path: str | Path | None = CALLS_FILE, *, retention: timedelta = RETENTION) -> None:
        self.path = Path(path) if path else None
        self.retention = retention
        self._calls: deque[RoutedCall] = deque(maxlen=MAX_IN_MEMORY)
        self._lock = threading.Lock()
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if self.path is None or not self.path.exists():
            return
        cutoff = datetime.now(timezone.utc) - self.retention
        kept: list[RoutedCall] = []
        try:
            with self.path.open(encoding="utf-8") as fh:
                for line in fh:
                    try:
                        call = RoutedCall.from_dict(json.loads(line))
                    except (ValueError, TypeError):
                        continue
                    if call.when >= cutoff:
                        kept.append(call)
        except OSError:
            logger.debug("could not read %s", self.path, exc_info=True)
            return
        self._calls.extend(kept[-MAX_IN_MEMORY:])
        self._compact(kept)

    def _compact(self, kept: list[RoutedCall]) -> None:
        """Rewrite the file without expired records (on first load only)."""
        if self.path is None:
            return
        try:
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text("".join(json.dumps(c.to_dict()) + "\n" for c in kept), encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError:
            logger.debug("could not compact %s", self.path, exc_info=True)

    def add(self, call: RoutedCall) -> None:
        with self._lock:
            self._load()
            self._calls.append(call)
            if self.path is None:
                return
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(call.to_dict()) + "\n")
            except OSError:
                logger.debug("could not write %s", self.path, exc_info=True)

    def calls(self, since: datetime | None = None) -> list[RoutedCall]:
        with self._lock:
            self._load()
            calls = list(self._calls)
        return [c for c in calls if since is None or c.when >= since]

    def get(self, call_id: str) -> RoutedCall | None:
        return next((c for c in reversed(self.calls()) if c.id == call_id), None)

    def session_calls(self, tool: str, session: str) -> list[RoutedCall]:
        return [c for c in self.calls() if c.tool == tool and c.session == session]


# ------------------------------------------------------------------ summaries


def _bucket() -> dict[str, Any]:
    return {
        "calls": 0,
        "routed": 0,
        "errors": 0,
        "fallbacks": 0,
        "tokens": 0,
        "cost_usd": 0.0,
        "baseline_usd": 0.0,
        "savings_usd": 0.0,
        "new_spend_usd": 0.0,
        "plan_equivalent_usd": 0.0,
        "cache_read_tokens": 0,
        "input_tokens": 0,
        "tool_results": 0,
        "tool_failures": 0,
        "escalations": 0,
    }


def _add(b: dict[str, Any], c: RoutedCall) -> None:
    b["calls"] += 1
    b["routed"] += c.route != "passthrough"
    b["errors"] += c.status != "ok"
    b["fallbacks"] += max(0, len(c.attempts) - 1)
    b["tokens"] += c.tokens
    b["cost_usd"] += c.cost_usd
    b["baseline_usd"] += c.baseline_usd
    b["savings_usd"] += c.savings_usd
    b["new_spend_usd"] += c.new_spend_usd
    b["plan_equivalent_usd"] += c.plan_equivalent_usd
    b["cache_read_tokens"] += c.cache_read_tokens
    b["input_tokens"] += c.input_tokens
    b["tool_results"] += c.tool_failed is not None
    b["tool_failures"] += bool(c.tool_failed)
    b["escalations"] += c.escalated


def _round(b: dict[str, Any]) -> dict[str, Any]:
    for k in ("cost_usd", "baseline_usd", "savings_usd", "new_spend_usd", "plan_equivalent_usd"):
        b[k] = round(b[k], 6)
    b["cache_hit"] = round(b["cache_read_tokens"] / b["input_tokens"], 4) if b["input_tokens"] else None
    # How often the tools the model chose worked: the success measure presets are compared by.
    b["tool_success"] = round(1 - b["tool_failures"] / b["tool_results"], 4) if b["tool_results"] else None
    return b


def rule_label(rule: dict[str, Any]) -> str:
    """A rule's name in summaries: ``"kind: background"``, ``"model: claude-haiku-*"``, ``"preset: economy"``."""
    source = str(rule.get("source") or "none")
    match = rule.get("match")
    return f"{source}: {match}" if match else source


def summarize_calls(calls: Iterable[RoutedCall], start: datetime) -> dict[str, Any]:
    """``/v1/router/savings``: totals, then per tool, project, rule, preset and destination."""
    total = _bucket()
    groups: dict[str, dict[str, dict[str, Any]]] = {"tool": {}, "project": {}, "rule": {}, "preset": {}, "served": {}}
    for c in calls:
        if c.when < start:
            continue
        _add(total, c)
        keys = {
            "tool": c.tool,
            "project": c.project or "",
            "rule": rule_label(c.rule),
            "preset": str(c.rule.get("preset") or "none"),
            "served": c.served,
        }
        for group, key in keys.items():
            _add(groups[group].setdefault(key, _bucket()), c)

    def listing(group: str, field: str) -> list[dict[str, Any]]:
        rows = [{field: (k or None), **_round(v)} for k, v in groups[group].items()]
        return sorted(rows, key=lambda r: (-r["cost_usd"], -r["calls"]))

    return {
        "start": start.isoformat(),
        "total": _round(total),
        "by_tool": listing("tool", "tool"),
        "by_project": listing("project", "project"),
        "by_rule": listing("rule", "rule"),
        "by_preset": listing("preset", "preset"),
        "by_served": listing("served", "served"),
    }
