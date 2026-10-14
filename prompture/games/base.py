"""GameHarness: let any model play a turn-based game without ever making an illegal move.

The model never writes a move freely. Each turn it is shown a menu of
candidate moves and must answer with one of them; the answer's JSON schema
lists the candidates as an ``enum``, so drivers with native structured output
(Ollama, llama.cpp, OpenAI, Claude ...) cannot produce anything else. Every
reply is still checked against the game's rules. A rejected reply is sent back
with the reason (a failed model call counts as a rejected reply); after
``max_retries`` the harness plays a fallback move and
marks the decision ``rescued``, so a model's rescue count measures how much it
leans on the harness.

Two independent dials shape the help a model gets:

``level`` — which moves it is shown (the safety rail):

    0  every legal move, untagged
    1  every legal move, tagged with facts (captures, checks, flips ...)
    2  legal moves minus blunders (needs an engine)
    3  the engine's top moves, unranked and without scores (top 5 by default)
    4  the engine's top moves, ranked, with evaluations (top 3 by default)
    5  only the engine's best move

``concepts`` — which ideas it knows: each game names its own (``"fork"`` in
chess, ``"double-threat"`` in Connect Four, ``"corner"`` in Othello). Each
enabled concept tags the candidate moves that show it, so a model "knows"
forks when forking moves say so.

At level 3 and above most of the strength is the engine's. To model a player
of a given strength, keep ``level`` low (0-2) and vary ``concepts``.

A game plugs in by implementing :class:`Game`; an engine by implementing
:class:`Engine` (``analyse(state, count)`` returning a :class:`Score` per move).
Optional capabilities (:class:`GameHistory`, ``dump_state``/``load_state``)
unlock replay identity, memory and checkpoints; a game without them still plays.

Turns can be bounded: ``deadline`` (seconds, covering every request, retry
and backoff), ``cancel`` (a :class:`~prompture.games.control.CancelToken`),
``retry`` (a separate transient-failure allowance with backoff) and ``budget``
(calls, tokens, cost, time). ``on_event`` receives sequenced progress events.
See :mod:`prompture.games.control` for what cancellation can and cannot stop.

``memory`` (``"none"``, ``"previous"`` or a window size) carries the side's
own accepted reasons and plans forward on the same game line (see
:mod:`prompture.games.memory`). ``context_budget`` caps the prompt's estimated
tokens: the position, legal menu and instructions always stay, and optional
hints and memory are added urgent-first until the budget is used.

Experimental options are off by default: ``strategy`` (see
:mod:`prompture.games.strategies`) and ``representation="text+image"`` (a
picture from the game's optional ``render_image`` alongside the unchanged
text; when no picture can be sent the turn records that it fell back to text).

Every turn produces a :class:`~prompture.games.records.TurnRecord` on
``decision.record``: each model call with its prompt, public response,
validation result, timing, finish reason, classified failure and usage, plus
honest cost completeness. ``failure_policy`` decides what an exhausted turn
does: ``"rescue"`` (the default) plays a marked fallback, ``"forfeit"`` and
``"stop"`` raise :class:`~prompture.games.records.TurnFailed`. Cancellation,
budget exhaustion and fatal configuration errors never become a played move.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import itertools
import logging
import random
import threading
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..exceptions import ExtractionError
from .control import CancelToken, EventHandler, GameBudget, Reservation, RetryPolicy, TurnEvent
from .records import (
    Accounting,
    FailureInfo,
    FailureKind,
    Fingerprint,
    TurnCancelled,
    TurnError,
    TurnFailed,
    TurnRecord,
    classify_failure,
    fingerprint,
    redact_text,
    safe_options,
    source_meta,
    text_digest,
    truncate,
    usage_status,
)
from .trace import TraceSink, TraceWriter

logger = logging.getLogger("prompture.games")

LEVELS = {
    0: "every legal move",
    1: "every legal move, tagged with facts",
    2: "legal moves minus blunders",
    3: "the engine's top moves, unranked",
    4: "the engine's top moves, ranked with evaluations",
    5: "only the engine's best move",
}
DEFAULT_TOP_N = {3: 5, 4: 3, 5: 1}
BLUNDER_DROP = 0.3  # drop in winning chances (-1..1) that Lichess calls a blunder
_USAGE_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens", "cost")
PROMPT_VERSION = "1"  # bump when the harness's prompt wording or schema changes
FAILURE_POLICIES = ("rescue", "forfeit", "stop")


SEVERITIES = ("urgent", "warning", "info")


@dataclass(frozen=True)
class Motif:
    """One tag on a move: its kind (a fact or concept name) and a sentence.

    ``severity`` (``urgent``, ``warning`` or ``info``) orders what a compact
    prompt keeps first. ``exact`` is False for heuristics (a local exchange
    count, a bounded search that ran out of work); ``evidence`` holds a short
    witness such as the exchange line, kept for traces.
    """

    kind: str
    text: str
    squares: tuple[str, ...] = ()
    severity: str = "info"
    exact: bool = True
    evidence: str = ""

    def __str__(self) -> str:
        return self.text


@dataclass(frozen=True)
class Score:
    """An engine's verdict on a move: winning chances for the mover (-1..1) and a label."""

    value: float
    label: str


class Engine(Protocol):
    """Scores moves. ``count`` is how many of the best moves are needed; more is fine."""

    def analyse(self, state: Any, count: int) -> dict[Any, Score]: ...


class Game(Protocol):
    """The rules, notation and move tags of one game. States may be mutable; ``apply`` must not mutate."""

    title: str
    concepts: tuple[str, ...]
    rules: str

    def new_state(self) -> Any: ...

    def player(self, state: Any) -> str: ...

    def legal_moves(self, state: Any) -> list[Any]: ...

    def move_name(self, state: Any, move: Any) -> str: ...

    def parse_move(self, state: Any, text: str) -> Any | None: ...

    def describe_move(self, state: Any, move: Any, concepts: Iterable[str], *, facts: bool = True) -> list[Motif]: ...

    def render(self, state: Any) -> list[str]: ...

    def apply(self, state: Any, move: Any) -> Any: ...

    def is_over(self, state: Any) -> bool: ...

    def winner(self, state: Any) -> str | None: ...

    def default_engine(self) -> Engine | None: ...


@dataclass(frozen=True)
class GameLine:
    """The identity of a position: where the game started and every move since.

    ``plies`` are ``(side, move name)`` pairs, so games where a side moves twice
    in a row (an Othello pass) or custom turn orders are described exactly.
    """

    root: str
    plies: tuple[tuple[str, str], ...] = ()

    def extends(self, other: GameLine) -> bool:
        """Whether this line continues ``other`` (same start, ``other``'s moves first)."""
        return self.root == other.root and self.plies[: len(other.plies)] == other.plies

    @property
    def digest(self) -> str:
        blob = self.root + "|" + ";".join(f"{side}:{name}" for side, name in self.plies)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {"root": self.root, "plies": [list(p) for p in self.plies]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GameLine:
        return cls(str(data["root"]), tuple((str(side), str(name)) for side, name in data.get("plies", [])))


@runtime_checkable
class GameHistory(Protocol):
    """Optional capability: report a state's :class:`GameLine` for replay identity and memory."""

    def history(self, state: Any) -> GameLine: ...


def game_line(game: Game, state: Any) -> GameLine | None:
    """``state``'s line if ``game`` reports one, else ``None``."""
    history = getattr(game, "history", None)
    return history(state) if callable(history) else None


def check_concepts(game: Game, concepts: Iterable[str]) -> set[str]:
    """``concepts`` as a set, rejecting names ``game`` does not know."""
    wanted = set(concepts)
    unknown = wanted - set(game.concepts)
    if unknown:
        raise ValueError(f"Unknown {game.title} concepts: {sorted(unknown)}. Known: {list(game.concepts)}")
    return wanted


@dataclass(frozen=True)
class RenderedImage:
    """A picture of a state from a game's optional ``render_image(state)``."""

    data: bytes
    media_type: str
    renderer: str  # name and version, for fingerprints and records

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.data).hexdigest()[:16]


@dataclass
class Candidate:
    """A move offered to the model, with its tags and (if analysed) its score."""

    move: Any
    name: str
    motifs: list[Motif] = field(default_factory=list)
    score: Score | None = None

    @property
    def san(self) -> str:
        """The move's name; chess calls it SAN."""
        return self.name

    @property
    def scored(self) -> bool:
        return self.score is not None

    def evaluation(self) -> str:
        return self.score.label if self.score else ""


@dataclass
class Attempt:
    """One model call and, if it was rejected, why.

    ``reply`` and ``error`` are the historical fields. The rest record the call:
    its ``purpose`` (``choose``, ``review``, ``proposal``, ``selector``), the
    rendered ``prompt`` and ``schema``, the public ``response_text`` (partial for
    a broken stream), ``failure`` (classified), timing, ``finish_reason`` and this
    call's own ``usage`` with its completeness. ``late`` marks a response that
    arrived after the turn ended; it was never applied.
    """

    reply: Any
    error: str | None = None
    failure: FailureInfo | None = None
    purpose: str = "choose"
    index: int = 0
    attempt_id: str = ""
    prompt: str | None = None
    schema: dict[str, Any] | None = None
    response_text: str | None = None
    reason: str = ""
    plan: str = ""
    finish_reason: str | None = None
    started_at: float = 0.0
    elapsed_ms: float = 0.0
    usage: dict[str, float] = field(default_factory=dict)
    usage_status: str = "unknown"
    cost_status: str = "unknown"
    abandoned: bool = False
    late: bool = False

    @property
    def valid(self) -> bool:
        """Whether this attempt produced an acceptable choice."""
        return self.error is None and self.failure is None

    def to_dict(self, *, include_prompt: bool = True, max_field_chars: int = 20_000) -> dict[str, Any]:
        def text(value: str | None) -> str | None:
            return truncate(redact_text(value), max_field_chars) if value is not None else None

        reply = self.reply
        if not (reply is None or isinstance(reply, (str, int, float, bool))):
            reply = str(reply)
        return {
            "attempt_id": self.attempt_id,
            "index": self.index,
            "purpose": self.purpose,
            "reply": text(reply) if isinstance(reply, str) else reply,
            "reason": text(self.reason),
            "plan": text(self.plan),
            "validation": "valid" if self.valid else "invalid",
            "error": text(self.error),
            "failure": self.failure.to_dict() if self.failure else None,
            "finish_reason": self.finish_reason,
            "started_at": self.started_at,
            "elapsed_ms": round(self.elapsed_ms, 3),
            "usage": dict(self.usage),
            "usage_status": self.usage_status,
            "cost_status": self.cost_status,
            "abandoned": self.abandoned,
            "late": self.late,
            "prompt": text(self.prompt) if include_prompt else None,
            "schema": self.schema if include_prompt else None,
            "response_text": text(self.response_text),
        }


@dataclass
class MoveDecision:
    """The move a player settled on and how it got there.

    ``rescued`` marks a harness substitute after the model failed; ``assistance``
    names any help that chose the move (``"rescue"``, ``"engine"``, ``"random"``).
    ``record`` is the full :class:`~prompture.games.records.TurnRecord` when the
    decision came from a harness.
    """

    move: Any
    name: str
    reason: str
    candidates: list[Candidate]
    attempts: list[Attempt]
    rescued: bool
    usage: dict[str, float]
    plan: str = ""
    record: TurnRecord | None = None
    assistance: str | None = None

    @property
    def san(self) -> str:
        return self.name

    @property
    def model_selected(self) -> bool:
        """True only for a move a model chose itself: no substitute, engine or random pick."""
        return not self.rescued and self.assistance is None and bool(self.attempts)

    @property
    def accounting(self) -> Accounting:
        if self.record is not None:
            return self.record.accounting
        accounting = Accounting()
        if self.usage:
            accounting.add(self.usage, "complete", "reported")
        return accounting


@dataclass
class _Turn:
    """Per-turn working state: identity, candidates, cancellation and deadline."""

    record: TurnRecord
    started: float
    candidates: list[Candidate] = field(default_factory=list)
    cancel: CancelToken | None = None
    deadline: float | None = None  # time.monotonic() value
    images: list[Any] | None = None

    @property
    def controlled(self) -> bool:
        return self.cancel is not None or self.deadline is not None

    def cancelled(self) -> bool:
        return self.cancel is not None and self.cancel.cancelled

    def remaining(self) -> float | None:
        return None if self.deadline is None else self.deadline - time.monotonic()

    def expired(self) -> bool:
        remaining = self.remaining()
        return remaining is not None and remaining <= 0

    def sleep(self, seconds: float) -> None:
        """Wait, but never past the deadline and never through a cancellation."""
        remaining = self.remaining()
        if remaining is not None:
            seconds = min(seconds, max(0.0, remaining))
        if self.cancel is not None:
            self.cancel.wait(seconds)
        elif seconds > 0:
            time.sleep(seconds)


_DEADLINE = "turn_deadline"
REPRESENTATIONS = ("text", "text+image")
IMAGE_NOTE = "\n\nA picture of the board is attached. The text above is authoritative if they ever disagree."
_SEVERITY_RANK = {"urgent": 0, "warning": 1, "info": 2}
_OMITTED = "(Some lower-priority hints were left out to fit the context budget.)"


def _estimate(text: str) -> int:
    """A labeled token estimate: characters / 4, rounded up."""
    return -(-len(text) // 4)


class GameHarness:
    """Chooses moves in ``game`` with a model, constrained to a menu of candidates.

    Streaming is enabled by default for supporting drivers, with the same
    completed JSON result, retries and usage accounting. ``stream=False`` opts
    out. Each decision includes a short ``reason`` and ``plan`` for the caller.

    ``failure_policy`` (``"rescue"``, ``"forfeit"`` or ``"stop"``) decides what
    happens after ``max_retries`` rejected replies; see the module docstring.
    ``trace`` receives turn records (see :mod:`prompture.games.trace`);
    ``trace_errors`` is ``"warn"`` or ``"raise"``; ``trace_prompts=False`` keeps
    rendered prompts out of traces. ``run_id`` groups records from one run.

    ``deadline`` is a total per-turn limit in seconds over every request, retry
    and backoff (distinct from a driver's own connect/read timeouts); an
    expired turn is handled by ``failure_policy``. ``cancel`` is a default
    :class:`~prompture.games.control.CancelToken` (``choose`` also takes one).
    With either set, each request runs on a worker thread so the turn can stop
    waiting; ``cleanup_timeout`` bounds how long it waits for a stopped request
    to wind down. ``retry`` gives transient failures their own allowance with
    backoff (without it they share ``max_retries`` and retry immediately, as
    before). ``budget`` is checked and reserved before every request.
    ``on_event`` gets :class:`~prompture.games.control.TurnEvent` objects, possibly
    from a worker thread; exceptions it raises are logged and ignored.
    """

    def __init__(
        self,
        game: Game,
        model: str | None = None,
        *,
        driver: Any = None,
        level: int = 0,
        concepts: Iterable[str] | str = (),
        engine: Engine | None = None,
        top_n: int | None = None,
        blunder_drop: float = BLUNDER_DROP,
        max_retries: int = 2,
        system_prompt: str | None = None,
        options: dict[str, Any] | None = None,
        seed: int | None = None,
        stream: bool = True,
        failure_policy: str = "rescue",
        trace: TraceSink | None = None,
        trace_errors: str = "warn",
        trace_prompts: bool = True,
        max_trace_chars: int = 20_000,
        run_id: str | None = None,
        deadline: float | None = None,
        cancel: CancelToken | None = None,
        retry: RetryPolicy | None = None,
        budget: GameBudget | None = None,
        on_event: EventHandler | None = None,
        cleanup_timeout: float = 0.25,
        memory: str | int | None = "none",
        context_budget: int | None = None,
        strategy: Any = None,
        representation: str = "text",
    ) -> None:
        if level not in LEVELS:
            raise ValueError(f"level must be one of {sorted(LEVELS)}, got {level!r}")
        if failure_policy not in FAILURE_POLICIES:
            raise ValueError(f"failure_policy must be one of {FAILURE_POLICIES}, got {failure_policy!r}")
        if representation not in REPRESENTATIONS:
            raise ValueError(f"representation must be one of {REPRESENTATIONS}, got {representation!r}")
        if level >= 2 and engine is None:
            engine = game.default_engine()
            if engine is None:
                raise ValueError(f"level {level} ({LEVELS[level]}) needs an engine")
        self.game = game
        self.concepts = frozenset(check_concepts(game, game.concepts if concepts == "all" else concepts))
        if driver is None:
            if model is None:
                raise ValueError("Pass a model string like 'ollama/llama3.2:3b' or a driver")
            from ..drivers import get_driver_for_model

            driver = get_driver_for_model(model)
        self.driver = driver
        self.model: str = model or str(getattr(driver, "model", ""))
        self.level = level
        self.engine = engine
        self.top_n = top_n or DEFAULT_TOP_N.get(level)
        self.blunder_drop = blunder_drop
        self.max_retries = max_retries
        self.system_prompt = system_prompt
        self.options = options or {}
        self.stream = stream
        self.seed = seed
        self.failure_policy = failure_policy
        self.trace = TraceWriter(trace, trace_errors)
        self.trace_prompts = trace_prompts
        self.max_trace_chars = max_trace_chars
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.game_id = uuid.uuid4().hex[:12]
        from .memory import DecisionMemory

        self.memory = DecisionMemory(game, memory)
        self.context_budget = context_budget
        self.strategy = strategy
        self.representation = representation
        self.deadline = deadline
        self.cancel = cancel
        self.retry = retry
        self.budget = budget
        self.on_event = on_event
        self.cleanup_timeout = cleanup_timeout
        self._run_announced = False
        self._seq = itertools.count(1)
        self._seq_lock = threading.Lock()
        self._rng = random.Random(seed)  # nosec B311 - fallback move choice, not crypto
        self._jitter = random.Random(seed)  # nosec B311 - backoff jitter, not crypto

    def __enter__(self) -> GameHarness:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Release the engine, if it holds anything (a chess engine process)."""
        close = getattr(self.engine, "close", None)
        if close is not None:
            close()

    def start_game(self, game_id: str | None = None) -> str:
        """Start a new game identity for the records (``play_match`` calls this)."""
        self.game_id = game_id or uuid.uuid4().hex[:12]
        return self.game_id

    def commit(self, state: Any, decision: MoveDecision) -> None:
        """Tell the harness ``decision`` (chosen in ``state``) was applied.

        Optional: memory also notices applied moves from the game line. Hosts
        that keep states without history should call this.
        """
        self.memory.commit(state, decision)

    # -- identity -------------------------------------------------------------

    def behavior(self) -> dict[str, Any]:
        """The configuration that changes how this harness plays (no secrets, names or model)."""
        engine_id = None
        if self.engine is not None and self.level >= 2:
            identity = getattr(self.engine, "identity", None)
            engine_id = identity() if callable(identity) else identity or type(self.engine).__name__
        return {
            "record_version": "1",
            "prompt_version": PROMPT_VERSION,
            "game": self.game.title,
            "game_version": getattr(self.game, "version", None),
            "hints_version": getattr(self.game, "hints_version", None),
            "rules": text_digest(self.game.rules),
            "level": self.level,
            "concepts": sorted(self.concepts),
            "top_n": self.top_n if self.level >= 3 else None,
            "blunder_drop": self.blunder_drop if self.level == 2 else None,
            "engine": engine_id,
            "assistance": "engine-menu" if self.level >= 2 else None,
            "max_retries": self.max_retries,
            "failure_policy": self.failure_policy,
            "retry": self.retry.to_dict() if self.retry else None,
            "deadline_s": self.deadline,
            "memory": self.memory.policy,
            "memory_version": "1" if self.memory.enabled else None,
            "context_budget": self.context_budget,
            "strategy": self.strategy.identity() if self.strategy is not None else None,
            "representation": self.representation,
            "options": safe_options(self.options),
            "system_prompt": text_digest(self.system_prompt),
        }

    def fingerprint(self) -> Fingerprint:
        """A digest of :meth:`behavior`: changed behavior changes it, renamed players do not."""
        return fingerprint(self.behavior())

    # -- candidates ---------------------------------------------------------

    def candidates(self, state: Any) -> list[Candidate]:
        """The moves this level offers in ``state``, tagged with facts and concepts."""
        moves = self.game.legal_moves(state)
        scores: dict[Any, Score] = {}
        if self.level >= 2:
            assert self.engine is not None  # levels >= 2 get one at construction
            scores = self.engine.analyse(state, len(moves) if self.level == 2 else self.top_n or 1)
            moves = self._filter(moves, scores)
        offered = [
            Candidate(
                move,
                self.game.move_name(state, move),
                self.game.describe_move(state, move, self.concepts, facts=self.level >= 1),
                scores.get(move),
            )
            for move in moves
        ]
        if self.level >= 4:
            offered.sort(key=lambda c: -c.score.value if c.score else 0)
        else:
            offered.sort(key=lambda c: c.name)  # no order hint below level 4
        return offered

    def _filter(self, moves: list[Any], scores: dict[Any, Score]) -> list[Any]:
        scored = [m for m in moves if m in scores]
        if self.level >= 3:
            return sorted(scored, key=lambda m: -scores[m].value)[: self.top_n]
        best = max(scores[m].value for m in scored)
        return [m for m in scored if best - scores[m].value < self.blunder_drop]

    # -- prompt -------------------------------------------------------------

    def prompt(self, state: Any, candidates: list[Candidate], feedback: str | None = None) -> str:
        """The turn's prompt: rules, the position as the game renders it, the menu and any memory."""
        return self._compose(state, candidates, feedback)[0]

    def _compose(
        self, state: Any, candidates: list[Candidate], feedback: str | None = None
    ) -> tuple[str, dict[str, Any]]:
        """The prompt and how it was fitted: ``(text, info)``.

        Required: the side, rules, the rendered position, every candidate name,
        engine evaluations at level 4+, rejection feedback and the instruction.
        Optional (only trimmed under ``context_budget``): hint notes by
        severity (urgent, then warnings), then memory, then informational notes.
        Token counts are estimates (characters / 4).
        """
        head = [f"You are playing {self.game.title} as {self.game.player(state)}."]
        if self.game.rules:
            head.append(self.game.rules)
        head += self.game.render(state)
        head += ["", "Your moves (pick exactly one, written as shown):"]
        required: list[list[str]] = []
        optional: list[list[tuple[int, str]]] = []
        for candidate in candidates:
            fixed = [f"engine: {candidate.score.label}"] if self.level >= 4 and candidate.score else []
            seen = set(fixed)
            notes = []
            for motif in candidate.motifs:
                if motif.text not in seen:
                    seen.add(motif.text)
                    notes.append((_SEVERITY_RANK.get(motif.severity, 2), motif.text))
            required.append(fixed)
            optional.append(notes)
        tail = ["", f"Your previous answer was rejected: {feedback}"] if feedback else []
        tail += ["", "Give a short reason and a brief plan for your next turn, then the move you choose."]
        recalled = self.memory.recall(state)
        memory_text = self.memory.render(recalled)
        keep: list[set[int]] = [set(range(len(notes))) for notes in optional]
        keep_memory = bool(memory_text)
        info: dict[str, Any] = {"estimate": "chars/4", "budget": self.context_budget, "memory_entries": len(recalled)}
        if self.context_budget is not None:
            keep, keep_memory = self._fit(head, candidates, required, optional, tail, memory_text, info)
        lines = list(head)
        for candidate, fixed, notes, kept in zip(candidates, required, optional, keep, strict=True):
            shown = fixed + [text for index, (_, text) in enumerate(notes) if index in kept]
            lines.append(f"- {candidate.name}" + (f": {'; '.join(shown)}" if shown else ""))
        if info.get("omitted_notes") or (memory_text and not keep_memory):
            lines += ["", _OMITTED]
        lines += tail
        text = "\n".join(lines)
        if memory_text and keep_memory:
            text += "\n\n" + memory_text
        info["estimated_tokens"] = _estimate(text)
        info["memory_shown"] = bool(memory_text and keep_memory)
        return text, info

    def _fit(
        self,
        head: list[str],
        candidates: list[Candidate],
        required: list[list[str]],
        optional: list[list[tuple[int, str]]],
        tail: list[str],
        memory_text: str,
        info: dict[str, Any],
    ) -> tuple[list[set[int]], bool]:
        """Choose which optional notes and memory fit ``context_budget`` (urgent first)."""
        assert self.context_budget is not None
        menu = [
            f"- {c.name}" + (f": {'; '.join(fixed)}" if fixed else "")
            for c, fixed in zip(candidates, required, strict=True)
        ]
        base = len("\n".join(head + menu + tail))
        has_optional = bool(memory_text) or any(optional)
        budget_chars = self.context_budget * 4 - (len(_OMITTED) + 2 if has_optional else 0)
        if base > budget_chars:
            raise TurnFailed(
                f"the required prompt (position, legal moves, instructions: ~{_estimate(' ' * base)} tokens) "
                f"does not fit context_budget={self.context_budget}",
                kind=FailureKind.CONFIGURATION,
            )
        used = base
        keep: list[set[int]] = [set() for _ in optional]
        items = []
        for row, notes in enumerate(optional):
            for index, (rank, text) in enumerate(notes):
                items.append((rank, row, index, text))
        items.sort(key=lambda item: (item[0], item[1], item[2]))
        keep_memory = False
        for phase in (0, 1, "memory", 2):
            if phase == "memory":
                if memory_text and used + len(memory_text) + 2 <= budget_chars:
                    used += len(memory_text) + 2
                    keep_memory = True
                continue
            for rank, row, index, text in items:
                if rank != phase:
                    continue
                cost = len(text) + 2  # the ": " or "; " separator
                if used + cost <= budget_chars:
                    keep[row].add(index)
                    used += cost
        omitted = sum(len(notes) for notes in optional) - sum(len(k) for k in keep)
        info["omitted_notes"] = omitted
        info["memory_omitted"] = bool(memory_text) and not keep_memory
        return keep, keep_memory

    @staticmethod
    def schema(candidates: list[Candidate]) -> dict[str, Any]:
        """JSON schema for the reply; the move must be one of the candidates' names."""
        return {
            "type": "object",
            "properties": {
                "reason": {"type": "string"},
                "plan": {"type": "string"},
                "move": {"type": "string", "enum": [c.name for c in candidates]},
            },
            "required": ["reason", "plan", "move"],
        }

    # -- choosing -----------------------------------------------------------

    def choose(self, state: Any, *, cancel: CancelToken | None = None) -> MoveDecision:
        """Ask the model for a move in ``state``.

        Returns a legal move: the model's, or a marked rescue under the default
        policy. Raises :class:`~prompture.games.records.TurnFailed` (forfeit/stop
        policies, fatal configuration errors, budget) or
        :class:`~prompture.games.records.TurnCancelled`; no move may be played then.
        ``cancel`` overrides the harness's default token for this turn.
        """
        if self.game.is_over(state):
            raise ValueError("The game is over; there is no move to choose.")
        turn = self._begin(state, cancel or self.cancel)
        try:
            if turn.cancelled():
                raise TurnCancelled("the turn was cancelled before it started", record=turn.record)
            turn.candidates = self.candidates(state)
            self._prepare_images(turn, state)
            decision = self._decide(turn, state)
            if turn.cancelled():  # a completion racing a cancellation is not applied
                raise TurnCancelled("the turn was cancelled before its move was returned", record=turn.record)
        except TurnError as exc:
            self._finish(turn, exc)
            raise
        self._finish(turn, None, decision)
        return decision

    async def achoose(self, state: Any) -> MoveDecision:
        """:meth:`choose` for asyncio callers.

        The turn runs on a worker thread. Cancelling the awaiting task cancels
        the turn's token: a streaming request is closed between chunks, a
        blocking one is abandoned (see :mod:`prompture.games.control`).
        """
        token = CancelToken(self.cancel)
        try:
            return await asyncio.to_thread(self.choose, state, cancel=token)
        except asyncio.CancelledError:
            token.cancel("the awaiting task was cancelled")
            raise

    def _decide(self, turn: _Turn, state: Any) -> MoveDecision:
        """One decision from the candidates: the model's (directly or via a strategy), or the failure policy's."""
        if self.strategy is not None:
            decision = self.strategy.decide(self, turn, state)
        else:
            chosen, reason, plan = self._select(turn, state, turn.candidates)
            if chosen is None:
                decision = self._exhausted(turn, turn.candidates)
            else:
                decision = self._accepted(turn, chosen, reason, plan)
        self.memory.propose(state, decision)
        if self.memory.enabled:
            turn.record.context["memory"] = {"policy": self.memory.policy, "version": "1"}
        return decision

    def _accepted(self, turn: _Turn, chosen: Candidate, reason: str, plan: str) -> MoveDecision:
        turn.record.outcome = "accepted"
        return MoveDecision(
            chosen.move,
            chosen.name,
            reason,
            turn.candidates,
            turn.record.attempts,
            False,
            turn.record.accounting.legacy(),
            plan,
            record=turn.record,
        )

    def _select(
        self, turn: _Turn, state: Any, candidates: list[Candidate], *, purpose: str = "choose"
    ) -> tuple[Candidate | None, str, str]:
        """Ask until a listed move comes back or the retry allowance is spent.

        Returns ``(candidate, reason, plan)``, or ``(None, "", "")`` when exhausted.
        Fatal configuration failures raise; they cannot become a rescue.
        """
        schema = self.schema(candidates)
        feedback: str | None = None
        failures = 0
        matched: list[Candidate] = []

        def check(answer: dict[str, Any]) -> str | None:
            chosen, error = self._match(state, candidates, answer.get("move"))
            if chosen is not None:
                matched.append(chosen)
            return error

        transient = 0
        while True:
            if not self._may_dispatch(turn):
                return None, "", ""
            prompt, prompt_info = self._compose(state, candidates, feedback)
            turn.record.context["prompt"] = prompt_info
            if turn.images:
                prompt += IMAGE_NOTE
            matched.clear()
            _answer, attempt = self._call(turn, prompt, schema, purpose, check, images=turn.images)
            if attempt.valid and matched:
                return matched[0], attempt.reason, attempt.plan
            failure = attempt.failure or FailureInfo(FailureKind.INVALID_OUTPUT, "invalid", attempt.error or "")
            turn.record.failure = failure
            if failure.kind is FailureKind.CANCELLED:
                raise TurnCancelled(self._cancel_message(turn), record=turn.record)
            if failure.category == _DEADLINE:
                return None, "", ""
            if failure.kind.fatal:
                raise TurnFailed(
                    f"{failure.kind.value} failure: {attempt.error}",
                    kind=failure.kind,
                    record=turn.record,
                    forfeit=self.failure_policy == "forfeit",
                )
            if failure.kind.transient and self.retry is not None:
                transient += 1
                if transient > self.retry.transient_retries:
                    return None, "", ""
                delay = self.retry.delay(transient, failure.retry_after, self._jitter)
                self._emit("retry_scheduled", turn, delay_s=round(delay, 3), failure=failure.kind.value)
                turn.sleep(delay)
                continue  # the request did not reach the model: no feedback to give
            failures += 1
            if failures > self.max_retries:
                return None, "", ""
            feedback = attempt.error

    def _prepare_images(self, turn: _Turn, state: Any) -> None:
        """Render the turn's picture, or record exactly why the turn is text-only."""
        if self.representation == "text":
            return
        info: dict[str, Any] = {"requested": self.representation, "effective": "text"}
        render = getattr(self.game, "render_image", None)
        if not callable(render):
            info["reason"] = f"{self.game.title} has no image renderer"
        elif not getattr(self.driver, "supports_vision", False):
            info["reason"] = "the driver does not accept images"
        else:
            try:
                image = render(state)
            except Exception as exc:  # e.g. an optional rasterizer is not installed
                info["reason"] = f"rendering failed: {exc}"
            else:
                from ..media.image import ImageContent

                turn.images = [ImageContent(base64.b64encode(image.data).decode("ascii"), image.media_type)]
                info.update(
                    effective="text+image", renderer=image.renderer, media_type=image.media_type, digest=image.digest
                )
        turn.record.context["representation"] = info

    def _may_dispatch(self, turn: _Turn) -> bool:
        """Whether another request may start; raises on cancellation, False when out of time."""
        if turn.cancelled():
            raise TurnCancelled(self._cancel_message(turn), record=turn.record)
        if turn.expired():
            turn.record.failure = FailureInfo(
                FailureKind.TIMEOUT, _DEADLINE, f"the {self.deadline:g}s turn deadline expired"
            )
            return False
        return True

    def _call(
        self,
        turn: _Turn,
        prompt: str,
        schema: dict[str, Any],
        purpose: str,
        check: Callable[[dict[str, Any]], str | None] | None = None,
        images: list[Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any] | None, Attempt]:
        """Send one request and record it as an :class:`Attempt`.

        ``check`` validates a parsed answer (returning an error or ``None``)
        before the attempt is recorded and traced. Returns the parsed answer
        object (or ``None``) and the attempt, whose ``error``/``failure`` are set
        when the call, its JSON or the check failed. The budget is reserved
        first (raising :class:`~prompture.games.records.BudgetExhausted`).
        """
        from ..extraction.core import _stop_reason

        attempt = self._new_attempt(turn, prompt, schema, purpose, images)
        reservation = self._reserve(turn, prompt)
        self._emit("attempt_started", turn, attempt_id=attempt.attempt_id, purpose=purpose, index=attempt.index)
        started = time.perf_counter()
        status, outcome = self._dispatch(turn, attempt, prompt, schema, images, options)
        meta: dict[str, Any] | None = None
        answer: dict[str, Any] | None = None
        if status == "abandoned":
            attempt.abandoned = True
            self._stopped(turn, attempt, "abandoned while still running; the provider may still finish and bill it")
        elif isinstance(outcome, ExtractionError):  # an unparseable reply was still paid for
            meta = source_meta(outcome.usage)
            attempt.response_text = outcome.raw_text
            attempt.finish_reason = outcome.stop_reason
            attempt.error = f"the reply was not valid JSON ({outcome})"
            attempt.failure = FailureInfo(FailureKind.INVALID_OUTPUT, "unparseable", str(outcome))
        elif isinstance(outcome, BaseException):  # a timeout or a dead backend must not abort a whole game
            attempt.response_text = getattr(outcome, "partial_text", None) or None
            attempt.error = f"the model call failed ({outcome})"
            attempt.failure = classify_failure(outcome)
        else:
            usage = outcome.get("usage") or {}
            meta = source_meta(usage)
            answer = self._read_answer(attempt, outcome)
            raw = usage.get("raw_response")
            attempt.finish_reason = _stop_reason(raw) if isinstance(raw, dict) else None
            if answer is not None and check is not None:
                error = check(answer)
                if error is not None:
                    attempt.error = error
                    attempt.failure = FailureInfo(FailureKind.INVALID_OUTPUT, "illegal_choice", error)
        if status == "stopped":  # finished while the turn was stopping: never applied
            answer = None
            self._stopped(turn, attempt, "stopped; any response was discarded")
        attempt.elapsed_ms = (time.perf_counter() - started) * 1000
        self._account(turn, attempt, meta, reservation)
        return answer, attempt

    @staticmethod
    def _cancel_message(turn: _Turn) -> str:
        reason = turn.cancel.reason if turn.cancel is not None else ""
        return f"the turn was cancelled ({reason})" if reason else "the turn was cancelled"

    def _stopped(self, turn: _Turn, attempt: Attempt, what: str) -> None:
        if turn.cancelled():
            attempt.error = f"the turn was cancelled; the request was {what}"
            attempt.failure = FailureInfo(FailureKind.CANCELLED, "cancelled", attempt.error)
        else:
            attempt.error = f"the {self.deadline:g}s turn deadline expired; the request was {what}"
            attempt.failure = FailureInfo(FailureKind.TIMEOUT, _DEADLINE, attempt.error)

    def _reserve(self, turn: _Turn, prompt: str) -> Reservation | None:
        if self.budget is None:
            return None
        output = self.options.get("max_tokens") or self.options.get("max_output_tokens") or self.budget.output_reserve
        try:
            return self.budget.reserve(len(prompt) / 4 + float(output))  # chars/4: a labeled estimate
        except TurnError as exc:
            exc.record = turn.record
            raise

    def _dispatch(
        self,
        turn: _Turn,
        attempt: Attempt,
        prompt: str,
        schema: dict[str, Any],
        images: list[Any] | None,
        options: dict[str, Any] | None = None,
    ) -> tuple[str, Any]:
        """Run the request. Returns ``(status, result or exception)``.

        ``status`` is ``"done"``; ``"stopped"`` (it ended while the turn was
        stopping, so its result must not be used); or ``"abandoned"`` (still
        running after ``cleanup_timeout``; a late result is only logged).
        """
        from ..extraction.core import ask_for_json

        def on_delta(text: str) -> None:
            self._emit("output_delta", turn, attempt_id=attempt.attempt_id, text=text)

        stop = threading.Event()
        kwargs: dict[str, Any] = {
            "ai_cleanup": False,
            "model_name": self.model,
            "options": self.options if options is None else options,
            "system_prompt": self.system_prompt,
            "stream": self.stream,
            "on_delta": on_delta if self.on_event is not None else None,
            "should_stop": stop.is_set if turn.controlled else None,
        }
        if images:
            kwargs["images"] = images

        def run() -> Any:
            try:
                return ask_for_json(self.driver, prompt, schema, **kwargs)
            except Exception as exc:
                return exc

        if not turn.controlled:
            return "done", run()

        box: dict[str, Any] = {"closed": False}
        done = threading.Event()
        lock = threading.Lock()

        def work() -> None:
            outcome = run()
            with lock:
                box["outcome"] = outcome
                late = box["closed"]
            done.set()
            if late:
                self._late(turn, attempt, outcome)

        threading.Thread(target=work, name=f"game-request-{attempt.attempt_id}", daemon=True).start()
        while not done.wait(0.02):
            if turn.cancelled() or turn.expired():
                break
        else:
            return "done", box["outcome"]
        stop.set()  # a stream closes at its next chunk
        done.wait(self.cleanup_timeout)
        with lock:
            if "outcome" not in box:
                box["closed"] = True
                return "abandoned", None
            return "stopped", box["outcome"]

    def _late(self, turn: _Turn, attempt: Attempt, outcome: Any) -> None:
        """A response that arrived after its turn stopped waiting: log it, never apply it."""
        attempt.late = True
        usage = None
        if isinstance(outcome, dict):
            meta = source_meta(outcome.get("usage")) or {}
            usage = {key: meta.get(key) for key in _USAGE_KEYS}
        self._emit("late_response", turn, attempt_id=attempt.attempt_id, usage=usage)
        self._trace("late_response", turn, attempt_id=attempt.attempt_id, usage=usage)

    def _new_attempt(
        self, turn: _Turn, prompt: str, schema: dict[str, Any], purpose: str, images: list[Any] | None = None
    ) -> Attempt:
        index = len(turn.record.attempts)
        return Attempt(
            None,
            purpose=purpose,
            index=index,
            attempt_id=f"{turn.record.turn_id}.{index}",
            prompt=prompt,
            schema=schema,
            started_at=time.time(),
        )

    @staticmethod
    def _read_answer(attempt: Attempt, result: dict[str, Any]) -> dict[str, Any] | None:
        attempt.response_text = result.get("json_string")
        parsed = result.get("json_object")
        if not isinstance(parsed, dict):
            attempt.reply = parsed
            attempt.error = 'the reply must be a JSON object with "reason", "plan" and "move"'
            attempt.failure = FailureInfo(FailureKind.INVALID_OUTPUT, "not_an_object", attempt.error)
            return None
        attempt.reply = parsed.get("move")
        attempt.reason = str(parsed.get("reason") or "")
        attempt.plan = str(parsed.get("plan") or "")
        return parsed

    def _account(
        self, turn: _Turn, attempt: Attempt, meta: dict[str, Any] | None, reservation: Reservation | None = None
    ) -> None:
        """Record ``attempt``'s usage exactly once, with its completeness, settle the budget and trace it."""
        attempt.usage = {key: float((meta or {}).get(key) or 0) for key in _USAGE_KEYS}
        attempt.usage_status, attempt.cost_status = usage_status(meta)
        turn.record.attempts.append(attempt)
        turn.record.accounting.add(
            attempt.usage, attempt.usage_status, attempt.cost_status, abandoned=attempt.abandoned
        )
        if self.budget is not None and reservation is not None:
            self.budget.settle(reservation, attempt.usage, attempt.cost_status)
        self._trace(
            "attempt",
            turn,
            attempt=attempt.to_dict(include_prompt=self.trace_prompts, max_field_chars=self.max_trace_chars),
        )
        self._emit(
            "attempt_finished",
            turn,
            attempt_id=attempt.attempt_id,
            purpose=attempt.purpose,
            valid=attempt.valid,
            error=attempt.error,
            failure=attempt.failure.kind.value if attempt.failure else None,
            elapsed_ms=round(attempt.elapsed_ms, 3),
            usage=dict(attempt.usage),
            usage_status=attempt.usage_status,
            cost_status=attempt.cost_status,
            abandoned=attempt.abandoned,
        )

    def _exhausted(self, turn: _Turn, candidates: list[Candidate]) -> MoveDecision:
        """Apply the failure policy once the retry allowance is spent."""
        failure = turn.record.failure
        if self.failure_policy != "rescue":
            raise TurnFailed(
                f"no acceptable move after {len(turn.record.attempts)} attempts ({self.failure_policy} policy)",
                kind=failure.kind if failure else FailureKind.INVALID_OUTPUT,
                record=turn.record,
                forfeit=self.failure_policy == "forfeit",
            )
        fallback = self._fallback(candidates)
        turn.record.outcome = "rescued"
        turn.record.notes.append("move substituted by the harness after the model failed; not model-selected")
        return MoveDecision(
            fallback.move,
            fallback.name,
            "",
            candidates,
            turn.record.attempts,
            True,
            turn.record.accounting.legacy(),
            record=turn.record,
            assistance="rescue",
        )

    # -- records ---------------------------------------------------------------

    def _begin(self, state: Any, cancel: CancelToken | None = None) -> _Turn:
        line = game_line(self.game, state)
        record = TurnRecord(
            run_id=self.run_id,
            game_id=self.game_id,
            turn_id=uuid.uuid4().hex[:16],
            side=self.game.player(state),
            ply=len(line.plies) if line else None,
            state_id=line.digest if line else None,
            fingerprint=self.fingerprint().digest,
            model=self.model,
            started_at=time.time(),
        )
        if not self._run_announced:
            self._run_announced = True
            fp = self.fingerprint()
            self.trace.write(
                {
                    "type": "run",
                    "run_id": self.run_id,
                    "fingerprint": fp.digest,
                    "config": fp.config,
                    "model": self.model,
                }
            )
        now = time.monotonic()
        turn = _Turn(record, time.perf_counter(), cancel=cancel)
        if self.deadline is not None:
            turn.deadline = now + self.deadline
            record.deadline_s = self.deadline
        sdk_retries = getattr(getattr(self.driver, "client", None), "max_retries", None)
        if isinstance(sdk_retries, int):
            record.context["sdk_max_retries"] = sdk_retries  # SDK retries happen inside one attempt
        self._trace("turn_started", turn, fingerprint=record.fingerprint, model=self.model)
        self._emit("turn_started", turn, ply=record.ply, state_id=record.state_id, deadline_s=self.deadline)
        return turn

    def _finish(self, turn: _Turn, error: TurnError | None, decision: MoveDecision | None = None) -> None:
        record = turn.record
        record.elapsed_ms = (time.perf_counter() - turn.started) * 1000
        if error is not None:
            record.outcome = "cancelled" if isinstance(error, TurnCancelled) else "failed"
            if record.failure is None or record.failure.kind != error.kind:
                record.failure = FailureInfo(error.kind, error.kind.value, str(error), error_type=type(error).__name__)
            error.record = record
        elif decision is not None:
            record.move = decision.name
        self._trace("turn_finished", turn, record=record.to_dict(include_prompts=False))
        kind = {"cancelled": "turn_cancelled", "failed": "turn_failed"}.get(record.outcome, "turn_completed")
        self._emit(
            kind,
            turn,
            outcome=record.outcome,
            move=record.move,
            failure=record.failure.kind.value if record.failure else None,
            attempts=len(record.attempts),
            elapsed_ms=round(record.elapsed_ms, 3),
            accounting=record.accounting.to_dict(),
        )
        self.trace.raise_pending()

    def _emit(self, kind: str, turn: _Turn, **data: Any) -> None:
        if self.on_event is None:
            return
        with self._seq_lock:
            seq = next(self._seq)
        record = turn.record
        event = TurnEvent(seq, kind, record.run_id, record.game_id, record.turn_id, record.side, time.time(), data)
        try:
            self.on_event(event)
        except Exception:
            logger.exception("on_event handler raised for %s", kind)

    def _trace(self, kind: str, turn: _Turn, **data: Any) -> None:
        record = turn.record
        self.trace.write(
            {
                "type": kind,
                "run_id": record.run_id,
                "game_id": record.game_id,
                "turn_id": record.turn_id,
                "side": record.side,
                "ply": record.ply,
                "state_id": record.state_id,
                "time": time.time(),
                **data,
            }
        )

    # -- matching -----------------------------------------------------------------

    def _match(self, state: Any, candidates: list[Candidate], reply: Any) -> tuple[Candidate | None, str | None]:
        if not isinstance(reply, (str, int)) or not str(reply).strip():
            return None, "no move was given"
        text = str(reply).strip()
        by_name = {c.name: c for c in candidates}
        if text in by_name:
            return by_name[text], None
        move = self.game.parse_move(state, text)
        if move is None:
            return None, f"{text} is not a legal move in this position"
        for candidate in candidates:
            if candidate.move == move:
                return candidate, None
        return None, f"{text} is legal but not one of your listed moves"

    def _fallback(self, candidates: list[Candidate]) -> Candidate:
        scored = [c for c in candidates if c.score]
        if scored:
            return max(scored, key=lambda c: c.score.value if c.score else 0)
        return self._rng.choice(candidates)
