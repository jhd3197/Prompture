"""Skill mining — learn reusable skills from what an agent actually did.

Prompture already has a *skill system* (:mod:`prompture.agents.skills`): you write
a ``SKILL.md``, it gets discovered, registered, and injected into an agent. What it
did **not** have is a way to notice *when a new skill is worth creating*. That is what
this module adds.

A :class:`SkillMiner` observes completed :class:`~prompture.agents.types.AgentResult`
trajectories. When the same multi-step tool procedure recurs (``min_occurrences``,
default 2) **and** an LLM judges it a generalizable, reusable procedure, the miner
drafts a :class:`SkillProposal` — a ready-to-save ``SKILL.md`` — so that next time the
agent already has the skill instead of rediscovering the steps.

Design (chosen defaults):

* **Trigger** — recurrence *and* an LLM generalization judgment. Pure repetition is not
  enough; the judge filters out non-meaningful or one-off chains.
* **Autonomy** — *propose & confirm*. A confirmed proposal is registered in-memory for
  the session (so the agent can use it immediately), but the durable ``SKILL.md`` file is
  only written when you call :meth:`SkillMiner.save`.

The miner is wired with zero changes to the agent loop: ``AgentResult`` is already handed
to the existing ``on_output`` callback, so::

    miner = SkillMiner(model="openai/gpt-4o-mini", on_proposal=print)
    agent = Agent(..., callbacks=AgentCallbacks(on_output=miner.observe))

Every run flows through the miner; the LLM is only consulted on the run that first crosses
the recurrence threshold. You can also mine a batch of past runs offline with
:meth:`SkillMiner.mine`.
"""

from __future__ import annotations

import dataclasses
import logging
import re
import threading
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .skills import SkillInfo, get_skill_registry_snapshot, register_skill
from .types import AgentResult, AgentState, StepType

if TYPE_CHECKING:
    from pydantic import BaseModel

logger = logging.getLogger("prompture.skill_miner")

# A procedure "signature" is the ordered tuple of tool names an agent invoked.
Signature = tuple[str, ...]

_GENERATED_BY = "prompture.skill_miner"
_DEFAULT_SKILLS_DIR = Path(".claude") / "skills"
_TRACE_RESULT_CHARS = 240
_TASK_CHARS = 400
_OUTPUT_CHARS = 600
_MAX_EXEMPLARS = 3


# ------------------------------------------------------------------
# Proposal
# ------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class SkillProposal:
    """A drafted, not-yet-persisted skill mined from agent trajectories.

    Convert to a :class:`~prompture.agents.skills.SkillInfo` with
    :meth:`to_skill_info`, or render the ``SKILL.md`` text with :meth:`to_markdown`.
    The :class:`SkillMiner` registers confirmed proposals in-memory automatically;
    persisting to disk is an explicit :meth:`SkillMiner.save` call.

    Attributes:
        name: kebab-case skill identifier (and on-disk directory name).
        description: One-line ``SKILL.md`` description / trigger.
        instructions: Markdown body — the generalized procedure the agent should follow.
        tool_sequence: Ordered tool names that defined the recurring procedure.
        occurrences: How many runs exhibited this procedure when it was proposed.
        confidence: LLM confidence (0..1) that this is a high-value reusable skill.
        when_to_use: Short "use this when ..." trigger phrase (may be empty).
        rationale: Why the miner judged this worth saving (for surfacing to a human).
    """

    name: str
    description: str
    instructions: str
    tool_sequence: tuple[str, ...] = ()
    occurrences: int = 1
    confidence: float = 0.0
    when_to_use: str = ""
    rationale: str = ""

    def to_skill_info(self) -> SkillInfo:
        """Build a registrable :class:`SkillInfo` from this proposal."""
        metadata: dict[str, Any] = {
            "generated_by": _GENERATED_BY,
            "tool_sequence": list(self.tool_sequence),
            "occurrences": self.occurrences,
            "confidence": round(float(self.confidence), 4),
        }
        if self.when_to_use:
            metadata["when_to_use"] = self.when_to_use
        return SkillInfo(
            name=self.name,
            description=self.description,
            instructions=self.instructions,
            metadata=metadata,
        )

    def to_markdown(self) -> str:
        """Render this proposal as ``SKILL.md`` text (YAML frontmatter + body)."""
        try:
            import yaml
        except ImportError:  # pragma: no cover - yaml is a skill dependency
            raise ImportError("pyyaml is required to render SKILL.md. Install with: pip install pyyaml") from None

        frontmatter: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "metadata": {
                "generated_by": _GENERATED_BY,
                "tool_sequence": list(self.tool_sequence),
                "occurrences": self.occurrences,
                "confidence": round(float(self.confidence), 4),
            },
        }
        if self.when_to_use:
            frontmatter["metadata"]["when_to_use"] = self.when_to_use
        fm = yaml.safe_dump(frontmatter, sort_keys=False, allow_unicode=True).strip()
        body = self.instructions.strip()
        return f"---\n{fm}\n---\n\n{body}\n"


# ------------------------------------------------------------------
# Internal draft schema (LLM reflection target)
# ------------------------------------------------------------------


def _draft_model() -> type[BaseModel]:
    """Build the Pydantic schema the LLM fills in when judging/drafting a skill.

    Defined lazily so importing this module does not require pydantic at import time.
    """
    from pydantic import BaseModel, Field

    class _SkillDraft(BaseModel):
        generalizable: bool = Field(
            ...,
            description=(
                "True only if this tool sequence is a reusable, generalizable procedure "
                "worth saving as a skill for future tasks. False if it is a one-off, "
                "trivial, or task-specific chain that would not help next time."
            ),
        )
        name: str = Field(
            ...,
            description="Short kebab-case skill name, e.g. 'web-research-summary'. Lowercase, hyphenated.",
        )
        description: str = Field(
            ...,
            description=(
                "One sentence describing what the skill does and when to use it — this becomes "
                "the SKILL.md description that decides when the skill is relevant."
            ),
        )
        when_to_use: str = Field(
            default="",
            description="Short trigger phrase, e.g. 'Use this when the user asks to research a topic and summarize sources.'",
        )
        instructions: str = Field(
            ...,
            description=(
                "Markdown body of the skill: numbered, generalized steps the agent should follow "
                "to perform this procedure. Generalize away the specific inputs/values from the "
                "example runs; describe the reusable method, not the one example."
            ),
        )
        confidence: float = Field(
            default=0.0,
            ge=0.0,
            le=1.0,
            description="0..1 confidence that this is a high-value reusable skill.",
        )

    return _SkillDraft


# ------------------------------------------------------------------
# Miner
# ------------------------------------------------------------------


class SkillMiner:
    """Detect when an agent's behavior should become a reusable skill, and draft it.

    Feed completed runs in via :meth:`observe` (or wire it as an agent's ``on_output``
    callback). The miner counts normalized tool-call sequences across runs; once a
    sequence recurs ``min_occurrences`` times and clears the heuristic gate, it asks an
    LLM whether the procedure is genuinely generalizable and, if so, drafts a
    :class:`SkillProposal`.

    Confirmed proposals are registered in-memory (``auto_register``) so the agent can use
    them this session; call :meth:`save` to persist the ``SKILL.md`` to disk for next time.

    Args:
        model: ``"provider/model"`` used for the generalization judgment / drafting.
        min_occurrences: Recurrence threshold before a procedure is considered (default 2).
        min_tools: Minimum number of *distinct* tools for a procedure to qualify (default 2).
        min_steps: Minimum number of tool calls for a procedure to qualify (default 2).
        min_confidence: Reject drafts below this LLM confidence (default 0.0).
        judge: When True (default), use the LLM to gate + draft. When False, skip the LLM
            and draft a heuristic skill purely from the tool sequence (useful offline/tests).
        auto_register: Register confirmed proposals in the global skill registry (default True).
        skills_dir: Default directory for :meth:`save` (default ``./.claude/skills``).
        on_proposal: Callback invoked with each confirmed :class:`SkillProposal`.
        routing: Optional routing strategy/config forwarded to extraction (when ``model`` is None).
        options: Extra driver options forwarded to extraction.
        signature_fn: Override how a run is reduced to a procedure signature.
        extract_fn / aextract_fn: Override the extraction functions (primarily for testing).
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        min_occurrences: int = 2,
        min_tools: int = 2,
        min_steps: int = 2,
        min_confidence: float = 0.0,
        judge: bool = True,
        auto_register: bool = True,
        skills_dir: str | Path | None = None,
        on_proposal: Callable[[SkillProposal], None] | None = None,
        routing: Any = None,
        options: dict[str, Any] | None = None,
        signature_fn: Callable[[AgentResult], Signature] | None = None,
        extract_fn: Callable[..., dict[str, Any]] | None = None,
        aextract_fn: Callable[..., Any] | None = None,
    ) -> None:
        self._model = model
        self._min_occurrences = max(1, int(min_occurrences))
        self._min_tools = max(1, int(min_tools))
        self._min_steps = max(1, int(min_steps))
        self._min_confidence = float(min_confidence)
        self._judge = bool(judge)
        self._auto_register = bool(auto_register)
        self._skills_dir = Path(skills_dir) if skills_dir is not None else _DEFAULT_SKILLS_DIR
        self._on_proposal = on_proposal
        self._routing = routing
        self._options = options
        self._signature_fn = signature_fn or default_signature
        self._extract_fn = extract_fn
        self._aextract_fn = aextract_fn

        self._lock = threading.Lock()
        self._counter: Counter[Signature] = Counter()
        self._exemplars: dict[Signature, list[AgentResult]] = {}
        self._verdicts: dict[Signature, bool] = {}
        self._proposals: list[SkillProposal] = []

    # -- public surface ------------------------------------------------------

    @property
    def proposals(self) -> list[SkillProposal]:
        """All confirmed proposals made so far (most recent last)."""
        with self._lock:
            return list(self._proposals)

    def recurring(self, min_occurrences: int | None = None) -> list[tuple[Signature, int]]:
        """Procedure signatures seen at least ``min_occurrences`` times (default: configured)."""
        threshold = self._min_occurrences if min_occurrences is None else int(min_occurrences)
        with self._lock:
            return sorted(
                ((sig, n) for sig, n in self._counter.items() if n >= threshold),
                key=lambda kv: kv[1],
                reverse=True,
            )

    def observe(self, result: AgentResult) -> SkillProposal | None:
        """Record a completed run and, if it triggers, propose a skill.

        Safe to use directly as an agent ``on_output`` callback — never raises; any
        synthesis failure is logged and yields ``None``.

        Returns the confirmed :class:`SkillProposal` on the run that crosses the
        threshold, else ``None``.
        """
        gated = self._record_and_gate(result)
        if gated is None:
            return None
        sig, exemplars = gated
        try:
            draft = self._synthesize(sig, exemplars)
        except Exception as exc:  # network/parse failure — retry on a later occurrence
            logger.warning("Skill synthesis failed for %s: %s", " -> ".join(sig), exc)
            return None
        return self._finalize(sig, exemplars, draft)

    async def aobserve(self, result: AgentResult) -> SkillProposal | None:
        """Async sibling of :meth:`observe`, for wiring into async agents."""
        gated = self._record_and_gate(result)
        if gated is None:
            return None
        sig, exemplars = gated
        try:
            draft = await self._asynthesize(sig, exemplars)
        except Exception as exc:
            logger.warning("Skill synthesis failed for %s: %s", " -> ".join(sig), exc)
            return None
        return self._finalize(sig, exemplars, draft)

    def mine(self, results: Iterable[AgentResult]) -> list[SkillProposal]:
        """Feed a batch of past runs through the miner and return confirmed proposals."""
        out: list[SkillProposal] = []
        for result in results:
            proposal = self.observe(result)
            if proposal is not None:
                out.append(proposal)
        return out

    def save(
        self,
        proposal: SkillProposal,
        *,
        skills_dir: str | Path | None = None,
        overwrite: bool = False,
    ) -> Path:
        """Persist a confirmed proposal as ``<skills_dir>/<name>/SKILL.md`` (the confirm step).

        The file lands where :func:`~prompture.agents.skills.discover_skills` looks, so the
        skill is available in future sessions.

        Args:
            proposal: The proposal to write.
            skills_dir: Target skills root (default: the miner's ``skills_dir``).
            overwrite: Allow replacing an existing ``SKILL.md`` (default False — refuse).

        Returns:
            Path to the written ``SKILL.md``.

        Raises:
            FileExistsError: If the skill already exists on disk and ``overwrite`` is False.
        """
        root = Path(skills_dir) if skills_dir is not None else self._skills_dir
        skill_dir = root / proposal.name
        skill_file = skill_dir / "SKILL.md"
        if skill_file.exists() and not overwrite:
            raise FileExistsError(f"Skill already exists: {skill_file} (pass overwrite=True to replace)")
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_file.write_text(proposal.to_markdown(), encoding="utf-8")
        logger.info("Saved mined skill '%s' to %s", proposal.name, skill_file)
        return skill_file

    # -- internals -----------------------------------------------------------

    def _record_and_gate(self, result: AgentResult) -> tuple[Signature, list[AgentResult]] | None:
        """Heuristic + recurrence gate. Returns (signature, exemplars) when it is time to
        synthesize, else None. Holds the lock only briefly; never calls the LLM."""
        if not self._passes_heuristics(result):
            return None
        sig = self._signature_fn(result)
        if len(sig) < self._min_steps or len(set(sig)) < self._min_tools:
            return None

        with self._lock:
            exemplars = self._exemplars.setdefault(sig, [])
            if len(exemplars) < _MAX_EXEMPLARS:
                exemplars.append(result)
            self._counter[sig] += 1

            if sig in self._verdicts:
                return None  # already decided (accepted or rejected) — don't re-judge
            if self._counter[sig] < self._min_occurrences:
                return None
            if _already_a_skill(sig):
                self._verdicts[sig] = False
                return None
            return sig, list(exemplars)

    def _passes_heuristics(self, result: AgentResult) -> bool:
        """Cheap pre-filter: only mine non-trivial, non-errored runs that produced output."""
        if getattr(result, "state", None) == AgentState.errored:
            return False
        if not (result.output_text or "").strip():
            return False
        return bool(result.all_tool_calls) or any(s.step_type == StepType.tool_call for s in result.steps)

    def _synthesize(self, sig: Signature, exemplars: Sequence[AgentResult]) -> Any:
        if not self._judge:
            return _heuristic_draft(sig, len(exemplars))
        text, system_prompt = self._build_synth_prompt(sig, exemplars)
        extract = self._extract_fn or _default_extract
        result = extract(
            _draft_model(),
            text=text,
            model_name=self._model,
            system_prompt=system_prompt,
            max_retries=2,
            **self._extract_kwargs(),
        )
        return result["model"]

    async def _asynthesize(self, sig: Signature, exemplars: Sequence[AgentResult]) -> Any:
        if not self._judge:
            return _heuristic_draft(sig, len(exemplars))
        text, system_prompt = self._build_synth_prompt(sig, exemplars)
        extract = self._aextract_fn or _default_aextract
        result = await extract(
            _draft_model(),
            text=text,
            model_name=self._model,
            system_prompt=system_prompt,
            max_retries=2,
            **self._extract_kwargs(),
        )
        return result["model"]

    def _extract_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if self._routing is not None:
            kwargs["routing"] = self._routing
        if self._options is not None:
            kwargs["options"] = self._options
        return kwargs

    def _finalize(
        self,
        sig: Signature,
        exemplars: Sequence[AgentResult],
        draft: Any,
    ) -> SkillProposal | None:
        generalizable = bool(getattr(draft, "generalizable", True))
        confidence = float(getattr(draft, "confidence", 0.0) or 0.0)
        with self._lock:
            self._verdicts[sig] = generalizable
        if not generalizable or confidence < self._min_confidence:
            logger.debug(
                "Skill rejected for %s (generalizable=%s, conf=%.2f)", " -> ".join(sig), generalizable, confidence
            )
            return None

        name = _slugify(getattr(draft, "name", "") or "-".join(sig))
        proposal = SkillProposal(
            name=name,
            description=(getattr(draft, "description", "") or "").strip(),
            instructions=(getattr(draft, "instructions", "") or "").strip(),
            tool_sequence=tuple(sig),
            occurrences=self._counter[sig],
            confidence=confidence,
            when_to_use=(getattr(draft, "when_to_use", "") or "").strip(),
            rationale=(getattr(draft, "description", "") or "").strip(),
        )
        with self._lock:
            self._proposals.append(proposal)
        if self._auto_register:
            register_skill(proposal.to_skill_info())
        if self._on_proposal is not None:
            try:
                self._on_proposal(proposal)
            except Exception as exc:  # a UI callback must never break mining
                logger.warning("on_proposal callback raised: %s", exc)
        return proposal

    def _build_synth_prompt(self, sig: Signature, exemplars: Sequence[AgentResult]) -> tuple[str, str]:
        system_prompt = (
            "You are a skill librarian for an AI agent. You are shown a procedure the agent "
            "performed across one or more tasks. Decide whether it is a REUSABLE, generalizable "
            "skill worth saving so the agent can reuse it next time, then draft it.\n"
            "Be conservative: reject trivial, one-off, or purely task-specific chains "
            "(set generalizable=false). When accepting, generalize away the specific inputs and "
            "values from the examples — describe the reusable method as numbered steps, not the "
            "single example. Name it in kebab-case."
        )
        parts: list[str] = [
            f"The agent ran this tool procedure {self._counter[sig]} time(s).",
            f"Tool sequence (in order): {' -> '.join(sig)}",
            "",
        ]
        for i, ex in enumerate(exemplars, start=1):
            parts.append(f"=== Example run {i} ===")
            task = _first_user_text(ex.messages)
            if task:
                parts.append(f"Task: {_truncate(task, _TASK_CHARS)}")
            parts.append("Trace:")
            parts.append(_render_trace(ex))
            if (ex.output_text or "").strip():
                parts.append(f"Final output: {_truncate(ex.output_text.strip(), _OUTPUT_CHARS)}")
            parts.append("")
        return "\n".join(parts), system_prompt


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def default_signature(result: AgentResult) -> Signature:
    """Reduce a run to its ordered tool-name sequence (the procedure signature)."""
    names: list[str] = []
    for call in result.all_tool_calls:
        name = call.get("name")
        if name:
            names.append(str(name))
    if names:
        return tuple(names)
    # Fall back to tool_call steps if all_tool_calls was not populated.
    return tuple(s.tool_name for s in result.steps if s.step_type == StepType.tool_call and s.tool_name)


def _already_a_skill(sig: Signature) -> bool:
    """True if a registered skill already encodes this exact tool sequence."""
    target = list(sig)
    return any(skill.metadata.get("tool_sequence") == target for skill in get_skill_registry_snapshot().values())


def _heuristic_draft(sig: Signature, occurrences: int) -> Any:
    """LLM-free draft used when ``judge=False``; deterministic, for offline/test use."""
    Draft = _draft_model()
    name = "-".join(_slugify(t) for t in sig) or "mined-skill"
    steps = "\n".join(f"{i}. Use the `{t}` tool." for i, t in enumerate(sig, start=1))
    return Draft(
        generalizable=True,
        name=name[:60],
        description=f"Reusable procedure: {' then '.join(sig)}.",
        when_to_use=f"Use this when a task needs: {', '.join(sig)}.",
        instructions=f"Follow these steps:\n\n{steps}",
        confidence=0.5,
    )


def _first_user_text(messages: Sequence[dict[str, Any]]) -> str:
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and isinstance(block.get("text"), str) and block["text"].strip():
                    return block["text"].strip()
    return ""


def _render_trace(result: AgentResult) -> str:
    lines: list[str] = []
    n = 0
    for step in result.steps:
        if step.step_type == StepType.tool_call:
            n += 1
            args = step.tool_args if step.tool_args is not None else {}
            lines.append(f"{n}. call {step.tool_name}({_truncate(str(args), 160)})")
        elif step.step_type == StepType.tool_result:
            lines.append(f"   -> {_truncate(_stringify(step.tool_result), _TRACE_RESULT_CHARS)}")
    if lines:
        return "\n".join(lines)
    # Fall back to the flat tool-call list when typed steps are unavailable.
    for i, call in enumerate(result.all_tool_calls, start=1):
        lines.append(f"{i}. call {call.get('name')}({_truncate(str(call.get('arguments', {})), 160)})")
    return "\n".join(lines)


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _truncate(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "mined-skill"


# ------------------------------------------------------------------
# Default extraction bindings (imported lazily to avoid import cycles)
# ------------------------------------------------------------------


def _default_extract(model_cls: Any, **kwargs: Any) -> dict[str, Any]:
    from ..extraction.core import extract_with_model

    return extract_with_model(model_cls, **kwargs)


async def _default_aextract(model_cls: Any, **kwargs: Any) -> dict[str, Any]:
    from ..extraction.async_core import extract_with_model as _aextract

    return await _aextract(model_cls, **kwargs)
