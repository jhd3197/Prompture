"""Turn user corrections into a tested instruction change.

An assistant does not learn from the corrections people give it in chat.  The
usual fix is manual: every week or two, ask the model to summarize the
feedback, paste the summary into the instructions, and hope it helped.  This
module does that loop with evidence:

1. :class:`FeedbackLog` records corrections as people give them.
2. :func:`summarize_feedback` turns corrections into short, deduplicated rules.
3. :func:`propose_from_feedback` writes those rules into a ``## Feedback rules``
   section of the instructions and returns an inert
   :class:`~prompture.execution.improve.candidates.Candidate`.
4. :func:`score_instructions` runs the old and new instructions on the same
   prompts and scores both with an :class:`~prompture.eval.LLMJudge` whose
   rubric includes every rule.
5. :class:`FeedbackLoop` records the development and held-out results in a
   :class:`~prompture.execution.improve.scorecard.Scorecard`, then promotes the
   revision through the :class:`~prompture.execution.improve.promotion.PromotionLedger`
   only if it wins on held-out prompts.  Rollback is one call.

Example::

    from prompture.execution.improve import FeedbackLog, FeedbackLoop, PromotionLedger

    log = FeedbackLog("feedback.jsonl")
    log.add("Cut the length in half.", target=assistant.name)
    log.add("No emojis.", target=assistant.name)

    loop = FeedbackLoop(
        assistant.name,
        base_instructions=assistant.instructions,
        driver=writer_driver,
        judge_driver=judge_driver,
        ledger=PromotionLedger("promotions.json"),
    )
    outcome = loop.run(log.entries(assistant.name), dev_prompts, heldout_prompts)
    print(outcome.scorecard.format())
    if outcome.promoted:
        assistant = assistant.with_instructions(loop.current_instructions())
"""

from __future__ import annotations

import json
import re
import statistics
import threading
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..harness import WorkloadSummary
from .candidates import Candidate, CandidateKind, CandidateStatus, CandidateStore, content_version
from .promotion import PromotionLedger
from .scorecard import Scorecard

__all__ = [
    "FEEDBACK_HEADING",
    "FeedbackEntry",
    "FeedbackLog",
    "FeedbackLoop",
    "FeedbackOutcome",
    "apply_feedback_rules",
    "driver_runner",
    "feedback_judge",
    "feedback_rules",
    "propose_from_feedback",
    "score_instructions",
    "summarize_feedback",
]

#: Section of the instructions that holds rules learned from feedback.
FEEDBACK_HEADING = "## Feedback rules"

#: ``(system_prompt, user_prompt) -> output`` or ``(output, meta)``.
Runner = Callable[[str, str], Any]


# ----------------------------------------------------------------------
# Recording feedback
# ----------------------------------------------------------------------


@dataclass
class FeedbackEntry:
    """One correction, with what prompted it."""

    correction: str
    target: str = ""
    prompt: str = ""
    output: str = ""
    rating: int | None = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeedbackEntry:
        known = {k: data[k] for k in ("correction", "target", "prompt", "output", "rating", "id", "at") if k in data}
        return cls(**known)


class FeedbackLog:
    """An append-only JSON-lines log of corrections.  In memory when ``path`` is ``None``."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self._entries: list[FeedbackEntry] = []
        self._lock = threading.Lock()
        if self.path is not None and self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self._entries.append(FeedbackEntry.from_dict(json.loads(line)))

    def add(
        self,
        correction: str,
        *,
        target: str = "",
        prompt: str = "",
        output: str = "",
        rating: int | None = None,
    ) -> FeedbackEntry:
        """Record a correction such as "too long" or "never use exclamation marks"."""
        if not correction.strip():
            raise ValueError("a correction cannot be empty")
        entry = FeedbackEntry(correction.strip(), target=target, prompt=prompt, output=output, rating=rating)
        with self._lock:
            self._entries.append(entry)
            if self.path is not None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(entry.to_dict()) + "\n")
        return entry

    def entries(self, target: str | None = None, *, since: float | None = None) -> list[FeedbackEntry]:
        """Entries for ``target`` (all targets when ``None``), optionally only those after ``since``."""
        with self._lock:
            return [
                e for e in self._entries if (target is None or e.target == target) and (since is None or e.at > since)
            ]

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


# ----------------------------------------------------------------------
# Rules
# ----------------------------------------------------------------------

SUMMARIZE_PROMPT = """You maintain the instructions of an AI assistant. Users corrected its answers.
Turn the corrections into short, specific, imperative rules the assistant can follow on every answer.

- One rule per distinct correction; merge corrections that say the same thing.
- Keep each rule under 25 words and testable ("Keep posts under 120 words", not "Be concise").
- Skip any correction already covered by an EXISTING RULE.
- Skip corrections that only applied to one specific request.
- Return at most {max_rules} rules.

EXISTING RULES:
{existing}

CORRECTIONS:
{corrections}

Output a JSON array of rule strings with no other text. Output [] if there is nothing new.
"""


def feedback_rules(instructions: str) -> list[str]:
    """Rules already in the ``## Feedback rules`` section of ``instructions``."""
    section = _feedback_section(instructions)
    if section is None:
        return []
    rules = []
    for line in section.splitlines():
        match = re.match(r"^\s*[-*]\s+(.*\S)", line)
        if match:
            rules.append(match.group(1))
    return rules


def apply_feedback_rules(instructions: str, rules: Iterable[str]) -> str:
    """Return ``instructions`` with ``rules`` merged into the ``## Feedback rules`` section.

    Existing rules keep their place; new ones are appended; exact duplicates
    (ignoring case and trailing punctuation) are dropped.  Everything outside
    the section is left untouched, so this never rewrites the author's text.
    """
    existing = feedback_rules(instructions)
    seen = {_rule_key(rule) for rule in existing}
    merged = list(existing)
    for rule in rules:
        rule = rule.strip()
        if rule and _rule_key(rule) not in seen:
            seen.add(_rule_key(rule))
            merged.append(rule)
    if not merged:
        return instructions
    block = FEEDBACK_HEADING + "\n" + "\n".join(f"- {rule}" for rule in merged)
    start, end = _feedback_span(instructions)
    if start < 0:
        return f"{instructions.rstrip()}\n\n{block}\n"
    after = instructions[end:].lstrip("\n")
    return instructions[:start] + block + (f"\n\n{after}" if after.strip() else "\n")


def summarize_feedback(
    entries: Iterable[FeedbackEntry | str],
    driver: Any,
    *,
    existing_rules: Iterable[str] = (),
    max_rules: int = 12,
    llm_options: dict[str, Any] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Ask ``driver`` to turn corrections into new rules.  Returns ``(rules, meta)``."""
    from ...eval.base import _accumulate_meta
    from ...eval.faithfulness import _parse_claims

    corrections = [e.correction if isinstance(e, FeedbackEntry) else str(e) for e in entries]
    corrections = [c.strip() for c in corrections if c and c.strip()]
    meta: dict[str, Any] = {}
    if not corrections:
        return [], meta
    existing = list(existing_rules)
    prompt = SUMMARIZE_PROMPT.format(
        max_rules=max_rules,
        existing="\n".join(f"- {r}" for r in existing) or "(none)",
        corrections="\n".join(f"- {c}" for c in corrections),
    )
    response = driver.generate(prompt, dict(llm_options or {"temperature": 0.0, "max_tokens": 800}))
    text = response.get("text", "") if isinstance(response, dict) else str(response or "")
    if isinstance(response, dict):
        _accumulate_meta(meta, response.get("meta"))
    known = {_rule_key(r) for r in existing}
    rules = [r for r in _parse_claims(text) if _rule_key(r) not in known]
    return rules[:max_rules], meta


def propose_from_feedback(
    target: str,
    base_instructions: str,
    entries: Iterable[FeedbackEntry],
    driver: Any,
    *,
    base_version: str = "",
    max_rules: int = 12,
) -> Candidate | None:
    """Build a prompt candidate from corrections.  ``None`` when they add no new rule."""
    entries = list(entries)
    existing = feedback_rules(base_instructions)
    new_rules, meta = summarize_feedback(entries, driver, existing_rules=existing, max_rules=max_rules)
    if not new_rules:
        return None
    proposed = apply_feedback_rules(base_instructions, new_rules)
    if proposed == base_instructions:
        return None
    return Candidate(
        target=target,
        kind=CandidateKind.PROMPT,
        content=proposed,
        base_content=base_instructions,
        base_version=base_version or content_version(base_instructions),
        origin="feedback",
        motivating_task_ids=[e.id for e in entries],
        rationale=f"{len(entries)} correction(s) summarized into {len(new_rules)} new rule(s)",
        metadata={"new_rules": new_rules, "rules": feedback_rules(proposed), "summary_meta": meta},
    )


# ----------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------


def driver_runner(driver: Any, options: dict[str, Any] | None = None) -> Runner:
    """A runner that sends the instructions as the system message."""
    opts = dict(options or {})

    def run(system_prompt: str, prompt: str) -> tuple[str, dict[str, Any]]:
        if hasattr(driver, "generate_messages"):
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
            response = driver.generate_messages(messages, dict(opts))
        else:
            response = driver.generate(f"{system_prompt}\n\n{prompt}", dict(opts))
        if isinstance(response, dict):
            return str(response.get("text") or ""), dict(response.get("meta") or {})
        return str(response or ""), {}

    return run


def feedback_judge(driver: Any, rules: Iterable[str], *, scale: int = 5, extra_criteria: Iterable[str] = ()) -> Any:
    """An :class:`~prompture.eval.LLMJudge` that checks the request was done well and every rule was followed."""
    from ...eval import LLMJudge

    criteria = ["task: the output does what the request asks, well", *extra_criteria]
    criteria.extend(f"rule_{i}: {rule}" for i, rule in enumerate(rules, start=1))
    return LLMJudge(driver, criteria=criteria, scale=scale)


def score_instructions(
    instructions: str,
    prompts: Iterable[str],
    *,
    run: Runner,
    judge: Any,
    workload: str = "",
    split: str = "dev",
    label: str = "",
) -> WorkloadSummary:
    """Run ``instructions`` on every prompt and score each output with ``judge``.

    Scores are normalized to 0..1 (``(score - 1) / (scale - 1)``) so they
    compare with the rest of the harness.  Cost covers both the runs and the
    judge calls; ``cost_complete`` is ``False`` if any call reported no cost.
    """
    prompts = list(prompts)
    scores: list[float] = []
    latencies: list[float] = []
    total_cost = 0.0
    calls = 0
    unpriced = 0
    failures: dict[str, int] = {}

    def account(meta: dict[str, Any] | None) -> None:
        nonlocal total_cost, calls, unpriced
        calls += int((meta or {}).get("calls", 1))
        cost = (meta or {}).get("cost")
        if isinstance(cost, (int, float)):
            total_cost += float(cost)
        else:
            unpriced += 1

    for prompt in prompts:
        started = time.perf_counter()
        try:
            result = run(instructions, prompt)
        except Exception as exc:  # a failed run scores nothing but is counted
            failures[type(exc).__name__] = failures.get(type(exc).__name__, 0) + 1
            continue
        latencies.append((time.perf_counter() - started) * 1000)
        output, meta = result if isinstance(result, tuple) else (result, {})
        account(meta)
        try:
            verdict = judge.evaluate(input_text=prompt, output=str(output))
        except Exception as exc:
            failures[f"judge_{type(exc).__name__}"] = failures.get(f"judge_{type(exc).__name__}", 0) + 1
            continue
        account(verdict.meta)
        scores.append((verdict.score - 1) / (judge.scale - 1))

    latencies.sort()
    summary = WorkloadSummary(
        workload=workload,
        split=split,
        label=label,
        n_cases=len(prompts),
        runs=len(prompts),
        scored_cases=len(scores),
        mean_score=statistics.fmean(scores) if scores else None,
        score_stddev=statistics.pstdev(scores) if len(scores) > 1 else None,
        mean_latency_ms=statistics.fmean(latencies) if latencies else None,
        p95_latency_ms=latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))] if latencies else None,
        total_cost=total_cost,
        mean_cost_per_task=total_cost / len(scores) if scores else None,
        total_calls=calls,
        unpriced_calls=unpriced,
        cost_complete=unpriced == 0,
        termination_counts=failures,
    )
    if failures:
        summary.incomplete_measurements.append(
            f"{sum(failures.values())} prompt(s) failed to run or be judged and were not scored"
        )
    return summary


# ----------------------------------------------------------------------
# The loop
# ----------------------------------------------------------------------


@dataclass
class FeedbackOutcome:
    """What one pass of :meth:`FeedbackLoop.run` did."""

    candidate: Candidate | None
    scorecard: Scorecard | None
    promoted: bool
    reason: str


class FeedbackLoop:
    """Propose, test and (maybe) promote instruction changes for one assistant.

    Args:
        target: Ledger key, usually the assistant's name.
        base_instructions: Recorded as the baseline when the ledger has no
            active version for ``target`` yet.
        driver: Writes the outputs being judged and summarizes feedback.
        judge_driver: Scores outputs.  Defaults to ``driver``; a stronger model is better.
        ledger: Where the active version lives.  In memory when omitted.
        store: Keeps every candidate, including rejected ones.
        run: Overrides how outputs are produced (defaults to :func:`driver_runner`).
        min_gain: Minimum held-out score gain (0..1 scale) needed to promote.
            Positive by default: a revision that does not measurably help is
            not worth a longer prompt.
        scale: Judge score scale.
    """

    def __init__(
        self,
        target: str,
        *,
        base_instructions: str,
        driver: Any,
        judge_driver: Any = None,
        ledger: PromotionLedger | None = None,
        store: CandidateStore | None = None,
        run: Runner | None = None,
        min_gain: float = 0.01,
        scale: int = 5,
    ) -> None:
        self.target = target
        self.driver = driver
        self.judge_driver = judge_driver or driver
        self.ledger = ledger or PromotionLedger()
        self.store = store or CandidateStore()
        self.run_fn = run or driver_runner(driver)
        self.min_gain = min_gain
        self.scale = scale
        if self.ledger.active(target) is None:
            self.ledger.set_baseline(target, base_instructions)

    def current_instructions(self) -> str:
        """The active instructions for this target."""
        return str(self.ledger.active_content(self.target))

    def propose(self, entries: Iterable[FeedbackEntry]) -> Candidate | None:
        active = self.ledger.active(self.target)
        assert active is not None
        candidate = propose_from_feedback(
            self.target, str(active.content), entries, self.driver, base_version=active.version
        )
        if candidate is not None:
            self.store.add(candidate)
        return candidate

    def evaluate(self, candidate: Candidate, dev_prompts: Iterable[str], heldout_prompts: Iterable[str]) -> Scorecard:
        """Score baseline and candidate on both splits with a judge that knows every rule."""
        judge = feedback_judge(self.judge_driver, candidate.metadata.get("rules") or [], scale=self.scale)
        card = Scorecard(
            candidate_id=candidate.id,
            target=self.target,
            candidate_version=candidate.version,
            base_version=candidate.base_version,
        )
        base = str(candidate.base_content)
        dev_prompts, heldout_prompts = list(dev_prompts), list(heldout_prompts)
        if not heldout_prompts:
            raise ValueError(
                "held-out prompts are required: promotion is decided on prompts the rules were not tuned on"
            )
        for split, prompts, record in (
            ("dev", dev_prompts, card.record_dev),
            ("heldout", heldout_prompts, card.record_heldout),
        ):
            if not prompts:
                continue
            baseline = score_instructions(
                base, prompts, run=self.run_fn, judge=judge, workload=self.target, split=split, label="baseline"
            )
            revised = score_instructions(
                str(candidate.content),
                prompts,
                run=self.run_fn,
                judge=judge,
                workload=self.target,
                split=split,
                label="candidate",
            )
            record(baseline, revised)
        candidate.status = CandidateStatus.EVALUATED
        self.store.update(candidate)
        return card

    def decide(self, candidate: Candidate, scorecard: Scorecard) -> tuple[bool, str]:
        """Promote when the held-out gain clears ``min_gain``; otherwise reject and keep the record."""
        ok, reason = scorecard.promotable(min_gain=self.min_gain)
        if ok:
            self.ledger.promote(candidate, scorecard, min_gain=self.min_gain)
        else:
            self.ledger.reject(candidate, scorecard, reason=reason)
        self.store.update(candidate)
        return ok, reason

    def rollback(self, reason: str = "manual rollback") -> str:
        """Restore the previous instructions and return them."""
        return str(self.ledger.rollback(self.target, reason=reason).content)

    def run(
        self,
        entries: Iterable[FeedbackEntry],
        dev_prompts: Iterable[str],
        heldout_prompts: Iterable[str],
    ) -> FeedbackOutcome:
        """Propose from ``entries``, evaluate, and promote or reject in one call."""
        candidate = self.propose(entries)
        if candidate is None:
            return FeedbackOutcome(None, None, False, "the feedback adds no rule the instructions do not already have")
        scorecard = self.evaluate(candidate, dev_prompts, heldout_prompts)
        promoted, reason = self.decide(candidate, scorecard)
        return FeedbackOutcome(candidate, scorecard, promoted, reason)


def _rule_key(rule: str) -> str:
    return re.sub(r"[\s.!;:,]+$", "", rule.strip().lower())


def _feedback_span(instructions: str) -> tuple[int, int]:
    match = re.search(rf"^{re.escape(FEEDBACK_HEADING)}[ \t]*$", instructions, re.MULTILINE)
    if match is None:
        return -1, -1
    rest = instructions[match.end() :]
    next_heading = re.search(r"^#{1,2} ", rest, re.MULTILINE)
    end = match.end() + (next_heading.start() if next_heading else len(rest))
    return match.start(), end


def _feedback_section(instructions: str) -> str | None:
    start, end = _feedback_span(instructions)
    if start < 0:
        return None
    return instructions[start:end]
