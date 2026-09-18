"""``retrieve_and_verify`` — ground the answer, then check the grounding.

The strategy is four steps: retrieve passages, answer *from those passages only*,
verify that the answer's citations are real and its claims are supported, and
return either a grounded answer or an explicit insufficient-evidence outcome.

The distinction this strategy exists to preserve
------------------------------------------------

"I could not answer from this evidence" is a **successful outcome**, not an
error.  It terminates as ``INSUFFICIENT_EVIDENCE`` with the retrieved sources
still attached, so a caller can tell it apart from ``PROVIDER_ERROR`` (the call
failed) and from ``VALIDATION_FAILED`` (an answer came back in the wrong shape).

Citations are verified structurally before anything else: a source id the model
invented is not evidence.  Claim-level support is a *separate*, optional check —
by default this strategy makes a deterministic structural verdict and does not
spend a second model call to grade itself.  Pass ``faithfulness=True`` (or your
own ``evidence_checker``) to add that, and the result records which checker ran
so a cheap structural pass is never mistaken for a claim-level one.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..outcomes import EvidenceReport, TerminationReason
from ..types import EvidencePassage, ExecutionRequest, ExecutionResult
from .base import (
    ExecutionStrategy,
    StrategyContext,
    StrategyError,
    schema_for,
    validate_against_model,
)

__all__ = ["RetrieveAndVerifyStrategy"]


_GROUNDED_INSTRUCTIONS = """Answer the question using ONLY the numbered passages below.

Rules:
- Cite the passage ids you actually used in "source_ids".
- If the passages do not contain enough information, set "sufficient" to false
  and leave "answer" null. Do not guess, and do not use outside knowledge.
- Never cite a passage id that is not listed below.

Passages:
{passages}

Question: {question}"""


def _answer_schema(inner: dict[str, Any] | None) -> dict[str, Any]:
    """The grounded-answer envelope, wrapping the caller's own schema if any."""
    answer_schema: dict[str, Any] = inner if inner is not None else {"type": ["string", "null"]}
    return {
        "type": "object",
        "properties": {
            "answer": answer_schema,
            "source_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Ids of the passages actually used, copied verbatim.",
            },
            "sufficient": {
                "type": "boolean",
                "description": "False when the passages do not support an answer.",
            },
        },
        "required": ["source_ids", "sufficient"],
    }


class RetrieveAndVerifyStrategy(ExecutionStrategy):
    """Retrieve passages, answer from them, and verify the grounding.

    Args:
        model / driver / driver_factory / options: See
            :class:`~prompture.execution.strategies.base.ExecutionStrategy`.
        retriever: Default retriever used when the request supplies neither
            passages nor its own retriever.  Anything with
            ``retrieve(query, k=...)`` qualifies, including every
            :class:`prompture.rag.Retriever`.
        top_k: Passages to retrieve when retrieval runs.
        min_support: Minimum claim-support fraction required to accept an
            answer, applied only when a claim-level checker ran.  ``None``
            disables the threshold.
        faithfulness: Run :class:`~prompture.eval.FaithfulnessEvaluator` over the
            answer.  Costs extra model calls, which are accounted for like any
            other.
        evidence_checker: Custom ``(answer, passages) -> EvidenceReport``.  Takes
            precedence over ``faithfulness``.
        require_citations: When ``True`` (default) an answer that cites nothing
            is treated as ungrounded.

    Example::

        strategy = RetrieveAndVerifyStrategy(model="openai/gpt-4o-mini", retriever=my_retriever)
        result = strategy.run(ExecutionRequest(task="What is the leave notice period?"))
        if result.termination is TerminationReason.INSUFFICIENT_EVIDENCE:
            print("no grounded answer; sources seen:", result.evidence.sources)
    """

    name = "retrieve_verify"
    version = "1"
    native_mode = "sync"

    def __init__(
        self,
        *,
        retriever: Any = None,
        top_k: int = 4,
        min_support: float | None = None,
        faithfulness: bool = False,
        evidence_checker: Callable[[str, tuple[EvidencePassage, ...]], EvidenceReport] | None = None,
        require_citations: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.retriever = retriever
        self.top_k = top_k
        self.min_support = min_support
        self.faithfulness = faithfulness
        self.evidence_checker = evidence_checker
        self.require_citations = require_citations

    # ------------------------------------------------------------------

    def _execute(self, ctx: StrategyContext) -> ExecutionResult:
        request = ctx.request
        passages = self._gather_passages(ctx)

        if not passages:
            ctx.note("retrieval returned no passages; cannot ground an answer")
            return ctx.finish(
                termination=TerminationReason.INSUFFICIENT_EVIDENCE,
                evidence=EvidenceReport(checked=True, sufficient=False, checker="structural", sources=[]),
            )

        known_ids = {p.id for p in passages}
        driver = self.resolve_driver(ctx.model)
        prompt = _GROUNDED_INSTRUCTIONS.format(
            passages="\n".join(f"[{p.id}] {p.text}" for p in passages),
            question=request.task,
        )

        ctx.stop_if_over_budget(prompt=prompt, model=ctx.model)

        from ...extraction.core import ask_for_json

        envelope = _answer_schema(schema_for(request))
        with ctx.step("generate", detail=f"grounded answer over {len(passages)} passage(s)") as step:
            response = ask_for_json(
                driver,
                prompt,
                envelope,
                ai_cleanup=True,
                model_name=ctx.model,
                options=self.merged_options(request),
                system_prompt=self.system_prompt_for(request),
                strategy=request.structured_output_strategy,
            )
            usage = (response or {}).get("usage") or {}
            ctx.book(step, usage)
            if usage.get("strategy"):
                ctx.structured_output_strategy = str(usage["strategy"])
        payload = (response or {}).get("json_object") or {}

        raw_answer = payload.get("answer")
        cited = [str(s) for s in (payload.get("source_ids") or [])]
        self_reported_sufficient = bool(payload.get("sufficient", True))

        # --- structural verification, before anything expensive ---------
        with ctx.step("evidence_check", detail="structural citation check") as step:
            invalid = [s for s in cited if s not in known_ids]
            valid = [s for s in cited if s in known_ids]
            step.data["cited"] = cited
            step.data["invalid_citations"] = invalid
            evidence = EvidenceReport(
                checked=True,
                sources=valid,
                sufficient=self_reported_sufficient and not invalid,
                checker="structural",
            )
            if invalid:
                # A fabricated citation is the clearest possible ungrounded
                # signal, and it is free to detect.
                ctx.note(f"model cited unknown source id(s): {invalid}")
                step.ok = False

        if invalid:
            evidence.sufficient = False
            return ctx.finish(
                termination=TerminationReason.INSUFFICIENT_EVIDENCE,
                answer=_as_text(raw_answer),
                evidence=evidence,
                error=None,
                raw=response,
            )

        if not self_reported_sufficient or raw_answer is None or _as_text(raw_answer).strip() == "":
            ctx.note("model reported the passages do not support an answer")
            evidence.sufficient = False
            return ctx.finish(
                termination=TerminationReason.INSUFFICIENT_EVIDENCE,
                evidence=evidence,
                raw=response,
            )

        if self.require_citations and not valid:
            ctx.note("answer cited no passage; treating it as ungrounded")
            evidence.sufficient = False
            return ctx.finish(
                termination=TerminationReason.INSUFFICIENT_EVIDENCE,
                answer=_as_text(raw_answer),
                evidence=evidence,
                raw=response,
            )

        # --- optional claim-level verification --------------------------
        evidence = self._claim_check(ctx, _as_text(raw_answer), passages, evidence)
        if evidence.sufficient is False:
            return ctx.finish(
                termination=TerminationReason.INSUFFICIENT_EVIDENCE,
                answer=_as_text(raw_answer),
                evidence=evidence,
                raw=response,
            )

        # --- typed contract, if one was asked for -----------------------
        validation = None
        output: Any = raw_answer
        if request.output_model is not None or request.output_schema is not None:
            with ctx.step("validate") as step:
                instance, validation = validate_against_model(request.output_model, raw_answer)
                step.ok = validation.ok is not False
                step.data["field_errors"] = dict(validation.field_errors)
            if validation.ok is False:
                ctx.note("grounded answer did not satisfy the output contract")
                return ctx.finish(
                    termination=TerminationReason.VALIDATION_FAILED,
                    output=raw_answer,
                    answer=_as_text(raw_answer),
                    validation=validation,
                    evidence=evidence,
                    raw=response,
                )
            output = instance if instance is not None else raw_answer

        ctx.note(f"answer grounded in {len(valid)} passage(s): {valid}")
        return ctx.finish(
            termination=TerminationReason.COMPLETED,
            output=output,
            answer=_as_text(raw_answer),
            validation=validation,
            evidence=evidence,
            raw=response,
        )

    # ---- helpers ------------------------------------------------------

    def _gather_passages(self, ctx: StrategyContext) -> tuple[EvidencePassage, ...]:
        """Use supplied passages, else retrieve.  Retrieval is its own step."""
        request = ctx.request
        if request.passages:
            ctx.note(f"using {len(request.passages)} caller-supplied passage(s); retrieval skipped")
            return tuple(request.passages)

        retriever = request.retriever or self.retriever
        if retriever is None:
            raise StrategyError(
                f"{self.name} needs evidence: supply `passages=` on the request, or a "
                "`retriever=` on the request or the strategy."
            )

        k = request.top_k or self.top_k
        with ctx.step("retrieve", detail=f"top_k={k}") as step:
            ctx.check_cancelled()
            hits = retriever.retrieve(request.task, k=k)
            passages = tuple(EvidencePassage.from_any(h, index=i) for i, h in enumerate(hits or ()))
            step.data["retrieved_ids"] = [p.id for p in passages]
        ctx.note(f"retrieved {len(passages)} passage(s)")
        return passages

    def _claim_check(
        self,
        ctx: StrategyContext,
        answer: str,
        passages: tuple[EvidencePassage, ...],
        evidence: EvidenceReport,
    ) -> EvidenceReport:
        """Run the configured claim-level checker, if any."""
        checker = self.evidence_checker
        if checker is None and not self.faithfulness:
            return evidence

        with ctx.step("evidence_check", detail="claim-level support") as step:
            if checker is not None:
                report = checker(answer, passages)
            else:
                from ...eval import FaithfulnessEvaluator
                from ...eval.base import EvalError

                evaluator = FaithfulnessEvaluator(self.resolve_driver(ctx.model))
                try:
                    outcome = evaluator.evaluate(answer, [p.text for p in passages])
                except EvalError as exc:
                    ctx.note(f"claim-level check could not run: {exc}")
                    step.ok = False
                    step.error = str(exc)
                    return evidence
                ctx.book(step, outcome.meta or {})
                report = EvidenceReport(
                    checked=True,
                    sources=list(evidence.sources),
                    supported_claims=list(outcome.supported),
                    unsupported_claims=list(outcome.unsupported),
                    contradicted_claims=list(outcome.contradicted),
                    support_score=outcome.score,
                    sufficient=True,
                    checker="faithfulness_evaluator",
                )
            report.sources = report.sources or list(evidence.sources)
            step.data["support_score"] = report.support_score

        if (
            self.min_support is not None
            and report.support_score is not None
            and report.support_score < self.min_support
        ):
            report.sufficient = False
            ctx.note(
                f"claim support {report.support_score:.2f} is below the required "
                f"{self.min_support:.2f}; refusing to present the answer as grounded"
            )
        return report


def _as_text(value: Any) -> str:
    """Render a possibly-structured answer as text for the ``answer`` field."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    from .base import compact_json

    return compact_json(value)


def retrieve_and_verify(request: ExecutionRequest, **kwargs: Any) -> ExecutionResult:
    """One-shot convenience wrapper around :class:`RetrieveAndVerifyStrategy`."""
    return RetrieveAndVerifyStrategy(**kwargs).run(request)
