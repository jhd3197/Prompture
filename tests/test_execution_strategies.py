"""Phase B contract checks for the three fixed execution strategies.

Every test drives a scripted stub driver, so the suite runs offline and asserts
on *behaviour*, not on model quality: which steps ran, how usage was aggregated,
what termination reason came back, and whether existing callers still see their
old return types.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from pydantic import BaseModel

from prompture.drivers.base import Driver
from prompture.execution import (
    CancellationToken,
    EvidencePassage,
    ExecutionRequest,
    ExecutionResult,
    ResourceLimits,
    TaskCategory,
    TerminationReason,
    compare_strategies,
)
from prompture.execution.outcomes import EvidenceReport
from prompture.execution.strategies import (
    DirectStrategy,
    DraftAndCritiqueStrategy,
    RetrieveAndVerifyStrategy,
    StrategyError,
    get_strategy,
    list_strategies,
    register_strategy,
    unregister_strategy,
)
from prompture.execution.strategies.base import ExecutionStrategy

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class Contact(BaseModel):
    name: str | None = None
    email: str | None = None
    company: str | None = None
    role: str | None = None
    phone: str | None = None


class StrictContact(BaseModel):
    name: str
    age: int


class _StubDriver(Driver):
    """Returns scripted responses in order; records every prompt it saw.

    Subclasses the real :class:`~prompture.drivers.base.Driver` so the strategies
    exercise the same ``generate_with_hooks`` path production code takes.

    ``responses`` entries may be a plain dict (serialised to JSON, i.e. a
    structured answer), a string (raw text), or an ``Exception`` to raise.
    """

    supports_json_mode = False
    supports_json_schema = False
    supports_tool_use = False
    model = "stub/model"

    def __init__(self, responses, *, cost=0.001, priced=True):
        self._responses = list(responses)
        self.prompts: list[str] = []
        self.options: list[dict] = []
        self._cost = cost
        self._priced = priced

    def generate(self, prompt, options):
        self.prompts.append(prompt)
        self.options.append(dict(options or {}))
        if not self._responses:
            raise AssertionError("stub driver called more times than scripted")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        text = item if isinstance(item, str) else json.dumps(item)
        meta = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        if self._priced:
            meta["cost"] = self._cost
        return {"text": text, "meta": meta}


def _direct(driver, **kwargs):
    return DirectStrategy(driver=driver, model="stub/model", **kwargs)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_the_three_roadmap_strategies_are_registered():
    assert list_strategies() == ["direct", "draft_critique", "retrieve_verify"]


def test_get_strategy_builds_a_configured_instance():
    strategy = get_strategy("direct", model="openai/gpt-4o-mini", max_repairs=0)
    assert isinstance(strategy, DirectStrategy)
    assert strategy.model == "openai/gpt-4o-mini"
    assert strategy.max_repairs == 0


def test_custom_strategies_can_be_registered_and_removed():
    class _Noop(ExecutionStrategy):
        name = "noop"

        def _execute(self, ctx):
            return ctx.finish(termination=TerminationReason.COMPLETED, answer="noop")

    register_strategy("noop", _Noop)
    try:
        assert isinstance(get_strategy("noop"), _Noop)
        with pytest.raises(ValueError, match="already registered"):
            register_strategy("noop", _Noop)
    finally:
        assert unregister_strategy("noop") is True
    with pytest.raises(KeyError):
        get_strategy("noop")


# ---------------------------------------------------------------------------
# direct
# ---------------------------------------------------------------------------


def test_direct_returns_a_validated_instance_and_one_generate_step():
    driver = _StubDriver([{"name": "Ana", "email": "ana@x.test"}])
    result = _direct(driver).run(ExecutionRequest(task="Ana is ana@x.test", output_model=Contact))

    assert result.termination is TerminationReason.COMPLETED
    assert result.ok is True
    assert isinstance(result.output, Contact)
    assert result.output.name == "Ana"
    assert result.validation.checked is True
    assert result.validation.schema_valid is True
    assert [s.kind for s in result.steps] == ["generate", "validate"]
    assert result.strategy == "direct"


def test_direct_aggregates_usage_across_the_repair_pass():
    driver = _StubDriver([{"name": "Ana"}, {"name": "Ana", "age": 30}])
    result = _direct(driver).run(ExecutionRequest(task="Ana is 30", output_model=StrictContact))

    assert result.termination is TerminationReason.COMPLETED
    assert result.usage.call_count == 2, "both the first attempt and the repair must be counted"
    assert result.usage.cost == pytest.approx(0.002)
    assert result.usage.cost_complete is True
    assert [s.kind for s in result.steps] == ["generate", "validate", "repair", "validate"]


def test_direct_repair_prompt_names_the_failing_fields():
    driver = _StubDriver([{"name": "Ana"}, {"name": "Ana", "age": 30}])
    _direct(driver).run(ExecutionRequest(task="Ana is 30", output_model=StrictContact))

    repair_prompt = driver.prompts[1]
    assert "age" in repair_prompt
    assert "Fix only the listed fields" in repair_prompt


def test_direct_terminates_as_validation_failed_when_repairs_run_out():
    driver = _StubDriver([{"name": "Ana"}, {"name": "Ana"}])
    result = _direct(driver, max_repairs=1).run(ExecutionRequest(task="Ana", output_model=StrictContact))

    assert result.termination is TerminationReason.VALIDATION_FAILED
    assert result.ok is False
    assert result.validation.schema_valid is False
    assert "age" in result.validation.field_errors
    assert result.validation.repair_attempts == 1


def test_direct_with_zero_repairs_makes_exactly_one_attempt():
    driver = _StubDriver([{"name": "Ana"}])
    result = _direct(driver, max_repairs=0).run(ExecutionRequest(task="Ana", output_model=StrictContact))

    assert result.termination is TerminationReason.VALIDATION_FAILED
    assert result.usage.call_count == 1


def test_direct_without_a_schema_makes_no_validity_claim():
    driver = _StubDriver(["a plain english answer"])
    result = _direct(driver).run(ExecutionRequest(task="say something"))

    assert result.answer == "a plain english answer"
    assert result.validation.checked is False
    assert result.validation.ok is None
    assert any("schema validity is unchecked" in d for d in result.decisions)


def test_direct_captures_a_provider_failure_as_a_termination_not_an_exception():
    driver = _StubDriver([RuntimeError("upstream 503")])
    result = _direct(driver).run(ExecutionRequest(task="x", output_model=Contact))

    assert result.termination is TerminationReason.PROVIDER_ERROR
    assert "upstream 503" in (result.error or "")
    assert result.ok is False


def test_a_misconfigured_strategy_raises_rather_than_returning_a_result():
    strategy = DirectStrategy()  # no model, no driver
    with pytest.raises(StrategyError, match="needs a model"):
        strategy.run(ExecutionRequest(task="x", output_model=Contact))


# ---------------------------------------------------------------------------
# Budgets, cancellation, accounting
# ---------------------------------------------------------------------------


def test_budget_stops_the_repair_pass_before_it_starts():
    driver = _StubDriver([{"name": "Ana"}, {"name": "Ana", "age": 30}])
    result = _direct(driver, max_repairs=3).run(
        ExecutionRequest(
            task="Ana",
            output_model=StrictContact,
            limits=ResourceLimits(max_llm_calls=1),
        )
    )

    assert result.termination is TerminationReason.BUDGET_EXHAUSTED
    assert result.usage.call_count == 1, "the second call must never have been dispatched"
    assert "call limit reached" in (result.error or "")


def test_budget_note_reports_that_an_in_flight_call_is_still_billed():
    driver = _StubDriver([{"name": "Ana", "email": "a@b.test"}])
    result = _direct(driver).run(
        ExecutionRequest(task="x", output_model=Contact, limits=ResourceLimits(max_llm_calls=5))
    )

    assert "in-flight call is still billed" in result.budget_note


def test_unpriced_calls_make_the_result_cost_a_lower_bound():
    driver = _StubDriver([{"name": "Ana"}], priced=False)
    result = _direct(driver).run(ExecutionRequest(task="x", output_model=Contact))

    assert result.usage.cost_complete is False
    assert result.usage.unpriced_calls == 1
    assert "LOWER BOUND" in result.explain()


def test_cancellation_before_the_first_call_terminates_as_cancelled():
    token = CancellationToken()
    token.cancel("user stopped it")
    driver = _StubDriver([{"name": "Ana"}])

    result = _direct(driver).run(ExecutionRequest(task="x", output_model=Contact, cancel=token))

    assert result.termination is TerminationReason.CANCELLED
    assert "user stopped it" in (result.error or "")
    assert result.usage.call_count == 0


def test_explain_lists_the_steps_and_the_decision_reasons():
    driver = _StubDriver([{"name": "Ana", "email": "a@b.test"}])
    result = _direct(driver).run(ExecutionRequest(task="x", output_model=Contact))

    explanation = result.explain()
    assert "strategy=direct" in explanation
    assert "generate" in explanation and "validate" in explanation
    assert "validated on attempt 0" in explanation


# ---------------------------------------------------------------------------
# retrieve_verify
# ---------------------------------------------------------------------------


_PASSAGES = (
    EvidencePassage(id="p1", text="Planned leave needs 14 days of notice."),
    EvidencePassage(id="p2", text="Expense claims are due within 30 days."),
)


class _StubRetriever:
    def __init__(self, passages):
        self.passages = passages
        self.queries: list[str] = []

    def retrieve(self, query, k=4):
        self.queries.append(query)
        return list(self.passages)[:k]


def test_retrieve_verify_grounds_an_answer_and_records_its_sources():
    driver = _StubDriver([{"answer": "14 days", "source_ids": ["p1"], "sufficient": True}])
    strategy = RetrieveAndVerifyStrategy(driver=driver, model="stub/model")

    result = strategy.run(
        ExecutionRequest(task="How much notice?", category=TaskCategory.DOCUMENT_QA, passages=_PASSAGES)
    )

    assert result.termination is TerminationReason.COMPLETED
    assert result.answer == "14 days"
    assert result.evidence.checked is True
    assert result.evidence.sources == ["p1"]
    assert result.evidence.checker == "structural"
    assert [s.kind for s in result.steps] == ["generate", "evidence_check"]


def test_retrieve_verify_runs_the_retriever_when_no_passages_are_supplied():
    retriever = _StubRetriever(_PASSAGES)
    driver = _StubDriver([{"answer": "14 days", "source_ids": ["p1"], "sufficient": True}])
    strategy = RetrieveAndVerifyStrategy(driver=driver, model="stub/model", retriever=retriever)

    result = strategy.run(ExecutionRequest(task="How much notice?", top_k=2))

    assert retriever.queries == ["How much notice?"]
    assert result.steps[0].kind == "retrieve"
    assert result.steps[0].data["retrieved_ids"] == ["p1", "p2"]


def test_insufficient_evidence_is_a_distinct_outcome_from_an_error():
    driver = _StubDriver([{"answer": None, "source_ids": [], "sufficient": False}])
    strategy = RetrieveAndVerifyStrategy(driver=driver, model="stub/model")

    result = strategy.run(ExecutionRequest(task="Sabbatical policy?", passages=_PASSAGES))

    assert result.termination is TerminationReason.INSUFFICIENT_EVIDENCE
    assert result.abstained is True
    assert result.error is None, "an honest refusal is not an error"
    assert result.evidence.sufficient is False


def test_a_fabricated_citation_is_rejected_as_ungrounded():
    driver = _StubDriver([{"answer": "42 days", "source_ids": ["p9"], "sufficient": True}])
    strategy = RetrieveAndVerifyStrategy(driver=driver, model="stub/model")

    result = strategy.run(ExecutionRequest(task="How much notice?", passages=_PASSAGES))

    assert result.termination is TerminationReason.INSUFFICIENT_EVIDENCE
    assert result.evidence.sources == []
    assert any("unknown source id" in d for d in result.decisions)


def test_an_answer_with_no_citation_is_ungrounded_by_default():
    driver = _StubDriver([{"answer": "14 days", "source_ids": [], "sufficient": True}])
    strategy = RetrieveAndVerifyStrategy(driver=driver, model="stub/model")

    result = strategy.run(ExecutionRequest(task="How much notice?", passages=_PASSAGES))
    assert result.termination is TerminationReason.INSUFFICIENT_EVIDENCE

    lenient = RetrieveAndVerifyStrategy(
        driver=_StubDriver([{"answer": "14 days", "source_ids": [], "sufficient": True}]),
        model="stub/model",
        require_citations=False,
    )
    assert (
        lenient.run(ExecutionRequest(task="How much notice?", passages=_PASSAGES)).termination
        is TerminationReason.COMPLETED
    )


def test_empty_retrieval_abstains_without_calling_the_model():
    driver = _StubDriver([])
    strategy = RetrieveAndVerifyStrategy(driver=driver, model="stub/model", retriever=_StubRetriever([]))

    result = strategy.run(ExecutionRequest(task="anything"))

    assert result.termination is TerminationReason.INSUFFICIENT_EVIDENCE
    assert result.usage.call_count == 0


def test_retrieve_verify_without_evidence_or_retriever_is_a_configuration_error():
    strategy = RetrieveAndVerifyStrategy(driver=_StubDriver([]), model="stub/model")
    with pytest.raises(StrategyError, match="needs evidence"):
        strategy.run(ExecutionRequest(task="anything"))


def test_a_claim_level_checker_below_min_support_refuses_the_answer():
    def checker(answer, passages):
        return EvidenceReport(
            checked=True,
            support_score=0.25,
            unsupported_claims=["most of it"],
            sufficient=True,
            checker="fake",
        )

    driver = _StubDriver([{"answer": "14 days and a pony", "source_ids": ["p1"], "sufficient": True}])
    strategy = RetrieveAndVerifyStrategy(driver=driver, model="stub/model", evidence_checker=checker, min_support=0.8)

    result = strategy.run(ExecutionRequest(task="How much notice?", passages=_PASSAGES))

    assert result.termination is TerminationReason.INSUFFICIENT_EVIDENCE
    assert result.evidence.checker == "fake"
    assert result.evidence.support_score == pytest.approx(0.25)
    assert any("below the required" in d for d in result.decisions)


def test_a_typed_contract_is_validated_on_top_of_the_grounded_answer():
    driver = _StubDriver([{"answer": {"name": "Ana", "email": "a@b.test"}, "source_ids": ["p1"], "sufficient": True}])
    strategy = RetrieveAndVerifyStrategy(driver=driver, model="stub/model")

    result = strategy.run(ExecutionRequest(task="who?", passages=_PASSAGES, output_model=Contact))

    assert result.termination is TerminationReason.COMPLETED
    assert isinstance(result.output, Contact)
    assert result.validation.schema_valid is True


# ---------------------------------------------------------------------------
# draft_critique
# ---------------------------------------------------------------------------


def _critique(driver, **kwargs):
    return DraftAndCritiqueStrategy(driver=driver, model="stub/model", **kwargs)


def test_draft_critique_stops_as_soon_as_the_reviewer_approves():
    driver = _StubDriver(
        [
            "first draft",
            {"approved": True, "score": 9, "issues": []},
        ]
    )
    result = _critique(driver).run(ExecutionRequest(task="write something"))

    assert result.termination is TerminationReason.COMPLETED
    assert result.answer == "first draft"
    assert [s.kind for s in result.steps] == ["generate", "review"]
    assert result.usage.call_count == 2


def test_draft_critique_revises_with_the_reviewers_issues():
    driver = _StubDriver(
        [
            "weak draft",
            {"approved": False, "score": 4, "issues": ["missing the conclusion"]},
            "better draft",
            {"approved": True, "score": 9, "issues": []},
        ]
    )
    result = _critique(driver).run(ExecutionRequest(task="write something"))

    assert result.termination is TerminationReason.COMPLETED
    assert result.answer == "better draft"
    assert [s.kind for s in result.steps] == ["generate", "review", "revise", "review"]
    assert "missing the conclusion" in driver.prompts[2]
    assert result.usage.call_count == 4


def test_an_unapproved_run_terminates_as_review_rejected_and_keeps_the_issues():
    driver = _StubDriver(
        [
            "d1",
            {"approved": False, "score": 3, "issues": ["still wrong"]},
            "d2",
            {"approved": False, "score": 4, "issues": ["still wrong"]},
        ]
    )
    result = _critique(driver, max_iterations=2).run(ExecutionRequest(task="write something"))

    assert result.termination is TerminationReason.REVIEW_REJECTED
    assert result.ok is False
    assert result.artifacts["review_issues"] == ["still wrong"]
    assert result.answer == "d2", "the last draft is still available for inspection"


def test_accept_unapproved_opts_into_returning_the_last_draft():
    driver = _StubDriver(["d1", {"approved": False, "score": 3, "issues": ["meh"]}])
    result = _critique(driver, max_iterations=1, accept_unapproved=True).run(ExecutionRequest(task="write something"))

    assert result.termination is TerminationReason.COMPLETED
    assert result.answer == "d1"
    assert result.artifacts["review_issues"] == ["meh"]


def test_min_score_can_override_a_generous_approval():
    driver = _StubDriver(["d1", {"approved": True, "score": 5, "issues": []}])
    result = _critique(driver, max_iterations=1, min_score=8).run(ExecutionRequest(task="write something"))

    assert result.termination is TerminationReason.REVIEW_REJECTED


def test_draft_critique_notes_that_self_review_is_not_independent():
    driver = _StubDriver(["d1", {"approved": True, "score": 9, "issues": []}])
    result = _critique(driver).run(ExecutionRequest(task="x"))

    assert any("not an independent check" in d for d in result.decisions)


def test_draft_critique_stops_on_budget_and_keeps_the_last_draft():
    driver = _StubDriver(
        [
            "d1",
            {"approved": False, "score": 3, "issues": ["fix it"]},
            "d2",
            {"approved": True, "score": 9, "issues": []},
        ]
    )
    result = _critique(driver, max_iterations=3).run(ExecutionRequest(task="x", limits=ResourceLimits(max_llm_calls=2)))

    assert result.termination is TerminationReason.BUDGET_EXHAUSTED
    assert result.answer == "d1"
    assert result.usage.call_count == 2


def test_draft_critique_is_async_native_and_run_refuses_a_running_loop():
    async def inner():
        driver = _StubDriver(["d1", {"approved": True, "score": 9, "issues": []}])
        strategy = _critique(driver)

        awaited = await strategy.arun(ExecutionRequest(task="x"))
        assert awaited.termination is TerminationReason.COMPLETED

        with pytest.raises(StrategyError, match="async-native"):
            strategy.run(ExecutionRequest(task="x"))

    asyncio.run(inner())


def test_sync_native_strategies_are_awaitable_too():
    async def inner():
        driver = _StubDriver([{"name": "Ana", "email": "a@b.test"}])
        result = await _direct(driver).arun(ExecutionRequest(task="x", output_model=Contact))
        assert result.termination is TerminationReason.COMPLETED

    asyncio.run(inner())


# ---------------------------------------------------------------------------
# Result envelope → outcome record
# ---------------------------------------------------------------------------


def test_result_projects_onto_an_outcome_record_without_inventing_correctness():
    driver = _StubDriver([{"name": "Ana", "email": "a@b.test"}])
    result = _direct(driver).run(ExecutionRequest(task="x", output_model=Contact))

    outcome = result.to_outcome(task_id="case-1", category=TaskCategory.EXTRACTION)

    assert outcome.task_id == "case-1"
    assert outcome.strategy == "direct"
    assert outcome.correct is None, "a strategy cannot know whether its own answer was right"
    assert outcome.validation.schema_valid is True
    assert outcome.usage.cost_complete is True
    assert outcome.metadata["steps"][0]["kind"] == "generate"


# ---------------------------------------------------------------------------
# Integration surfaces
# ---------------------------------------------------------------------------


def test_compare_strategies_isolates_each_run_from_the_others():
    request = ExecutionRequest(task="Ana is ana@x.test", output_model=Contact, passages=_PASSAGES)
    results = compare_strategies(
        request,
        {
            "direct": _direct(_StubDriver([{"name": "Ana", "email": "ana@x.test"}])),
            "retrieve": RetrieveAndVerifyStrategy(
                driver=_StubDriver([{"answer": {"name": "Ana"}, "source_ids": ["p1"], "sufficient": True}]),
                model="stub/model",
            ),
        },
    )

    assert set(results) == {"direct", "retrieve"}
    assert all(isinstance(r, ExecutionResult) for r in results.values())
    assert results["direct"].strategy == "direct"
    assert results["retrieve"].strategy == "retrieve_verify"


def test_assistant_gains_strategy_entry_points_without_changing_arun():
    from prompture import Assistant, Persona

    assistant = Assistant(
        name="policy-reader",
        persona=Persona(name="reader", system_prompt="You read {{doc_type}} documents."),
        model="stub/model",
        variables={"doc_type": "policy"},
    )

    request = assistant.execution_request("What is the notice period?")
    assert request.model == "stub/model"
    assert request.variables["doc_type"] == "policy"
    assert "policy documents" in (request.persona.render(**request.variables))

    driver = _StubDriver(["14 days"])
    result = assistant.run_strategy("What is the notice period?", strategy=_direct(driver))
    assert result.answer == "14 days"
    # The original API is untouched.
    assert hasattr(assistant, "arun")


def test_assistant_skills_reach_the_strategy_persona():
    from prompture import Assistant, Persona
    from prompture.agents.skills import SkillInfo

    assistant = Assistant(
        name="with-skill",
        persona=Persona(name="base", system_prompt="Base prompt."),
        skills=(SkillInfo(name="citing", description="Cite sources", instructions="Always cite."),),
        model="stub/model",
    )

    rendered = assistant.execution_request("x").persona.render()
    assert "Base prompt." in rendered
    assert "Always cite." in rendered


def test_coding_agent_assistants_are_refused_with_a_clear_message():
    from prompture import Assistant, Persona

    assistant = Assistant(
        name="cli",
        persona=Persona(name="p", system_prompt="x"),
        coding_agent="claude",
    )
    with pytest.raises(ValueError, match="coding_agent"):
        assistant.execution_request("x")


def test_a_strategy_node_reports_an_unresolvable_model_instead_of_raising():
    """A node that cannot reach a provider fails as a NodeResult, not an exception."""
    from prompture.workflow import Graph, Node, run_graph

    graph = Graph(id="wf")
    graph.add_node(
        Node(
            id="step",
            type="strategy",
            config={"strategy": "direct", "task": "say something", "model": "no-such-provider/x"},
        )
    )
    graph.outputs = {"answer": "{{step.outputs.answer}}"}

    run = run_graph(graph)

    assert run.ok is False
    node = run.node_results["step"]
    assert node.status.value == "failed"
    assert node.error, "the node must carry the reason it failed"


def test_strategy_workflow_node_accepts_a_prebuilt_strategy_instance():
    from prompture.workflow import Graph, Node, run_graph

    driver = _StubDriver(["a workflow answer"])
    graph = Graph(id="wf")
    graph.add_node(
        Node(
            id="step",
            type="strategy",
            config={"task": "say something"},
        )
    )
    # Pre-build the strategy and hand it to the node via config.
    graph.node("step").config["strategy"] = _direct(driver)
    graph.outputs = {"answer": "{{step.outputs.answer}}"}

    run = run_graph(graph)

    assert run.ok is True
    assert run.outputs["answer"] == "a workflow answer"
    assert run.node_results["step"].outputs["termination"] == "completed"
    assert run.total_usage["cost"] == pytest.approx(0.001)


def test_an_abstaining_strategy_node_completes_so_a_graph_can_branch_on_it():
    """"No grounded answer" is an outcome, not a crash — the node completes."""
    from prompture.workflow import Graph, Node, run_graph

    driver = _StubDriver([{"answer": None, "source_ids": [], "sufficient": False}])
    strategy = RetrieveAndVerifyStrategy(driver=driver, model="stub/model")

    graph = Graph(id="wf")
    graph.add_node(
        Node(
            id="ask",
            type="strategy",
            config={
                "strategy": strategy,
                "task": "What is the sabbatical policy?",
                "passages": [{"id": "p1", "text": "Leave needs 14 days of notice."}],
            },
        )
    )
    graph.outputs = {"ok": "{{ask.outputs.ok}}", "why": "{{ask.outputs.termination}}"}

    run = run_graph(graph)

    assert run.ok is True, "an abstention must not fail the graph"
    assert run.outputs["ok"] is False
    assert run.outputs["why"] == "insufficient_evidence"


def test_a_failed_strategy_node_skips_whatever_depended_on_it():
    from prompture.workflow import Graph, Node, run_graph

    failing = _direct(_StubDriver([RuntimeError("provider down")]))
    graph = Graph(id="wf")
    graph.add_node(Node(id="first", type="strategy", config={"strategy": failing, "task": "x"}))
    graph.add_node(
        Node(id="second", type="strategy", config={"strategy": failing, "task": "{{first.outputs.answer}}"})
    )

    run = run_graph(graph)

    assert run.ok is False
    assert run.node_results["first"].status.value == "failed"
    assert run.node_results["second"].status.value == "skipped", "downstream must not run on a failed upstream"
    assert "provider down" in (run.node_results["first"].error or "")
