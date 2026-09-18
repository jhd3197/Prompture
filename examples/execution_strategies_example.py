"""One persona, three execution strategies, one comparable report.

Demonstrates :mod:`prompture.execution`:

1. :class:`~prompture.execution.strategies.direct.DirectStrategy` — one attempt
   plus schema validation, with a bounded repair pass.
2. :class:`~prompture.execution.strategies.retrieve.RetrieveAndVerifyStrategy` —
   a grounded answer with verified citations, or an explicit
   insufficient-evidence outcome.
3. :class:`~prompture.execution.strategies.critique.DraftAndCritiqueStrategy` —
   draft, structured critique, bounded revision.

Then it runs the bundled evaluation fixtures through the benchmark harness and
prints a report that keeps quality, schema validity, evidence support, latency
and cost separate — and says plainly which of those it could not measure.

By default the example uses Ollama, so it runs with no cloud API key. When
Ollama is not reachable it falls back to a scripted driver: the *contract* is
still demonstrated end to end, but the numbers are then a demonstration of the
plumbing, not a measurement of model quality. The report says so.

Run it
~~~~~~

::

    ollama pull llama3.1:8b
    OLLAMA_MODEL=llama3.1:8b python examples/execution_strategies_example.py
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from pydantic import BaseModel, Field

from prompture import Assistant, Persona
from prompture.drivers.base import Driver
from prompture.execution import (
    EvidencePassage,
    ExecutionRequest,
    JsonlOutcomeStore,
    ResourceLimits,
    Split,
    TaskCategory,
    compare_strategies,
    load_bundled_fixture_set,
    run_fixture_set,
)
from prompture.execution.strategies import (
    DirectStrategy,
    DraftAndCritiqueStrategy,
    RetrieveAndVerifyStrategy,
)

# =============================================================================
# Output contract
# =============================================================================


class Contact(BaseModel):
    """The typed contract the extraction workload must satisfy."""

    name: str | None = Field(None, description="Full name of the primary contact.")
    email: str | None = Field(None, description="Email address, or null when absent.")
    company: str | None = Field(None, description="Employer, or null when absent.")
    role: str | None = Field(None, description="Job title, or null when absent.")
    phone: str | None = Field(None, description="Phone number, or null when absent.")


# =============================================================================
# Driver selection
# =============================================================================


class _ScriptedDriver(Driver):
    """Offline stand-in that returns shape-correct answers.

    It exists so the example always runs. It knows nothing about the questions,
    so its *answers* are not meaningful — only the plumbing it exercises is.
    """

    model = "scripted/demo"
    model_name = "scripted/demo"
    supports_json_mode = False
    supports_json_schema = False

    def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        return {
            "text": json.dumps(self._respond(prompt)),
            "meta": {
                "prompt_tokens": max(1, len(prompt) // 4),
                "completion_tokens": 24,
                "total_tokens": max(1, len(prompt) // 4) + 24,
                "cost": 0.0,
                "raw_response": {},
            },
        }

    @staticmethod
    def _respond(prompt: str) -> dict[str, Any]:
        if "approved" in prompt and "issues" in prompt:
            return {"approved": True, "score": 9, "issues": [], "suggestions": []}
        if "source_ids" in prompt or "Passages:" in prompt:
            return {"answer": None, "source_ids": [], "sufficient": False}
        return {"name": None, "email": None, "company": None, "role": None, "phone": None}


def build_driver() -> tuple[Any, str, bool]:
    """Return ``(driver, model_string, is_live)``.

    Falls back to the scripted driver when Ollama is not reachable.
    """
    model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    model_string = f"ollama/{model}"
    try:
        import requests

        from prompture.drivers import get_driver_for_model

        driver = get_driver_for_model(model_string)
        endpoint = getattr(driver, "endpoint", "") or ""
        base = endpoint.split("/api/")[0] if "/api/" in endpoint else endpoint.rstrip("/")
        if base:
            requests.head(base, timeout=3)
        # A reachable server is not the same as a usable model — probe once so
        # the example falls back cleanly instead of erroring on every call.
        driver.generate("ping", {"max_tokens": 1})
        return driver, model_string, True
    except Exception as exc:
        print(f"  (Ollama unavailable - {type(exc).__name__}: {exc}; using the scripted driver)")
        return _ScriptedDriver(), "scripted/demo", False


# =============================================================================
# 1. One persona, three strategies
# =============================================================================


def section_one_persona_three_strategies(driver: Any, model: str) -> None:
    print("\n" + "=" * 78)
    print("1. One persona, three strategies")
    print("=" * 78)

    analyst = Assistant(
        name="policy-analyst",
        persona=Persona(
            name="analyst",
            system_prompt=(
                "You are a careful {{domain}} analyst. You never state anything the source material does not support."
            ),
        ),
        model=model,
        variables={"domain": "HR policy"},
    )

    passages = (
        EvidencePassage(id="hb-leave-1", text="Planned leave must be requested at least 14 calendar days in advance."),
        EvidencePassage(id="hb-expenses-1", text="Expense claims must be filed within 30 days of the expenditure."),
    )

    request = analyst.execution_request(
        "How much notice is required before taking planned leave?",
        category=TaskCategory.DOCUMENT_QA,
        passages=passages,
        limits=ResourceLimits(max_cost_usd=0.05, max_llm_calls=6, max_seconds=120),
    )

    results = compare_strategies(
        request,
        {
            "direct": DirectStrategy(driver=driver, model=model),
            "retrieve_verify": RetrieveAndVerifyStrategy(driver=driver, model=model),
            "draft_critique": DraftAndCritiqueStrategy(driver=driver, model=model, max_iterations=2),
        },
    )

    for label, result in results.items():
        print(f"\n--- {label} ---")
        print(result.explain())
        print(f"  answer: {result.answer[:160]!r}")
        print(f"  calls={result.usage.call_count} cost=${result.usage.cost:.6f} complete={result.usage.cost_complete}")


# =============================================================================
# 2. Refusing to answer without evidence
# =============================================================================


def section_insufficient_evidence(driver: Any, model: str) -> None:
    print("\n" + "=" * 78)
    print("2. An honest refusal is an outcome, not an error")
    print("=" * 78)

    strategy = RetrieveAndVerifyStrategy(driver=driver, model=model)
    result = strategy.run(
        ExecutionRequest(
            task="What is the company's paid sabbatical policy?",
            category=TaskCategory.DOCUMENT_QA,
            passages=(
                EvidencePassage(id="hb-leave-1", text="Planned leave needs 14 days of notice."),
                EvidencePassage(id="hb-benefits-1", text="Employees receive 25 days of paid annual leave."),
            ),
        )
    )

    print(f"  termination : {result.termination.value}")
    print(f"  abstained   : {result.abstained}")
    print(f"  error       : {result.error!r}  <- an honest refusal carries no error")
    print(f"  sources seen: {result.evidence.sources}")


# =============================================================================
# 3. Budgets stop work before it starts
# =============================================================================


def section_budget(driver: Any, model: str) -> None:
    print("\n" + "=" * 78)
    print("3. Budgets are checked before more work starts")
    print("=" * 78)

    strategy = DraftAndCritiqueStrategy(driver=driver, model=model, max_iterations=5)
    result = strategy.run(
        ExecutionRequest(
            task="Write a two-sentence summary of the leave policy.",
            limits=ResourceLimits(max_llm_calls=2),
        )
    )

    print(f"  termination : {result.termination.value}")
    print(f"  calls made  : {result.usage.call_count} (limit was 2)")
    print(f"  budget note : {result.budget_note}")


# =============================================================================
# 4. The benchmark harness
# =============================================================================


def section_benchmark(driver: Any, model: str, live: bool, out_dir: Path) -> None:
    print("\n" + "=" * 78)
    print("4. Benchmark: the dev split, three repeats, against the declared gate")
    print("=" * 78)

    fixtures = load_bundled_fixture_set("extraction_contacts")
    store = JsonlOutcomeStore(out_dir / "outcomes.jsonl")

    report = run_fixture_set(
        fixtures,
        DirectStrategy(driver=driver, model=model, max_repairs=1),
        split=Split.DEV,
        repeats=3,
        label=f"direct/{model}",
        model=model,
        output_model=Contact,
        limits=ResourceLimits(max_cost_usd=0.05, max_llm_calls=4),
        outcome_store=store,
    )

    print(report.format())
    saved = report.save(out_dir / "report.json")
    print(f"\n  report written to  : {saved}")
    print(f"  outcome records at : {store.path} ({len(store)} record(s))")
    if not live:
        print(
            "\n  NOTE: this run used the scripted fallback driver. The quality numbers\n"
            "        above describe the harness, not a model. Point OLLAMA_MODEL at a\n"
            "        running Ollama (or wire in any other provider) for real figures."
        )


def main() -> None:
    # Console encodings vary (cp1252 on Windows); never let a dash crash a demo.
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(errors="replace")

    print("Prompture - execution strategies")
    driver, model, live = build_driver()
    print(f"  model: {model} ({'live' if live else 'scripted fallback'})")

    # Written to a temp directory so running the example leaves the repo clean.
    out_dir = Path(tempfile.mkdtemp(prefix="prompture-execution-"))

    section_one_persona_three_strategies(driver, model)
    section_insufficient_evidence(driver, model)
    section_budget(driver, model)
    section_benchmark(driver, model, live, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
