"""Phase 3: HyDE, MultiQuery, StepBack retrievers + the shared fusion helper.

Every test stubs both the LLM and the base retriever so the suite never
hits any provider. The point is to verify that:

* The query the LLM sees matches our documented prompt contract.
* The query (or queries) the base retriever sees matches what the
  technique requires.
* Results are RRF-fused (Multi-Query / Step-Back) or passed through
  (HyDE).
* Failure modes degrade gracefully — a broken LLM never makes the
  retriever worse than the baseline.
"""

from __future__ import annotations

from typing import Any

import pytest

from prompture.rag.documents import Document
from prompture.rag.retrievers._fusion import _document_id, fuse_search_results
from prompture.rag.retrievers.base import Retriever
from prompture.rag.retrievers.hyde import HyDERetriever
from prompture.rag.retrievers.multi_query import MultiQueryRetriever, _parse_alternatives
from prompture.rag.retrievers.step_back import StepBackRetriever
from prompture.rag.vectorstores.base import VectorSearchResult

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _ScriptedLLM:
    """LLM stub that returns canned responses in order. Records every prompt."""

    def __init__(self, texts: list[str]) -> None:
        self._texts = list(texts)
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((prompt, options))
        text = self._texts.pop(0) if self._texts else ""
        return {"text": text, "meta": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0.0}}


class _RaisingLLM:
    def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("boom")


class _RecordingRetriever(Retriever):
    """Base retriever stub that records every query and returns canned hits.

    ``answers`` maps query strings (substring match) to result lists. A
    catch-all default list is returned for unrecognised queries.
    """

    def __init__(
        self,
        answers: dict[str, list[VectorSearchResult]] | None = None,
        default: list[VectorSearchResult] | None = None,
    ) -> None:
        self.answers = answers or {}
        self.default = default or []
        self.calls: list[tuple[str, int]] = []

    def retrieve(
        self,
        query: str,
        k: int = 4,
        filter: dict | None = None,
        **options: Any,
    ) -> list[VectorSearchResult]:
        self.calls.append((query, k))
        for key, results in self.answers.items():
            if key.casefold() in query.casefold():
                return results[:k]
        return self.default[:k]


def _result(content: str, score: float, doc_id: str | None = None) -> VectorSearchResult:
    md: dict[str, Any] = {}
    if doc_id is not None:
        md["id"] = doc_id
    return VectorSearchResult(document=Document(content=content, metadata=md), score=score)


# ---------------------------------------------------------------------------
# Fusion helper
# ---------------------------------------------------------------------------


class TestFusionHelper:
    def test_document_id_uses_metadata_id_when_present(self) -> None:
        r = _result("foo", 0.9, doc_id="explicit-1")
        assert _document_id(r) == "explicit-1"

    def test_document_id_falls_back_to_content_hash(self) -> None:
        r = _result("foo", 0.9)
        rid = _document_id(r)
        assert rid.startswith("sha1:")
        # Same content -> same id, deterministic across calls.
        assert _document_id(_result("foo", 0.1)) == rid

    def test_fuse_returns_empty_for_no_inputs(self) -> None:
        assert fuse_search_results([], top_n=4) == []
        assert fuse_search_results([[], []], top_n=4) == []

    def test_fuse_keeps_best_score_per_doc(self) -> None:
        a = _result("doc-a", 0.3, doc_id="a")
        a_hi = _result("doc-a", 0.9, doc_id="a")
        out = fuse_search_results([[a], [a_hi]], top_n=2)
        assert len(out) == 1
        # The single surviving result carries the *better* upstream score.
        assert out[0].score == 0.9

    def test_fuse_ranks_by_rrf(self) -> None:
        # doc-a appears first in both lists -> higher fused score than doc-b
        # which is rank 1 only in the second list.
        a = _result("doc-a", 0.5, doc_id="a")
        b = _result("doc-b", 0.5, doc_id="b")
        out = fuse_search_results([[a, b], [a, b]], top_n=2)
        assert [r.document.metadata["id"] for r in out] == ["a", "b"]

    def test_fuse_top_n_caps_output(self) -> None:
        rs = [_result(f"d{i}", 1.0 - i * 0.1, doc_id=str(i)) for i in range(5)]
        out = fuse_search_results([rs], top_n=2)
        assert len(out) == 2


# ---------------------------------------------------------------------------
# HyDE
# ---------------------------------------------------------------------------


class TestHyDERetriever:
    def test_query_swap_passes_hypothetical_to_base(self) -> None:
        llm = _ScriptedLLM(["Paris is the capital of France."])
        base = _RecordingRetriever(default=[_result("doc1", 1.0)])
        ret = HyDERetriever(base=base, llm=llm)

        ret.retrieve("What is the capital of France?", k=2)

        # LLM saw the rendered template, base retriever saw the hypothetical answer.
        assert "What is the capital of France?" in llm.calls[0][0]
        assert base.calls == [("Paris is the capital of France.", 2)]

    def test_empty_llm_response_falls_back_to_original_query(self) -> None:
        llm = _ScriptedLLM([""])
        base = _RecordingRetriever(default=[_result("doc1", 1.0)])
        ret = HyDERetriever(base=base, llm=llm)

        ret.retrieve("Original?", k=3)
        assert base.calls == [("Original?", 3)]

    def test_llm_failure_falls_back_to_original_query(self) -> None:
        base = _RecordingRetriever(default=[_result("doc1", 1.0)])
        ret = HyDERetriever(base=base, llm=_RaisingLLM())
        out = ret.retrieve("Original?", k=2)
        assert base.calls == [("Original?", 2)]
        assert len(out) == 1

    def test_long_response_is_truncated(self) -> None:
        # Hypothetical is far over the 2000-char cap.
        big = "x" * 5000
        llm = _ScriptedLLM([big])
        base = _RecordingRetriever(default=[_result("doc1", 1.0)])
        ret = HyDERetriever(base=base, llm=llm, max_hypothetical_chars=200)
        ret.retrieve("q", k=1)
        sent_query, _ = base.calls[0]
        assert len(sent_query) == 200

    def test_prompt_must_include_query_placeholder(self) -> None:
        with pytest.raises(ValueError, match="\\{query\\}"):
            HyDERetriever(base=_RecordingRetriever(), llm=_ScriptedLLM([]), prompt="no placeholder here")

    def test_filter_and_options_forwarded_to_base(self) -> None:
        llm = _ScriptedLLM(["hypo"])
        base = _RecordingRetriever(default=[])

        captured: dict[str, Any] = {}

        def _patched(query: str, k: int = 4, filter: dict | None = None, **opts: Any) -> list[VectorSearchResult]:
            captured["query"] = query
            captured["filter"] = filter
            captured["opts"] = opts
            return []

        base.retrieve = _patched  # type: ignore[method-assign]
        ret = HyDERetriever(base=base, llm=llm)
        ret.retrieve("q", k=3, filter={"category": "history"}, foo="bar")
        assert captured["filter"] == {"category": "history"}
        assert captured["opts"] == {"foo": "bar"}


# ---------------------------------------------------------------------------
# MultiQuery
# ---------------------------------------------------------------------------


class TestParseAlternatives:
    def test_plain_lines(self) -> None:
        raw = "Foo bar?\nBaz qux?\nQuux quuux?\n"
        assert _parse_alternatives(raw, max_count=5) == ["Foo bar?", "Baz qux?", "Quux quuux?"]

    def test_numbered_lines(self) -> None:
        raw = "1. Foo?\n2) Bar?\n- Baz?\n* Qux?"
        assert _parse_alternatives(raw, max_count=5) == ["Foo?", "Bar?", "Baz?", "Qux?"]

    def test_dedupes_case_insensitive(self) -> None:
        raw = "Foo?\nFOO?\nfoo?\nBar?"
        assert _parse_alternatives(raw, max_count=5) == ["Foo?", "Bar?"]

    def test_caps_at_max_count(self) -> None:
        raw = "\n".join(f"Q{i}?" for i in range(10))
        assert len(_parse_alternatives(raw, max_count=3)) == 3

    def test_strips_quotes(self) -> None:
        raw = '"Foo?"\n\'Bar?\''
        assert _parse_alternatives(raw, max_count=5) == ["Foo?", "Bar?"]


class TestMultiQueryRetriever:
    def test_runs_base_per_variant_including_original(self) -> None:
        llm = _ScriptedLLM(["Alt one?\nAlt two?\nAlt three?"])
        base = _RecordingRetriever(default=[_result("doc1", 1.0)])
        ret = MultiQueryRetriever(base=base, llm=llm, n_queries=3)

        ret.retrieve("Original?", k=2)

        # 1 original + 3 alternatives = 4 base calls.
        queries = [c[0] for c in base.calls]
        assert queries == ["Original?", "Alt one?", "Alt two?", "Alt three?"]

    def test_skips_original_when_disabled(self) -> None:
        llm = _ScriptedLLM(["Alt one?\nAlt two?"])
        base = _RecordingRetriever(default=[_result("doc1", 1.0)])
        ret = MultiQueryRetriever(base=base, llm=llm, n_queries=2, include_original=False)

        ret.retrieve("Original?", k=2)
        queries = [c[0] for c in base.calls]
        assert "Original?" not in queries
        assert queries == ["Alt one?", "Alt two?"]

    def test_drops_alternative_equal_to_original(self) -> None:
        llm = _ScriptedLLM(["Original?\nAlt two?"])  # first alt is a dupe of input
        base = _RecordingRetriever(default=[_result("doc1", 1.0)])
        ret = MultiQueryRetriever(base=base, llm=llm, n_queries=2)

        ret.retrieve("Original?", k=2)
        queries = [c[0] for c in base.calls]
        # Original + the one non-dupe alternative.
        assert queries == ["Original?", "Alt two?"]

    def test_llm_failure_still_runs_original(self) -> None:
        base = _RecordingRetriever(default=[_result("doc1", 1.0)])
        ret = MultiQueryRetriever(base=base, llm=_RaisingLLM(), n_queries=3)

        out = ret.retrieve("Original?", k=2)
        # Falls back to just the original query.
        assert base.calls == [("Original?", 2)]
        assert len(out) == 1

    def test_fuses_results_across_variants(self) -> None:
        # Two variants produce overlapping ranked lists; RRF should rank
        # the doc seen by both higher than docs seen by only one.
        shared = _result("shared", 0.5, doc_id="shared")
        unique_a = _result("unique-a", 0.5, doc_id="ua")
        unique_b = _result("unique-b", 0.5, doc_id="ub")

        answers = {
            "Original?": [shared, unique_a],
            "Alt?": [shared, unique_b],
        }
        base = _RecordingRetriever(answers=answers)
        llm = _ScriptedLLM(["Alt?"])  # one reformulation
        ret = MultiQueryRetriever(base=base, llm=llm, n_queries=1)

        out = ret.retrieve("Original?", k=3)
        ids = [r.document.metadata.get("id") or r.document.content for r in out]
        # Shared doc fuses to the top.
        assert ids[0] == "shared"
        # And both unique docs are kept (order between them is implementation-defined).
        assert set(ids[1:]) == {"ua", "ub"}

    def test_prompt_must_include_query_and_n_placeholders(self) -> None:
        with pytest.raises(ValueError, match="\\{query\\}"):
            MultiQueryRetriever(
                base=_RecordingRetriever(), llm=_ScriptedLLM([]), prompt="no placeholders"
            )

    def test_invalid_n_queries_raises(self) -> None:
        with pytest.raises(ValueError, match="n_queries"):
            MultiQueryRetriever(base=_RecordingRetriever(), llm=_ScriptedLLM([]), n_queries=0)


# ---------------------------------------------------------------------------
# StepBack
# ---------------------------------------------------------------------------


class TestStepBackRetriever:
    def test_runs_original_and_broader(self) -> None:
        llm = _ScriptedLLM(["What are the major European capitals?"])
        base = _RecordingRetriever(default=[_result("doc1", 1.0)])
        ret = StepBackRetriever(base=base, llm=llm)

        ret.retrieve("What is the capital of France?", k=2)

        queries = [c[0] for c in base.calls]
        assert queries == [
            "What is the capital of France?",
            "What are the major European capitals?",
        ]

    def test_llm_failure_still_returns_original_results(self) -> None:
        base = _RecordingRetriever(default=[_result("doc1", 1.0)])
        ret = StepBackRetriever(base=base, llm=_RaisingLLM())

        out = ret.retrieve("Q?", k=2)
        assert base.calls == [("Q?", 2)]
        assert len(out) == 1

    def test_broader_identical_to_original_is_skipped(self) -> None:
        llm = _ScriptedLLM(["Q?"])  # broader == original
        base = _RecordingRetriever(default=[_result("doc1", 1.0)])
        ret = StepBackRetriever(base=base, llm=llm)

        ret.retrieve("Q?", k=2)
        assert base.calls == [("Q?", 2)]

    def test_strips_leading_label_and_quotes(self) -> None:
        llm = _ScriptedLLM(['"What are European capitals?"'])
        base = _RecordingRetriever(default=[_result("doc1", 1.0)])
        ret = StepBackRetriever(base=base, llm=llm)

        ret.retrieve("Capital of France?", k=1)
        # Quotes stripped; first non-empty line used.
        assert base.calls[1][0] == "What are European capitals?"

    def test_fuses_two_ranked_lists(self) -> None:
        shared = _result("shared", 0.5, doc_id="shared")
        narrow = _result("narrow", 0.5, doc_id="narrow")
        broad = _result("broad", 0.5, doc_id="broad")

        answers = {
            "Narrow?": [shared, narrow],
            "Broader?": [shared, broad],
        }
        base = _RecordingRetriever(answers=answers)
        llm = _ScriptedLLM(["Broader?"])
        ret = StepBackRetriever(base=base, llm=llm)

        out = ret.retrieve("Narrow?", k=3)
        ids = [r.document.metadata["id"] for r in out]
        assert ids[0] == "shared"
        assert set(ids[1:]) == {"narrow", "broad"}

    def test_prompt_must_include_query_placeholder(self) -> None:
        with pytest.raises(ValueError, match="\\{query\\}"):
            StepBackRetriever(
                base=_RecordingRetriever(), llm=_ScriptedLLM([]), prompt="no placeholder"
            )


# ---------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------


class TestRetrieverRegistry:
    def test_new_retrievers_registered(self) -> None:
        from prompture.rag.retrievers import RETRIEVER_REGISTRY, get_retriever

        for name in ("hyde", "multi_query", "multi-query", "step_back", "step-back"):
            assert name in RETRIEVER_REGISTRY, f"missing registry entry for {name}"

        # Factory call works for one to prove the wire is hot.
        ret = get_retriever(
            "hyde",
            base=_RecordingRetriever(default=[_result("doc1", 1.0)]),
            llm=_ScriptedLLM(["hypo"]),
        )
        assert isinstance(ret, HyDERetriever)
