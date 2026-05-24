"""Tests for prompture.citations."""

from __future__ import annotations

import pytest

from prompture.citations import (
    CITATION_INSTRUCTION,
    CitationTracker,
    CitedAnswer,
    Source,
    extract_citations,
)
from prompture.rag.documents import Document
from prompture.rag.vectorstores.base import VectorSearchResult


def _src(i: str, text: str, **kw) -> Source:
    return Source(id=i, text=text, **kw)


# ----------------------------------------------------------------------
# extract_citations
# ----------------------------------------------------------------------


class TestExtractCitations:
    def test_single_bracket(self) -> None:
        sources = [_src("1", "Paris is the capital of France.")]
        text = "Paris is the capital of France [1]."
        cites, clean, cited = extract_citations(text, sources)
        assert len(cites) == 1
        assert cites[0].source_ids == ["1"]
        assert cites[0].text == "Paris is the capital of France"
        assert clean == "Paris is the capital of France ."
        assert cited == {"1"}

    def test_multi_id_bracket(self) -> None:
        sources = [_src("1", "a"), _src("2", "b"), _src("3", "c")]
        text = "Both routes are viable [1, 3]."
        cites, _, cited = extract_citations(text, sources)
        assert cites[0].source_ids == ["1", "3"]
        assert cited == {"1", "3"}

    def test_unknown_id_recorded_but_not_in_cited_set(self) -> None:
        sources = [_src("1", "a")]
        text = "Made-up claim [9]."
        cites, _, cited = extract_citations(text, sources)
        assert cites[0].source_ids == ["9"]
        assert cited == set()

    def test_no_citations(self) -> None:
        cites, clean, cited = extract_citations("No marks here.", [_src("1", "a")])
        assert cites == []
        assert clean == "No marks here."
        assert cited == set()

    def test_claim_extraction_walks_to_sentence_boundary(self) -> None:
        sources = [_src("1", "x"), _src("2", "y")]
        text = "First fact [1]. Second separate fact [2]."
        cites, _, _ = extract_citations(text, sources)
        assert cites[0].text == "First fact"
        assert cites[1].text == "Second separate fact"

    def test_claim_extraction_uses_newline_boundary(self) -> None:
        sources = [_src("1", "x")]
        text = "Header line\nThe claim [1]."
        cites, _, _ = extract_citations(text, sources)
        assert cites[0].text == "The claim"

    def test_clean_text_collapses_extra_spaces(self) -> None:
        sources = [_src("1", "x"), _src("2", "y")]
        text = "A [1] and B [2]."
        _, clean, _ = extract_citations(text, sources)
        # Markers stripped; surrounding spaces collapsed.
        assert "  " not in clean

    def test_offsets_align_to_input(self) -> None:
        sources = [_src("1", "x")]
        text = "Hello world [1]."
        cites, _, _ = extract_citations(text, sources)
        assert text[cites[0].start : cites[0].end].endswith("[1]")


# ----------------------------------------------------------------------
# CitationTracker
# ----------------------------------------------------------------------


class TestCitationTracker:
    def test_from_documents_assigns_sequential_ids(self) -> None:
        docs = [
            Document(content="alpha", metadata={"title": "A"}),
            Document(content="beta", metadata={"title": "B"}),
        ]
        tracker = CitationTracker.from_documents(docs)
        sources = tracker.sources
        assert [s.id for s in sources] == ["1", "2"]
        assert sources[0].title == "A"
        assert sources[0].text == "alpha"

    def test_from_retrieval_results_preserves_scores(self) -> None:
        results = [
            VectorSearchResult(document=Document(content="a", metadata={}), score=0.91),
            VectorSearchResult(document=Document(content="b", metadata={}), score=0.42),
        ]
        tracker = CitationTracker.from_retrieval_results(results)
        scores = [s.score for s in tracker.sources]
        assert scores == [0.91, 0.42]

    def test_build_prompt_includes_instruction_and_sources(self) -> None:
        docs = [Document(content="The Eiffel Tower is in Paris.", metadata={"title": "Wiki"})]
        tracker = CitationTracker.from_documents(docs)
        prompt = tracker.build_prompt(question="Where is the Eiffel Tower?")
        assert CITATION_INSTRUCTION.strip() in prompt
        assert "[1] Wiki" in prompt
        assert "Eiffel Tower is in Paris" in prompt
        assert "Question: Where is the Eiffel Tower?" in prompt

    def test_build_prompt_omits_question_when_absent(self) -> None:
        tracker = CitationTracker(sources=[_src("1", "x")])
        prompt = tracker.build_prompt()
        assert "Question:" not in prompt

    def test_build_prompt_uses_stored_question_when_no_arg(self) -> None:
        tracker = CitationTracker(sources=[_src("1", "x")], question="stored?")
        prompt = tracker.build_prompt()
        assert "Question: stored?" in prompt

    def test_no_sources_renders_placeholder(self) -> None:
        tracker = CitationTracker()
        prompt = tracker.build_prompt(question="hi?")
        assert "(no sources available)" in prompt

    def test_add_source_assigns_next_id(self) -> None:
        tracker = CitationTracker(sources=[_src("1", "x")])
        added = tracker.add_source(Document(content="y", metadata={}))
        assert added.id == "2"
        assert tracker.sources[-1].id == "2"

    def test_parse_response_builds_cited_answer(self) -> None:
        tracker = CitationTracker(
            sources=[_src("1", "a"), _src("2", "b")],
            question="why?",
        )
        ans = tracker.parse_response("Claim one [1]. Claim two [2].", usage={"total_tokens": 7})
        assert isinstance(ans, CitedAnswer)
        assert ans.is_grounded
        assert ans.coverage == 1.0
        assert ans.cited_source_ids == {"1", "2"}
        assert ans.usage["total_tokens"] == 7
        # answer preserves brackets; clean_answer strips them.
        assert "[1]" in ans.answer
        assert "[1]" not in ans.clean_answer

    def test_coverage_partial(self) -> None:
        tracker = CitationTracker(sources=[_src("1", "a"), _src("2", "b"), _src("3", "c")])
        ans = tracker.parse_response("Only one is cited [1].")
        assert pytest.approx(ans.coverage) == 1 / 3
        uncited = {s.id for s in ans.uncited_sources()}
        assert uncited == {"2", "3"}

    def test_not_grounded_when_no_citations(self) -> None:
        tracker = CitationTracker(sources=[_src("1", "a")])
        ans = tracker.parse_response("No citations at all.")
        assert not ans.is_grounded
        assert ans.coverage == 0.0

    def test_to_dict_round_trip_shape(self) -> None:
        tracker = CitationTracker(sources=[_src("1", "a")])
        ans = tracker.parse_response("Yes [1].")
        d = ans.to_dict()
        assert d["is_grounded"] is True
        assert d["citations"][0]["source_ids"] == ["1"]
        assert d["sources"][0]["id"] == "1"

    def test_unsupported_source_type_raises(self) -> None:
        with pytest.raises(TypeError, match="Source"):
            CitationTracker(sources=[12345])  # type: ignore[list-item]
