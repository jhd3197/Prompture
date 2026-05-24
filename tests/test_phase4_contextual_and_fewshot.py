"""Phase 4: ContextualChunker + FewShotExampleStore + extract_with_model wiring.

Three independent feature areas:

1. ``ContextualChunker`` — wraps a base chunker and prepends per-chunk
   LLM-generated context. Uses the system-role path when the driver
   exposes ``generate_messages`` so Anthropic prompt caching kicks in.
2. ``FewShotExampleStore`` — embedding-backed example bank with
   ``add`` / ``add_many`` / ``select``.
3. Wiring — ``extract_with_model(examples_store=...)`` retrieves
   top-K examples and prepends them to the instruction. Tests use a
   recording driver to verify what reaches the LLM.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from prompture.extraction.few_shot import (
    Example,
    FewShotExampleStore,
    _NgramEmbedder,
    format_examples_for_prompt,
)
from prompture.rag.chunkers.base import TextChunker
from prompture.rag.chunkers.contextual import (
    DEFAULT_CONTEXTUAL_PROMPT,
    ContextualChunker,
)
from prompture.rag.documents import Document

# ---------------------------------------------------------------------------
# Shared LLM stubs
# ---------------------------------------------------------------------------


class _ScriptedLLM:
    """Returns canned text in order, optionally distinguishing path used."""

    def __init__(self, texts: list[str], *, has_generate_messages: bool = True) -> None:
        self._texts = list(texts)
        self.calls: list[tuple[str, str, Any]] = []  # (path, payload, options)
        if not has_generate_messages:
            # Hide generate_messages so ContextualChunker falls back to generate().
            self.generate_messages = None  # type: ignore[assignment]

    def _pop(self) -> str:
        return self._texts.pop(0) if self._texts else ""

    def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(("generate", prompt, options))
        return {"text": self._pop()}

    def generate_messages(self, messages: list[dict[str, Any]], options: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(("generate_messages", messages, options))
        return {"text": self._pop()}


class _RaisingLLM:
    def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("boom")

    def generate_messages(self, messages: list[dict[str, Any]], options: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("boom")


class _FixedSplitChunker(TextChunker):
    """Splits on '|' so test inputs can exercise exact chunk counts deterministically."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 0) -> None:
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def split_text(self, text: str) -> list[str]:
        return [p.strip() for p in text.split("|") if p.strip()]


# ---------------------------------------------------------------------------
# 1. ContextualChunker
# ---------------------------------------------------------------------------


class TestContextualChunkerSplitPath:
    def test_prompt_must_have_chunk_placeholder(self) -> None:
        with pytest.raises(ValueError, match="\\{chunk\\}"):
            ContextualChunker(
                base=_FixedSplitChunker(),
                driver=_ScriptedLLM([]),
                prompt="no placeholder",
            )

    def test_split_text_delegates_to_base_uncontextualised(self) -> None:
        chunker = ContextualChunker(base=_FixedSplitChunker(), driver=_ScriptedLLM([]))
        # split_text never calls the LLM — it operates on raw text without doc context.
        assert chunker.split_text("a|b|c") == ["a", "b", "c"]

    def test_split_documents_prepends_context_to_each_chunk(self) -> None:
        llm = _ScriptedLLM(["context for A", "context for B", "context for C"])
        chunker = ContextualChunker(base=_FixedSplitChunker(), driver=llm)
        doc = Document(content="A|B|C", metadata={"source": "doc1"})
        out = chunker.split_documents([doc])

        assert [d.content for d in out] == [
            "context for A\n\nA",
            "context for B\n\nB",
            "context for C\n\nC",
        ]
        # Each chunk gets its context stamped in metadata for transparency.
        for i, d in enumerate(out):
            assert d.metadata["chunk_index"] == i
            assert d.metadata["chunk_count"] == 3
            assert d.metadata["parent_source"] == "doc1"
            assert "context for" in d.metadata["contextual_context"]

    def test_single_chunk_document_skipped_by_default(self) -> None:
        # When the base only produces one chunk, contextualisation is a no-op
        # (no LLM call) unless the user explicitly opts in.
        llm = _ScriptedLLM(["should not be used"])
        chunker = ContextualChunker(base=_FixedSplitChunker(), driver=llm)
        out = chunker.split_documents([Document(content="just one")])

        assert llm.calls == []
        assert out[0].content == "just one"
        assert "contextual_context" not in out[0].metadata

    def test_single_chunk_contextualised_when_opted_in(self) -> None:
        llm = _ScriptedLLM(["single-chunk context"])
        chunker = ContextualChunker(
            base=_FixedSplitChunker(),
            driver=llm,
            skip_if_only_one_chunk=False,
        )
        out = chunker.split_documents([Document(content="just one")])
        assert out[0].content == "single-chunk context\n\njust one"


class TestContextualChunkerLLMPath:
    def test_uses_generate_messages_with_system_role_when_available(self) -> None:
        llm = _ScriptedLLM(["context A", "context B"])
        chunker = ContextualChunker(base=_FixedSplitChunker(), driver=llm)
        chunker.split_documents([Document(content="A|B")])

        # Both calls went through generate_messages.
        assert all(c[0] == "generate_messages" for c in llm.calls)
        # System role carries the full document; user role carries the chunk prompt.
        first_messages = llm.calls[0][1]
        assert first_messages[0]["role"] == "system"
        assert first_messages[0]["content"] == "A|B"  # full doc, not just the chunk
        assert first_messages[1]["role"] == "user"
        assert "<chunk>\nA\n</chunk>" in first_messages[1]["content"]

    def test_fallback_to_generate_when_no_generate_messages(self) -> None:
        llm = _ScriptedLLM(["ctx A", "ctx B"], has_generate_messages=False)
        chunker = ContextualChunker(base=_FixedSplitChunker(), driver=llm)
        chunker.split_documents([Document(content="A|B")])

        # Both calls landed on generate() (the messages path was unavailable).
        assert all(c[0] == "generate" for c in llm.calls)
        # The fallback combined prompt includes the document.
        assert "<document>\nA|B\n</document>" in llm.calls[0][1]

    def test_document_truncated_for_llm_call(self) -> None:
        # Document is enormous; the LLM should see at most max_document_chars.
        big = "X" * 5000
        doc = Document(content=f"{big}|Y")
        llm = _ScriptedLLM(["ctx 1", "ctx 2"])
        chunker = ContextualChunker(
            base=_FixedSplitChunker(),
            driver=llm,
            max_document_chars=100,
        )
        chunker.split_documents([doc])

        first_messages = llm.calls[0][1]
        assert len(first_messages[0]["content"]) == 100

    def test_long_context_is_truncated(self) -> None:
        long_ctx = "Y" * 1000
        llm = _ScriptedLLM([long_ctx, long_ctx])
        chunker = ContextualChunker(
            base=_FixedSplitChunker(),
            driver=llm,
            max_context_chars=50,
        )
        out = chunker.split_documents([Document(content="A|B")])
        # Context capped to 50 chars, then "\n\n" + chunk.
        for d in out:
            assert d.metadata["contextual_context"].startswith("YYY")
            assert len(d.metadata["contextual_context"]) <= 50

    def test_strips_label_prefixes_from_llm_output(self) -> None:
        llm = _ScriptedLLM(["Context: this is the situating context"])
        chunker = ContextualChunker(base=_FixedSplitChunker(), driver=llm)
        out = chunker.split_documents([Document(content="A|B")])
        # The "Context:" prefix was stripped.
        assert out[0].metadata["contextual_context"] == "this is the situating context"

    def test_fail_open_emits_empty_context_on_llm_failure(self) -> None:
        chunker = ContextualChunker(
            base=_FixedSplitChunker(),
            driver=_RaisingLLM(),
            fail_open=True,
        )
        out = chunker.split_documents([Document(content="A|B")])
        # Failure -> empty context, chunk content unmodified.
        assert [d.content for d in out] == ["A", "B"]
        for d in out:
            assert "contextual_context" not in d.metadata

    def test_fail_closed_propagates_error(self) -> None:
        # When both generate_messages and generate fail, fail_open=False re-raises.
        chunker = ContextualChunker(
            base=_FixedSplitChunker(),
            driver=_RaisingLLM(),
            fail_open=False,
        )
        with pytest.raises(RuntimeError, match="boom"):
            chunker.split_documents([Document(content="A|B")])


class TestContextualChunkerRegistry:
    def test_registered_factory(self) -> None:
        from prompture.rag.chunkers import CHUNKER_REGISTRY, get_chunker

        assert "contextual" in CHUNKER_REGISTRY
        ck = get_chunker(
            "contextual",
            base=_FixedSplitChunker(),
            driver=_ScriptedLLM(["ctx"]),
        )
        assert isinstance(ck, ContextualChunker)


class TestDefaultPrompt:
    def test_default_prompt_contains_chunk_placeholder(self) -> None:
        assert "{chunk}" in DEFAULT_CONTEXTUAL_PROMPT
        # The default prompt is *also* a self-contained instruction.
        assert "retrieval" in DEFAULT_CONTEXTUAL_PROMPT.lower()


# ---------------------------------------------------------------------------
# 2. FewShotExampleStore
# ---------------------------------------------------------------------------


class TestNgramEmbedderFallback:
    def test_consistent_dimensions(self) -> None:
        emb = _NgramEmbedder(dimensions=64)
        out = emb.embed(["foo bar", "baz qux"], {})
        assert "embeddings" in out
        assert len(out["embeddings"]) == 2
        assert all(len(v) == 64 for v in out["embeddings"])

    def test_similar_inputs_have_higher_similarity(self) -> None:
        # Same string should embed identically -> cosine 1.0.
        emb = _NgramEmbedder(dimensions=256)
        out = emb.embed(["hello world", "hello world", "totally different content here"], {})
        a, b, c = out["embeddings"]
        from prompture.extraction.few_shot import _cosine

        assert _cosine(a, b) > 0.99
        assert _cosine(a, c) < _cosine(a, b)


class TestFewShotExampleStore:
    def test_add_and_len(self) -> None:
        store = FewShotExampleStore()
        store.add("Maria, 32, developer", {"name": "Maria", "age": 32})
        store.add("Bob, 47, accountant", {"name": "Bob", "age": 47})
        assert len(store) == 2

    def test_add_many_batches_embedder(self) -> None:
        class _CountingEmbedder:
            def __init__(self) -> None:
                self.calls = 0

            def embed(self, texts: list[str], options: dict[str, Any]) -> dict[str, Any]:
                self.calls += 1
                return {"embeddings": [[float(len(t)), 0.0] for t in texts]}

        emb = _CountingEmbedder()
        store = FewShotExampleStore(embedder=emb)
        store.add_many([("a", 1), ("ab", 2), ("abc", 3)])
        # One batched call instead of three single-text calls.
        assert emb.calls == 1
        assert len(store) == 3

    def test_select_returns_top_k_by_similarity(self) -> None:
        # Use a controllable embedder: each example gets an explicit vector so
        # we can verify the ranking logic without relying on the n-gram hash
        # collisions of the fallback embedder.
        class _ManualEmbedder:
            """Returns the per-text vector configured at construction."""

            def __init__(self, by_text: dict[str, list[float]]) -> None:
                self._by_text = by_text

            def embed(self, texts: list[str], options: dict[str, Any]) -> dict[str, Any]:
                return {"embeddings": [self._by_text[t] for t in texts]}

        emb = _ManualEmbedder(
            {
                "ex1": [1.0, 0.0],   # close to query
                "ex2": [0.9, 0.1],   # also close
                "ex3": [0.0, 1.0],   # orthogonal — should NOT make the top 2
                "query": [1.0, 0.0],
            }
        )
        store = FewShotExampleStore(embedder=emb)
        store.add("ex1", {"id": 1})
        store.add("ex2", {"id": 2})
        store.add("ex3", {"id": 3})

        hits = store.select("query", k=2)
        ids = [h.expected_output["id"] for h in hits]
        assert ids == [1, 2]

    def test_select_uses_ngram_fallback_for_obvious_overlap(self) -> None:
        """Smoke test using the default n-gram embedder.

        With the default fallback embedder, we don't promise textbook
        recall — only that *very* obvious overlaps rank above unrelated
        content. The closest input wins.
        """
        store = FewShotExampleStore()
        store.add("Alice is a software engineer in LA.", {"closest": True})
        store.add("Completely unrelated: nuclear physics formulas.", {"closest": False})
        hits = store.select("Alice is a software engineer in LA.", k=1)
        assert hits[0].expected_output == {"closest": True}

    def test_select_respects_k(self) -> None:
        store = FewShotExampleStore()
        for i in range(5):
            store.add(f"input {i}", {"i": i})
        assert len(store.select("query", k=0)) == 0
        assert len(store.select("query", k=2)) == 2
        # k larger than store size is clamped, not an error.
        assert len(store.select("query", k=99)) == 5

    def test_select_empty_store_returns_empty(self) -> None:
        store = FewShotExampleStore()
        assert store.select("anything", k=3) == []

    def test_clear(self) -> None:
        store = FewShotExampleStore()
        store.add("x", 1)
        store.add("y", 2)
        store.clear()
        assert len(store) == 0

    def test_embedder_returning_wrong_count_raises(self) -> None:
        class _BadEmbedder:
            def embed(self, texts: list[str], options: dict[str, Any]) -> dict[str, Any]:
                return {"embeddings": [[0.0]] * (len(texts) - 1)}  # one short

        store = FewShotExampleStore(embedder=_BadEmbedder())
        with pytest.raises(ValueError, match="vectors for"):
            store.add_many([("a", 1), ("b", 2)])

    def test_embedder_returning_bad_shape_raises(self) -> None:
        class _BadShape:
            def embed(self, texts: list[str], options: dict[str, Any]) -> dict[str, Any]:
                return ["not", "a", "dict"]  # type: ignore[return-value]

        store = FewShotExampleStore(embedder=_BadShape())
        with pytest.raises(ValueError, match="unexpected shape"):
            store.add("anything", 1)


# ---------------------------------------------------------------------------
# 3. format_examples_for_prompt
# ---------------------------------------------------------------------------


class TestFormatExamplesForPrompt:
    def test_empty_returns_empty_string(self) -> None:
        assert format_examples_for_prompt([]) == ""

    def test_renders_inputs_and_outputs(self) -> None:
        ex1 = Example(input_text="Maria, 32, dev", expected_output={"name": "Maria", "age": 32})
        ex2 = Example(input_text="Bob, 47, acct", expected_output={"name": "Bob", "age": 47})
        text = format_examples_for_prompt([ex1, ex2])
        assert "Maria, 32, dev" in text
        assert "Bob, 47, acct" in text
        # Outputs serialised as indented JSON.
        assert '"name": "Maria"' in text
        assert '"age": 32' in text

    def test_pydantic_model_output_is_dumped(self) -> None:
        class Person(BaseModel):
            name: str
            age: int

        ex = Example(input_text="x", expected_output=Person(name="A", age=1))
        text = format_examples_for_prompt([ex])
        assert '"name": "A"' in text
        assert '"age": 1' in text

    def test_string_output_passes_through(self) -> None:
        ex = Example(input_text="x", expected_output="literal answer")
        text = format_examples_for_prompt([ex])
        assert "literal answer" in text

    def test_uses_custom_header_and_footer(self) -> None:
        ex = Example(input_text="x", expected_output=1)
        text = format_examples_for_prompt(
            [ex],
            header="CUSTOM HEAD",
            footer="CUSTOM FOOT",
        )
        assert "CUSTOM HEAD" in text
        assert "CUSTOM FOOT" in text


# ---------------------------------------------------------------------------
# 4. extract_with_model + examples_store wiring
# ---------------------------------------------------------------------------


class _PersonForExtract(BaseModel):
    name: str
    age: int


class TestExtractWithModelFewShotWiring:
    def _build_store(self) -> FewShotExampleStore:
        store = FewShotExampleStore()
        store.add("Maria is 32, a developer in NYC.", {"name": "Maria", "age": 32})
        store.add("Bob, 47, accountant.", {"name": "Bob", "age": 47})
        store.add("The forecast says rain tomorrow.", {"unrelated": True})
        return store

    def test_examples_block_prepended_to_instruction(self) -> None:
        """The selected examples should appear in the prompt sent to the driver."""
        store = self._build_store()

        captured_prompts: list[str] = []

        def _fake_extract_and_jsonify(*, text, instruction_template, **kw):
            captured_prompts.append(instruction_template)
            return {
                "json_string": '{"name": "Alice", "age": 25}',
                "json_object": {"name": "Alice", "age": 25},
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0.0},
            }

        with patch("prompture.extraction.core.extract_and_jsonify", _fake_extract_and_jsonify):
            from prompture.extraction.core import extract_with_model

            result = extract_with_model(
                _PersonForExtract,
                "Alice is 25, a designer in LA.",
                model_name="fake/model",
                examples_store=store,
                examples_k=2,
            )

        assert captured_prompts, "extract_and_jsonify was not called"
        prompt = captured_prompts[0]
        # The instruction now starts with the examples block.
        assert "Input:" in prompt
        assert "Output:" in prompt
        # The 2 person examples beat the unrelated weather sentence.
        assert "Maria" in prompt or "Bob" in prompt
        # The store metadata is surfaced on the result for transparency.
        assert "few_shot" in result["usage"]
        assert result["usage"]["few_shot"]["count"] == 2
        assert result["usage"]["few_shot"]["k_requested"] == 2

    def test_no_examples_store_means_no_block(self) -> None:
        """Without examples_store the instruction template is unchanged."""
        captured: list[str] = []

        def _fake_extract_and_jsonify(*, text, instruction_template, **kw):
            captured.append(instruction_template)
            return {
                "json_string": '{"name": "Alice", "age": 25}',
                "json_object": {"name": "Alice", "age": 25},
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0.0},
            }

        with patch("prompture.extraction.core.extract_and_jsonify", _fake_extract_and_jsonify):
            from prompture.extraction.core import extract_with_model

            extract_with_model(
                _PersonForExtract,
                "Alice is 25, a designer in LA.",
                model_name="fake/model",
            )

        assert "Input:" not in captured[0]
        assert "Output:" not in captured[0]

    def test_examples_k_zero_disables_injection(self) -> None:
        store = self._build_store()
        captured: list[str] = []

        def _fake_extract_and_jsonify(*, text, instruction_template, **kw):
            captured.append(instruction_template)
            return {
                "json_string": '{"name": "Alice", "age": 25}',
                "json_object": {"name": "Alice", "age": 25},
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0.0},
            }

        with patch("prompture.extraction.core.extract_and_jsonify", _fake_extract_and_jsonify):
            from prompture.extraction.core import extract_with_model

            result = extract_with_model(
                _PersonForExtract,
                "Alice is 25.",
                model_name="fake/model",
                examples_store=store,
                examples_k=0,
            )

        assert "Input:" not in captured[0]
        assert "few_shot" not in result["usage"]

    def test_failed_select_continues_extraction(self) -> None:
        """If the store's select() blows up, extraction still runs without examples."""

        class _BadStore:
            def select(self, query: str, k: int = 3):
                raise RuntimeError("store on fire")

        captured: list[str] = []

        def _fake_extract_and_jsonify(*, text, instruction_template, **kw):
            captured.append(instruction_template)
            return {
                "json_string": '{"name": "Alice", "age": 25}',
                "json_object": {"name": "Alice", "age": 25},
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0.0},
            }

        with patch("prompture.extraction.core.extract_and_jsonify", _fake_extract_and_jsonify):
            from prompture.extraction.core import extract_with_model

            result = extract_with_model(
                _PersonForExtract,
                "Alice is 25.",
                model_name="fake/model",
                examples_store=_BadStore(),
                examples_k=2,
            )

        # Extraction succeeded with no examples injected.
        assert "Input:" not in captured[0]
        assert "few_shot" not in result["usage"]
        assert result["json_object"]["name"] == "Alice"
