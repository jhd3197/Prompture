"""Phase 8C: synthetic dataset modernisation.

Covers four orthogonal feature areas:

* :class:`QualityFilter` — shape / length / refusal-gating filters
* :func:`dedup_exact` / :func:`dedup_shingle` / :func:`dedup_semantic` —
  three dedup strategies + the unified :class:`DedupConfig` entry
* :class:`EvolInstruct` — depth / breadth / reasoning evolution
* :class:`SelfInstruct` — seed-based expansion
* :func:`generate_qa_dataset` wiring — personas, quality_filter, and
  dedup parameters

Every test stubs the LLM via :func:`unittest.mock.patch` so the suite
never touches a provider.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from prompture.dataset import (
    DedupConfig,
    EvolInstruct,
    QAPair,
    QualityFilter,
    SelfInstruct,
    apply_dedup,
    dedup_exact,
    dedup_semantic,
    dedup_shingle,
    generate_qa_dataset,
    length_filter,
    refusal_filter,
    shape_filter,
)
from prompture.dataset.filters import FilterDecision
from prompture.rag.documents import Document

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _pair(q: str, a: str) -> QAPair:
    return QAPair(question=q, answer=a)


class _ManualEmbedder:
    """Returns a fixed vector per text. Used to control dedup tests."""

    def __init__(self, by_text: dict[str, list[float]]) -> None:
        self._by_text = by_text

    def embed(self, texts: list[str], options: dict[str, Any]) -> dict[str, Any]:
        return {"embeddings": [self._by_text[t] for t in texts]}


# ---------------------------------------------------------------------------
# QualityFilter
# ---------------------------------------------------------------------------


class TestShapeFilter:
    def test_drops_empty_question(self) -> None:
        decision = shape_filter()(_pair("", "answer"))
        assert decision.keep is False
        assert "empty_field" in decision.reason

    def test_drops_empty_answer(self) -> None:
        decision = shape_filter()(_pair("What?", ""))
        assert decision.keep is False

    def test_drops_identical_q_and_a(self) -> None:
        decision = shape_filter()(_pair("Same", "Same"))
        assert decision.keep is False
        assert "identical" in decision.reason

    def test_drops_non_question(self) -> None:
        # No ``?``, no interrogative prefix.
        decision = shape_filter()(_pair("This is a statement.", "Some answer."))
        assert decision.keep is False
        assert "not_a_question" in decision.reason

    def test_keeps_interrogative_without_questionmark(self) -> None:
        # Starts with "What" / "Why" / etc. — keep even without "?".
        decision = shape_filter()(_pair("Define osmosis", "Osmosis is..."))
        assert decision.keep is True

    def test_keeps_normal_question(self) -> None:
        decision = shape_filter()(_pair("What is the capital of France?", "Paris."))
        assert decision.keep is True


class TestLengthFilter:
    def test_drops_short_question(self) -> None:
        d = length_filter(min_question_chars=10)(_pair("Hi?", "answer text"))
        assert d.keep is False
        assert "short_question" in d.reason

    def test_drops_long_answer(self) -> None:
        d = length_filter(max_answer_chars=20)(_pair("What is X?", "y" * 1000))
        assert d.keep is False
        assert "long_answer" in d.reason

    def test_keeps_within_bounds(self) -> None:
        d = length_filter()(_pair("What is the capital of France?", "Paris is the capital."))
        assert d.keep is True

    def test_disabling_upper_bound(self) -> None:
        d = length_filter(max_answer_chars=None)(
            _pair("What is X?", "x" * 100_000)
        )
        assert d.keep is True


class TestRefusalFilter:
    def test_drops_refusal_answer(self) -> None:
        f = refusal_filter()
        d = f(_pair("How do I hack X?", "I'm sorry, but I can't help with that."))
        assert d.keep is False
        assert "refusal" in d.reason

    def test_keeps_genuine_answer(self) -> None:
        f = refusal_filter()
        d = f(_pair("What is the capital of France?", "Paris is the capital of France."))
        assert d.keep is True


class TestQualityFilterPipeline:
    def test_filter_collects_stats(self) -> None:
        pairs = [
            _pair("What is the capital of France?", "Paris."),
            _pair("", "empty Q"),
            _pair("Hi?", "y"),  # short answer (default min=1, but pass — adjust)
            _pair("This is not a question.", "Some prose."),
        ]
        clean, stats = QualityFilter().filter(pairs)
        assert stats.total_in == 4
        assert len(clean) <= stats.total_in
        assert stats.total_out == len(clean)
        # At least the empty-Q and non-question rows must drop.
        assert stats.total_in - stats.total_out >= 2
        assert sum(stats.dropped_by_reason.values()) == stats.total_in - stats.total_out

    def test_iter_yields_only_kept(self) -> None:
        pairs = [_pair("What is the capital of France?", "Paris."), _pair("", "")]
        out = list(QualityFilter().iter(pairs))
        assert len(out) == 1

    def test_filter_decision_helpers(self) -> None:
        assert FilterDecision.keep_().keep
        assert not FilterDecision.drop("x").keep
        assert FilterDecision.drop("foo").reason == "drop:foo"

    def test_first_failing_predicate_short_circuits(self) -> None:
        calls = {"count": 0}

        def _counting_filter():
            def _p(_pair: Any) -> FilterDecision:
                calls["count"] += 1
                return FilterDecision.keep_()

            return _p

        filt = QualityFilter(predicates=[shape_filter(), _counting_filter()])
        # Run a row that fails the first predicate.
        filt.evaluate(_pair("", ""))
        # The second predicate must NOT have been called.
        assert calls["count"] == 0


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------


class TestDedupExact:
    def test_drops_normalised_duplicates(self) -> None:
        pairs = [
            _pair("What is the Capital of France?", "Paris."),
            _pair("what is the capital of france?", "Paris."),  # case-only dup
            _pair("What about Germany?", "Berlin."),
        ]
        kept, removed = dedup_exact(pairs)
        assert len(kept) == 2
        assert removed == 1

    def test_empty_keys_dropped(self) -> None:
        kept, _ = dedup_exact([QAPair(question="", answer="a")])
        assert kept == []

    def test_preserves_order(self) -> None:
        pairs = [_pair("Q1?", "A1"), _pair("Q2?", "A2"), _pair("Q3?", "A3")]
        kept, _ = dedup_exact(pairs)
        assert [p.question for p in kept] == ["Q1?", "Q2?", "Q3?"]


class TestDedupShingle:
    def test_catches_paraphrase(self) -> None:
        pairs = [
            _pair("What is the capital of France?", "Paris."),
            _pair("What's the capital of France?", "Paris."),  # paraphrase
            _pair("Describe the moon.", "It's a satellite."),
        ]
        kept, removed = dedup_shingle(pairs, threshold=0.55, shingle_size=4)
        assert removed >= 1
        # The unrelated moon question survives.
        assert any("moon" in p.question.lower() for p in kept)

    def test_high_threshold_keeps_paraphrases(self) -> None:
        # At threshold=0.99, only carbon copies merge.
        pairs = [
            _pair("What is X?", "Y."),
            _pair("What about X?", "Y."),
        ]
        kept, removed = dedup_shingle(pairs, threshold=0.99)
        assert removed == 0
        assert len(kept) == 2


class TestDedupSemantic:
    def test_drops_close_embeddings(self) -> None:
        embedder = _ManualEmbedder(
            {
                "Q1": [1.0, 0.0],
                "Q2": [0.99, 0.01],  # close to Q1
                "Q3": [0.0, 1.0],  # orthogonal
            }
        )
        pairs = [_pair("Q1", "A"), _pair("Q2", "A"), _pair("Q3", "A")]
        kept, removed = dedup_semantic(pairs, embedder=embedder, threshold=0.9)
        assert removed == 1
        assert {p.question for p in kept} == {"Q1", "Q3"}

    def test_handles_empty_keys(self) -> None:
        embedder = _ManualEmbedder({"Q": [1.0, 0.0]})
        pairs = [_pair("", "A"), _pair("Q", "A")]
        kept, removed = dedup_semantic(pairs, embedder=embedder)
        assert removed == 1  # empty-Q dropped
        assert len(kept) == 1

    def test_wrong_vector_count_raises(self) -> None:
        class _Bad:
            def embed(self, texts: list[str], options: dict) -> dict:
                return {"embeddings": [[0.0]] * (len(texts) - 1)}

        with pytest.raises(ValueError, match="vectors"):
            dedup_semantic([_pair("A", "B")], embedder=_Bad())


class TestApplyDedup:
    def test_exact(self) -> None:
        kept, _ = apply_dedup([_pair("A?", "x"), _pair("A?", "x")], DedupConfig(strategy="exact"))
        assert len(kept) == 1

    def test_none_passthrough(self) -> None:
        pairs = [_pair("A?", "x"), _pair("A?", "x")]
        kept, removed = apply_dedup(pairs, DedupConfig(strategy="none"))
        assert kept == pairs
        assert removed == 0

    def test_semantic_requires_embedder(self) -> None:
        with pytest.raises(ValueError, match="embedder"):
            apply_dedup([_pair("Q", "A")], DedupConfig(strategy="semantic"))

    def test_unknown_strategy_raises(self) -> None:
        with pytest.raises(ValueError, match="strategy"):
            apply_dedup([], DedupConfig(strategy="bogus"))


# ---------------------------------------------------------------------------
# EvolInstruct
# ---------------------------------------------------------------------------


class TestEvolInstruct:
    def test_evolve_pair_returns_new_qapair(self) -> None:
        def _fake_extract(*args: Any, **kwargs: Any) -> dict[str, Any]:
            return {
                "model": QAPair(question="Evolved Q?", answer="Evolved A."),
                "json_string": "",
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.001},
            }

        with patch("prompture.dataset.evol.extract_with_model", _fake_extract):
            evol = EvolInstruct(model="fake/model")
            evolved, meta = evol.evolve_pair(_pair("Original Q?", "Original A."), operator="depth")
            assert evolved is not None
            assert evolved.question == "Evolved Q?"
            assert meta.get("prompt_tokens") == 10

    def test_evolve_pair_drops_identical_question(self) -> None:
        identical = QAPair(question="Original?", answer="Same.")

        def _fake(*args: Any, **kwargs: Any) -> dict[str, Any]:
            return {"model": identical, "json_string": "", "usage": {}}

        with patch("prompture.dataset.evol.extract_with_model", _fake):
            evol = EvolInstruct(model="fake")
            evolved, _ = evol.evolve_pair(_pair("Original?", "Original."), operator="depth")
            assert evolved is None

    def test_evolve_pair_returns_none_on_extract_failure(self) -> None:
        def _boom(*args: Any, **kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("driver down")

        with patch("prompture.dataset.evol.extract_with_model", _boom):
            evol = EvolInstruct(model="fake")
            evolved, _ = evol.evolve_pair(_pair("Q?", "A."))
            assert evolved is None

    def test_evolve_batch_runs_all_operators(self) -> None:
        outputs = iter(["d1", "b1", "r1", "d2", "b2", "r2"])

        def _fake(*args: Any, **kwargs: Any) -> dict[str, Any]:
            n = next(outputs)
            return {
                "model": QAPair(question=f"Evolved {n}?", answer=f"Answer {n}"),
                "json_string": "",
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2, "cost": 0.0},
            }

        with patch("prompture.dataset.evol.extract_with_model", _fake):
            evol = EvolInstruct(model="fake")
            result = evol.evolve_batch(
                [_pair("S1?", "a"), _pair("S2?", "b")],
                operators=["depth", "breadth", "reasoning"],
                rounds=1,
            )
            assert len(result.evolved) == 6
            assert result.failures == 0
            assert result.meta["calls"] == 6

    def test_evolve_batch_rounds_compound(self) -> None:
        # Two sources × one operator × two rounds = 4 evolved.
        outputs = iter(["e1", "e2", "e3", "e4"])

        def _fake(*args: Any, **kwargs: Any) -> dict[str, Any]:
            n = next(outputs)
            return {
                "model": QAPair(question=f"Q-{n}?", answer="."),
                "json_string": "",
                "usage": {},
            }

        with patch("prompture.dataset.evol.extract_with_model", _fake):
            evol = EvolInstruct(model="fake")
            result = evol.evolve_batch(
                [_pair("S1?", "a"), _pair("S2?", "b")],
                operators=["depth"],
                rounds=2,
            )
            assert len(result.evolved) == 4

    def test_init_validates_operator_prompts(self) -> None:
        with pytest.raises(ValueError, match="depth"):
            EvolInstruct(model="x", operator_prompts={"breadth": "{question} {answer}"})
        with pytest.raises(ValueError, match=r"\{question\}"):
            EvolInstruct(
                model="x",
                operator_prompts={
                    "depth": "no placeholders",
                    "breadth": "{question} {answer}",
                    "reasoning": "{question} {answer}",
                },
            )

    def test_invalid_operator_arg_raises(self) -> None:
        with patch("prompture.dataset.evol.extract_with_model", lambda *a, **k: {}):
            evol = EvolInstruct(model="x")
            with pytest.raises(ValueError, match="Unknown operator"):
                evol.evolve_batch([_pair("Q?", "A")], operators=["bogus"])  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# SelfInstruct
# ---------------------------------------------------------------------------


class TestSelfInstruct:
    def _fake_batch(self, pairs: list[QAPair]) -> Any:
        from prompture.dataset.schemas import QAPairBatch

        def _make(*args: Any, **kwargs: Any) -> dict[str, Any]:
            return {
                "model": QAPairBatch(pairs=list(pairs)),
                "json_string": "",
                "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10, "cost": 0.0},
            }

        return _make

    def test_expand_produces_target_count(self) -> None:
        new = [_pair(f"Q{i}?", f"A{i}") for i in range(5)]
        with patch("prompture.dataset.seed.extract_with_model", self._fake_batch(new)):
            si = SelfInstruct(model="fake", seed_random=__import__("random").Random(0))
            result = si.expand(
                [_pair("Seed?", "answer")],
                target_count=3,
                batch_size=5,
            )
            assert len(result.pairs) == 3
            # Seeds should not appear in the result.
            assert all(p.question != "Seed?" for p in result.pairs)

    def test_seed_required(self) -> None:
        with pytest.raises(ValueError, match="seed"):
            SelfInstruct(model="x").expand([], target_count=5)

    def test_target_count_bounds(self) -> None:
        with pytest.raises(ValueError, match="target_count"):
            SelfInstruct(model="x").expand([_pair("S?", "a")], target_count=0)

    def test_dedup_against_seeds(self) -> None:
        # Generated pair collides with seed -> dropped.
        seed = _pair("Match?", "OK.")
        with patch("prompture.dataset.seed.extract_with_model", self._fake_batch([seed])):
            si = SelfInstruct(model="fake", seed_random=__import__("random").Random(0))
            result = si.expand([seed], target_count=2, batch_size=2, max_rounds=2)
            # The fake always returns the seed itself — every gen gets dropped.
            assert result.pairs == []

    def test_max_rounds_caps_loop(self) -> None:
        # Fake returns the SAME pair every call so dedup keeps culling.
        with patch(
            "prompture.dataset.seed.extract_with_model",
            self._fake_batch([_pair("Repeat?", "OK.")]),
        ):
            si = SelfInstruct(model="fake", seed_random=__import__("random").Random(0))
            result = si.expand([_pair("Seed?", "a")], target_count=10, max_rounds=3)
            # The first round produces one new pair; subsequent rounds find
            # only duplicates of that pair. We stop at max_rounds = 3.
            assert result.rounds_completed == 3
            assert len(result.pairs) == 1


# ---------------------------------------------------------------------------
# generate_qa_dataset wiring
# ---------------------------------------------------------------------------


def _fake_chunker_class():
    from prompture.rag.chunkers.base import TextChunker

    class _SplitOnPipe(TextChunker):
        def __init__(self) -> None:
            super().__init__(chunk_size=1000, chunk_overlap=0)

        def split_text(self, text: str) -> list[str]:
            return [p.strip() for p in text.split("|") if p.strip()]

    return _SplitOnPipe()


class TestGenerateQADatasetWiring:
    def _stub_extract(self, pairs_per_call: list[QAPair]) -> Any:
        from prompture.dataset.schemas import QAPairBatch

        def _fake(*args: Any, **kwargs: Any) -> dict[str, Any]:
            return {
                "model": QAPairBatch(pairs=list(pairs_per_call)),
                "json_string": "",
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0.0},
            }

        return _fake

    def test_personas_multiply_calls(self) -> None:
        # Two personas, two chunks → 4 extract calls.
        class _StubPersona:
            def __init__(self, name: str, voice: str) -> None:
                self.name = name
                self._voice = voice

            def render(self) -> str:
                return self._voice

        calls: list[str] = []

        def _capture(*args: Any, **kwargs: Any) -> dict[str, Any]:
            from prompture.dataset.schemas import QAPairBatch

            # ``text`` positional captures the rendered prompt.
            calls.append(args[1] if len(args) > 1 else kwargs.get("text", ""))
            return {
                "model": QAPairBatch(pairs=[_pair("Q?", "A.")]),
                "json_string": "",
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0.0},
            }

        with patch("prompture.dataset.synth.extract_with_model", _capture):
            doc = Document(content="alpha | beta")
            pairs = generate_qa_dataset(
                [doc],
                model="fake/model",
                chunker=_fake_chunker_class(),
                personas=[_StubPersona("p1", "VOICE_ONE"), _StubPersona("p2", "VOICE_TWO")],
            )
        # 2 chunks * 2 personas = 4 extraction calls.
        assert len(calls) == 4
        # Each persona prefix appears in some prompt.
        assert any("VOICE_ONE" in p for p in calls)
        assert any("VOICE_TWO" in p for p in calls)
        # All 4 pairs come back (default dedup is exact on identical "Q?"),
        # so we expect 1 after dedup.
        assert len(pairs) == 1

    def test_quality_filter_drops_junk_pairs(self) -> None:
        from prompture.dataset.schemas import QAPairBatch

        # First call returns 2 pairs: one good, one junk.
        def _fake(*args: Any, **kwargs: Any) -> dict[str, Any]:
            return {
                "model": QAPairBatch(
                    pairs=[
                        _pair("What is the capital of France?", "Paris."),
                        _pair("", "junk"),  # empty Q → dropped by shape_filter
                    ]
                ),
                "json_string": "",
                "usage": {},
            }

        with patch("prompture.dataset.synth.extract_with_model", _fake):
            doc = Document(content="single chunk")
            result, stats = generate_qa_dataset(
                [doc],
                model="fake",
                chunker=_fake_chunker_class(),
                quality_filter=QualityFilter(),
                return_stats=True,
            )
        assert len(result) == 1
        assert stats["filter"]["dropped"] == 1
        assert stats["filter"]["total_in"] == 2

    def test_dedup_config_supersedes_legacy_dedupe(self) -> None:
        from prompture.dataset.schemas import QAPairBatch

        # 3 identical pairs across one chunk.
        def _fake(*args: Any, **kwargs: Any) -> dict[str, Any]:
            return {
                "model": QAPairBatch(
                    pairs=[
                        _pair("Same Q?", "A1."),
                        _pair("Same Q?", "A2."),
                        _pair("Same Q?", "A3."),
                    ]
                ),
                "json_string": "",
                "usage": {},
            }

        with patch("prompture.dataset.synth.extract_with_model", _fake):
            doc = Document(content="single chunk")
            # dedup=None_strategy means no dedup; legacy `dedupe=True` ignored.
            pairs = generate_qa_dataset(
                [doc],
                model="fake",
                chunker=_fake_chunker_class(),
                dedup=DedupConfig(strategy="none"),
                dedupe=True,  # would be exact by itself; superseded
            )
            assert len(pairs) == 3

    def test_return_stats_shape(self) -> None:
        from prompture.dataset.schemas import QAPairBatch

        def _fake(*args: Any, **kwargs: Any) -> dict[str, Any]:
            return {
                "model": QAPairBatch(pairs=[_pair("Q?", "A.")]),
                "json_string": "",
                "usage": {},
            }

        with patch("prompture.dataset.synth.extract_with_model", _fake):
            doc = Document(content="x")
            pairs, stats = generate_qa_dataset(
                [doc],
                model="fake",
                chunker=_fake_chunker_class(),
                return_stats=True,
            )
            assert isinstance(pairs, list)
            assert "raw_count" in stats


# ---------------------------------------------------------------------------
# Package exports
# ---------------------------------------------------------------------------


class TestDatasetExports:
    def test_all_new_symbols_exported(self) -> None:
        from prompture.dataset import (  # noqa: F401
            DedupConfig,
            EvolInstruct,
            EvolResult,
            FilterDecision,
            FilterStats,
            QualityFilter,
            SelfInstruct,
            SelfInstructResult,
            apply_dedup,
            dedup_exact,
            dedup_semantic,
            dedup_shingle,
            length_filter,
            refusal_filter,
            shape_filter,
        )
