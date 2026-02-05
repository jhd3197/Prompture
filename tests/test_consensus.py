"""Tests for extraction consensus module."""

from unittest.mock import patch

import pytest

from prompture.consensus import (
    ConsensusResult,
    ModelVote,
    _compute_highest_confidence_consensus,
    _compute_majority_consensus,
    _compute_unanimous_consensus,
    _compute_weighted_average_consensus,
    _find_majority_value,
    _values_equal,
    extract_with_consensus,
)


class TestValuesEqual:
    """Tests for value equality comparison."""

    def test_equal_strings(self):
        """Should match equal strings."""
        assert _values_equal("hello", "hello")
        assert not _values_equal("hello", "world")

    def test_equal_integers(self):
        """Should match equal integers."""
        assert _values_equal(42, 42)
        assert not _values_equal(42, 43)

    def test_equal_floats_exact(self):
        """Should match exactly equal floats."""
        assert _values_equal(3.14, 3.14)

    def test_equal_floats_tolerance(self):
        """Should match floats within tolerance."""
        assert _values_equal(100.0, 100.5, tolerance=0.01)  # Within 1%
        assert not _values_equal(100.0, 105.0, tolerance=0.01)  # Outside 1%

    def test_equal_lists(self):
        """Should compare lists element-wise."""
        assert _values_equal([1, 2, 3], [1, 2, 3])
        assert not _values_equal([1, 2, 3], [1, 2, 4])
        assert not _values_equal([1, 2], [1, 2, 3])

    def test_equal_dicts(self):
        """Should compare dicts recursively."""
        assert _values_equal({"a": 1, "b": 2}, {"a": 1, "b": 2})
        assert _values_equal({"a": 1, "b": 2}, {"b": 2, "a": 1})  # Order independent
        assert not _values_equal({"a": 1}, {"a": 2})
        assert not _values_equal({"a": 1}, {"a": 1, "b": 2})

    def test_equal_nested(self):
        """Should compare nested structures."""
        d1 = {"person": {"name": "John", "age": 30}}
        d2 = {"person": {"name": "John", "age": 30}}
        d3 = {"person": {"name": "John", "age": 31}}

        assert _values_equal(d1, d2)
        assert not _values_equal(d1, d3)

    def test_type_mismatch(self):
        """Should return False for different types."""
        assert not _values_equal("42", 42)
        assert not _values_equal([1], {"0": 1})

    def test_int_float_coercion(self):
        """Should allow int/float comparison."""
        assert _values_equal(42, 42.0)
        assert _values_equal(100, 100.5, tolerance=0.01)


class TestFindMajorityValue:
    """Tests for majority value detection."""

    def test_clear_majority(self):
        """Should find clear majority value."""
        value, count = _find_majority_value(["a", "a", "a", "b"])
        assert value == "a"
        assert count == 3

    def test_tie(self):
        """Should return one of the tied values."""
        value, count = _find_majority_value(["a", "a", "b", "b"])
        assert value in ("a", "b")
        assert count == 2

    def test_empty_list(self):
        """Should handle empty list."""
        value, count = _find_majority_value([])
        assert value is None
        assert count == 0

    def test_single_value(self):
        """Should handle single value."""
        value, count = _find_majority_value(["only"])
        assert value == "only"
        assert count == 1

    def test_dict_values(self):
        """Should handle dict values (unhashable)."""
        value, count = _find_majority_value([{"a": 1}, {"a": 1}, {"b": 2}])
        assert value == {"a": 1}
        assert count == 2

    def test_list_values(self):
        """Should handle list values (unhashable)."""
        value, count = _find_majority_value([[1, 2], [1, 2], [3, 4]])
        assert value == [1, 2]
        assert count == 2


class TestComputeMajorityConsensus:
    """Tests for majority consensus computation."""

    def test_unanimous_agreement(self):
        """Should achieve full agreement when all match."""
        votes = [
            ModelVote(model="m1", result={"name": "John", "age": 30}, success=True),
            ModelVote(model="m2", result={"name": "John", "age": 30}, success=True),
            ModelVote(model="m3", result={"name": "John", "age": 30}, success=True),
        ]

        consensus, ratio, disagreements = _compute_majority_consensus(votes)

        assert consensus == {"name": "John", "age": 30}
        assert ratio == 1.0
        assert disagreements == {}

    def test_partial_disagreement(self):
        """Should identify partial disagreements."""
        votes = [
            ModelVote(model="m1", result={"name": "John", "age": 30}, success=True),
            ModelVote(model="m2", result={"name": "John", "age": 31}, success=True),
            ModelVote(model="m3", result={"name": "John", "age": 30}, success=True),
        ]

        consensus, ratio, disagreements = _compute_majority_consensus(votes)

        assert consensus["name"] == "John"
        assert consensus["age"] == 30  # Majority wins
        assert ratio < 1.0
        assert "age" in disagreements
        assert set(disagreements["age"]) == {30, 31}

    def test_full_disagreement(self):
        """Should handle complete disagreement."""
        votes = [
            ModelVote(model="m1", result={"name": "John"}, success=True),
            ModelVote(model="m2", result={"name": "Jane"}, success=True),
            ModelVote(model="m3", result={"name": "Bob"}, success=True),
        ]

        consensus, _ratio, disagreements = _compute_majority_consensus(votes)

        assert consensus["name"] in ("John", "Jane", "Bob")
        assert "name" in disagreements
        assert len(disagreements["name"]) == 3

    def test_empty_votes(self):
        """Should handle empty votes list."""
        consensus, ratio, disagreements = _compute_majority_consensus([])

        assert consensus == {}
        assert ratio == 0.0
        assert disagreements == {}

    def test_filters_failed_votes(self):
        """Should only use successful votes."""
        votes = [
            ModelVote(model="m1", result={"name": "John"}, success=True),
            ModelVote(model="m2", result={}, success=False),
            ModelVote(model="m3", result={"name": "John"}, success=True),
        ]

        consensus, ratio, _disagreements = _compute_majority_consensus(votes)

        assert consensus == {"name": "John"}
        assert ratio == 1.0


class TestComputeUnanimousConsensus:
    """Tests for unanimous consensus computation."""

    def test_unanimous_success(self):
        """Should succeed when all agree."""
        votes = [
            ModelVote(model="m1", result={"name": "John"}, success=True),
            ModelVote(model="m2", result={"name": "John"}, success=True),
        ]

        consensus, ratio, disagreements = _compute_unanimous_consensus(votes)

        assert consensus == {"name": "John"}
        assert ratio == 1.0
        assert disagreements == {}

    def test_unanimous_failure(self):
        """Should raise when not unanimous."""
        votes = [
            ModelVote(model="m1", result={"name": "John"}, success=True),
            ModelVote(model="m2", result={"name": "Jane"}, success=True),
        ]

        with pytest.raises(ValueError, match="Unanimous consensus not achieved"):
            _compute_unanimous_consensus(votes)


class TestComputeHighestConfidenceConsensus:
    """Tests for highest confidence consensus."""

    def test_selects_highest_confidence(self):
        """Should select result from most confident model."""
        votes = [
            ModelVote(model="m1", result={"name": "John"}, confidence=0.7, success=True),
            ModelVote(model="m2", result={"name": "Jane"}, confidence=0.9, success=True),
            ModelVote(model="m3", result={"name": "Bob"}, confidence=0.5, success=True),
        ]

        consensus, confidence, _disagreements = _compute_highest_confidence_consensus(votes)

        assert consensus == {"name": "Jane"}
        assert confidence == 0.9

    def test_handles_none_confidence(self):
        """Should treat None confidence as 0."""
        votes = [
            ModelVote(model="m1", result={"name": "John"}, confidence=None, success=True),
            ModelVote(model="m2", result={"name": "Jane"}, confidence=0.5, success=True),
        ]

        consensus, confidence, _ = _compute_highest_confidence_consensus(votes)

        assert consensus == {"name": "Jane"}
        assert confidence == 0.5


class TestComputeWeightedAverageConsensus:
    """Tests for weighted average consensus."""

    def test_averages_numeric_fields(self):
        """Should compute weighted average for numeric fields."""
        votes = [
            ModelVote(model="openai/gpt-4o", result={"value": 100}, success=True),
            ModelVote(model="openai/gpt-4o-mini", result={"value": 80}, success=True),
        ]

        # Default weights: standard=1.0, mini=0.7
        consensus, _, _ = _compute_weighted_average_consensus(votes)

        # Weighted average: (100*1.0 + 80*0.7) / (1.0 + 0.7) = 156/1.7 = 91.76
        assert 85 < consensus["value"] < 100

    def test_majority_for_non_numeric(self):
        """Should use majority vote for non-numeric fields."""
        votes = [
            ModelVote(model="m1", result={"name": "John"}, success=True),
            ModelVote(model="m2", result={"name": "John"}, success=True),
            ModelVote(model="m3", result={"name": "Jane"}, success=True),
        ]

        consensus, _, _ = _compute_weighted_average_consensus(votes)

        assert consensus["name"] == "John"

    def test_custom_weights(self):
        """Should use custom model weights."""
        votes = [
            ModelVote(model="m1", result={"value": 100}, success=True),
            ModelVote(model="m2", result={"value": 200}, success=True),
        ]

        weights = {"m1": 3.0, "m2": 1.0}
        consensus, _, _ = _compute_weighted_average_consensus(votes, weights)

        # Weighted average: (100*3 + 200*1) / 4 = 500/4 = 125
        assert consensus["value"] == 125

    def test_preserves_int_type(self):
        """Should preserve int type when all inputs are ints."""
        votes = [
            ModelVote(model="m1", result={"count": 10}, success=True),
            ModelVote(model="m2", result={"count": 10}, success=True),
        ]

        consensus, _, _ = _compute_weighted_average_consensus(votes)

        assert isinstance(consensus["count"], int)


class TestModelVote:
    """Tests for ModelVote dataclass."""

    def test_successful_vote(self):
        """Should create successful vote."""
        vote = ModelVote(
            model="openai/gpt-4o",
            result={"name": "John"},
            confidence=0.9,
            cost=0.001,
            tokens=100,
            success=True,
        )

        assert vote.model == "openai/gpt-4o"
        assert vote.result == {"name": "John"}
        assert vote.confidence == 0.9
        assert vote.success
        assert vote.error is None

    def test_failed_vote(self):
        """Should create failed vote."""
        vote = ModelVote(
            model="openai/gpt-4o",
            result={},
            success=False,
            error="API error",
        )

        assert not vote.success
        assert vote.error == "API error"


class TestConsensusResult:
    """Tests for ConsensusResult dataclass."""

    def test_to_dict(self):
        """Should serialize to dict."""
        result = ConsensusResult(
            consensus={"name": "John"},
            confidence=0.9,
            agreement_ratio=0.8,
            votes=[ModelVote(model="m1", result={"name": "John"}, success=True)],
            disagreements={},
            strategy="majority_vote",
            total_cost=0.01,
            total_tokens=500,
        )

        d = result.to_dict()

        assert d["consensus"] == {"name": "John"}
        assert d["confidence"] == 0.9
        assert d["agreement_ratio"] == 0.8
        assert len(d["votes"]) == 1
        assert d["strategy"] == "majority_vote"
        assert d["total_cost"] == 0.01
        assert d["total_tokens"] == 500


class TestExtractWithConsensus:
    """Tests for extract_with_consensus function."""

    def test_requires_models(self):
        """Should require at least one model."""
        from pydantic import BaseModel

        class Person(BaseModel):
            name: str

        with pytest.raises(ValueError, match="(?i)at least one model"):
            extract_with_consensus(Person, "text", models=[])

    @patch("prompture.consensus._extract_single_model")
    def test_parallel_execution(self, mock_extract):
        """Should execute models in parallel by default."""
        from pydantic import BaseModel

        class Person(BaseModel):
            name: str
            age: int

        mock_extract.return_value = ModelVote(
            model="test",
            result={"name": "John", "age": 30},
            cost=0.001,
            tokens=100,
            success=True,
        )

        result = extract_with_consensus(
            Person,
            "John is 30 years old",
            models=["openai/gpt-4o", "claude/claude-sonnet-4"],
            parallel=True,
        )

        assert mock_extract.call_count == 2
        assert result.consensus == {"name": "John", "age": 30}
        assert result.total_cost == 0.002
        assert result.total_tokens == 200

    @patch("prompture.consensus._extract_single_model")
    def test_sequential_execution(self, mock_extract):
        """Should execute models sequentially when parallel=False."""
        from pydantic import BaseModel

        class Person(BaseModel):
            name: str

        mock_extract.return_value = ModelVote(
            model="test",
            result={"name": "John"},
            cost=0.001,
            tokens=100,
            success=True,
        )

        extract_with_consensus(
            Person,
            "John",
            models=["m1", "m2"],
            parallel=False,
        )

        assert mock_extract.call_count == 2

    @patch("prompture.consensus._extract_single_model")
    def test_majority_vote_strategy(self, mock_extract):
        """Should use majority vote by default."""
        from pydantic import BaseModel

        class Person(BaseModel):
            name: str

        # Two say John, one says Jane
        mock_extract.side_effect = [
            ModelVote(model="m1", result={"name": "John"}, cost=0.001, tokens=100, success=True),
            ModelVote(model="m2", result={"name": "John"}, cost=0.001, tokens=100, success=True),
            ModelVote(model="m3", result={"name": "Jane"}, cost=0.001, tokens=100, success=True),
        ]

        result = extract_with_consensus(
            Person,
            "text",
            models=["m1", "m2", "m3"],
            strategy="majority_vote",
        )

        assert result.consensus["name"] == "John"
        assert result.strategy == "majority_vote"

    @patch("prompture.consensus._extract_single_model")
    def test_unanimous_strategy_fails(self, mock_extract):
        """Should fail unanimous strategy when disagreement exists."""
        from pydantic import BaseModel

        class Person(BaseModel):
            name: str

        mock_extract.side_effect = [
            ModelVote(model="m1", result={"name": "John"}, cost=0.001, tokens=100, success=True),
            ModelVote(model="m2", result={"name": "Jane"}, cost=0.001, tokens=100, success=True),
        ]

        with pytest.raises(ValueError, match="Unanimous consensus not achieved"):
            extract_with_consensus(
                Person,
                "text",
                models=["m1", "m2"],
                strategy="unanimous",
            )

    @patch("prompture.consensus._extract_single_model")
    def test_min_agreement_threshold(self, mock_extract):
        """Should fail when below min_agreement threshold."""
        from pydantic import BaseModel

        class Person(BaseModel):
            name: str

        # All different - low agreement
        mock_extract.side_effect = [
            ModelVote(model="m1", result={"name": "John"}, cost=0.001, tokens=100, success=True),
            ModelVote(model="m2", result={"name": "Jane"}, cost=0.001, tokens=100, success=True),
            ModelVote(model="m3", result={"name": "Bob"}, cost=0.001, tokens=100, success=True),
        ]

        with pytest.raises(ValueError, match="Agreement ratio .* below minimum"):
            extract_with_consensus(
                Person,
                "text",
                models=["m1", "m2", "m3"],
                min_agreement=0.9,  # Require 90% agreement
            )

    @patch("prompture.consensus._extract_single_model")
    def test_handles_extraction_failures(self, mock_extract):
        """Should handle some extraction failures gracefully."""
        from pydantic import BaseModel

        class Person(BaseModel):
            name: str

        mock_extract.side_effect = [
            ModelVote(model="m1", result={"name": "John"}, cost=0.001, tokens=100, success=True),
            ModelVote(model="m2", result={}, success=False, error="API error"),
            ModelVote(model="m3", result={"name": "John"}, cost=0.001, tokens=100, success=True),
        ]

        result = extract_with_consensus(
            Person,
            "text",
            models=["m1", "m2", "m3"],
        )

        # Should still succeed with 2 of 3 models
        assert result.consensus["name"] == "John"
        assert len([v for v in result.votes if v.success]) == 2
        assert len([v for v in result.votes if not v.success]) == 1

    @patch("prompture.consensus._extract_single_model")
    def test_all_extractions_fail(self, mock_extract):
        """Should fail when all extractions fail."""
        from pydantic import BaseModel

        class Person(BaseModel):
            name: str

        mock_extract.return_value = ModelVote(model="m1", result={}, success=False, error="API error")

        with pytest.raises(ValueError, match="All model extractions failed"):
            extract_with_consensus(
                Person,
                "text",
                models=["m1", "m2"],
            )

    @patch("prompture.consensus._extract_single_model")
    def test_disagreement_detection(self, mock_extract):
        """Should detect and report disagreements."""
        from pydantic import BaseModel

        class Person(BaseModel):
            name: str
            age: int

        mock_extract.side_effect = [
            ModelVote(model="m1", result={"name": "John", "age": 30}, cost=0.001, tokens=100, success=True),
            ModelVote(model="m2", result={"name": "John", "age": 31}, cost=0.001, tokens=100, success=True),
        ]

        result = extract_with_consensus(
            Person,
            "text",
            models=["m1", "m2"],
            min_agreement=0.0,  # Allow any agreement
        )

        assert "name" not in result.disagreements  # Both agree on name
        assert "age" in result.disagreements
        assert set(result.disagreements["age"]) == {30, 31}

    @patch("prompture.consensus._extract_single_model")
    def test_cost_aggregation(self, mock_extract):
        """Should aggregate costs across all models."""
        from pydantic import BaseModel

        class Person(BaseModel):
            name: str

        mock_extract.side_effect = [
            ModelVote(model="m1", result={"name": "John"}, cost=0.001, tokens=100, success=True),
            ModelVote(model="m2", result={"name": "John"}, cost=0.002, tokens=200, success=True),
            ModelVote(model="m3", result={"name": "John"}, cost=0.003, tokens=300, success=True),
        ]

        result = extract_with_consensus(
            Person,
            "text",
            models=["m1", "m2", "m3"],
        )

        assert result.total_cost == 0.006
        assert result.total_tokens == 600


class TestExtractWithConsensusAsync:
    """Tests for async consensus extraction."""

    @pytest.mark.asyncio
    @patch("prompture.consensus._extract_single_model")
    async def test_async_execution(self, mock_extract):
        """Should run asynchronously."""
        from pydantic import BaseModel

        from prompture.consensus import extract_with_consensus_async

        class Person(BaseModel):
            name: str

        mock_extract.return_value = ModelVote(
            model="m1",
            result={"name": "John"},
            cost=0.001,
            tokens=100,
            success=True,
        )

        result = await extract_with_consensus_async(
            Person,
            "text",
            models=["m1", "m2"],
        )

        assert result.consensus["name"] == "John"
