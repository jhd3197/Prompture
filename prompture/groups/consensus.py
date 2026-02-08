"""Extraction consensus across multiple models.

Run the same extraction task across multiple LLM models and find consensus
on the results. Useful for high-stakes extractions where accuracy is critical.

Features:
- Multiple consensus strategies (majority_vote, unanimous, highest_confidence, weighted_average)
- Parallel or sequential execution
- Per-field disagreement detection
- Aggregated cost tracking
- Sync and async variants
"""

from __future__ import annotations

import asyncio
import logging
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from pydantic import BaseModel

logger = logging.getLogger("prompture.consensus")

# Type alias for consensus strategies
ConsensusStrategy = Literal["majority_vote", "unanimous", "highest_confidence", "weighted_average"]


@dataclass
class ModelVote:
    """Single model's extraction result.

    Args:
        model: The model string that produced this result.
        result: The extracted JSON object.
        confidence: Self-reported or inferred confidence (0.0-1.0).
        cost: Cost in USD for this extraction.
        tokens: Total tokens used.
        success: Whether extraction completed successfully.
        error: Error message if extraction failed.
    """

    model: str
    result: dict[str, Any]
    confidence: float | None = None
    cost: float = 0.0
    tokens: int = 0
    success: bool = True
    error: str | None = None


@dataclass
class ConsensusResult:
    """Result of consensus extraction.

    Args:
        consensus: The agreed-upon result dictionary.
        confidence: Overall confidence in the consensus (0.0-1.0).
        agreement_ratio: Fraction of models that agree with consensus (0.0-1.0).
        votes: Individual model results.
        disagreements: Fields where models disagreed, mapping field name to list of different values.
        strategy: The consensus strategy that was used.
        total_cost: Sum of costs across all model calls.
        total_tokens: Sum of tokens across all model calls.
    """

    consensus: dict[str, Any]
    confidence: float
    agreement_ratio: float
    votes: list[ModelVote]
    disagreements: dict[str, list[Any]]
    strategy: ConsensusStrategy
    total_cost: float
    total_tokens: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dictionary."""
        return {
            "consensus": self.consensus,
            "confidence": self.confidence,
            "agreement_ratio": self.agreement_ratio,
            "votes": [
                {
                    "model": v.model,
                    "result": v.result,
                    "confidence": v.confidence,
                    "cost": v.cost,
                    "tokens": v.tokens,
                    "success": v.success,
                    "error": v.error,
                }
                for v in self.votes
            ],
            "disagreements": self.disagreements,
            "strategy": self.strategy,
            "total_cost": self.total_cost,
            "total_tokens": self.total_tokens,
        }


def _values_equal(a: Any, b: Any, tolerance: float = 0.01) -> bool:
    """Check if two values are equal, with tolerance for floats.

    Args:
        a: First value.
        b: Second value.
        tolerance: Relative tolerance for float comparison.

    Returns:
        True if values are considered equal.
    """
    if type(a) is not type(b):
        # Try type coercion for common cases
        try:
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                return abs(float(a) - float(b)) <= tolerance * max(abs(float(a)), abs(float(b)), 1.0)
        except (TypeError, ValueError):
            pass
        return False

    if isinstance(a, float):
        if a == 0 and b == 0:
            return True
        return abs(a - b) <= tolerance * max(abs(a), abs(b), 1.0)

    if isinstance(a, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(_values_equal(a[k], b[k], tolerance) for k in a)

    if isinstance(a, list):
        if len(a) != len(b):
            return False
        return all(_values_equal(ai, bi, tolerance) for ai, bi in zip(a, b))

    return a == b


def _find_majority_value(values: list[Any]) -> tuple[Any, int]:
    """Find the most common value in a list.

    Args:
        values: List of values to analyze.

    Returns:
        Tuple of (majority_value, count).
    """
    # Convert unhashable types to strings for counting
    hashable_values = []
    for v in values:
        if isinstance(v, (dict, list)):
            import json

            hashable_values.append(json.dumps(v, sort_keys=True))
        else:
            hashable_values.append(v)

    counter = Counter(hashable_values)
    if not counter:
        return None, 0

    most_common_hashable, count = counter.most_common(1)[0]

    # Map back to original value
    for v, h in zip(values, hashable_values):
        if h == most_common_hashable:
            return v, count

    return most_common_hashable, count


def _compute_majority_consensus(
    votes: list[ModelVote],
) -> tuple[dict[str, Any], float, dict[str, list[Any]]]:
    """Compute consensus using majority voting per field.

    Args:
        votes: List of successful model votes.

    Returns:
        Tuple of (consensus_dict, agreement_ratio, disagreements).
    """
    if not votes:
        return {}, 0.0, {}

    # Get all field names across all results
    all_fields: set[str] = set()
    for vote in votes:
        if vote.success and isinstance(vote.result, dict):
            all_fields.update(vote.result.keys())

    consensus: dict[str, Any] = {}
    disagreements: dict[str, list[Any]] = {}
    total_agreement = 0
    field_count = 0

    for field_name in all_fields:
        # Collect values for this field from all votes
        field_values = []
        for vote in votes:
            if vote.success and isinstance(vote.result, dict) and field_name in vote.result:
                field_values.append(vote.result[field_name])

        if not field_values:
            continue

        field_count += 1
        majority_value, _ = _find_majority_value(field_values)
        consensus[field_name] = majority_value

        # Check for disagreements
        unique_values = []
        for v in field_values:
            if not any(_values_equal(v, uv) for uv in unique_values):
                unique_values.append(v)

        if len(unique_values) > 1:
            disagreements[field_name] = unique_values

        # Track agreement ratio
        agreement_count = sum(1 for v in field_values if _values_equal(v, majority_value))
        total_agreement += agreement_count / len(field_values)

    agreement_ratio = total_agreement / field_count if field_count > 0 else 0.0

    return consensus, agreement_ratio, disagreements


def _compute_unanimous_consensus(
    votes: list[ModelVote],
) -> tuple[dict[str, Any], float, dict[str, list[Any]]]:
    """Compute consensus requiring unanimous agreement.

    Args:
        votes: List of successful model votes.

    Returns:
        Tuple of (consensus_dict, agreement_ratio, disagreements).

    Raises:
        ValueError: If unanimous agreement is not achieved.
    """
    if not votes:
        return {}, 0.0, {}

    consensus, _, disagreements = _compute_majority_consensus(votes)

    if disagreements:
        raise ValueError(f"Unanimous consensus not achieved. Disagreements on fields: {list(disagreements.keys())}")

    return consensus, 1.0, {}


def _compute_highest_confidence_consensus(
    votes: list[ModelVote],
) -> tuple[dict[str, Any], float, dict[str, list[Any]]]:
    """Use result from model with highest confidence.

    Args:
        votes: List of successful model votes.

    Returns:
        Tuple of (consensus_dict, confidence, disagreements).
    """
    if not votes:
        return {}, 0.0, {}

    # Find vote with highest confidence
    best_vote = max(votes, key=lambda v: v.confidence if v.confidence is not None else 0.0)

    # Still compute disagreements for informational purposes
    _, _, disagreements = _compute_majority_consensus(votes)

    return (
        best_vote.result if isinstance(best_vote.result, dict) else {},
        best_vote.confidence or 0.0,
        disagreements,
    )


def _compute_weighted_average_consensus(
    votes: list[ModelVote],
    model_weights: dict[str, float] | None = None,
) -> tuple[dict[str, Any], float, dict[str, list[Any]]]:
    """Compute weighted average for numeric fields.

    Non-numeric fields fall back to majority voting.

    Args:
        votes: List of successful model votes.
        model_weights: Optional weights per model string.

    Returns:
        Tuple of (consensus_dict, agreement_ratio, disagreements).
    """
    if not votes:
        return {}, 0.0, {}

    # Default weights based on model tier
    if model_weights is None:
        model_weights = {}
        for vote in votes:
            model_lower = vote.model.lower()
            if any(x in model_lower for x in ["opus", "pro", "4.1", "o1", "o3"]):
                model_weights[vote.model] = 1.5
            elif any(x in model_lower for x in ["mini", "flash", "haiku", "instant"]):
                model_weights[vote.model] = 0.7
            else:
                model_weights[vote.model] = 1.0

    # Get all field names
    all_fields: set[str] = set()
    for vote in votes:
        if vote.success and isinstance(vote.result, dict):
            all_fields.update(vote.result.keys())

    consensus: dict[str, Any] = {}
    disagreements: dict[str, list[Any]] = {}

    for field_name in all_fields:
        field_values = []
        field_weights = []

        for vote in votes:
            if vote.success and isinstance(vote.result, dict) and field_name in vote.result:
                field_values.append(vote.result[field_name])
                field_weights.append(model_weights.get(vote.model, 1.0))

        if not field_values:
            continue

        # Check if all values are numeric
        all_numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in field_values)

        if all_numeric and field_values:
            # Weighted average for numeric fields
            total_weight = sum(field_weights)
            if total_weight > 0:
                weighted_sum = sum(v * w for v, w in zip(field_values, field_weights))
                avg_value = weighted_sum / total_weight
                # Preserve int type if all inputs were ints
                if all(isinstance(v, int) for v in field_values):
                    consensus[field_name] = round(avg_value)
                else:
                    consensus[field_name] = avg_value
            else:
                consensus[field_name] = field_values[0]
        else:
            # Majority vote for non-numeric
            majority_value, _ = _find_majority_value(field_values)
            consensus[field_name] = majority_value

        # Track disagreements
        unique_values = []
        for v in field_values:
            if not any(_values_equal(v, uv) for uv in unique_values):
                unique_values.append(v)
        if len(unique_values) > 1:
            disagreements[field_name] = unique_values

    # Compute agreement ratio
    _, agreement_ratio, _ = _compute_majority_consensus(votes)

    return consensus, agreement_ratio, disagreements


def _extract_single_model(
    model_cls: type[BaseModel],
    text: str,
    model_name: str,
    extract_kwargs: dict[str, Any],
) -> ModelVote:
    """Run extraction for a single model.

    Args:
        model_cls: Pydantic model class.
        text: Text to extract from.
        model_name: Model string to use.
        extract_kwargs: Additional kwargs for extract_with_model.

    Returns:
        ModelVote with extraction result.
    """
    from ..extraction.core import extract_with_model

    try:
        result = extract_with_model(
            model_cls,
            text,
            model_name,
            **extract_kwargs,
        )

        # Extract usage info
        usage = result.get("usage", {})
        cost = usage.get("cost", 0.0)
        tokens = usage.get("total_tokens", 0)

        # Try to get confidence if the model includes it
        confidence = None
        json_obj = result.get("json_object", {})
        if isinstance(json_obj, dict):
            confidence = json_obj.pop("_confidence", None)

        return ModelVote(
            model=model_name,
            result=json_obj,
            confidence=confidence,
            cost=cost,
            tokens=tokens,
            success=True,
        )

    except Exception as e:
        logger.warning("Extraction failed for model %s: %s", model_name, e)
        return ModelVote(
            model=model_name,
            result={},
            success=False,
            error=str(e),
        )


def extract_with_consensus(
    model_cls: type[BaseModel],
    text: str,
    models: list[str],
    strategy: ConsensusStrategy = "majority_vote",
    min_agreement: float = 0.5,
    parallel: bool = True,
    max_workers: int | None = None,
    model_weights: dict[str, float] | None = None,
    **extract_kwargs: Any,
) -> ConsensusResult:
    """Extract using multiple models and find consensus.

    Runs the same extraction task across multiple LLM models and
    combines their results using the specified consensus strategy.

    Args:
        model_cls: Pydantic model to extract into.
        text: Text to extract from.
        models: List of model strings (e.g., ["openai/gpt-4o", "claude/claude-sonnet-4"]).
        strategy: How to determine consensus:
            - "majority_vote": Most common value per field wins
            - "unanimous": All models must agree (raises if not)
            - "highest_confidence": Use result from most confident model
            - "weighted_average": Weighted average for numeric fields
        min_agreement: Minimum agreement ratio required (raises if not met).
        parallel: Run models in parallel (True) or sequential (False).
        max_workers: Max parallel threads (default: number of models).
        model_weights: Optional weights for weighted_average strategy.
        **extract_kwargs: Passed to extract_with_model().

    Returns:
        ConsensusResult with consensus values and metadata.

    Raises:
        ValueError: If min_agreement is not met or unanimous fails.

    Example:
        >>> from pydantic import BaseModel
        >>> class Person(BaseModel):
        ...     name: str
        ...     age: int
        >>> result = extract_with_consensus(
        ...     Person,
        ...     "John Smith is a 35-year-old engineer.",
        ...     models=["openai/gpt-4o", "claude/claude-sonnet-4", "google/gemini-2.0-flash"],
        ...     strategy="majority_vote"
        ... )
        >>> print(result.consensus)  # {"name": "John Smith", "age": 35}
        >>> print(result.agreement_ratio)  # 1.0
    """
    if not models:
        raise ValueError("At least one model must be specified")

    votes: list[ModelVote] = []

    if parallel:
        # Parallel execution
        workers = max_workers or len(models)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_extract_single_model, model_cls, text, model, extract_kwargs): model
                for model in models
            }
            for future in as_completed(futures):
                vote = future.result()
                votes.append(vote)
    else:
        # Sequential execution
        for model in models:
            vote = _extract_single_model(model_cls, text, model, extract_kwargs)
            votes.append(vote)

    # Filter successful votes for consensus
    successful_votes = [v for v in votes if v.success]

    if not successful_votes:
        raise ValueError("All model extractions failed")

    # Compute consensus based on strategy
    if strategy == "majority_vote":
        consensus, agreement_ratio, disagreements = _compute_majority_consensus(successful_votes)
        confidence = agreement_ratio
    elif strategy == "unanimous":
        consensus, agreement_ratio, disagreements = _compute_unanimous_consensus(successful_votes)
        confidence = 1.0
    elif strategy == "highest_confidence":
        consensus, confidence, disagreements = _compute_highest_confidence_consensus(successful_votes)
        agreement_ratio = _compute_majority_consensus(successful_votes)[1]
    elif strategy == "weighted_average":
        consensus, agreement_ratio, disagreements = _compute_weighted_average_consensus(successful_votes, model_weights)
        confidence = agreement_ratio
    else:
        raise ValueError(f"Unknown consensus strategy: {strategy}")

    # Check minimum agreement
    if agreement_ratio < min_agreement:
        raise ValueError(
            f"Agreement ratio {agreement_ratio:.2f} below minimum {min_agreement:.2f}. "
            f"Disagreements: {list(disagreements.keys())}"
        )

    # Aggregate costs
    total_cost = sum(v.cost for v in votes)
    total_tokens = sum(v.tokens for v in votes)

    return ConsensusResult(
        consensus=consensus,
        confidence=confidence,
        agreement_ratio=agreement_ratio,
        votes=votes,
        disagreements=disagreements,
        strategy=strategy,
        total_cost=total_cost,
        total_tokens=total_tokens,
    )


async def extract_with_consensus_async(
    model_cls: type[BaseModel],
    text: str,
    models: list[str],
    strategy: ConsensusStrategy = "majority_vote",
    min_agreement: float = 0.5,
    model_weights: dict[str, float] | None = None,
    **extract_kwargs: Any,
) -> ConsensusResult:
    """Async version of extract_with_consensus.

    Uses asyncio.gather for true parallelism with async drivers.
    Falls back to thread pool for sync drivers.

    Args:
        model_cls: Pydantic model to extract into.
        text: Text to extract from.
        models: List of model strings.
        strategy: Consensus strategy to use.
        min_agreement: Minimum agreement ratio required.
        model_weights: Optional weights for weighted_average strategy.
        **extract_kwargs: Passed to extract_with_model().

    Returns:
        ConsensusResult with consensus values and metadata.

    Example:
        >>> result = await extract_with_consensus_async(
        ...     Person,
        ...     "John is 35",
        ...     models=["openai/gpt-4o", "claude/claude-sonnet-4"],
        ... )
    """
    # Run extractions in parallel using run_in_executor for sync extract_with_model
    loop = asyncio.get_running_loop()

    async def extract_one(model: str) -> ModelVote:
        return await loop.run_in_executor(
            None,
            lambda: _extract_single_model(model_cls, text, model, extract_kwargs),
        )

    votes = await asyncio.gather(*[extract_one(model) for model in models])
    votes = list(votes)

    # Filter successful votes
    successful_votes = [v for v in votes if v.success]

    if not successful_votes:
        raise ValueError("All model extractions failed")

    # Compute consensus (same logic as sync version)
    if strategy == "majority_vote":
        consensus, agreement_ratio, disagreements = _compute_majority_consensus(successful_votes)
        confidence = agreement_ratio
    elif strategy == "unanimous":
        consensus, agreement_ratio, disagreements = _compute_unanimous_consensus(successful_votes)
        confidence = 1.0
    elif strategy == "highest_confidence":
        consensus, confidence, disagreements = _compute_highest_confidence_consensus(successful_votes)
        agreement_ratio = _compute_majority_consensus(successful_votes)[1]
    elif strategy == "weighted_average":
        consensus, agreement_ratio, disagreements = _compute_weighted_average_consensus(successful_votes, model_weights)
        confidence = agreement_ratio
    else:
        raise ValueError(f"Unknown consensus strategy: {strategy}")

    if agreement_ratio < min_agreement:
        raise ValueError(
            f"Agreement ratio {agreement_ratio:.2f} below minimum {min_agreement:.2f}. "
            f"Disagreements: {list(disagreements.keys())}"
        )

    total_cost = sum(v.cost for v in votes)
    total_tokens = sum(v.tokens for v in votes)

    return ConsensusResult(
        consensus=consensus,
        confidence=confidence,
        agreement_ratio=agreement_ratio,
        votes=votes,
        disagreements=disagreements,
        strategy=strategy,
        total_cost=total_cost,
        total_tokens=total_tokens,
    )
