"""Smart model routing for Prompture.

Auto-selects optimal LLM model based on task complexity, cost constraints,
and routing strategy. Integrates with extract_with_model() for transparent
model selection.

Features:
- Complexity analysis (text length, schema depth, reasoning requirements)
- Multiple routing strategies (cost_optimized, quality_first, balanced, fast)
- Provider preferences and model exclusions
- Cost budget enforcement
- Routing metadata in responses
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from pydantic import BaseModel

logger = logging.getLogger("prompture.routing")

# Type alias for routing strategies
RoutingStrategy = Literal["cost_optimized", "quality_first", "balanced", "fast"]


@dataclass
class RoutingConfig:
    """Configuration for model routing.

    Args:
        strategy: Routing strategy to use. Options:
            - "cost_optimized": Select cheapest model capable of the task
            - "quality_first": Select highest quality model available
            - "balanced": Balance cost and quality (default)
            - "fast": Select fastest model (lowest latency)
        fallback_model: Model to use if routing fails.
        max_cost_per_call: Maximum cost budget in USD per extraction call.
        preferred_providers: List of provider names to prefer (e.g., ["openai", "claude"]).
        excluded_models: List of model strings to exclude from selection.
    """

    strategy: RoutingStrategy = "balanced"
    fallback_model: str | None = None
    max_cost_per_call: float | None = None
    preferred_providers: list[str] | None = None
    excluded_models: list[str] | None = None


@dataclass
class ComplexityAnalysis:
    """Result of analyzing input complexity.

    All fields are computed from the input text and schema to help
    determine the optimal model for extraction.

    Args:
        text_length: Character count of input text.
        estimated_tokens: Rough token estimate (chars / 4).
        schema_complexity: Number of fields plus nesting depth score.
        has_nested_objects: Whether schema contains nested objects.
        has_arrays: Whether schema contains array types.
        has_enums: Whether schema contains enum constraints.
        requires_reasoning: Heuristic for whether task needs reasoning.
        complexity_score: Normalized 0.0-1.0 score (higher = more complex).
    """

    text_length: int
    estimated_tokens: int
    schema_complexity: int
    has_nested_objects: bool
    has_arrays: bool
    has_enums: bool
    requires_reasoning: bool
    complexity_score: float


@dataclass
class RoutingResult:
    """Metadata about model routing decision.

    Included in extraction responses when routing is used.

    Args:
        strategy: The routing strategy that was applied.
        selected_model: The model string that was selected.
        complexity_score: The computed complexity score (0.0-1.0).
        reason: Human-readable explanation of the selection.
        considered_models: Models that were evaluated during selection.
        excluded_models: Models that were filtered out.
    """

    strategy: RoutingStrategy
    selected_model: str
    complexity_score: float
    reason: str
    considered_models: list[str] = field(default_factory=list)
    excluded_models: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dictionary."""
        return {
            "strategy": self.strategy,
            "selected_model": self.selected_model,
            "complexity_score": self.complexity_score,
            "reason": self.reason,
            "considered_models": self.considered_models,
            "excluded_models": self.excluded_models,
        }


class ModelRouter:
    """Intelligent model selection based on task complexity and constraints.

    Analyzes input text and schema complexity to select the optimal model
    for extraction tasks. Supports multiple routing strategies and respects
    cost/provider constraints.

    Args:
        config: Routing configuration. Uses defaults if not provided.

    Example:
        >>> router = ModelRouter(RoutingConfig(strategy="cost_optimized"))
        >>> model = router.select_model("Short text", {"name": {"type": "string"}})
        >>> print(model)
        'openai/gpt-4o-mini'
    """

    def __init__(self, config: RoutingConfig | None = None) -> None:
        self.config = config or RoutingConfig()
        self._model_tiers = self._build_model_tiers()

    def _build_model_tiers(self) -> dict[str, list[str]]:
        """Build model tiers: budget, standard, premium based on capabilities/cost.

        Returns:
            Dict mapping tier name to list of model strings.
        """
        return {
            "budget": [
                "openai/gpt-4o-mini",
                "groq/llama-3.1-8b-instant",
                "groq/llama-3.2-3b-preview",
                "claude/claude-3-5-haiku-latest",
                "google/gemini-2.0-flash-lite",
            ],
            "standard": [
                "openai/gpt-4o",
                "claude/claude-sonnet-4-20250514",
                "google/gemini-2.0-flash",
                "groq/llama-3.3-70b-versatile",
            ],
            "premium": [
                "openai/gpt-4.1",
                "openai/o3-mini",
                "claude/claude-opus-4-20250514",
                "google/gemini-2.5-pro-preview-06-05",
            ],
        }

    def _count_schema_fields(self, schema: dict[str, Any], depth: int = 0) -> tuple[int, int]:
        """Count fields and max nesting depth in a JSON schema.

        Args:
            schema: JSON schema dictionary.
            depth: Current nesting depth (for recursion).

        Returns:
            Tuple of (field_count, max_depth).
        """
        field_count = 0
        max_depth = depth

        properties = schema.get("properties", {})
        field_count += len(properties)

        for prop_schema in properties.values():
            prop_type = prop_schema.get("type", "")

            # Handle nested objects
            if prop_type == "object" or "properties" in prop_schema:
                sub_count, sub_depth = self._count_schema_fields(prop_schema, depth + 1)
                field_count += sub_count
                max_depth = max(max_depth, sub_depth)

            # Handle arrays of objects
            if prop_type == "array":
                items = prop_schema.get("items", {})
                if isinstance(items, dict) and ("properties" in items or items.get("type") == "object"):
                    sub_count, sub_depth = self._count_schema_fields(items, depth + 1)
                    field_count += sub_count
                    max_depth = max(max_depth, sub_depth)

        return field_count, max_depth

    def _schema_has_feature(self, schema: dict[str, Any], feature: str) -> bool:
        """Check if schema contains a specific feature.

        Args:
            schema: JSON schema dictionary.
            feature: Feature to check ("nested", "arrays", "enums").

        Returns:
            True if feature is present.
        """
        properties = schema.get("properties", {})

        for prop_schema in properties.values():
            prop_type = prop_schema.get("type", "")

            if feature == "nested" and (prop_type == "object" or "properties" in prop_schema):
                return True

            if feature == "arrays" and prop_type == "array":
                return True

            if feature == "enums" and "enum" in prop_schema:
                return True

            # Recurse into nested structures
            if (prop_type == "object" or "properties" in prop_schema) and self._schema_has_feature(
                prop_schema, feature
            ):
                return True

            if prop_type == "array":
                items = prop_schema.get("items", {})
                if isinstance(items, dict) and self._schema_has_feature(items, feature):
                    return True

        return False

    def _detect_reasoning_need(self, text: str, schema: dict[str, Any]) -> bool:
        """Heuristic detection of whether task requires reasoning.

        Checks for:
        - Long text requiring comprehension
        - Complex relationships in schema
        - Analysis-suggesting keywords

        Args:
            text: Input text to analyze.
            schema: JSON schema for extraction.

        Returns:
            True if reasoning capabilities are likely needed.
        """
        # Long text often requires understanding context
        if len(text) > 5000:
            return True

        # Deep nesting suggests complex relationships
        _, max_depth = self._count_schema_fields(schema)
        if max_depth >= 3:
            return True

        # Analysis keywords suggest reasoning
        reasoning_keywords = [
            r"\bwhy\b",
            r"\bhow\b",
            r"\bexplain\b",
            r"\banalyze\b",
            r"\bcompare\b",
            r"\bevaluate\b",
            r"\binfer\b",
            r"\breason\b",
            r"\brelationship\b",
            r"\bcause\b",
            r"\beffect\b",
        ]
        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in reasoning_keywords)

    def analyze_complexity(self, text: str, schema: dict[str, Any]) -> ComplexityAnalysis:
        """Analyze input text and schema to determine complexity.

        Computes various metrics about the extraction task to inform
        model selection.

        Args:
            text: The input text to be processed.
            schema: JSON schema defining expected output structure.

        Returns:
            ComplexityAnalysis with computed metrics.

        Example:
            >>> router = ModelRouter()
            >>> analysis = router.analyze_complexity(
            ...     "John is 35 years old.",
            ...     {"properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}
            ... )
            >>> print(f"Score: {analysis.complexity_score:.2f}")
        """
        text_length = len(text)
        estimated_tokens = text_length // 4  # Rough approximation

        field_count, max_depth = self._count_schema_fields(schema)
        schema_complexity = field_count + (max_depth * 2)

        has_nested = self._schema_has_feature(schema, "nested")
        has_arrays = self._schema_has_feature(schema, "arrays")
        has_enums = self._schema_has_feature(schema, "enums")
        requires_reasoning = self._detect_reasoning_need(text, schema)

        # Compute normalized complexity score (0.0 to 1.0)
        # Factors: text length, schema complexity, special features
        score = 0.0

        # Text length contribution (0-0.3)
        if text_length < 500:
            score += 0.05
        elif text_length < 2000:
            score += 0.15
        elif text_length < 5000:
            score += 0.25
        else:
            score += 0.30

        # Schema complexity contribution (0-0.3)
        if schema_complexity < 5:
            score += 0.05
        elif schema_complexity < 10:
            score += 0.15
        elif schema_complexity < 20:
            score += 0.25
        else:
            score += 0.30

        # Special features contribution (0-0.4)
        if has_nested:
            score += 0.10
        if has_arrays:
            score += 0.10
        if has_enums:
            score += 0.05
        if requires_reasoning:
            score += 0.15

        # Clamp to [0.0, 1.0]
        score = max(0.0, min(1.0, score))

        return ComplexityAnalysis(
            text_length=text_length,
            estimated_tokens=estimated_tokens,
            schema_complexity=schema_complexity,
            has_nested_objects=has_nested,
            has_arrays=has_arrays,
            has_enums=has_enums,
            requires_reasoning=requires_reasoning,
            complexity_score=score,
        )

    def _get_available_models(self) -> list[str]:
        """Get list of available models based on configuration.

        Checks provider availability and applies exclusions.

        Returns:
            List of model strings that can be used.
        """
        try:
            from ..infra.discovery import get_available_models

            available = get_available_models()
        except Exception:
            logger.debug("Could not fetch available models, using tier defaults")
            available = []
            for tier_models in self._model_tiers.values():
                available.extend(tier_models)

        # Apply provider preferences
        if self.config.preferred_providers:
            preferred = []
            others = []
            for model in available:
                provider = model.split("/")[0] if "/" in model else model
                if provider in self.config.preferred_providers:
                    preferred.append(model)
                else:
                    others.append(model)
            available = preferred + others

        # Apply exclusions
        if self.config.excluded_models:
            available = [m for m in available if m not in self.config.excluded_models]

        return available

    def _get_model_tier(self, model: str) -> str:
        """Determine which tier a model belongs to.

        Args:
            model: Model string to classify.

        Returns:
            Tier name ("budget", "standard", "premium", or "unknown").
        """
        for tier, models in self._model_tiers.items():
            if model in models:
                return tier
        # Default classification based on model name patterns
        model_lower = model.lower()
        if any(x in model_lower for x in ["mini", "flash", "instant", "8b", "3b", "haiku"]):
            return "budget"
        if any(x in model_lower for x in ["opus", "pro", "o1", "o3", "4.1"]):
            return "premium"
        return "standard"

    def _select_cheapest_capable(self, analysis: ComplexityAnalysis, available: list[str]) -> str:
        """Select cheapest model that can handle the task complexity.

        Args:
            analysis: Complexity analysis results.
            available: List of available models.

        Returns:
            Selected model string.
        """
        # Simple tasks can use budget models
        if analysis.complexity_score < 0.3:
            for model in available:
                if self._get_model_tier(model) == "budget":
                    return model

        # Medium complexity prefers standard
        if analysis.complexity_score < 0.6:
            for model in available:
                tier = self._get_model_tier(model)
                if tier in ("budget", "standard"):
                    return model

        # Complex tasks need premium but try standard first
        for model in available:
            if self._get_model_tier(model) == "standard":
                return model

        # Fall through to first available
        return available[0] if available else self.config.fallback_model or "openai/gpt-4o-mini"

    def _select_highest_quality(self, analysis: ComplexityAnalysis, available: list[str]) -> str:
        """Select highest quality model regardless of cost.

        Args:
            analysis: Complexity analysis results (unused but kept for API consistency).
            available: List of available models.

        Returns:
            Selected model string.
        """
        # Prefer premium tier
        for model in available:
            if self._get_model_tier(model) == "premium":
                return model

        # Fall back to standard
        for model in available:
            if self._get_model_tier(model) == "standard":
                return model

        return available[0] if available else self.config.fallback_model or "openai/gpt-4o"

    def _select_fastest(self, analysis: ComplexityAnalysis, available: list[str]) -> str:
        """Select fastest model (typically smallest/cheapest).

        Args:
            analysis: Complexity analysis results.
            available: List of available models.

        Returns:
            Selected model string.
        """
        # Fast models are typically budget tier
        fast_patterns = ["mini", "flash", "instant", "3b", "8b", "haiku", "lite"]

        for model in available:
            model_lower = model.lower()
            if any(p in model_lower for p in fast_patterns):
                return model

        # Just return first budget model
        for model in available:
            if self._get_model_tier(model) == "budget":
                return model

        return available[0] if available else self.config.fallback_model or "groq/llama-3.2-3b-preview"

    def _select_balanced(self, analysis: ComplexityAnalysis, available: list[str]) -> str:
        """Select model balancing cost and quality based on complexity.

        Args:
            analysis: Complexity analysis results.
            available: List of available models.

        Returns:
            Selected model string.
        """
        # Map complexity to appropriate tier
        if analysis.complexity_score < 0.25:
            target_tier = "budget"
        elif analysis.complexity_score < 0.55:
            target_tier = "standard"
        else:
            target_tier = "premium"

        # Try target tier first
        for model in available:
            if self._get_model_tier(model) == target_tier:
                return model

        # Fall back to adjacent tiers
        if target_tier == "budget":
            fallback_order = ["standard", "premium"]
        elif target_tier == "premium":
            fallback_order = ["standard", "budget"]
        else:
            fallback_order = ["budget", "premium"]

        for tier in fallback_order:
            for model in available:
                if self._get_model_tier(model) == tier:
                    return model

        return available[0] if available else self.config.fallback_model or "openai/gpt-4o"

    def select_model(
        self,
        text: str,
        schema: dict[str, Any],
    ) -> tuple[str, RoutingResult]:
        """Select optimal model for the extraction task.

        Analyzes the input and applies the configured routing strategy
        to select the best model.

        Args:
            text: Input text to be processed.
            schema: JSON schema defining expected output structure.

        Returns:
            Tuple of (selected_model, routing_metadata).

        Example:
            >>> router = ModelRouter(RoutingConfig(strategy="balanced"))
            >>> model, metadata = router.select_model(
            ...     "Extract the person's name and age from: John is 35.",
            ...     {"properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}
            ... )
            >>> print(f"Selected: {model}, Score: {metadata.complexity_score:.2f}")
        """
        analysis = self.analyze_complexity(text, schema)
        available = self._get_available_models()

        excluded = []
        if self.config.excluded_models:
            excluded = list(self.config.excluded_models)

        # Select based on strategy
        if self.config.strategy == "cost_optimized":
            selected = self._select_cheapest_capable(analysis, available)
            reason = f"Cost-optimized selection for complexity score {analysis.complexity_score:.2f}"
        elif self.config.strategy == "quality_first":
            selected = self._select_highest_quality(analysis, available)
            reason = "Quality-first selection (highest capability model)"
        elif self.config.strategy == "fast":
            selected = self._select_fastest(analysis, available)
            reason = "Fast selection (lowest latency model)"
        else:  # balanced
            selected = self._select_balanced(analysis, available)
            reason = f"Balanced selection for complexity score {analysis.complexity_score:.2f}"

        routing_result = RoutingResult(
            strategy=self.config.strategy,
            selected_model=selected,
            complexity_score=analysis.complexity_score,
            reason=reason,
            considered_models=available[:10],  # Limit for readability
            excluded_models=excluded,
        )

        logger.debug(
            "Model routing: strategy=%s, score=%.2f, selected=%s",
            self.config.strategy,
            analysis.complexity_score,
            selected,
        )

        return selected, routing_result

    def select_model_for_pydantic(
        self,
        model_cls: type[BaseModel],
        text: str,
    ) -> tuple[str, RoutingResult]:
        """Select optimal model for Pydantic model extraction.

        Convenience method that converts the Pydantic model to its
        JSON schema before routing.

        Args:
            model_cls: Pydantic BaseModel class defining the extraction target.
            text: Input text to be processed.

        Returns:
            Tuple of (selected_model, routing_metadata).

        Example:
            >>> from pydantic import BaseModel
            >>> class Person(BaseModel):
            ...     name: str
            ...     age: int
            >>> router = ModelRouter()
            >>> model, metadata = router.select_model_for_pydantic(Person, "John is 35")
        """
        schema = model_cls.model_json_schema()
        return self.select_model(text, schema)


def route_model(
    text: str,
    schema: dict[str, Any],
    strategy: RoutingStrategy = "balanced",
    **config_kwargs: Any,
) -> tuple[str, RoutingResult]:
    """Convenience function for one-off model routing.

    Creates a temporary ModelRouter and selects a model.

    Args:
        text: Input text to be processed.
        schema: JSON schema defining expected output structure.
        strategy: Routing strategy to use.
        **config_kwargs: Additional RoutingConfig parameters.

    Returns:
        Tuple of (selected_model, routing_metadata).

    Example:
        >>> model, meta = route_model(
        ...     "John is 35 years old",
        ...     {"properties": {"name": {"type": "string"}}},
        ...     strategy="cost_optimized"
        ... )
    """
    config = RoutingConfig(strategy=strategy, **config_kwargs)
    router = ModelRouter(config)
    return router.select_model(text, schema)
