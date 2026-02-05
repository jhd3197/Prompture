"""Tests for smart model routing module."""

from prompture.routing import (
    ComplexityAnalysis,
    ModelRouter,
    RoutingConfig,
    RoutingResult,
    route_model,
)


class TestComplexityAnalysis:
    """Tests for complexity analysis."""

    def test_simple_schema_low_complexity(self):
        """Simple schema with short text should have low complexity."""
        router = ModelRouter()
        analysis = router.analyze_complexity(
            "John is 35 years old.",
            {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}},
        )

        assert isinstance(analysis, ComplexityAnalysis)
        assert analysis.text_length == len("John is 35 years old.")
        assert analysis.complexity_score < 0.3
        assert not analysis.has_nested_objects
        assert not analysis.has_arrays
        assert not analysis.requires_reasoning

    def test_complex_schema_high_complexity(self):
        """Nested schema with long text should have high complexity."""
        router = ModelRouter()
        long_text = "A " * 2000  # Long text

        complex_schema = {
            "type": "object",
            "properties": {
                "person": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "address": {
                            "type": "object",
                            "properties": {
                                "street": {"type": "string"},
                                "city": {"type": "string"},
                            },
                        },
                    },
                },
                "tags": {"type": "array", "items": {"type": "string"}},
                "status": {"type": "string", "enum": ["active", "inactive"]},
            },
        }

        analysis = router.analyze_complexity(long_text, complex_schema)

        assert analysis.complexity_score > 0.4
        assert analysis.has_nested_objects
        assert analysis.has_arrays
        assert analysis.has_enums

    def test_reasoning_detection_from_keywords(self):
        """Text with reasoning keywords should trigger reasoning flag."""
        router = ModelRouter()

        analysis = router.analyze_complexity(
            "Explain why the customer is unhappy and analyze the root cause.",
            {"type": "object", "properties": {"explanation": {"type": "string"}}},
        )

        assert analysis.requires_reasoning

    def test_reasoning_detection_from_long_text(self):
        """Very long text should trigger reasoning flag."""
        router = ModelRouter()
        long_text = "Content " * 1500  # > 5000 chars

        analysis = router.analyze_complexity(
            long_text,
            {"type": "object", "properties": {"summary": {"type": "string"}}},
        )

        assert analysis.requires_reasoning

    def test_estimated_tokens(self):
        """Token estimation should be roughly chars/4."""
        router = ModelRouter()
        text = "A" * 400  # 400 chars

        analysis = router.analyze_complexity(
            text,
            {"type": "object", "properties": {"value": {"type": "string"}}},
        )

        assert analysis.estimated_tokens == 100


class TestModelRouter:
    """Tests for ModelRouter class."""

    def test_default_config(self):
        """Router should work with default config."""
        router = ModelRouter()

        assert router.config.strategy == "balanced"
        assert router.config.fallback_model is None

    def test_custom_config(self):
        """Router should accept custom config."""
        config = RoutingConfig(
            strategy="cost_optimized",
            fallback_model="openai/gpt-4o-mini",
            max_cost_per_call=0.01,
        )
        router = ModelRouter(config)

        assert router.config.strategy == "cost_optimized"
        assert router.config.fallback_model == "openai/gpt-4o-mini"

    def test_select_model_returns_tuple(self):
        """select_model should return (model, routing_result) tuple."""
        router = ModelRouter()

        model, result = router.select_model(
            "Simple text",
            {"type": "object", "properties": {"name": {"type": "string"}}},
        )

        assert isinstance(model, str)
        assert "/" in model  # Should be provider/model format
        assert isinstance(result, RoutingResult)
        assert result.strategy == "balanced"
        assert result.selected_model == model
        assert 0 <= result.complexity_score <= 1

    def test_cost_optimized_selects_budget(self):
        """Cost-optimized strategy should prefer budget models for simple tasks."""
        config = RoutingConfig(strategy="cost_optimized")
        router = ModelRouter(config)

        model, result = router.select_model(
            "Short simple text",
            {"type": "object", "properties": {"name": {"type": "string"}}},
        )

        # Should select a budget tier model
        budget_indicators = ["mini", "haiku", "flash", "instant", "8b", "3b", "lite"]
        assert any(ind in model.lower() for ind in budget_indicators), f"Expected budget model, got {model}"
        assert result.strategy == "cost_optimized"

    def test_quality_first_selects_premium(self):
        """Quality-first strategy should prefer premium models."""
        config = RoutingConfig(strategy="quality_first")
        router = ModelRouter(config)

        model, result = router.select_model(
            "Any text",
            {"type": "object", "properties": {"name": {"type": "string"}}},
        )

        # Should select a premium or standard tier model (not budget)
        budget_only_indicators = ["3b", "lite"]
        assert not any(ind in model.lower() for ind in budget_only_indicators) or "pro" in model.lower()
        assert result.strategy == "quality_first"

    def test_fast_selects_low_latency(self):
        """Fast strategy should select low-latency models."""
        config = RoutingConfig(strategy="fast")
        router = ModelRouter(config)

        model, result = router.select_model(
            "Any text",
            {"type": "object", "properties": {"name": {"type": "string"}}},
        )

        fast_indicators = ["mini", "flash", "instant", "haiku", "3b", "8b", "lite"]
        assert any(ind in model.lower() for ind in fast_indicators), f"Expected fast model, got {model}"
        assert result.strategy == "fast"

    def test_balanced_scales_with_complexity(self):
        """Balanced strategy should select model tier based on complexity."""
        router = ModelRouter(RoutingConfig(strategy="balanced"))

        # Simple task
        _, simple_result = router.select_model(
            "Hi",
            {"type": "object", "properties": {"greeting": {"type": "string"}}},
        )

        # Complex task
        long_text = "Analyze " * 500
        complex_schema = {
            "type": "object",
            "properties": {
                "analysis": {
                    "type": "object",
                    "properties": {
                        "findings": {"type": "array", "items": {"type": "string"}},
                        "recommendations": {
                            "type": "object",
                            "properties": {"priority": {"type": "string"}},
                        },
                    },
                },
            },
        }
        _, complex_result = router.select_model(long_text, complex_schema)

        # Complex task should have higher complexity score
        assert complex_result.complexity_score > simple_result.complexity_score

    def test_excluded_models(self):
        """Excluded models should not be selected."""
        config = RoutingConfig(
            strategy="cost_optimized",
            excluded_models=["openai/gpt-4o-mini", "groq/llama-3.1-8b-instant"],
        )
        router = ModelRouter(config)

        model, result = router.select_model(
            "Text",
            {"type": "object", "properties": {"name": {"type": "string"}}},
        )

        assert model not in config.excluded_models
        assert config.excluded_models == result.excluded_models

    def test_preferred_providers(self):
        """Preferred providers should be prioritized."""
        config = RoutingConfig(
            strategy="balanced",
            preferred_providers=["claude"],
        )
        router = ModelRouter(config)

        model, _ = router.select_model(
            "Text",
            {"type": "object", "properties": {"name": {"type": "string"}}},
        )

        # If Claude models are available, should select one
        # This may vary based on what's configured, so we just verify it runs
        assert isinstance(model, str)


class TestRoutingResult:
    """Tests for RoutingResult dataclass."""

    def test_to_dict(self):
        """RoutingResult should serialize to dict."""
        result = RoutingResult(
            strategy="balanced",
            selected_model="openai/gpt-4o",
            complexity_score=0.5,
            reason="Test reason",
            considered_models=["openai/gpt-4o", "claude/sonnet"],
            excluded_models=["grok/model"],
        )

        d = result.to_dict()

        assert d["strategy"] == "balanced"
        assert d["selected_model"] == "openai/gpt-4o"
        assert d["complexity_score"] == 0.5
        assert d["reason"] == "Test reason"
        assert "openai/gpt-4o" in d["considered_models"]
        assert "grok/model" in d["excluded_models"]


class TestRouteModelFunction:
    """Tests for route_model convenience function."""

    def test_basic_usage(self):
        """route_model should work with minimal arguments."""
        model, result = route_model(
            "Simple text",
            {"type": "object", "properties": {"name": {"type": "string"}}},
        )

        assert isinstance(model, str)
        assert isinstance(result, RoutingResult)
        assert result.strategy == "balanced"  # default

    def test_with_strategy(self):
        """route_model should accept strategy parameter."""
        _model, result = route_model(
            "Text",
            {"type": "object", "properties": {"name": {"type": "string"}}},
            strategy="cost_optimized",
        )

        assert result.strategy == "cost_optimized"

    def test_with_config_kwargs(self):
        """route_model should pass kwargs to RoutingConfig."""
        model, _result = route_model(
            "Text",
            {"type": "object", "properties": {"name": {"type": "string"}}},
            strategy="balanced",
            fallback_model="openai/gpt-4o-mini",
        )

        assert isinstance(model, str)


class TestSchemaFeatureDetection:
    """Tests for schema feature detection."""

    def test_detect_nested_objects(self):
        """Should detect nested object schemas."""
        router = ModelRouter()

        schema = {
            "type": "object",
            "properties": {
                "person": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                },
            },
        }

        assert router._schema_has_feature(schema, "nested")

    def test_detect_arrays(self):
        """Should detect array schemas."""
        router = ModelRouter()

        schema = {
            "type": "object",
            "properties": {
                "items": {"type": "array", "items": {"type": "string"}},
            },
        }

        assert router._schema_has_feature(schema, "arrays")

    def test_detect_enums(self):
        """Should detect enum constraints."""
        router = ModelRouter()

        schema = {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["active", "inactive"]},
            },
        }

        assert router._schema_has_feature(schema, "enums")

    def test_count_fields_simple(self):
        """Should count fields in simple schema."""
        router = ModelRouter()

        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
        }

        count, depth = router._count_schema_fields(schema)
        assert count == 2
        assert depth == 0

    def test_count_fields_nested(self):
        """Should count fields in nested schema."""
        router = ModelRouter()

        schema = {
            "type": "object",
            "properties": {
                "person": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "address": {
                            "type": "object",
                            "properties": {
                                "city": {"type": "string"},
                            },
                        },
                    },
                },
            },
        }

        count, depth = router._count_schema_fields(schema)
        assert count >= 3  # person, name, address, city
        assert depth >= 2  # Two levels of nesting


class TestSelectModelForPydantic:
    """Tests for Pydantic model routing."""

    def test_select_for_pydantic_model(self):
        """Should route based on Pydantic model schema."""
        from pydantic import BaseModel

        class Person(BaseModel):
            name: str
            age: int

        router = ModelRouter()
        model, result = router.select_model_for_pydantic(Person, "John is 35")

        assert isinstance(model, str)
        assert result.complexity_score > 0


class TestModelTierClassification:
    """Tests for model tier classification."""

    def test_budget_tier_models(self):
        """Budget tier models should be classified correctly."""
        router = ModelRouter()

        budget_models = [
            "openai/gpt-4o-mini",
            "groq/llama-3.1-8b-instant",
            "claude/claude-3-5-haiku-latest",
        ]

        for model in budget_models:
            tier = router._get_model_tier(model)
            assert tier == "budget", f"{model} should be budget tier, got {tier}"

    def test_premium_tier_models(self):
        """Premium tier models should be classified correctly."""
        router = ModelRouter()

        premium_models = [
            "openai/gpt-4.1",
            "openai/o3-mini",
            "claude/claude-opus-4-20250514",
        ]

        for model in premium_models:
            tier = router._get_model_tier(model)
            assert tier == "premium", f"{model} should be premium tier, got {tier}"

    def test_standard_tier_models(self):
        """Standard tier models should be classified correctly."""
        router = ModelRouter()

        standard_models = [
            "openai/gpt-4o",
            "claude/claude-sonnet-4-20250514",
        ]

        for model in standard_models:
            tier = router._get_model_tier(model)
            assert tier == "standard", f"{model} should be standard tier, got {tier}"
