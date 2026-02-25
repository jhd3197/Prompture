"""Tests for the model resolution system (prompture.pipeline.resolver)."""

import pytest

from prompture.pipeline.resolver import (
    DEFAULT_FALLBACK_SLOTS,
    ModelResolver,
    NoModelConfiguredError,
    SLOT_AUDIO,
    SLOT_DEFAULT,
    SLOT_EMBEDDING,
    SLOT_IMAGE,
    SLOT_STRUCTURED,
    SLOT_UTILITY,
    attr_layer,
    dict_layer,
    resolve_model,
)


# ── dict_layer tests ──────────────────────────────────────────────────────


class TestDictLayer:
    def test_resolves_known_slot(self):
        layer = dict_layer({"utility": "openai/gpt-4o-mini", "default": "openai/gpt-4o"})
        assert layer("utility") == "openai/gpt-4o-mini"
        assert layer("default") == "openai/gpt-4o"

    def test_returns_none_for_unknown_slot(self):
        layer = dict_layer({"utility": "openai/gpt-4o-mini"})
        assert layer("image") is None

    def test_returns_none_for_empty_string(self):
        layer = dict_layer({"utility": ""})
        assert layer("utility") is None

    def test_returns_none_for_none_value(self):
        layer = dict_layer({"utility": None})
        assert layer("utility") is None

    def test_strips_whitespace(self):
        layer = dict_layer({"utility": "  openai/gpt-4o-mini  "})
        assert layer("utility") == "openai/gpt-4o-mini"


# ── attr_layer tests ──────────────────────────────────────────────────────


class _MockConfig:
    model = "openai/gpt-4o"
    utility_model = "openai/gpt-4o-mini"


class _EmptyConfig:
    model = ""
    utility_model = None


class TestAttrLayer:
    def test_resolves_default_slot(self):
        layer = attr_layer(_MockConfig())
        assert layer("default") == "openai/gpt-4o"

    def test_resolves_utility_slot(self):
        layer = attr_layer(_MockConfig())
        assert layer("utility") == "openai/gpt-4o-mini"

    def test_returns_none_for_unmapped_slot(self):
        layer = attr_layer(_MockConfig())
        assert layer("image") is None

    def test_returns_none_for_empty_attr(self):
        layer = attr_layer(_EmptyConfig())
        assert layer("default") is None
        assert layer("utility") is None

    def test_custom_attr_map(self):
        class Custom:
            main = "claude/claude-3-opus"
            fast = "groq/llama3"

        layer = attr_layer(Custom(), attr_map={"default": "main", "utility": "fast"})
        assert layer("default") == "claude/claude-3-opus"
        assert layer("utility") == "groq/llama3"

    def test_missing_attribute_returns_none(self):
        class Minimal:
            pass

        layer = attr_layer(Minimal())
        assert layer("default") is None
        assert layer("utility") is None


# ── ModelResolver tests ───────────────────────────────────────────────────


class TestModelResolver:
    def test_resolves_from_single_layer(self):
        resolver = ModelResolver(layers=[
            dict_layer({"default": "openai/gpt-4o"}),
        ])
        assert resolver.resolve("default") == "openai/gpt-4o"

    def test_first_layer_wins(self):
        resolver = ModelResolver(layers=[
            dict_layer({"default": "first-wins"}),
            dict_layer({"default": "second-loses"}),
        ])
        assert resolver.resolve("default") == "first-wins"

    def test_skips_empty_layer(self):
        resolver = ModelResolver(layers=[
            dict_layer({"default": ""}),  # empty → skip
            dict_layer({"default": "second-wins"}),
        ])
        assert resolver.resolve("default") == "second-wins"

    def test_fallback_utility_to_default(self):
        resolver = ModelResolver(layers=[
            dict_layer({"default": "openai/gpt-4o"}),
            # no "utility" in any layer
        ])
        # utility falls back to default
        assert resolver.resolve("utility") == "openai/gpt-4o"

    def test_fallback_structured_to_default(self):
        resolver = ModelResolver(layers=[
            dict_layer({"default": "openai/gpt-4o"}),
        ])
        assert resolver.resolve("structured") == "openai/gpt-4o"

    def test_no_fallback_for_image(self):
        resolver = ModelResolver(layers=[
            dict_layer({"default": "openai/gpt-4o"}),
        ])
        with pytest.raises(NoModelConfiguredError, match="image"):
            resolver.resolve("image")

    def test_raises_when_all_layers_empty(self):
        resolver = ModelResolver(layers=[
            dict_layer({}),
            dict_layer({"utility": ""}),
        ])
        with pytest.raises(NoModelConfiguredError, match="default"):
            resolver.resolve("default")

    def test_raises_when_no_layers(self):
        resolver = ModelResolver(layers=[])
        with pytest.raises(NoModelConfiguredError):
            resolver.resolve("default")

    def test_resolve_or_returns_default(self):
        resolver = ModelResolver(layers=[])
        assert resolver.resolve_or("default", "fallback/model") == "fallback/model"

    def test_resolve_or_returns_resolved_value(self):
        resolver = ModelResolver(layers=[
            dict_layer({"default": "openai/gpt-4o"}),
        ])
        assert resolver.resolve_or("default", "fallback") == "openai/gpt-4o"

    def test_add_layer_appends_by_default(self):
        resolver = ModelResolver(layers=[
            dict_layer({"default": "first"}),
        ])
        resolver.add_layer(dict_layer({"default": "appended"}))
        # First layer still wins
        assert resolver.resolve("default") == "first"

    def test_add_layer_with_priority_zero_prepends(self):
        resolver = ModelResolver(layers=[
            dict_layer({"default": "original"}),
        ])
        resolver.add_layer(dict_layer({"default": "prepended"}), priority=0)
        # Prepended layer now wins
        assert resolver.resolve("default") == "prepended"

    def test_custom_fallback_slots(self):
        resolver = ModelResolver(
            layers=[dict_layer({"default": "openai/gpt-4o"})],
            fallback_slots={"image": ["default"]},
        )
        # "image" now falls back to "default"
        assert resolver.resolve("image") == "openai/gpt-4o"
        # But "utility" no longer has a fallback
        with pytest.raises(NoModelConfiguredError):
            resolver.resolve("utility")


# ── Mixed layer types ─────────────────────────────────────────────────────


class TestMixedLayers:
    def test_dict_plus_attr_layers(self):
        """Dict layer (higher priority) + attr layer (lower priority)."""
        config = _MockConfig()
        resolver = ModelResolver(layers=[
            dict_layer({"utility": "override/fast-model"}),
            attr_layer(config),
        ])
        # utility resolved from dict layer (higher priority)
        assert resolver.resolve("utility") == "override/fast-model"
        # default resolved from attr layer (lower priority, dict has no "default")
        assert resolver.resolve("default") == "openai/gpt-4o"

    def test_attr_layer_with_fallback(self):
        """Attr layer provides only 'default', utility falls back to it."""
        class OnlyDefault:
            model = "claude/claude-3-sonnet"

        resolver = ModelResolver(layers=[attr_layer(OnlyDefault())])
        assert resolver.resolve("utility") == "claude/claude-3-sonnet"


# ── Convenience function ──────────────────────────────────────────────────


class TestResolveModelFunction:
    def test_basic_resolve(self):
        result = resolve_model("default", [dict_layer({"default": "openai/gpt-4o"})])
        assert result == "openai/gpt-4o"

    def test_with_fallback(self):
        result = resolve_model("utility", [dict_layer({"default": "openai/gpt-4o"})])
        assert result == "openai/gpt-4o"

    def test_custom_fallback_slots(self):
        result = resolve_model(
            "image",
            [dict_layer({"default": "openai/dall-e-3"})],
            fallback_slots={"image": ["default"]},
        )
        assert result == "openai/dall-e-3"

    def test_raises_on_failure(self):
        with pytest.raises(NoModelConfiguredError):
            resolve_model("default", [dict_layer({})])


# ── Slot constants ────────────────────────────────────────────────────────


class TestSlotConstants:
    def test_slot_values(self):
        assert SLOT_DEFAULT == "default"
        assert SLOT_UTILITY == "utility"
        assert SLOT_IMAGE == "image"
        assert SLOT_AUDIO == "audio"
        assert SLOT_EMBEDDING == "embedding"
        assert SLOT_STRUCTURED == "structured"

    def test_default_fallback_slots(self):
        assert "utility" in DEFAULT_FALLBACK_SLOTS
        assert "structured" in DEFAULT_FALLBACK_SLOTS
        assert DEFAULT_FALLBACK_SLOTS["utility"] == ["default"]
        assert DEFAULT_FALLBACK_SLOTS["structured"] == ["default"]


# ── NoModelConfiguredError ────────────────────────────────────────────────


class TestNoModelConfiguredError:
    def test_inherits_from_configuration_error(self):
        from prompture.exceptions import ConfigurationError

        err = NoModelConfiguredError("utility")
        assert isinstance(err, ConfigurationError)
        assert isinstance(err, ValueError)  # ConfigurationError inherits ValueError

    def test_stores_slot(self):
        err = NoModelConfiguredError("utility")
        assert err.slot == "utility"

    def test_message_includes_slot(self):
        err = NoModelConfiguredError("utility")
        assert "utility" in str(err)
