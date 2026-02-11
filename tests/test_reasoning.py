"""Tests for the reasoning strategy system."""

from __future__ import annotations

import concurrent.futures
import threading

import pytest

from prompture.extraction.reasoning import (
    PLAN_AND_SOLVE,
    SELF_DISCOVER,
    ReasoningStrategy,
    ReasoningStrategyProtocol,
    apply_reasoning_strategy,
    auto_select_reasoning_strategy,
    get_reasoning_strategy,
    list_reasoning_strategies,
    register_reasoning_strategy,
    reset_reasoning_strategy_registry,
    unregister_reasoning_strategy,
)


@pytest.fixture(autouse=True)
def _reset_registry():
    """Reset the reasoning strategy registry before and after each test."""
    reset_reasoning_strategy_registry()
    yield
    reset_reasoning_strategy_registry()


# ---------------------------------------------------------------------------
# ReasoningStrategy dataclass
# ---------------------------------------------------------------------------


class TestReasoningStrategy:
    def test_construction(self):
        s = ReasoningStrategy(name="test", template="Hello {content_prompt}")
        assert s.name == "test"
        assert s.template == "Hello {content_prompt}"
        assert s.description == ""

    def test_construction_with_description(self):
        s = ReasoningStrategy(name="test", template="{content_prompt}", description="A test strategy")
        assert s.description == "A test strategy"

    def test_frozen_immutability(self):
        s = ReasoningStrategy(name="test", template="{content_prompt}")
        with pytest.raises(AttributeError):
            s.name = "modified"
        with pytest.raises(AttributeError):
            s.template = "modified"

    def test_augment_prompt(self):
        s = ReasoningStrategy(name="test", template="Before. {content_prompt} After.")
        result = s.augment_prompt("Extract this.")
        assert result == "Before. Extract this. After."

    def test_implements_protocol(self):
        s = ReasoningStrategy(name="test", template="{content_prompt}")
        assert isinstance(s, ReasoningStrategyProtocol)


# ---------------------------------------------------------------------------
# Built-in strategies
# ---------------------------------------------------------------------------


class TestBuiltinStrategies:
    def test_plan_and_solve_has_placeholder(self):
        assert "{content_prompt}" in PLAN_AND_SOLVE.template

    def test_self_discover_has_placeholder(self):
        assert "{content_prompt}" in SELF_DISCOVER.template

    def test_plan_and_solve_augment(self):
        result = PLAN_AND_SOLVE.augment_prompt("Extract data.")
        assert "Extract data." in result
        assert "plan" in result.lower()

    def test_self_discover_augment(self):
        result = SELF_DISCOVER.augment_prompt("Extract data.")
        assert "Extract data." in result
        assert "reasoning modules" in result.lower()

    def test_builtins_registered_on_import(self):
        names = list_reasoning_strategies()
        assert "plan-and-solve" in names
        assert "self-discover" in names


# ---------------------------------------------------------------------------
# Registry CRUD
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_register_and_get(self):
        s = ReasoningStrategy(name="custom", template="{content_prompt}")
        register_reasoning_strategy("custom", s)
        assert get_reasoning_strategy("custom") is s

    def test_register_lowercases_name(self):
        s = ReasoningStrategy(name="Mixed", template="{content_prompt}")
        register_reasoning_strategy("Mixed-Case", s)
        assert get_reasoning_strategy("mixed-case") is s

    def test_duplicate_raises(self):
        s = ReasoningStrategy(name="dup", template="{content_prompt}")
        register_reasoning_strategy("dup", s)
        with pytest.raises(ValueError, match="already registered"):
            register_reasoning_strategy("dup", s)

    def test_overwrite(self):
        s1 = ReasoningStrategy(name="ow", template="first {content_prompt}")
        s2 = ReasoningStrategy(name="ow", template="second {content_prompt}")
        register_reasoning_strategy("ow", s1)
        register_reasoning_strategy("ow", s2, overwrite=True)
        assert get_reasoning_strategy("ow") is s2

    def test_unregister_existing(self):
        s = ReasoningStrategy(name="rm", template="{content_prompt}")
        register_reasoning_strategy("rm", s)
        assert unregister_reasoning_strategy("rm") is True
        with pytest.raises(KeyError):
            get_reasoning_strategy("rm")

    def test_unregister_missing(self):
        assert unregister_reasoning_strategy("nonexistent") is False

    def test_list(self):
        names = list_reasoning_strategies()
        # Built-ins should be present
        assert "plan-and-solve" in names
        assert "self-discover" in names
        # Add a custom one
        register_reasoning_strategy("aaa", ReasoningStrategy(name="aaa", template="{content_prompt}"))
        names = list_reasoning_strategies()
        assert names == sorted(names)
        assert "aaa" in names

    def test_get_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown reasoning strategy"):
            get_reasoning_strategy("nonexistent")

    def test_reset(self):
        register_reasoning_strategy("extra", ReasoningStrategy(name="extra", template="{content_prompt}"))
        reset_reasoning_strategy_registry()
        names = list_reasoning_strategies()
        # Built-ins re-registered, custom gone
        assert "plan-and-solve" in names
        assert "self-discover" in names
        assert "extra" not in names


# ---------------------------------------------------------------------------
# Custom protocol class
# ---------------------------------------------------------------------------


class TestCustomProtocol:
    def test_custom_class_implements_protocol(self):
        class MyStrategy:
            def augment_prompt(self, content_prompt: str) -> str:
                return f"CUSTOM: {content_prompt}"

        s = MyStrategy()
        assert isinstance(s, ReasoningStrategyProtocol)
        assert s.augment_prompt("hello") == "CUSTOM: hello"

    def test_custom_class_works_with_apply(self):
        class MyStrategy:
            def augment_prompt(self, content_prompt: str) -> str:
                return f"CUSTOM: {content_prompt}"

        result = apply_reasoning_strategy("hello", MyStrategy())
        assert result == "CUSTOM: hello"


# ---------------------------------------------------------------------------
# apply_reasoning_strategy
# ---------------------------------------------------------------------------


class TestApplyReasoningStrategy:
    def test_none_passthrough(self):
        assert apply_reasoning_strategy("hello", None) == "hello"

    def test_string_lookup(self):
        result = apply_reasoning_strategy("hello", "plan-and-solve")
        assert "hello" in result
        assert result != "hello"

    def test_direct_instance(self):
        s = ReasoningStrategy(name="direct", template="PREFIX: {content_prompt}")
        result = apply_reasoning_strategy("hello", s)
        assert result == "PREFIX: hello"

    def test_unknown_string_raises(self):
        with pytest.raises(KeyError, match="Unknown reasoning strategy"):
            apply_reasoning_strategy("hello", "nonexistent-strategy")

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError, match="reasoning_strategy must be"):
            apply_reasoning_strategy("hello", 42)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_registrations(self):
        errors: list[Exception] = []
        barrier = threading.Barrier(20)

        def register_one(i: int):
            try:
                barrier.wait(timeout=5)
                s = ReasoningStrategy(name=f"thread-{i}", template="{content_prompt}")
                register_reasoning_strategy(f"thread-{i}", s)
            except Exception as exc:
                errors.append(exc)

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as pool:
            futures = [pool.submit(register_one, i) for i in range(20)]
            concurrent.futures.wait(futures)

        assert not errors
        names = list_reasoning_strategies()
        for i in range(20):
            assert f"thread-{i}" in names


# ---------------------------------------------------------------------------
# Integration: verify augmented prompt flows through extract_and_jsonify
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_extract_and_jsonify_receives_augmented_prompt(self, monkeypatch):
        """Mock the driver to capture the prompt and verify it was augmented."""
        captured_prompts: list[str] = []

        # Minimal mock driver implementing the hooks interface
        class MockDriver:
            model = "mock-model"
            supports_json_mode = False
            supports_json_schema = False

            def generate(self, prompt, options=None):
                captured_prompts.append(prompt)
                return {
                    "text": '{"name": "Alice"}',
                    "meta": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                        "cost": 0.0,
                        "raw_response": {},
                    },
                }

            def generate_with_hooks(self, prompt, options=None):
                return self.generate(prompt, options)

            def generate_messages_with_hooks(self, messages, options=None):
                prompt = messages[-1]["content"] if messages else ""
                return self.generate(prompt, options)

        # Patch get_driver_for_model to return our mock
        from prompture.extraction import core as core_mod

        monkeypatch.setattr(core_mod, "get_driver_for_model", lambda model_name: MockDriver())

        from prompture.extraction.core import extract_and_jsonify

        schema = {"name": {"type": "string"}}
        result = extract_and_jsonify(
            text="Some text about Alice",
            json_schema=schema,
            model_name="mock/model",
            reasoning_strategy="plan-and-solve",
        )

        assert len(captured_prompts) == 1
        prompt = captured_prompts[0]
        # The prompt should contain both the reasoning strategy preamble and the original text
        assert "plan" in prompt.lower()
        assert "Some text about Alice" in prompt
        # Should still produce a result
        assert result["json_object"]["name"] == "Alice"
        # Reasoning strategy should be tracked in usage metadata
        assert result["usage"]["reasoning_strategy"] == "plan-and-solve"

    def test_none_strategy_tracked_in_usage(self, monkeypatch):
        """When no strategy is used, usage tracks None."""

        class MockDriver:
            model = "mock-model"
            supports_json_mode = False
            supports_json_schema = False

            def generate(self, prompt, options=None):
                return {
                    "text": '{"name": "Alice"}',
                    "meta": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                        "cost": 0.0,
                        "raw_response": {},
                    },
                }

            def generate_with_hooks(self, prompt, options=None):
                return self.generate(prompt, options)

            def generate_messages_with_hooks(self, messages, options=None):
                prompt = messages[-1]["content"] if messages else ""
                return self.generate(prompt, options)

        from prompture.extraction import core as core_mod

        monkeypatch.setattr(core_mod, "get_driver_for_model", lambda model_name: MockDriver())

        from prompture.extraction.core import extract_and_jsonify

        schema = {"name": {"type": "string"}}
        result = extract_and_jsonify(
            text="Some text about Alice",
            json_schema=schema,
            model_name="mock/model",
            reasoning_strategy=None,
        )
        assert result["usage"]["reasoning_strategy"] is None


# ---------------------------------------------------------------------------
# Auto reasoning strategy selection
# ---------------------------------------------------------------------------


class TestAutoSelectReasoningStrategy:
    """Tests for auto_select_reasoning_strategy()."""

    def test_simple_text_returns_none(self):
        """Short text + flat schema → None (no strategy)."""
        text = "John is 30 years old."
        schema = {"properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}
        result = auto_select_reasoning_strategy(text, schema)
        assert result is None

    def test_medium_complexity_returns_plan_and_solve(self):
        """Medium-length text + moderately complex schema → plan-and-solve."""
        text = "A " * 600  # ~1200 chars, pushes text-length score up
        schema = {
            "properties": {
                "field1": {"type": "string"},
                "field2": {"type": "integer"},
                "field3": {"type": "string"},
                "field4": {"type": "string"},
                "field5": {"type": "string"},
                "nested": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "string"},
                        "b": {"type": "string"},
                    },
                },
            }
        }
        result = auto_select_reasoning_strategy(text, schema)
        assert result == "plan-and-solve"

    def test_high_complexity_returns_self_discover(self):
        """Long text + deeply nested schema → self-discover."""
        text = "word " * 2000  # ~10000 chars
        schema = {
            "properties": {
                "level1": {
                    "type": "object",
                    "properties": {
                        "level2": {
                            "type": "object",
                            "properties": {
                                "level3": {
                                    "type": "object",
                                    "properties": {
                                        "value": {"type": "string"},
                                    },
                                },
                            },
                        },
                    },
                },
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "tags": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                },
                "category": {"type": "string", "enum": ["A", "B", "C"]},
            }
        }
        result = auto_select_reasoning_strategy(text, schema)
        assert result == "self-discover"

    def test_reasoning_keyword_triggers_plan_and_solve(self):
        """Even short text with reasoning keywords → plan-and-solve."""
        text = "Explain why the revenue dropped last quarter."
        schema = {"properties": {"explanation": {"type": "string"}}}
        result = auto_select_reasoning_strategy(text, schema)
        # requires_reasoning should be True due to "explain" + "why"
        assert result == "plan-and-solve"

    def test_auto_in_list_strategies(self):
        """'auto' should appear in list_reasoning_strategies()."""
        names = list_reasoning_strategies()
        assert "auto" in names

    def test_auto_sentinel_raises_on_direct_augment(self):
        """Calling augment_prompt on the auto sentinel should raise RuntimeError."""
        strategy = get_reasoning_strategy("auto")
        with pytest.raises(RuntimeError, match="cannot be applied directly"):
            strategy.augment_prompt("hello")


class TestAutoIntegration:
    """End-to-end: reasoning_strategy='auto' through extract_and_jsonify."""

    def test_auto_strategy_end_to_end(self, monkeypatch):
        """Verify 'auto' resolves and the extraction completes."""
        captured_prompts: list[str] = []

        class MockDriver:
            model = "mock-model"
            supports_json_mode = False
            supports_json_schema = False

            def generate(self, prompt, options=None):
                captured_prompts.append(prompt)
                return {
                    "text": '{"name": "Alice"}',
                    "meta": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                        "cost": 0.0,
                        "raw_response": {},
                    },
                }

            def generate_with_hooks(self, prompt, options=None):
                return self.generate(prompt, options)

            def generate_messages_with_hooks(self, messages, options=None):
                prompt = messages[-1]["content"] if messages else ""
                return self.generate(prompt, options)

        from prompture.extraction import core as core_mod

        monkeypatch.setattr(core_mod, "get_driver_for_model", lambda model_name: MockDriver())

        from prompture.extraction.core import extract_and_jsonify

        schema = {"properties": {"name": {"type": "string"}}}
        result = extract_and_jsonify(
            text="Some text about Alice",
            json_schema=schema,
            model_name="mock/model",
            reasoning_strategy="auto",
        )

        # For short text + simple schema, auto should resolve to None → no augmentation
        assert len(captured_prompts) == 1
        assert result["json_object"]["name"] == "Alice"
        # Auto resolved to None for this simple case — tracked in usage
        assert result["usage"]["reasoning_strategy"] is None
