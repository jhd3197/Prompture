"""Tests for the persona templates module."""

from __future__ import annotations

import threading
import warnings
from unittest.mock import MagicMock

import pytest

from prompture.agents.persona import (
    BASE_PERSONAS,
    PERSONAS,
    Persona,
    clear_persona_registry,
    get_persona,
    get_persona_names,
    get_persona_registry_snapshot,
    get_trait,
    get_trait_names,
    load_personas_from_directory,
    register_persona,
    register_trait,
    reset_persona_registry,
    reset_trait_registry,
)

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_registries():
    """Reset persona and trait registries before each test."""
    reset_persona_registry()
    reset_trait_registry()
    yield
    reset_persona_registry()
    reset_trait_registry()


@pytest.fixture
def basic_persona():
    return Persona(
        name="test",
        system_prompt="You are a test assistant.",
        description="A test persona.",
    )


@pytest.fixture
def persona_with_vars():
    return Persona(
        name="templated",
        system_prompt="The year is {{current_year}}. Limit to {{max_items}} items.",
        variables={"max_items": "5"},
    )


@pytest.fixture
def persona_with_constraints():
    return Persona(
        name="constrained",
        system_prompt="You are precise.",
        constraints=["Be concise.", "No markdown."],
    )


# ==================================================================
# Phase 4a: Persona Data Model & Template Rendering
# ==================================================================


class TestPersonaConstruction:
    def test_basic_construction(self, basic_persona):
        assert basic_persona.name == "test"
        assert basic_persona.system_prompt == "You are a test assistant."
        assert basic_persona.description == "A test persona."

    def test_defaults(self):
        p = Persona(name="minimal", system_prompt="Hello.")
        assert p.description == ""
        assert p.traits == ()
        assert p.variables == {}
        assert p.constraints == []
        assert p.model_hint is None
        assert p.settings == {}

    def test_frozen_immutability(self, basic_persona):
        with pytest.raises(AttributeError):
            basic_persona.name = "changed"
        with pytest.raises(AttributeError):
            basic_persona.system_prompt = "changed"

    def test_full_construction(self):
        p = Persona(
            name="full",
            system_prompt="Full prompt.",
            description="Full description.",
            traits=("polite", "formal"),
            variables={"key": "value"},
            constraints=["Constraint 1"],
            model_hint="openai/gpt-4",
            settings={"temperature": 0.5},
        )
        assert p.traits == ("polite", "formal")
        assert p.variables == {"key": "value"}
        assert p.model_hint == "openai/gpt-4"
        assert p.settings == {"temperature": 0.5}


class TestPersonaRendering:
    def test_render_plain(self, basic_persona):
        result = basic_persona.render()
        assert result == "You are a test assistant."

    def test_render_builtin_vars(self):
        from datetime import datetime

        p = Persona(name="t", system_prompt="Year: {{current_year}}")
        rendered = p.render()
        assert str(datetime.now().year) in rendered

    def test_render_custom_vars(self, persona_with_vars):
        rendered = persona_with_vars.render()
        assert "5" in rendered

    def test_render_override_vars(self, persona_with_vars):
        rendered = persona_with_vars.render(max_items="10")
        assert "10" in rendered
        assert "5" not in rendered

    def test_render_constraints(self, persona_with_constraints):
        rendered = persona_with_constraints.render()
        assert "## Constraints" in rendered
        assert "- Be concise." in rendered
        assert "- No markdown." in rendered

    def test_render_constraints_with_template_vars(self):
        p = Persona(
            name="t",
            system_prompt="Prompt.",
            constraints=["Year is {{current_year}}."],
        )
        rendered = p.render()
        from datetime import datetime

        assert str(datetime.now().year) in rendered

    def test_render_with_traits(self):
        register_trait("polite", "Always be polite and respectful.")
        p = Persona(name="t", system_prompt="Base prompt.", traits=("polite",))
        rendered = p.render()
        assert "Base prompt." in rendered
        assert "Always be polite and respectful." in rendered

    def test_render_missing_trait_ignored(self):
        p = Persona(name="t", system_prompt="Base.", traits=("nonexistent",))
        rendered = p.render()
        assert rendered == "Base."

    def test_render_traits_between_prompt_and_constraints(self):
        register_trait("formal", "Use formal language.")
        p = Persona(
            name="t",
            system_prompt="Base prompt.",
            traits=("formal",),
            constraints=["No slang."],
        )
        rendered = p.render()
        # Traits should come after main prompt but before constraints
        prompt_idx = rendered.index("Base prompt.")
        trait_idx = rendered.index("Use formal language.")
        constraint_idx = rendered.index("## Constraints")
        assert prompt_idx < trait_idx < constraint_idx


# ==================================================================
# Phase 4b: Composition & Extension
# ==================================================================


class TestPersonaComposition:
    def test_extend_returns_new_instance(self, basic_persona):
        extended = basic_persona.extend("Also be helpful.")
        assert extended is not basic_persona
        assert "Also be helpful." in extended.system_prompt
        assert "Also be helpful." not in basic_persona.system_prompt

    def test_extend_preserves_other_fields(self, basic_persona):
        extended = basic_persona.extend("Extra.")
        assert extended.name == basic_persona.name
        assert extended.description == basic_persona.description

    def test_with_constraints_returns_new_instance(self, basic_persona):
        constrained = basic_persona.with_constraints(["Be brief."])
        assert constrained is not basic_persona
        assert constrained.constraints == ["Be brief."]
        assert basic_persona.constraints == []

    def test_with_constraints_appends(self, persona_with_constraints):
        new = persona_with_constraints.with_constraints(["New constraint."])
        assert len(new.constraints) == 3
        assert new.constraints[-1] == "New constraint."
        assert len(persona_with_constraints.constraints) == 2

    def test_add_merge_operator(self):
        p1 = Persona(name="a", system_prompt="Prompt A.", variables={"x": "1"}, constraints=["C1"])
        p2 = Persona(name="b", system_prompt="Prompt B.", variables={"y": "2"}, constraints=["C2"])
        merged = p1 + p2
        assert merged.name == "a+b"
        assert "Prompt A." in merged.system_prompt
        assert "Prompt B." in merged.system_prompt
        assert merged.variables == {"x": "1", "y": "2"}
        assert merged.constraints == ["C1", "C2"]

    def test_add_right_wins_variables(self):
        p1 = Persona(name="a", system_prompt="A", variables={"key": "left"})
        p2 = Persona(name="b", system_prompt="B", variables={"key": "right"})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            merged = p1 + p2
            assert len(w) == 1
            assert "conflict" in str(w[0].message).lower()
        assert merged.variables["key"] == "right"

    def test_add_traits_deduped(self):
        p1 = Persona(name="a", system_prompt="A", traits=("polite", "formal"))
        p2 = Persona(name="b", system_prompt="B", traits=("formal", "brief"))
        merged = p1 + p2
        assert merged.traits == ("polite", "formal", "brief")

    def test_add_model_hint_right_wins(self):
        p1 = Persona(name="a", system_prompt="A", model_hint="openai/gpt-3.5")
        p2 = Persona(name="b", system_prompt="B", model_hint="openai/gpt-4")
        merged = p1 + p2
        assert merged.model_hint == "openai/gpt-4"

    def test_add_model_hint_fallback_left(self):
        p1 = Persona(name="a", system_prompt="A", model_hint="openai/gpt-4")
        p2 = Persona(name="b", system_prompt="B")
        merged = p1 + p2
        assert merged.model_hint == "openai/gpt-4"

    def test_add_settings_merged(self):
        p1 = Persona(name="a", system_prompt="A", settings={"temperature": 0.5, "top_p": 0.9})
        p2 = Persona(name="b", system_prompt="B", settings={"temperature": 0.0})
        merged = p1 + p2
        assert merged.settings == {"temperature": 0.0, "top_p": 0.9}

    def test_add_not_implemented_for_non_persona(self, basic_persona):
        result = basic_persona.__add__("not a persona")
        assert result is NotImplemented


# ==================================================================
# Phase 4b: Trait Registry
# ==================================================================


class TestTraitRegistry:
    def test_register_and_get(self):
        register_trait("friendly", "Be warm and approachable.")
        assert get_trait("friendly") == "Be warm and approachable."

    def test_get_nonexistent(self):
        assert get_trait("nonexistent") is None

    def test_get_names(self):
        register_trait("a", "A")
        register_trait("b", "B")
        names = get_trait_names()
        assert "a" in names
        assert "b" in names

    def test_reset(self):
        register_trait("tmp", "Temporary.")
        reset_trait_registry()
        assert get_trait("tmp") is None
        assert get_trait_names() == []


# ==================================================================
# Phase 4c: Thread-Safe Global Persona Registry
# ==================================================================


class TestPersonaRegistry:
    def test_register_and_get(self, basic_persona):
        register_persona(basic_persona)
        result = get_persona("test")
        assert result is basic_persona

    def test_get_nonexistent(self):
        assert get_persona("nonexistent") is None

    def test_get_names_includes_builtins(self):
        names = get_persona_names()
        assert "json_extractor" in names
        assert "data_analyst" in names

    def test_snapshot(self):
        snapshot = get_persona_registry_snapshot()
        assert isinstance(snapshot, dict)
        assert "json_extractor" in snapshot

    def test_clear_removes_all(self):
        clear_persona_registry()
        assert get_persona_names() == []
        assert get_persona("json_extractor") is None

    def test_reset_restores_builtins(self):
        clear_persona_registry()
        reset_persona_registry()
        assert "json_extractor" in get_persona_names()

    def test_thread_safety(self):
        errors = []

        def register_many(prefix: str, count: int):
            try:
                for i in range(count):
                    p = Persona(name=f"{prefix}_{i}", system_prompt=f"Prompt {prefix}_{i}")
                    register_persona(p)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=register_many, args=(f"t{t}", 50)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        # Should have 5 builtins + 200 custom
        assert len(get_persona_names()) == 5 + 200


class TestPersonaRegistryProxy:
    def test_getitem(self):
        p = PERSONAS["json_extractor"]
        assert isinstance(p, Persona)
        assert p.name == "json_extractor"

    def test_getitem_missing(self):
        with pytest.raises(KeyError, match="not found"):
            PERSONAS["nonexistent"]

    def test_setitem(self, basic_persona):
        PERSONAS["custom"] = basic_persona
        assert get_persona("custom") is basic_persona

    def test_setitem_type_check(self):
        with pytest.raises(TypeError):
            PERSONAS["bad"] = "not a persona"

    def test_delitem(self, basic_persona):
        register_persona(basic_persona)
        del PERSONAS["test"]
        assert get_persona("test") is None

    def test_delitem_missing(self):
        with pytest.raises(KeyError):
            del PERSONAS["nonexistent"]

    def test_contains(self):
        assert "json_extractor" in PERSONAS
        assert "nonexistent" not in PERSONAS

    def test_iter(self):
        names = list(PERSONAS)
        assert "json_extractor" in names

    def test_len(self):
        assert len(PERSONAS) == 5  # 5 built-ins

    def test_keys_values_items(self):
        assert "json_extractor" in PERSONAS
        values = PERSONAS.values()
        assert all(isinstance(v, Persona) for v in values)
        items = PERSONAS.items()
        assert all(isinstance(k, str) and isinstance(v, Persona) for k, v in items)

    def test_get_with_default(self):
        assert PERSONAS.get("nonexistent", "default") == "default"
        assert isinstance(PERSONAS.get("json_extractor"), Persona)

    def test_repr(self):
        r = repr(PERSONAS)
        assert "PERSONAS" in r


# ==================================================================
# Phase 4d: Built-in Personas
# ==================================================================


class TestBuiltinPersonas:
    def test_all_five_exist(self):
        expected = {"json_extractor", "data_analyst", "text_summarizer", "code_reviewer", "concise_assistant"}
        actual = set(get_persona_names())
        assert expected.issubset(actual)

    def test_all_render_without_error(self):
        for name in BASE_PERSONAS:
            persona = get_persona(name)
            rendered = persona.render()
            assert isinstance(rendered, str)
            assert len(rendered) > 0

    def test_json_extractor_has_temperature(self):
        p = get_persona("json_extractor")
        assert p.settings.get("temperature") == 0.0

    def test_text_summarizer_default_variable(self):
        p = get_persona("text_summarizer")
        rendered = p.render()
        assert "3" in rendered  # default max_sentences

    def test_text_summarizer_variable_override(self):
        p = get_persona("text_summarizer")
        rendered = p.render(max_sentences="5")
        assert "5" in rendered

    def test_reset_restores_builtins(self):
        clear_persona_registry()
        assert get_persona("json_extractor") is None
        reset_persona_registry()
        assert get_persona("json_extractor") is not None
        assert len(get_persona_names()) == 5


# ==================================================================
# Phase 4e: Integration with Conversation & Agent
# ==================================================================


class TestConversationPersonaIntegration:
    def test_persona_string_lookup(self):
        """Conversation(persona="json_extractor") should resolve from registry."""
        from prompture.agents.conversation import Conversation

        driver = MagicMock()
        conv = Conversation(driver=driver, persona="json_extractor")
        assert conv._system_prompt is not None
        assert "extraction" in conv._system_prompt.lower()

    def test_persona_instance(self):
        """Conversation(persona=my_persona) should render directly."""
        from prompture.agents.conversation import Conversation

        driver = MagicMock()
        p = Persona(name="test", system_prompt="Custom persona prompt.")
        conv = Conversation(driver=driver, persona=p)
        assert conv._system_prompt == "Custom persona prompt."

    def test_persona_and_system_prompt_raises(self):
        """Providing both persona and system_prompt should raise ValueError."""
        from prompture.agents.conversation import Conversation

        driver = MagicMock()
        with pytest.raises(ValueError, match="Cannot provide both"):
            Conversation(driver=driver, system_prompt="Hello", persona="json_extractor")

    def test_persona_settings_as_defaults(self):
        """persona.settings should be applied as default options."""
        from prompture.agents.conversation import Conversation

        driver = MagicMock()
        conv = Conversation(driver=driver, persona="json_extractor")
        assert conv._options.get("temperature") == 0.0

    def test_persona_settings_overridden_by_explicit_options(self):
        """Explicit options should override persona.settings."""
        from prompture.agents.conversation import Conversation

        driver = MagicMock()
        conv = Conversation(driver=driver, persona="json_extractor", options={"temperature": 0.7})
        assert conv._options["temperature"] == 0.7

    def test_persona_model_hint_used(self):
        """persona.model_hint should be used if model_name is not provided."""
        from prompture.agents.conversation import Conversation

        p = Persona(name="hinted", system_prompt="With model hint.", model_hint="ollama/test:latest")
        # This would normally call get_driver_for_model("ollama/test:latest")
        # which would fail without the driver. We test the ValueError path instead.
        driver = MagicMock()
        conv = Conversation(driver=driver, persona=p)
        assert conv._system_prompt == "With model hint."

    def test_persona_string_not_found_raises(self):
        """Passing an unregistered persona string should raise ValueError."""
        from prompture.agents.conversation import Conversation

        driver = MagicMock()
        with pytest.raises(ValueError, match="not found"):
            Conversation(driver=driver, persona="nonexistent_persona")


class TestAsyncConversationPersonaIntegration:
    def test_persona_string_lookup(self):
        from prompture.agents.async_conversation import AsyncConversation

        driver = MagicMock()
        conv = AsyncConversation(driver=driver, persona="json_extractor")
        assert conv._system_prompt is not None
        assert "extraction" in conv._system_prompt.lower()

    def test_persona_and_system_prompt_raises(self):
        from prompture.agents.async_conversation import AsyncConversation

        driver = MagicMock()
        with pytest.raises(ValueError, match="Cannot provide both"):
            AsyncConversation(driver=driver, system_prompt="Hello", persona="json_extractor")


class TestAgentPersonaIntegration:
    def test_agent_with_persona_instance(self):
        """Agent(system_prompt=my_persona) should use Persona."""
        from prompture.agents.agent import Agent

        p = Persona(name="agent_test", system_prompt="I am an agent persona.")
        agent = Agent(model="test/model", driver=MagicMock(), system_prompt=p)
        resolved = agent._resolve_system_prompt()
        assert "I am an agent persona." in resolved

    def test_agent_persona_render_with_context(self):
        """Persona should render with RunContext variables."""
        from prompture.agents.agent import Agent
        from prompture.agents.types import RunContext

        p = Persona(name="ctx_test", system_prompt="Model: {{model}}")
        agent = Agent(model="test/model", driver=MagicMock(), system_prompt=p)

        ctx = RunContext(deps=None, model="test/model", usage={}, messages=[], iteration=0, prompt="test")
        resolved = agent._resolve_system_prompt(ctx)
        assert "test/model" in resolved

    def test_agent_persona_with_output_type(self):
        """Persona + output_type should concatenate correctly."""
        from pydantic import BaseModel

        from prompture.agents.agent import Agent

        class Output(BaseModel):
            value: str

        p = Persona(name="typed", system_prompt="Be structured.")
        agent = Agent(model="test/model", driver=MagicMock(), system_prompt=p, output_type=Output)
        resolved = agent._resolve_system_prompt()
        assert "Be structured." in resolved
        assert "JSON" in resolved


class TestAsyncAgentPersonaIntegration:
    def test_async_agent_with_persona(self):
        from prompture.agents.async_agent import AsyncAgent

        p = Persona(name="async_test", system_prompt="Async persona.")
        agent = AsyncAgent(model="test/model", driver=MagicMock(), system_prompt=p)
        resolved = agent._resolve_system_prompt()
        assert "Async persona." in resolved


# ==================================================================
# Phase 4f: Serialization & Persistence
# ==================================================================


class TestPersonaSerialization:
    def test_to_dict_and_from_dict(self):
        p = Persona(
            name="ser",
            system_prompt="Serialized.",
            description="For serialization.",
            traits=("polite",),
            variables={"key": "val"},
            constraints=["Be brief."],
            model_hint="openai/gpt-4",
            settings={"temperature": 0.5},
        )
        d = p.to_dict()
        assert d["version"] == 1
        assert d["name"] == "ser"

        restored = Persona.from_dict(d)
        assert restored.name == p.name
        assert restored.system_prompt == p.system_prompt
        assert restored.traits == p.traits
        assert restored.variables == p.variables
        assert restored.constraints == p.constraints
        assert restored.model_hint == p.model_hint
        assert restored.settings == p.settings

    def test_to_dict_minimal(self):
        p = Persona(name="min", system_prompt="Minimal.")
        d = p.to_dict()
        assert "description" not in d
        assert "traits" not in d
        assert "variables" not in d
        assert "constraints" not in d
        assert "model_hint" not in d
        assert "settings" not in d

    def test_json_file_roundtrip(self, tmp_path):
        p = Persona(
            name="json_test",
            system_prompt="JSON file test.",
            constraints=["C1"],
            settings={"temperature": 0.2},
        )
        file_path = tmp_path / "persona.json"
        p.save_json(file_path)
        loaded = Persona.load_json(file_path)
        assert loaded.name == p.name
        assert loaded.system_prompt == p.system_prompt
        assert loaded.constraints == p.constraints
        assert loaded.settings == p.settings

    def test_yaml_file_roundtrip(self, tmp_path):
        pytest.importorskip("yaml")
        p = Persona(
            name="yaml_test",
            system_prompt="YAML file test.",
            variables={"x": "1"},
        )
        file_path = tmp_path / "persona.yaml"
        p.save_yaml(file_path)
        loaded = Persona.load_yaml(file_path)
        assert loaded.name == p.name
        assert loaded.system_prompt == p.system_prompt
        assert loaded.variables == p.variables

    def test_load_personas_from_directory(self, tmp_path):
        # Create JSON persona
        p1 = Persona(name="dir_json", system_prompt="From JSON.")
        p1.save_json(tmp_path / "p1.json")

        # Optionally create YAML persona
        try:
            import importlib.util

            expect_yaml = importlib.util.find_spec("yaml") is not None
        except Exception:
            expect_yaml = False
        if expect_yaml:
            p2 = Persona(name="dir_yaml", system_prompt="From YAML.")
            p2.save_yaml(tmp_path / "p2.yaml")

        loaded = load_personas_from_directory(tmp_path)
        assert any(p.name == "dir_json" for p in loaded)
        assert get_persona("dir_json") is not None

        if expect_yaml:
            assert any(p.name == "dir_yaml" for p in loaded)
            assert get_persona("dir_yaml") is not None

    def test_yaml_import_error(self, tmp_path, monkeypatch):
        """save_yaml/load_yaml should raise ImportError if pyyaml missing."""
        import sys

        # Only test if we can actually hide yaml
        if "yaml" in sys.modules:
            pytest.skip("yaml is installed, cannot test ImportError path easily")


# ==================================================================
# Smoke test (from plan verification)
# ==================================================================


class TestSmokeTest:
    def test_import_and_render(self):
        """Smoke test from the plan verification section."""
        from prompture import PERSONAS

        p = PERSONAS["json_extractor"]
        rendered = p.render()
        assert isinstance(rendered, str)
        assert len(rendered) > 0
