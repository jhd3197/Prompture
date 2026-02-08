"""Persona templates module for Prompture.

Reusable, composable system prompt definitions with template variables,
layered composition, and a thread-safe registry — mirroring the
``field_definitions.py`` pattern.

Features:
- Frozen ``Persona`` dataclass with template rendering
- Composition via ``extend()``, ``with_constraints()``, and ``+`` operator
- Thread-safe trait registry for reusable prompt fragments
- Thread-safe global persona registry with dict-like proxy
- 5 built-in personas for common use cases
- JSON/YAML serialization and directory loading

Usage with :class:`~prompture.agent.Agent`::

    from prompture import Agent, Persona

    persona = Persona(
        name="cachibot",
        system_prompt="You are {{agent_name}}, a helpful assistant.\\nWorkspace: {{workspace}}",
        variables={"agent_name": "Cachibot"},
        constraints=["Only access files within the workspace", "Use python_execute for code"],
    )

    agent = Agent("openai/gpt-4o", system_prompt=persona)
    result = agent.run("What files are in the workspace?")
"""

from __future__ import annotations

import collections.abc
import dataclasses
import json
import logging
import threading
import warnings
from pathlib import Path
from typing import Any

from ..extraction.fields import _apply_templates, _get_template_variables

logger = logging.getLogger("prompture.persona")

_SERIALIZATION_VERSION = 1


# ------------------------------------------------------------------
# Persona dataclass
# ------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Persona:
    """A reusable system prompt template with metadata.

    Instances are immutable (frozen).  Use :meth:`extend`,
    :meth:`with_constraints`, or the ``+`` operator to derive new
    personas.

    Args:
        name: Short identifier for this persona.
        system_prompt: The system prompt template.  May contain
            ``{{variable}}`` placeholders.
        description: Human-readable description of the persona's purpose.
        traits: Tuple of trait names to resolve from the trait registry
            during :meth:`render`.
        variables: Default template variable values.
        constraints: List of constraint strings appended as a
            ``## Constraints`` section during :meth:`render`.
        model_hint: Suggested model string (e.g. ``"openai/gpt-4"``).
            Used as a default when no model is explicitly provided.
        settings: Default driver options (e.g. ``{"temperature": 0.0}``).
    """

    name: str
    system_prompt: str
    description: str = ""
    traits: tuple[str, ...] = ()
    variables: dict[str, Any] = dataclasses.field(default_factory=dict)
    constraints: list[str] = dataclasses.field(default_factory=list)
    model_hint: str | None = None
    settings: dict[str, Any] = dataclasses.field(default_factory=dict)

    def render(self, **kwargs: Any) -> str:
        """Render the system prompt with template variable substitution.

        Variable precedence (highest wins):
        1. ``kwargs`` passed to this method
        2. ``self.variables``
        3. Built-in template variables (``{{current_year}}``, etc.)

        Registered traits (from the trait registry) are appended between
        the main prompt body and the constraints section.
        """
        # Merge variables: built-in < self.variables < kwargs
        merged_vars = _get_template_variables()
        merged_vars.update(self.variables)
        merged_vars.update(kwargs)

        rendered = _apply_templates(self.system_prompt, merged_vars)

        # Append registered traits
        if self.traits:
            trait_texts: list[str] = []
            for trait_name in self.traits:
                text = get_trait(trait_name)
                if text is not None:
                    trait_texts.append(_apply_templates(text, merged_vars))
            if trait_texts:
                rendered += "\n\n" + "\n\n".join(trait_texts)

        # Append constraints
        if self.constraints:
            rendered_constraints = [_apply_templates(c, merged_vars) for c in self.constraints]
            rendered += "\n\n## Constraints\n" + "\n".join(f"- {c}" for c in rendered_constraints)

        return rendered

    # ------------------------------------------------------------------
    # Composition helpers
    # ------------------------------------------------------------------

    def extend(self, additional_instructions: str) -> Persona:
        """Return a new Persona with *additional_instructions* appended."""
        return dataclasses.replace(
            self,
            system_prompt=self.system_prompt + "\n\n" + additional_instructions,
        )

    def with_constraints(self, new_constraints: list[str]) -> Persona:
        """Return a new Persona with *new_constraints* appended."""
        return dataclasses.replace(
            self,
            constraints=[*self.constraints, *new_constraints],
        )

    def __add__(self, other: Persona) -> Persona:
        """Merge two personas.  Right-side values win on conflict."""
        if not isinstance(other, Persona):
            return NotImplemented

        # Warn on variable conflicts
        for key in self.variables:
            if key in other.variables and self.variables[key] != other.variables[key]:
                warnings.warn(
                    f"Persona variable '{key}' conflict: "
                    f"'{self.variables[key]}' overridden by '{other.variables[key]}'",
                    stacklevel=2,
                )

        merged_vars = {**self.variables, **other.variables}
        merged_settings = {**self.settings, **other.settings}
        merged_traits = tuple(dict.fromkeys(self.traits + other.traits))  # dedupe, preserve order

        return Persona(
            name=f"{self.name}+{other.name}",
            system_prompt=self.system_prompt + "\n\n" + other.system_prompt,
            description=f"{self.description}; {other.description}" if self.description and other.description else self.description or other.description,
            traits=merged_traits,
            variables=merged_vars,
            constraints=[*self.constraints, *other.constraints],
            model_hint=other.model_hint or self.model_hint,
            settings=merged_settings,
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        data: dict[str, Any] = {
            "version": _SERIALIZATION_VERSION,
            "name": self.name,
            "system_prompt": self.system_prompt,
        }
        if self.description:
            data["description"] = self.description
        if self.traits:
            data["traits"] = list(self.traits)
        if self.variables:
            data["variables"] = dict(self.variables)
        if self.constraints:
            data["constraints"] = list(self.constraints)
        if self.model_hint is not None:
            data["model_hint"] = self.model_hint
        if self.settings:
            data["settings"] = dict(self.settings)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Persona:
        """Deserialize from a dictionary."""
        return cls(
            name=data["name"],
            system_prompt=data["system_prompt"],
            description=data.get("description", ""),
            traits=tuple(data.get("traits", ())),
            variables=dict(data.get("variables", {})),
            constraints=list(data.get("constraints", [])),
            model_hint=data.get("model_hint"),
            settings=dict(data.get("settings", {})),
        )

    def save_json(self, path: str | Path) -> None:
        """Write this persona to a JSON file."""
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> Persona:
        """Load a persona from a JSON file."""
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(data)

    def save_yaml(self, path: str | Path) -> None:
        """Write this persona to a YAML file.  Requires ``pyyaml``."""
        try:
            import yaml
        except ImportError:
            raise ImportError("pyyaml is required for YAML support. Install with: pip install pyyaml") from None
        path = Path(path)
        path.write_text(yaml.safe_dump(self.to_dict(), default_flow_style=False), encoding="utf-8")

    @classmethod
    def load_yaml(cls, path: str | Path) -> Persona:
        """Load a persona from a YAML file.  Requires ``pyyaml``."""
        try:
            import yaml
        except ImportError:
            raise ImportError("pyyaml is required for YAML support. Install with: pip install pyyaml") from None
        path = Path(path)
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls.from_dict(data)

    @classmethod
    def from_skill(cls, path: str | Path) -> Persona:
        """Create a Persona from a SKILL.md file.

        Loads the skill and converts it to a Persona using :meth:`SkillInfo.as_persona`.

        Args:
            path: Path to the SKILL.md file.

        Returns:
            A Persona instance configured with the skill's data.
        """
        from .skills import load_skill

        skill = load_skill(path)
        return skill.as_persona()


# ------------------------------------------------------------------
# Trait registry
# ------------------------------------------------------------------

_trait_registry_lock = threading.Lock()
_trait_registry: dict[str, str] = {}


def register_trait(name: str, text: str) -> None:
    """Register a named trait text fragment."""
    with _trait_registry_lock:
        _trait_registry[name] = text


def get_trait(name: str) -> str | None:
    """Retrieve a trait by name, or ``None`` if not found."""
    with _trait_registry_lock:
        return _trait_registry.get(name)


def get_trait_names() -> list[str]:
    """Return all registered trait names."""
    with _trait_registry_lock:
        return list(_trait_registry.keys())


def reset_trait_registry() -> None:
    """Clear all registered traits."""
    with _trait_registry_lock:
        _trait_registry.clear()


# ------------------------------------------------------------------
# Global persona registry
# ------------------------------------------------------------------

_persona_registry_lock = threading.Lock()
_persona_global_registry: dict[str, Persona] = {}


def register_persona(persona: Persona) -> None:
    """Register a persona in the global registry."""
    with _persona_registry_lock:
        _persona_global_registry[persona.name] = persona


def get_persona(name: str) -> Persona | None:
    """Retrieve a persona by name, or ``None`` if not found."""
    with _persona_registry_lock:
        return _persona_global_registry.get(name)


def get_persona_names() -> list[str]:
    """Return all registered persona names."""
    with _persona_registry_lock:
        return list(_persona_global_registry.keys())


def get_persona_registry_snapshot() -> dict[str, Persona]:
    """Return a shallow copy of the current persona registry."""
    with _persona_registry_lock:
        return dict(_persona_global_registry)


def clear_persona_registry() -> None:
    """Remove all personas from the global registry."""
    with _persona_registry_lock:
        _persona_global_registry.clear()


def reset_persona_registry() -> None:
    """Reset the global registry to contain only built-in personas."""
    with _persona_registry_lock:
        _persona_global_registry.clear()
        _persona_global_registry.update(BASE_PERSONAS)


# ------------------------------------------------------------------
# Persona registry proxy (dict-like access)
# ------------------------------------------------------------------


class _PersonaRegistryProxy(dict, collections.abc.MutableMapping):
    """Dict-like proxy for the global persona registry.

    Allows ``PERSONAS["json_extractor"]`` style access.
    """

    def __getitem__(self, key: str) -> Persona:
        persona = get_persona(key)
        if persona is None:
            raise KeyError(f"Persona '{key}' not found in registry. Available: {', '.join(get_persona_names())}")
        return persona

    def __setitem__(self, key: str, value: Persona) -> None:
        if not isinstance(value, Persona):
            raise TypeError(f"Expected Persona instance, got {type(value).__name__}")
        with _persona_registry_lock:
            _persona_global_registry[key] = value

    def __delitem__(self, key: str) -> None:
        with _persona_registry_lock:
            if key in _persona_global_registry:
                del _persona_global_registry[key]
            else:
                raise KeyError(f"Persona '{key}' not found in registry")

    def __contains__(self, key: object) -> bool:
        return key in get_persona_names()

    def __iter__(self):
        return iter(get_persona_names())

    def keys(self):
        return get_persona_names()

    def values(self):
        with _persona_registry_lock:
            return list(_persona_global_registry.values())

    def items(self):
        with _persona_registry_lock:
            return list(_persona_global_registry.items())

    def __len__(self) -> int:
        with _persona_registry_lock:
            return len(_persona_global_registry)

    def get(self, key: str, default: Any = None) -> Any:
        persona = get_persona(key)
        return persona if persona is not None else default

    def __repr__(self) -> str:
        return f"PERSONAS({get_persona_names()})"


# ------------------------------------------------------------------
# Built-in personas
# ------------------------------------------------------------------

BASE_PERSONAS: dict[str, Persona] = {
    "json_extractor": Persona(
        name="json_extractor",
        system_prompt=(
            "You are a precise data extraction assistant. "
            "Your sole task is to extract structured information from the provided text "
            "and return it as valid JSON. Do not add commentary, explanations, or markdown formatting."
        ),
        description="Precise JSON extraction with strict output formatting.",
        constraints=[
            "Output ONLY valid JSON — no markdown fences, no prose.",
            "Use null for unknown or missing values.",
            "Preserve original data types (numbers as numbers, booleans as booleans).",
        ],
        settings={"temperature": 0.0},
    ),
    "data_analyst": Persona(
        name="data_analyst",
        system_prompt=(
            "You are a quantitative data analyst. "
            "Analyze data rigorously, cite sources for claims, and present findings with precision. "
            "Use statistical reasoning where appropriate."
        ),
        description="Quantitative analysis with cited sources.",
        traits=(),
        constraints=[
            "Cite the source of any factual claim.",
            "Distinguish between correlation and causation.",
            "State confidence levels when making inferences.",
        ],
    ),
    "text_summarizer": Persona(
        name="text_summarizer",
        system_prompt=(
            "You are a text summarization assistant. "
            "Produce concise summaries that capture the key points of the input text. "
            "Limit your summary to {{max_sentences}} sentences unless instructed otherwise."
        ),
        description="Configurable text summarization.",
        variables={"max_sentences": "3"},
        constraints=[
            "Stay within the sentence limit.",
            "Do not introduce information not present in the source text.",
        ],
    ),
    "code_reviewer": Persona(
        name="code_reviewer",
        system_prompt=(
            "You are an expert code reviewer. "
            "Analyze code for correctness, performance, security, and readability. "
            "Structure your feedback using the following format:\n\n"
            "## Summary\n"
            "Brief overview of the code.\n\n"
            "## Issues\n"
            "Numbered list of issues found.\n\n"
            "## Suggestions\n"
            "Numbered list of improvement suggestions."
        ),
        description="Structured code review feedback.",
        constraints=[
            "Focus on substantive issues, not style preferences.",
            "Provide concrete fix suggestions, not vague advice.",
        ],
    ),
    "concise_assistant": Persona(
        name="concise_assistant",
        system_prompt=(
            "You are a concise assistant. Answer questions directly and briefly. "
            "Do not elaborate unless explicitly asked."
        ),
        description="Brief, no-elaboration responses.",
        constraints=[
            "Keep responses under 3 sentences when possible.",
            "No filler phrases or unnecessary politeness.",
        ],
    ),
}


def _initialize_persona_registry() -> None:
    """Populate the global registry with built-in personas."""
    with _persona_registry_lock:
        if not _persona_global_registry:
            _persona_global_registry.update(BASE_PERSONAS)


# Initialize on import
_initialize_persona_registry()

# Public proxy instance
PERSONAS = _PersonaRegistryProxy()


# ------------------------------------------------------------------
# Directory loading
# ------------------------------------------------------------------


def load_personas_from_directory(path: str | Path) -> list[Persona]:
    """Bulk-load persona files from a directory and register them.

    Supports ``.json``, ``.yaml``, and ``.yml`` files.
    Returns a list of loaded personas.
    """
    directory = Path(path)
    loaded: list[Persona] = []

    for file_path in sorted(directory.iterdir()):
        if file_path.suffix == ".json":
            persona = Persona.load_json(file_path)
            register_persona(persona)
            loaded.append(persona)
        elif file_path.suffix in (".yaml", ".yml"):
            persona = Persona.load_yaml(file_path)
            register_persona(persona)
            loaded.append(persona)

    return loaded
