"""Tests for the skills module."""

from __future__ import annotations

import threading

import pytest

from prompture.skills import (
    SKILLS,
    SkillInfo,
    clear_skill_registry,
    discover_skills,
    get_skill,
    get_skill_names,
    get_skill_registry_snapshot,
    load_skill,
    load_skill_from_directory,
    load_skills_from_directory,
    register_skill,
    unregister_skill,
)

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_registry():
    """Reset skill registry before and after each test."""
    clear_skill_registry()
    yield
    clear_skill_registry()


@pytest.fixture
def basic_skill():
    return SkillInfo(
        name="test-skill",
        description="A test skill.",
        instructions="Do testing things.",
    )


@pytest.fixture
def skill_with_metadata():
    return SkillInfo(
        name="meta-skill",
        description="Skill with metadata.",
        instructions="Instructions here.",
        user_invocable=False,
        metadata={"author": "test", "version": "1.0"},
    )


@pytest.fixture
def sample_skill_md(tmp_path):
    """Create a sample SKILL.md file."""
    skill_dir = tmp_path / "sample-skill"
    skill_dir.mkdir()
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(
        """---
name: sample-skill
description: A sample skill for testing
user-invocable: true
metadata:
  author: prompture
  version: "1.0"
---

# Sample Skill

This is the skill instructions body.

## Usage

Use this skill for testing.
""",
        encoding="utf-8",
    )
    return skill_file


@pytest.fixture
def skill_without_name(tmp_path):
    """Create a SKILL.md without a name field."""
    skill_dir = tmp_path / "unnamed-skill"
    skill_dir.mkdir()
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(
        """---
description: Skill without explicit name
---

# Instructions

Just some instructions.
""",
        encoding="utf-8",
    )
    return skill_file


@pytest.fixture
def skill_no_frontmatter(tmp_path):
    """Create a SKILL.md without frontmatter."""
    skill_dir = tmp_path / "no-frontmatter"
    skill_dir.mkdir()
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(
        """# Just a Markdown File

No YAML frontmatter here.
""",
        encoding="utf-8",
    )
    return skill_file


@pytest.fixture
def skills_directory(tmp_path):
    """Create a directory with multiple skills."""
    skills_root = tmp_path / "skills"
    skills_root.mkdir()

    # Skill 1
    skill1_dir = skills_root / "skill-one"
    skill1_dir.mkdir()
    (skill1_dir / "SKILL.md").write_text(
        """---
name: skill-one
description: First skill
---

Skill one instructions.
""",
        encoding="utf-8",
    )

    # Skill 2
    skill2_dir = skills_root / "skill-two"
    skill2_dir.mkdir()
    (skill2_dir / "SKILL.md").write_text(
        """---
name: skill-two
description: Second skill
user-invocable: false
---

Skill two instructions.
""",
        encoding="utf-8",
    )

    # Non-skill directory (no SKILL.md)
    other_dir = skills_root / "not-a-skill"
    other_dir.mkdir()
    (other_dir / "README.md").write_text("Not a skill", encoding="utf-8")

    return skills_root


# ==================================================================
# SkillInfo Dataclass
# ==================================================================


class TestSkillInfoConstruction:
    def test_basic_construction(self, basic_skill):
        assert basic_skill.name == "test-skill"
        assert basic_skill.description == "A test skill."
        assert basic_skill.instructions == "Do testing things."
        assert basic_skill.user_invocable is True
        assert basic_skill.metadata == {}
        assert basic_skill.source_path is None
        assert basic_skill.skill_dir is None

    def test_full_construction(self, skill_with_metadata):
        assert skill_with_metadata.name == "meta-skill"
        assert skill_with_metadata.user_invocable is False
        assert skill_with_metadata.metadata["author"] == "test"
        assert skill_with_metadata.metadata["version"] == "1.0"

    def test_frozen_immutability(self, basic_skill):
        with pytest.raises(AttributeError):
            basic_skill.name = "changed"
        with pytest.raises(AttributeError):
            basic_skill.instructions = "changed"

    def test_defaults(self):
        skill = SkillInfo(name="min", description="", instructions="")
        assert skill.user_invocable is True
        assert skill.metadata == {}
        assert skill.source_path is None
        assert skill.skill_dir is None


class TestSkillInfoAsPersona:
    def test_as_persona_basic(self, basic_skill):
        persona = basic_skill.as_persona()
        assert persona.name == "test_skill"  # hyphens converted to underscores
        assert persona.system_prompt == "Do testing things."
        assert persona.description == "A test skill."

    def test_as_persona_with_metadata(self, skill_with_metadata):
        persona = skill_with_metadata.as_persona()
        assert "skill_metadata" in persona.settings
        assert persona.settings["skill_metadata"]["author"] == "test"

    def test_as_persona_no_metadata(self, basic_skill):
        persona = basic_skill.as_persona()
        assert persona.settings == {}


class TestSkillInfoResourcePath:
    def test_get_resource_path_with_skill_dir(self, tmp_path):
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        resource = skill_dir / "template.txt"
        resource.write_text("template content", encoding="utf-8")

        skill = SkillInfo(
            name="test",
            description="",
            instructions="",
            skill_dir=skill_dir,
        )
        path = skill.get_resource_path("template.txt")
        assert path is not None
        assert path == skill_dir / "template.txt"

    def test_get_resource_path_no_skill_dir(self, basic_skill):
        assert basic_skill.get_resource_path("anything.txt") is None


class TestSkillInfoSerialization:
    def test_to_dict_full(self, skill_with_metadata):
        d = skill_with_metadata.to_dict()
        assert d["version"] == 1
        assert d["name"] == "meta-skill"
        assert d["description"] == "Skill with metadata."
        assert d["instructions"] == "Instructions here."
        assert d["user_invocable"] is False
        assert d["metadata"]["author"] == "test"

    def test_to_dict_minimal(self, basic_skill):
        d = basic_skill.to_dict()
        assert d["name"] == "test-skill"
        assert "metadata" not in d  # Empty metadata not included
        assert "source_path" not in d
        assert "skill_dir" not in d

    def test_from_dict_roundtrip(self, skill_with_metadata):
        d = skill_with_metadata.to_dict()
        restored = SkillInfo.from_dict(d)
        assert restored.name == skill_with_metadata.name
        assert restored.description == skill_with_metadata.description
        assert restored.instructions == skill_with_metadata.instructions
        assert restored.user_invocable == skill_with_metadata.user_invocable
        assert restored.metadata == skill_with_metadata.metadata

    def test_from_dict_with_paths(self, tmp_path):
        d = {
            "name": "test",
            "description": "desc",
            "instructions": "inst",
            "source_path": str(tmp_path / "SKILL.md"),
            "skill_dir": str(tmp_path),
        }
        skill = SkillInfo.from_dict(d)
        assert skill.source_path == tmp_path / "SKILL.md"
        assert skill.skill_dir == tmp_path


# ==================================================================
# Parsing
# ==================================================================


class TestLoadSkill:
    def test_load_skill_basic(self, sample_skill_md):
        skill = load_skill(sample_skill_md)
        assert skill.name == "sample-skill"
        assert skill.description == "A sample skill for testing"
        assert skill.user_invocable is True
        assert skill.metadata["author"] == "prompture"
        assert "# Sample Skill" in skill.instructions
        assert skill.source_path == sample_skill_md.resolve()
        assert skill.skill_dir == sample_skill_md.parent.resolve()

    def test_load_skill_fallback_name(self, skill_without_name):
        skill = load_skill(skill_without_name)
        # Should fall back to directory name
        assert skill.name == "unnamed-skill"

    def test_load_skill_no_frontmatter(self, skill_no_frontmatter):
        skill = load_skill(skill_no_frontmatter)
        # Should fall back to directory name
        assert skill.name == "no-frontmatter"
        assert "# Just a Markdown File" in skill.instructions

    def test_load_skill_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_skill(tmp_path / "nonexistent" / "SKILL.md")

    def test_load_skill_from_directory(self, sample_skill_md):
        skill = load_skill_from_directory(sample_skill_md.parent)
        assert skill.name == "sample-skill"

    def test_load_skill_from_directory_no_skill_md(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="No SKILL.md found"):
            load_skill_from_directory(empty_dir)


class TestLoadSkillsFromDirectory:
    def test_load_skills_from_directory(self, skills_directory):
        skills = load_skills_from_directory(skills_directory, register=False)
        assert len(skills) == 2
        names = {s.name for s in skills}
        assert "skill-one" in names
        assert "skill-two" in names

    def test_load_skills_from_directory_with_register(self, skills_directory):
        skills = load_skills_from_directory(skills_directory, register=True)
        assert len(skills) == 2
        assert get_skill("skill-one") is not None
        assert get_skill("skill-two") is not None

    def test_load_skills_from_nonexistent_directory(self, tmp_path):
        skills = load_skills_from_directory(tmp_path / "nonexistent")
        assert skills == []


# ==================================================================
# Registry
# ==================================================================


class TestSkillRegistry:
    def test_register_and_get(self, basic_skill):
        register_skill(basic_skill)
        result = get_skill("test-skill")
        assert result is basic_skill

    def test_get_nonexistent(self):
        assert get_skill("nonexistent") is None

    def test_get_names(self, basic_skill, skill_with_metadata):
        register_skill(basic_skill)
        register_skill(skill_with_metadata)
        names = get_skill_names()
        assert "test-skill" in names
        assert "meta-skill" in names

    def test_unregister(self, basic_skill):
        register_skill(basic_skill)
        assert unregister_skill("test-skill") is True
        assert get_skill("test-skill") is None
        # Unregistering again returns False
        assert unregister_skill("test-skill") is False

    def test_clear(self, basic_skill, skill_with_metadata):
        register_skill(basic_skill)
        register_skill(skill_with_metadata)
        clear_skill_registry()
        assert get_skill_names() == []

    def test_snapshot(self, basic_skill):
        register_skill(basic_skill)
        snapshot = get_skill_registry_snapshot()
        assert isinstance(snapshot, dict)
        assert "test-skill" in snapshot
        # Modifying snapshot shouldn't affect registry
        del snapshot["test-skill"]
        assert get_skill("test-skill") is not None

    def test_thread_safety(self):
        errors = []

        def register_many(prefix: str, count: int):
            try:
                for i in range(count):
                    skill = SkillInfo(
                        name=f"{prefix}-{i}",
                        description=f"Skill {prefix}-{i}",
                        instructions=f"Do {prefix}-{i}",
                    )
                    register_skill(skill)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=register_many, args=(f"t{t}", 50)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(get_skill_names()) == 200


class TestSkillRegistryProxy:
    def test_getitem(self, basic_skill):
        register_skill(basic_skill)
        skill = SKILLS["test-skill"]
        assert skill is basic_skill

    def test_getitem_missing(self):
        with pytest.raises(KeyError, match="not found"):
            SKILLS["nonexistent"]

    def test_setitem(self, basic_skill):
        SKILLS["custom"] = basic_skill
        assert get_skill("custom") is basic_skill

    def test_setitem_type_check(self):
        with pytest.raises(TypeError):
            SKILLS["bad"] = "not a skill"

    def test_delitem(self, basic_skill):
        register_skill(basic_skill)
        del SKILLS["test-skill"]
        assert get_skill("test-skill") is None

    def test_delitem_missing(self):
        with pytest.raises(KeyError):
            del SKILLS["nonexistent"]

    def test_contains(self, basic_skill):
        register_skill(basic_skill)
        assert "test-skill" in SKILLS
        assert "nonexistent" not in SKILLS

    def test_iter(self, basic_skill, skill_with_metadata):
        register_skill(basic_skill)
        register_skill(skill_with_metadata)
        names = list(SKILLS)
        assert "test-skill" in names
        assert "meta-skill" in names

    def test_len(self, basic_skill, skill_with_metadata):
        register_skill(basic_skill)
        register_skill(skill_with_metadata)
        assert len(SKILLS) == 2

    def test_keys_values_items(self, basic_skill):
        register_skill(basic_skill)
        assert "test-skill" in SKILLS
        values = SKILLS.values()
        assert all(isinstance(v, SkillInfo) for v in values)
        items = SKILLS.items()
        assert all(isinstance(k, str) and isinstance(v, SkillInfo) for k, v in items)

    def test_get_with_default(self, basic_skill):
        register_skill(basic_skill)
        assert SKILLS.get("nonexistent", "default") == "default"
        assert SKILLS.get("test-skill") is basic_skill

    def test_repr(self, basic_skill):
        register_skill(basic_skill)
        r = repr(SKILLS)
        assert "SKILLS" in r


# ==================================================================
# Discovery
# ==================================================================


class TestDiscoverSkills:
    def test_discover_skills_with_additional_paths(self, skills_directory):
        skills = discover_skills(additional_paths=[skills_directory], register=True)
        assert len(skills) >= 2
        assert get_skill("skill-one") is not None
        assert get_skill("skill-two") is not None

    def test_discover_skills_no_register(self, skills_directory):
        skills = discover_skills(additional_paths=[skills_directory], register=False)
        assert len(skills) >= 2
        # Should not be registered
        assert get_skill("skill-one") is None

    def test_discover_skills_deduplicates_paths(self, skills_directory):
        # Pass the same path twice
        skills = discover_skills(additional_paths=[skills_directory, skills_directory], register=True)
        # Should only load each skill once
        names = [s.name for s in skills]
        assert names.count("skill-one") == 1


# ==================================================================
# Persona Integration
# ==================================================================


class TestPersonaFromSkill:
    def test_persona_from_skill(self, sample_skill_md):
        from prompture.persona import Persona

        persona = Persona.from_skill(sample_skill_md)
        assert persona.name == "sample_skill"  # hyphens to underscores
        assert "# Sample Skill" in persona.system_prompt
        assert persona.description == "A sample skill for testing"


# ==================================================================
# Package Exports
# ==================================================================


class TestPackageExports:
    def test_all_exports_available(self):
        from prompture import (
            SKILLS,
            SkillInfo,
            clear_skill_registry,
            discover_skills,
            get_skill,
            get_skill_names,
            get_skill_registry_snapshot,
            load_skill,
            load_skill_from_directory,
            load_skills_from_directory,
            register_skill,
            unregister_skill,
        )

        # Just verify they're importable
        assert SKILLS is not None
        assert SkillInfo is not None
        assert clear_skill_registry is not None
        assert discover_skills is not None
        assert get_skill is not None
        assert get_skill_names is not None
        assert get_skill_registry_snapshot is not None
        assert load_skill is not None
        assert load_skill_from_directory is not None
        assert load_skills_from_directory is not None
        assert register_skill is not None
        assert unregister_skill is not None
