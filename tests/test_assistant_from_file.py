"""Assistant definitions loaded from YAML, JSON and Markdown files."""

from __future__ import annotations

import json

import pytest

from prompture.agents.assistant import Assistant, KnowledgeDoc


def _write(path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def workspace(tmp_path):
    _write(tmp_path / "docs" / "customer.md", "Homeowners in Austin who just found a leak.")
    _write(tmp_path / "docs" / "cases" / "a.txt", "Case A: replaced a roof in two days.")
    _write(tmp_path / "docs" / "cases" / "b.txt", "Case B: fixed a skylight leak.")
    _write(tmp_path / "brand.md", "## Basics\n- Name: Northwind Roofing\n- Website: northwindroofing.com\n")
    _write(
        tmp_path / "skills" / "hooks" / "SKILL.md",
        "---\nname: hooks\ndescription: Write scroll-stopping first lines\n---\nOpen with a concrete image.\n",
    )
    return tmp_path


YAML_DEF = """
name: linkedin-copywriter
description: Writes LinkedIn posts
model: openai/gpt-4o
instructions: |
  You write LinkedIn posts for {{brand_name}} aimed at {{audience}}.
constraints: ["Short sentences."]
variables: {audience: homeowners}
knowledge: [docs/customer.md, docs/cases]
conversation_starters: ["Write this week's post", "Turn a case study into a post"]
tools: ["web:search"]
brand: brand.md
skills: [skills/hooks/SKILL.md]
options: {temperature: 0.7}
agent_options: {max_iterations: 3}
"""


def test_yaml_definition(workspace) -> None:
    assistant = Assistant.from_file(_write(workspace / "copywriter.yaml", YAML_DEF))

    assert assistant.name == "linkedin-copywriter"
    assert assistant.model == "openai/gpt-4o"
    assert assistant.conversation_starters == ("Write this week's post", "Turn a case study into a post")
    assert [doc.name for doc in assistant.knowledge] == ["customer", "a", "b"]
    assert assistant.tools == ("web:search",)
    assert assistant.brand.name == "Northwind Roofing"
    assert assistant.options == {"temperature": 0.7}
    assert assistant.agent_options == {"max_iterations": 3}
    assert assistant.skills[0].name == "hooks"

    rendered = assistant._composed_persona({}).render()
    assert rendered.startswith("You write LinkedIn posts for Northwind Roofing aimed at homeowners.")
    assert "## Brand context" in rendered
    assert "## Knowledge" in rendered
    assert "### customer\nHomeowners in Austin" in rendered
    assert "Case B: fixed a skylight leak." in rendered
    assert "## Skill: hooks" in rendered
    assert "- Short sentences." in rendered
    # Starters are for users, never the model.
    assert "Write this week's post" not in rendered


def test_call_vars_override_file_vars(workspace) -> None:
    assistant = Assistant.from_file(_write(workspace / "copywriter.yaml", YAML_DEF))
    assert "aimed at roofers." in assistant._composed_persona({"audience": "roofers"}).render()


def test_markdown_definition_uses_body_as_instructions(workspace) -> None:
    path = _write(
        workspace / "support.md",
        "---\nname: support\nmodel: openai/gpt-4o-mini\nknowledge: [docs/customer.md]\n---\n\nYou answer support emails.\n",
    )
    assistant = Assistant.from_file(path)
    assert assistant.instructions == "You answer support emails."
    assert assistant.brand is None
    assert len(assistant.knowledge) == 1


def test_json_definition_with_instructions_file(workspace) -> None:
    _write(workspace / "prompts" / "email.md", "You write emails.")
    path = _write(
        workspace / "email.json",
        json.dumps({"name": "email", "coding_agent": "claude", "instructions_file": "prompts/email.md"}),
    )
    assistant = Assistant.from_file(path)
    assert assistant.coding_agent == "claude"
    assert assistant.instructions == "You write emails."


def test_overrides_and_extra_tools(workspace) -> None:
    def lookup(order_id: str) -> str:
        """Look up an order."""
        return order_id

    path = _write(workspace / "copywriter.yaml", YAML_DEF)
    assistant = Assistant.from_file(path, model="claude/claude-sonnet-4-6", tools=[lookup], brand={"name": "Other Co"})
    assert assistant.model == "claude/claude-sonnet-4-6"
    assert assistant.tools == ("web:search", lookup)
    assert assistant.brand.name == "Other Co"


def test_knowledge_limit(workspace) -> None:
    path = _write(workspace / "big.yaml", YAML_DEF + "knowledge_max_chars: 20\n")
    with pytest.raises(ValueError, match="over the 20 limit"):
        Assistant.from_file(path)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("model: openai/gpt-4o\ninstructions: hi\n", "needs a 'name'"),
        ("name: x\nmodel: openai/gpt-4o\n", "needs 'instructions'"),
    ],
)
def test_required_keys(tmp_path, text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Assistant.from_file(_write(tmp_path / "a.yaml", text))


def test_missing_knowledge_file(tmp_path) -> None:
    path = _write(tmp_path / "a.yaml", "name: x\nmodel: openai/gpt-4o\ninstructions: hi\nknowledge: [nope.md]\n")
    with pytest.raises(FileNotFoundError):
        Assistant.from_file(path)


def test_with_instructions_keeps_everything_else(workspace) -> None:
    assistant = Assistant.from_file(_write(workspace / "copywriter.yaml", YAML_DEF))
    revised = assistant.with_instructions("New instructions.")
    assert revised.instructions == "New instructions."
    assert revised.knowledge == assistant.knowledge
    assert assistant.instructions.startswith("You write LinkedIn posts")


def test_knowledge_doc_direct() -> None:
    assistant = Assistant(
        name="a",
        persona=__import__("prompture").Persona(name="a", system_prompt="Base."),
        model="openai/gpt-4o",
        knowledge=[KnowledgeDoc("faq", "Q: hours? A: 9-5.")],
        conversation_starters=["What are your hours?"],
    )
    assert isinstance(assistant.knowledge, tuple)
    assert "### faq\nQ: hours? A: 9-5." in assistant._composed_persona({}).render()
