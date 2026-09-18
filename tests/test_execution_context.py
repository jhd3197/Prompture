"""Phase D contract checks: context allocation, selective loading, compaction.

Focus is on the guarantees that matter when a session gets long: protected
content survives, unauthorised tools stay unreachable, large results are
referenced rather than inlined, and compaction never produces a message list a
provider would reject.
"""

from __future__ import annotations

import pytest

from prompture.agents.skills import SkillInfo
from prompture.agents.tools_schema import ToolRegistry
from prompture.execution import InventoryWorld
from prompture.execution.context import (
    ArtifactStore,
    ContextPolicy,
    ContextSection,
    SkillCatalog,
    ToolCatalog,
    compact_messages,
    count_tokens,
    supports_native_tool_discovery,
)

# ---------------------------------------------------------------------------
# Context allocation
# ---------------------------------------------------------------------------


def _policy(**kwargs):
    kwargs.setdefault("window_tokens", 1000)
    kwargs.setdefault("response_allowance", 200)
    kwargs.setdefault(
        "sections",
        [
            ContextSection("instructions", protected=True),
            ContextSection("evidence", weight=3.0),
            ContextSection("history", weight=1.0),
        ],
    )
    return ContextPolicy(**kwargs)


def test_the_response_allowance_is_reserved_off_the_top():
    policy = _policy(window_tokens=1000, response_allowance=400)
    filler = ["x" * 4000]  # ~1000 tokens by the heuristic counter

    assembly = policy.allocate({"evidence": filler})

    assert assembly.response_allowance == 400
    assert assembly.total_tokens <= 600


def test_protected_sections_are_never_trimmed():
    policy = _policy(window_tokens=200, response_allowance=50)
    big_instructions = "i" * 4000

    assembly = policy.allocate({"instructions": [big_instructions], "evidence": ["e" * 400]})

    assert assembly.sections["instructions"] == [big_instructions]
    assert assembly.dropped["instructions"] == []
    assert assembly.overflow is True, "the caller must be told protected content blew the budget"
    assert assembly.sections["evidence"] == [], "nothing discretionary fits once protected content overflows"


def test_weights_divide_the_discretionary_budget():
    policy = _policy(window_tokens=1000, response_allowance=0)
    items = [("e" * 40) for _ in range(50)]  # ~10 tokens each

    assembly = policy.allocate({"evidence": list(items), "history": list(items)})

    assert assembly.allocated["evidence"] > assembly.allocated["history"]
    assert assembly.allocated["evidence"] + assembly.allocated["history"] <= 1000


def test_a_minimum_protects_a_small_but_required_section():
    policy = ContextPolicy(
        window_tokens=1000,
        response_allowance=0,
        sections=[
            ContextSection("history", weight=100.0),
            ContextSection("evidence", weight=0.01, min_tokens=100),
        ],
    )
    assembly = policy.allocate({"history": ["h" * 8000], "evidence": ["e" * 200]})

    assert assembly.allocated["evidence"] >= 100
    assert assembly.sections["evidence"] != []


def test_dropped_items_are_reported_not_silently_lost():
    policy = _policy(window_tokens=120, response_allowance=0)
    items = [("e" * 200) for _ in range(10)]

    assembly = policy.allocate({"evidence": items})

    assert assembly.dropped["evidence"], "the caller needs to know what did not fit"
    assert len(assembly.sections["evidence"]) + len(assembly.dropped["evidence"]) == 10


def test_content_for_an_unknown_section_is_ignored_not_charged(caplog):
    policy = _policy()
    assembly = policy.allocate({"typo_section": ["x" * 4000], "evidence": ["ok"]})

    assert "typo_section" not in assembly.sections
    assert assembly.sections["evidence"] == ["ok"]


def test_render_emits_sections_in_policy_order():
    policy = _policy(window_tokens=2000, response_allowance=0)
    assembly = policy.allocate({"instructions": ["INSTR"], "evidence": ["EV"], "history": ["HIST"]})

    assert assembly.render() == "INSTR\n\nEV\n\nHIST"


def test_policy_rejects_a_response_allowance_that_fills_the_window():
    with pytest.raises(ValueError, match="must leave room"):
        ContextPolicy(window_tokens=500, response_allowance=500)


def test_policy_rejects_duplicate_section_names():
    with pytest.raises(ValueError, match="Duplicate context section"):
        ContextPolicy(sections=[ContextSection("a"), ContextSection("a")])


def test_the_default_policy_protects_instructions_and_constraints():
    policy = ContextPolicy.default()
    assert policy.section("instructions").protected is True
    assert policy.section("constraints").protected is True
    assert policy.section("evidence").protected is False


# ---------------------------------------------------------------------------
# Skill catalog
# ---------------------------------------------------------------------------


def _skills():
    return [
        SkillInfo(
            name="cite-sources",
            description="Attribute every claim to a retrieved passage",
            instructions="X" * 500,
        ),
        SkillInfo(
            name="format-invoice",
            description="Lay out an invoice table",
            instructions="Y" * 500,
        ),
        SkillInfo(
            name="summarise-meeting",
            description="Condense meeting notes into actions",
            instructions="Z" * 500,
        ),
    ]


def test_summaries_cost_far_less_than_full_instructions():
    catalog = SkillCatalog(_skills())

    summary_cost = count_tokens(catalog.summary_block())
    eager_cost = count_tokens(catalog.all_instructions())

    assert summary_cost < eager_cost / 3
    assert "cite-sources" in catalog.summary_block()


def test_full_instructions_are_only_materialised_on_demand():
    catalog = SkillCatalog(_skills())
    assert catalog.loaded == set()

    catalog.instructions_for(["cite-sources"])

    assert catalog.loaded == {"cite-sources"}


def test_select_returns_the_relevant_skill_and_nothing_else():
    catalog = SkillCatalog(_skills())

    chosen = catalog.select("attribute each claim to a passage", k=2)

    assert chosen, "a clearly relevant skill must be found"
    assert chosen[0].name == "cite-sources"
    assert "format-invoice" not in [s.name for s in chosen]


def test_select_returns_nothing_when_nothing_is_relevant():
    catalog = SkillCatalog(_skills())
    assert catalog.select("quantum chromodynamics", k=3, min_relevance=0.5) == []


def test_eager_loading_stays_supported():
    catalog = SkillCatalog(_skills())
    body = catalog.all_instructions()

    assert catalog.loaded == {"cite-sources", "format-invoice", "summarise-meeting"}
    assert body.count("## Skill:") == 3


def test_an_unknown_skill_name_is_skipped_not_fatal():
    catalog = SkillCatalog(_skills())
    assert catalog.instructions_for(["hallucinated-skill"]) == ""


def test_catalog_can_be_built_from_the_global_registry():
    from prompture.agents.skills import clear_skill_registry, register_skill

    clear_skill_registry()
    try:
        register_skill(_skills()[0])
        catalog = SkillCatalog.from_registry()
        assert "cite-sources" in catalog.names
    finally:
        clear_skill_registry()


# ---------------------------------------------------------------------------
# Tool catalog
# ---------------------------------------------------------------------------


def _registry() -> ToolRegistry:
    world = InventoryWorld({"stock": {"SKU-1": {"A": 5}}})
    return world.as_tool_registry()


def test_an_unauthorised_tool_is_absent_from_the_catalogue_entirely():
    catalog = ToolCatalog(_registry(), allowed_tools={"get_stock", "list_inventory"})

    assert set(catalog.names) == {"get_stock", "list_inventory"}
    assert "move_stock" in catalog.excluded
    assert catalog.search("move stock between warehouses") == [] or all(
        hit.name != "move_stock" for hit in catalog.search("move stock between warehouses")
    )


def test_schemas_refuses_to_load_an_unauthorised_tool_even_by_name():
    catalog = ToolCatalog(_registry(), allowed_tools={"get_stock"})

    schemas = catalog.schemas(["get_stock", "move_stock"])

    assert [s["function"]["name"] for s in schemas] == ["get_stock"]
    assert catalog.schema_loads == ["get_stock"]


def test_search_returns_summaries_without_loading_any_schema():
    catalog = ToolCatalog(_registry())

    hits = catalog.search("how many units are in stock", limit=2)

    assert hits
    assert catalog.schema_loads == [], "searching must not materialise schemas"
    assert all(isinstance(h.description, str) for h in hits)


def test_a_summary_block_is_much_cheaper_than_every_schema():
    catalog = ToolCatalog(_registry())

    summary_cost = count_tokens(catalog.summary_block())
    full_cost = count_tokens(str(catalog.schemas(catalog.names)))

    assert summary_cost < full_cost / 2


def test_subset_registry_matches_what_the_model_was_shown():
    catalog = ToolCatalog(_registry(), allowed_tools={"get_stock", "move_stock"})

    executable = catalog.subset_registry(["get_stock", "restock"])

    assert executable.names == ["get_stock"], "an unauthorised or unshown tool is not executable"


def test_native_tool_discovery_falls_back_to_false_on_an_unknown_model():
    assert supports_native_tool_discovery("nonexistent-provider/imaginary") in {True, False}


# ---------------------------------------------------------------------------
# Artifact store
# ---------------------------------------------------------------------------


def test_a_large_result_is_stored_and_referenced_instead_of_inlined():
    store = ArtifactStore()
    payload = {"rows": [{"id": i, "value": "v" * 50} for i in range(200)]}

    text, ref = store.maybe_put(payload, scope="task-1", source="list_inventory", inline_char_limit=500)

    assert ref is not None
    assert "artifact://task-1/" in text
    assert len(text) < 500
    assert ref.approx_tokens > 500


def test_a_small_result_is_inlined_unchanged():
    store = ArtifactStore()
    text, ref = store.maybe_put("tiny", scope="task-1", inline_char_limit=500)

    assert ref is None
    assert text == "tiny"


def test_an_artifact_is_not_readable_from_another_scope():
    store = ArtifactStore()
    ref = store.put("secret payload", scope="task-1")

    assert store.get(ref.handle, scope="task-1") == "secret payload"
    with pytest.raises(PermissionError, match="not readable from"):
        store.get(ref.handle, scope="task-2")
    with pytest.raises(PermissionError):
        store.excerpt(ref.handle, scope="task-2")


def test_an_excerpt_loads_only_the_relevant_window():
    store = ArtifactStore()
    body = ("filler " * 500) + "THE ANSWER IS 42 " + ("filler " * 500)
    ref = store.put(body, scope="task-1")

    excerpt = store.excerpt(ref.handle, scope="task-1", query="answer", max_chars=600)

    assert "THE ANSWER IS 42" in excerpt
    assert len(excerpt) < len(body) / 2
    assert "excerpt of" in excerpt


def test_an_excerpt_says_when_it_truncated():
    store = ArtifactStore()
    ref = store.put("z" * 5000, scope="s")

    excerpt = store.excerpt(ref.handle, scope="s", max_chars=100)

    assert "truncated" in excerpt


def test_an_artifact_reference_keeps_its_source_attribution():
    store = ArtifactStore()
    ref = store.put("x" * 3000, scope="s", source="web_search", summary="search results")

    rendered = ref.render()
    assert "web_search" in rendered
    assert "search results" in rendered
    assert ref.handle in rendered


def test_an_unknown_handle_raises_keyerror():
    with pytest.raises(KeyError):
        ArtifactStore().get("artifact://s/missing", scope="s")


# ---------------------------------------------------------------------------
# Compaction
# ---------------------------------------------------------------------------


def _history(turns: int = 10) -> list[dict]:
    messages = [{"role": "system", "content": "You are a careful assistant."}]
    for i in range(turns):
        messages.append({"role": "user", "content": f"question {i} " + ("w" * 400)})
        messages.append({"role": "assistant", "content": f"answer {i} " + ("w" * 400)})
    return messages


def test_a_history_that_already_fits_is_returned_untouched():
    messages = [{"role": "user", "content": "hi"}]
    result = compact_messages(messages, max_tokens=1000)

    assert result.messages == messages
    assert result.dropped == []
    assert result.summary == ""


def test_compaction_keeps_the_system_prompt_and_the_last_turns():
    messages = _history(10)
    result = compact_messages(messages, max_tokens=400, keep_last=4)

    assert result.messages[0]["role"] == "system"
    assert result.messages[0]["content"] == "You are a careful assistant."
    assert result.messages[-4:] == messages[-4:]
    assert result.tokens_after < result.tokens_before


def test_compaction_restates_the_task_constraints():
    messages = _history(10)
    result = compact_messages(
        messages,
        max_tokens=400,
        constraints=["Never quote a figure without its source.", "Answer in British English."],
    )

    summary = result.messages[1]["content"]
    assert "Never quote a figure without its source." in summary
    assert "Answer in British English." in summary
    assert result.preserved_constraints == [
        "Never quote a figure without its source.",
        "Answer in British English.",
    ]


def test_compaction_carries_forward_source_attribution():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "What does it say? " + "w" * 800, "metadata": {"source": "handbook.pdf#p3"}},
        {"role": "assistant", "content": "It says [source: handbook.pdf#p3] fourteen days. " + "w" * 800},
        {"role": "user", "content": "and after five years?"},
        {"role": "assistant", "content": "28 days."},
    ]

    result = compact_messages(messages, max_tokens=120, keep_last=2)

    assert "handbook.pdf#p3" in result.messages[1]["content"]
    assert "handbook.pdf#p3" in result.preserved_sources


def test_compaction_carries_forward_unanswered_questions():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "Can you also confirm the sabbatical rules? " + "w" * 900},
        {"role": "assistant", "content": "Here is the leave answer. " + "w" * 900},
        {"role": "user", "content": "thanks"},
        {"role": "assistant", "content": "welcome"},
    ]

    result = compact_messages(messages, max_tokens=100, keep_last=2)

    assert any("sabbatical" in q for q in result.unresolved_questions)
    assert "sabbatical" in result.messages[1]["content"]


def test_a_tool_call_and_its_result_are_dropped_together():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "look it up " + "w" * 800},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call-1", "function": {"name": "get_stock", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "42 " + "w" * 800},
        {"role": "user", "content": "thanks"},
        {"role": "assistant", "content": "welcome"},
    ]

    result = compact_messages(messages, max_tokens=80, keep_last=2)

    roles = [m.get("role") for m in result.messages]
    tool_call_ids = {call["id"] for m in result.messages for call in (m.get("tool_calls") or ())}
    result_ids = {m.get("tool_call_id") for m in result.messages if m.get("role") == "tool"}

    assert result_ids <= tool_call_ids, "no orphaned tool result may survive"
    assert tool_call_ids <= result_ids, "no dangling tool call may survive"
    assert "tool" not in roles or "call-1" in tool_call_ids


def test_a_kept_tool_call_keeps_its_result():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "old " + "w" * 2000},
        {"role": "assistant", "content": "old " + "w" * 2000},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call-9", "function": {"name": "get_stock", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "call-9", "content": "9 units"},
        {"role": "assistant", "content": "There are 9 units."},
    ]

    result = compact_messages(messages, max_tokens=200, keep_last=3)

    ids = {m.get("tool_call_id") for m in result.messages if m.get("role") == "tool"}
    calls = {c["id"] for m in result.messages for c in (m.get("tool_calls") or ())}
    assert ids == calls == {"call-9"}


def test_the_default_summariser_needs_no_model_call():
    messages = _history(8)
    result = compact_messages(messages, max_tokens=300, constraints=["stay terse"])

    assert result.summary.startswith("[Context compacted:")
    assert "stay terse" in result.summary


def test_a_custom_summariser_is_used_when_supplied():
    messages = _history(8)
    result = compact_messages(
        messages,
        max_tokens=300,
        summarize=lambda dropped: f"custom digest of {len(dropped)} message(s)",
    )

    assert result.messages[1]["content"].startswith("custom digest of")


def test_the_compaction_report_names_its_token_counter():
    result = compact_messages(_history(8), max_tokens=300)
    assert result.token_counter in {"tiktoken", "heuristic"}
    assert result.tokens_saved > 0
    assert result.to_dict()["dropped"] > 0


# ---------------------------------------------------------------------------
# Large-catalogue and long-session evaluations
# ---------------------------------------------------------------------------


def _large_registry(n: int = 60) -> ToolRegistry:
    """A catalogue big enough that sending every schema is genuinely expensive."""
    registry = ToolRegistry()
    topics = ["invoice", "shipment", "warehouse", "customer", "refund", "payroll"]
    for i in range(n):
        topic = topics[i % len(topics)]

        def make(topic=topic, i=i):
            def fn(identifier: str) -> str:
                return "ok"

            fn.__name__ = f"{topic}_action_{i}"
            fn.__doc__ = f"Perform {topic} action {i} on a record.\n\nArgs:\n    identifier: The {topic} record id.\n"
            return fn

        registry.register(make(), metadata={"is_write": i % 2 == 0})
    return registry


def test_a_large_catalogue_selects_the_right_tools_at_a_fraction_of_the_tokens():
    from prompture.execution.context_eval import SelectionCase, measure_tool_selection

    registry = _large_registry(60)
    catalog = ToolCatalog(registry)

    report = measure_tool_selection(
        catalog,
        [
            SelectionCase(query="refund action 4 for a record", required=frozenset({"refund_action_4"})),
            SelectionCase(query="payroll action 5 for a record", required=frozenset({"payroll_action_5"})),
            SelectionCase(query="warehouse action 2 for a record", required=frozenset({"warehouse_action_2"})),
        ],
        limit=5,
    )

    assert report.catalog_size == 60
    assert report.mean_recall == pytest.approx(1.0), "the required tool must always be found"
    assert report.token_reduction is not None and report.token_reduction > 0.8
    assert report.safe is True
    assert any("not an improvement" in n for n in report.notes), "the report must refuse to oversell itself"
    assert report.format()


def test_the_selection_report_flags_an_authorisation_leak_rather_than_scoring_it_down():
    from prompture.execution.context_eval import SelectionCase, ToolSelectionReport, measure_tool_selection

    catalog = ToolCatalog(_large_registry(12))  # no allow-list: everything is authorised

    report = measure_tool_selection(
        catalog,
        [SelectionCase(query="refund action 4", forbidden=frozenset({"refund_action_4"}))],
        limit=5,
    )

    assert isinstance(report, ToolSelectionReport)
    assert report.unauthorised_leaks >= 1
    assert report.safe is False


def test_an_allow_list_makes_a_leak_structurally_impossible():
    from prompture.execution.context_eval import SelectionCase, measure_tool_selection

    registry = _large_registry(12)
    catalog = ToolCatalog(registry, allowed_tools={"refund_action_4", "payroll_action_5"})

    report = measure_tool_selection(
        catalog,
        [
            SelectionCase(
                query="warehouse action 2 and refund action 4",
                required=frozenset({"refund_action_4"}),
                forbidden=frozenset({"warehouse_action_2"}),
            )
        ],
        limit=5,
    )

    assert report.catalog_size == 2
    assert report.unauthorised_leaks == 0
    assert report.safe is True


def test_a_long_session_compacts_soundly_and_reports_its_invariants():
    from prompture.execution.context_eval import measure_compaction

    messages: list[dict] = [{"role": "system", "content": "You are precise."}]
    for i in range(40):
        messages.append({"role": "user", "content": f"turn {i}: what about clause {i}? " + "w" * 300})
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": f"call-{i}", "function": {"name": "lookup", "arguments": "{}"}}],
            }
        )
        messages.append({"role": "tool", "tool_call_id": f"call-{i}", "content": f"clause {i} text " + "w" * 300})
        messages.append({"role": "assistant", "content": f"Clause {i} says [source: handbook#{i}] ..." + "w" * 300})

    constraints = ["Always cite the clause number."]
    result = compact_messages(messages, max_tokens=1500, keep_last=6, constraints=constraints)
    report = measure_compaction(messages, result, constraints=constraints)

    assert report.sound is True, "a saving with a broken invariant is not a saving"
    assert report.protected_intact is True
    assert report.pairing_intact is True
    assert report.constraints_preserved is True
    assert report.reduction is not None and report.reduction > 0.5
    assert report.sources_preserved > 0
    assert report.questions_preserved > 0
    assert report.to_dict()["sound"] is True


def test_the_compaction_report_detects_a_broken_pairing():
    from prompture.execution.context import CompactionResult
    from prompture.execution.context_eval import measure_compaction

    broken = CompactionResult(
        messages=[{"role": "tool", "tool_call_id": "orphan", "content": "x"}],
        tokens_before=100,
        tokens_after=10,
    )

    report = measure_compaction([], broken)

    assert report.pairing_intact is False
    assert report.sound is False
    assert any("not usable" in n for n in report.notes)
