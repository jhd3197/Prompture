#!/usr/bin/env python3
"""DeepAgent example — agentic research with planning, virtual filesystem,
sub-agent delegation, and automatic summarization.

DeepAgent extends :class:`Agent` with four built-in capabilities:

1. **Planning** via the ``write_todos`` tool — externalises multi-step plans.
2. **Virtual filesystem** — six tools (``read_file``, ``write_file``, ``edit_file``,
   ``ls``, ``glob``, ``grep``) backed by an in-memory ``dict[str, str]``.
3. **Sub-agents** — the ``task`` tool dispatches scoped subproblems to
   specialist agents configured via :class:`SubAgentSpec`.
4. **Summarization** — when the context window grows past a threshold,
   the older messages are collapsed into a single summary message before
   the next driver call.

Each capability is independently toggleable and shares a
:class:`DeepAgentState` that is mutated in place by built-in tools and
surfaced on :class:`DeepAgentResult`.

Usage:
    python examples/deep_agent_example.py

Requires:
    A valid API key for a supported provider (OpenAI, Anthropic, etc.).
    Set OPENAI_API_KEY (or change MODEL below) before running.
"""

from __future__ import annotations

from prompture import (
    BudgetExceededError,
    DeepAgentResult,
    Persona,
    SubAgentSpec,
    create_deep_agent,
)

MODEL = "openai/gpt-4o-mini"


# ── Section 1: Basic deep agent ───────────────────────────────────────────

print("=" * 60)
print("Section 1: Basic DeepAgent (planning + VFS, no sub-agents)")
print("=" * 60)


def web_search_stub(query: str) -> str:
    """Simulated web search — replace with a real provider in production.

    Args:
        query: The search query.
    """
    fake_results = {
        "EU AI Act foundation model deadlines": (
            "The EU AI Act published Aug 2024 sets phased deadlines for "
            "general-purpose AI models: transparency obligations apply 12 "
            "months after entry into force; full obligations after 24 months."
        ),
        "default": "(no results)",
    }
    return fake_results.get(query, fake_results["default"])


agent = create_deep_agent(
    model=MODEL,
    tools=[web_search_stub],
    enable_summarization=False,  # keep the example deterministic
    max_iterations=15,
)

result: DeepAgentResult = agent.run(
    "Briefly: when do EU AI Act obligations apply to foundation models? "
    "Use the search tool, write your finding to notes/eu_ai_act.md, "
    "then give a one-sentence summary."
)

print(f"Final answer: {result.output_text}")
print(f"Todos created: {len(result.todos)}")
for t in result.todos:
    print(f"  [{t.status}] {t.content}")
print(f"Files written: {list(result.files.keys())}")
for path, content in result.files.items():
    snippet = content[:120].replace("\n", " ")
    print(f"  {path}: {snippet}...")
print(f"Cost: ${result.run_usage.get('cost', 0):.4f}")
print()


# ── Section 2: With persona and sub-agents ───────────────────────────────

print("=" * 60)
print("Section 2: DeepAgent with Persona + Sub-agents")
print("=" * 60)


def fetch_url_stub(url: str) -> str:
    """Stub for a real URL fetcher. Replace with httpx or similar."""
    if "example.com" in url:
        return "<html><body>Example Domain</body></html>"
    return "(no content)"


research_persona = Persona(
    name="research_analyst",
    system_prompt=("You are a senior research analyst at a policy think tank. Be rigorous and cite sources by URL."),
    constraints=[
        "Never speculate without flagging the speculation.",
        "Prefer primary sources over secondary commentary.",
    ],
)

agent2 = create_deep_agent(
    model=MODEL,
    persona=research_persona,
    tools=[web_search_stub, fetch_url_stub],
    subagents=[
        SubAgentSpec(
            name="fact_checker",
            description="Verifies a single factual claim against primary sources.",
            system_prompt=(
                "You are a meticulous fact-checker. Given one claim, search for "
                "supporting evidence and return a one-line verdict: VERIFIED / "
                "UNVERIFIED / CONTRADICTED, plus the supporting URL."
            ),
        ),
    ],
    enable_summarization=False,
    max_iterations=15,
)

result2 = agent2.run(
    "Find when the EU AI Act's transparency obligations for foundation models "
    "take effect and have the fact_checker confirm the date."
)

print(f"Final answer: {result2.output_text}")
print(f"Sub-agent calls: {len(result2.sub_agent_calls)}")
for record in result2.sub_agent_calls:
    print(f"  {record.subagent_name} ({record.iterations} iter): {record.result_summary[:120]}...")
print()


# ── Section 3: With summarization on a long task ─────────────────────────

print("=" * 60)
print("Section 3: DeepAgent with summarization enabled")
print("=" * 60)

# Set a low threshold to see summarization triggered on a realistic task.
agent3 = create_deep_agent(
    model=MODEL,
    tools=[web_search_stub],
    enable_summarization=True,
    summarize_at_tokens=3_000,  # trigger early for demo
    summarize_keep_last_n=4,
    max_iterations=15,
)

try:
    result3 = agent3.run(
        "Plan a 5-step research workflow for a single topic of your choice "
        "and execute the first 3 steps, writing intermediate notes to files."
    )
    print(f"Final answer (first 200 chars): {result3.output_text[:200]}...")
    print(f"Summary events: {len(result3.summary_events)}")
    for ev in result3.summary_events:
        print(f"  collapsed {ev.summarized_message_count} messages -> {len(ev.summary_text)} char summary")
    print(f"Files: {list(result3.files.keys())}")
except BudgetExceededError as e:
    print(f"Budget exceeded: {e}")


# ── Section 4: Streaming events ──────────────────────────────────────────

print()
print("=" * 60)
print("Section 4: Streaming a DeepAgent run")
print("=" * 60)

agent4 = create_deep_agent(
    model=MODEL,
    tools=[web_search_stub],
    enable_summarization=False,
    max_iterations=8,
)

stream = agent4.run_stream("What was the headline finding in section 1? Reply in one sentence.")
for event in stream:
    if event.event_type.value == "text_delta":
        print(event.data, end="", flush=True)
    elif event.event_type.value == "tool_call":
        print(f"\n[tool_call] {event.data.get('name')}({event.data.get('arguments')})")
    elif event.event_type.value == "tool_result":
        result_preview = str(event.data.get("result", ""))[:80]
        print(f"[tool_result] {result_preview}...")
print()
final_result = stream.result
if final_result is not None:
    print(f"\nFinal usage: ${final_result.run_usage.get('cost', 0):.4f}")
