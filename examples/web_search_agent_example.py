#!/usr/bin/env python3
"""Web search tool — let an LLM look things up online.

Demonstrates:
1. Direct ``WebSearchTool`` use (no LLM).
2. Auto-detect the configured backend from environment variables.
3. ``DeepAgent`` with web search to research a topic and cite sources.

Run::

    pip install prompture
    export TAVILY_API_KEY=tvly-...           # or SERPER_API_KEY / BRAVE_SEARCH_API_KEY
    python examples/web_search_agent_example.py

Requires:
    - One configured backend: Tavily (recommended), Serper, Brave, or SearXNG.
    - A configured LLM provider (defaults to openai/gpt-4o-mini).
"""

from __future__ import annotations

import os

from prompture import (
    SubAgentSpec,
    ToolRegistry,
    WebSearchTool,
    create_deep_agent,
)

MODEL = os.environ.get("PROMPTURE_EXAMPLE_MODEL", "openai/gpt-4o-mini")


# ── Section 1: Direct tool use ─────────────────────────────────────────────

print("=" * 60)
print("Section 1: Direct WebSearchTool use")
print("=" * 60)

tool = WebSearchTool()  # auto-detects from env
print(f"Using provider: {tool.provider}")
print(tool("latest LangChain release notes"))


# ── Section 2: DeepAgent with web search ───────────────────────────────────

print("\n" + "=" * 60)
print("Section 2: DeepAgent research with citations")
print("=" * 60)

registry = ToolRegistry()
WebSearchTool().register_on(registry)

agent = create_deep_agent(
    model=MODEL,
    tools=registry,
    subagents=[
        SubAgentSpec(
            name="fact_checker",
            description="Verifies factual claims against primary sources.",
            system_prompt=("You are a rigorous fact-checker. Use web_search to confirm or refute claims, citing URLs."),
        ),
    ],
)

result = agent.run(
    "Research the EU AI Act's compliance deadlines for foundation models. Cite each fact with a URL inline."
)
print(result.output_text)
