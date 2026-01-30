#!/usr/bin/env python3
"""Agent framework example — using Prompture's Agent for tool-augmented LLM tasks.

Demonstrates:
1. Simple agent with no tools (text Q&A)
2. Agent with tools (registration via list and @agent.tool decorator)
3. Agent with output_type (structured Pydantic output)
4. Inspecting AgentResult (steps, tool_calls, usage)

Usage:
    python examples/agent_example.py

Requires:
    - A valid API key for a supported provider (OpenAI, Ollama, etc.)
"""

from pydantic import BaseModel, Field

from prompture import Agent

MODEL = "openai/gpt-4o"

# ── Section 1: Simple agent (no tools) ─────────────────────────────────────

print("=" * 60)
print("Section 1: Simple Agent (No Tools)")
print("=" * 60)

agent = Agent(MODEL, system_prompt="You are a concise geography assistant.")
result = agent.run("What is the capital of France?")

print(f"Output: {result.output}")
print(f"State:  {result.state}")
print(f"Tokens: {result.usage.get('total_tokens', 'N/A')}")
print()

# ── Section 2: Agent with tools ────────────────────────────────────────────

print("=" * 60)
print("Section 2: Agent with Tools")
print("=" * 60)


def get_population(city: str) -> str:
    """Look up the population of a city."""
    populations = {
        "Paris": "2,161,000",
        "London": "8,982,000",
        "Tokyo": "13,960,000",
    }
    return populations.get(city, "Unknown")


def get_country(city: str) -> str:
    """Look up which country a city is in."""
    countries = {
        "Paris": "France",
        "London": "United Kingdom",
        "Tokyo": "Japan",
    }
    return countries.get(city, "Unknown")


# Register tools via list
agent_with_tools = Agent(
    MODEL,
    system_prompt="Use the available tools to answer questions about cities.",
    tools=[get_population, get_country],
)

result = agent_with_tools.run("What is the population of Tokyo and what country is it in?")

print(f"Output: {result.output}")
print(f"Tool calls made: {len(result.all_tool_calls)}")
for tc in result.all_tool_calls:
    print(f"  - {tc['name']}({tc['arguments']})")
print()

# ── Section 3: Tool registration via decorator ─────────────────────────────

print("=" * 60)
print("Section 3: Tool Registration via @agent.tool")
print("=" * 60)

math_agent = Agent(MODEL, system_prompt="You are a math assistant. Use tools for calculations.")


@math_agent.tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b


@math_agent.tool
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


result = math_agent.run("What is 7 * 8 + 3?")
print(f"Output: {result.output}")
print(f"Tool calls: {len(result.all_tool_calls)}")
print()

# ── Section 4: Structured output with output_type ──────────────────────────

print("=" * 60)
print("Section 4: Structured Output (output_type)")
print("=" * 60)


class CityInfo(BaseModel):
    name: str = Field(description="Name of the city")
    country: str = Field(description="Country the city is in")
    population: int | None = Field(default=None, description="Population estimate")
    famous_for: str = Field(description="What the city is famous for")


structured_agent = Agent(
    MODEL,
    system_prompt="You are a geography expert.",
    output_type=CityInfo,
)

result = structured_agent.run("Tell me about Paris, France.")

print(f"Type:       {type(result.output).__name__}")
print(f"Name:       {result.output.name}")
print(f"Country:    {result.output.country}")
print(f"Population: {result.output.population}")
print(f"Famous for: {result.output.famous_for}")
print()

# ── Section 5: Inspecting AgentResult ──────────────────────────────────────

print("=" * 60)
print("Section 5: Inspecting AgentResult")
print("=" * 60)

print(f"Result state: {result.state}")
print(f"Raw text:     {result.output_text[:80]}...")
print(f"Messages:     {len(result.messages)} message(s)")
print(f"Steps:        {len(result.steps)} step(s)")
for step in result.steps:
    print(f"  [{step.step_type.value}] {step.content[:60]}...")
print(f"Usage:        {result.usage}")
