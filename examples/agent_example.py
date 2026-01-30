#!/usr/bin/env python3
"""Agent framework example — using Prompture's Agent for tool-augmented LLM tasks.

Demonstrates:
1. Simple agent with no tools (text Q&A)
2. Agent with tools (registration via list and @agent.tool decorator)
3. Agent with output_type (structured Pydantic output)
4. Inspecting AgentResult (steps, tool_calls, usage)
5. Agent with RunContext-aware tools and deps
6. Agent with input/output guardrails
7. Agent with AgentCallbacks for observability
8. Inspecting per-run usage via result.run_usage

Usage:
    python examples/agent_example.py

Requires:
    - A valid API key for a supported provider (OpenAI, Ollama, etc.)
"""

from dataclasses import dataclass

from pydantic import BaseModel, Field

from prompture import Agent, AgentCallbacks, GuardrailError, ModelRetry, RunContext

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
print()

# ── Section 6: Agent with RunContext and deps ──────────────────────────────

print("=" * 60)
print("Section 6: RunContext-Aware Tools with deps")
print("=" * 60)


@dataclass
class AppDeps:
    """Application dependencies passed into every tool via RunContext."""

    user_name: str
    locale: str = "en"


def personalized_greeting(ctx: RunContext[AppDeps], topic: str) -> str:
    """Generate a personalized greeting for a topic."""
    return f"Hello {ctx.deps.user_name}! Here's info about {topic} (locale: {ctx.deps.locale})"


# Dynamic system prompt using RunContext
def dynamic_system_prompt(ctx: RunContext[AppDeps]) -> str:
    return f"You are an assistant for {ctx.deps.user_name}. Respond in locale: {ctx.deps.locale}."


deps_agent = Agent(
    MODEL,
    system_prompt=dynamic_system_prompt,
    tools=[personalized_greeting],
)

result = deps_agent.run(
    "Greet me about Python programming",
    deps=AppDeps(user_name="Alice", locale="en-US"),
)

print(f"Output: {result.output}")
print(f"Run usage: {result.run_usage.get('formatted', 'N/A')}")
print()

# ── Section 7: Agent with Guardrails ──────────────────────────────────────

print("=" * 60)
print("Section 7: Input and Output Guardrails")
print("=" * 60)


# Input guardrail: reject dangerous prompts
def safety_guardrail(ctx: RunContext, prompt: str) -> str | None:
    """Block prompts containing 'hack'."""
    if "hack" in prompt.lower():
        raise GuardrailError("Prompt contains blocked content")
    return None  # pass through unchanged


# Input guardrail: add prefix
def prefix_guardrail(ctx: RunContext, prompt: str) -> str:
    """Add a safety prefix to every prompt."""
    return f"[SAFE MODE] {prompt}"


# Output guardrail: ensure response isn't empty
def non_empty_guardrail(ctx: RunContext, result) -> None:
    """Reject empty responses."""
    if not result.output_text.strip():
        raise ModelRetry("Response was empty, please provide a substantive answer")
    return None


guarded_agent = Agent(
    MODEL,
    system_prompt="You are a helpful assistant.",
    input_guardrails=[safety_guardrail, prefix_guardrail],
    output_guardrails=[non_empty_guardrail],
)

result = guarded_agent.run("Tell me about Python")
print(f"Output: {result.output[:100]}...")

# Test rejection
try:
    guarded_agent.run("How to hack a server")
except GuardrailError as e:
    print(f"Blocked: {e.message}")
print()

# ── Section 8: Agent with Callbacks ───────────────────────────────────────

print("=" * 60)
print("Section 8: AgentCallbacks for Observability")
print("=" * 60)

callback_log: list[str] = []

callbacks = AgentCallbacks(
    on_iteration=lambda i: callback_log.append(f"iteration:{i}"),
    on_step=lambda s: callback_log.append(f"step:{s.step_type.value}"),
    on_output=lambda r: callback_log.append(f"output:{r.output_text[:30]}"),
)

observed_agent = Agent(
    MODEL,
    system_prompt="Be concise.",
    agent_callbacks=callbacks,
)

result = observed_agent.run("What is 2 + 2?")

print(f"Output: {result.output}")
print(f"Callback log: {callback_log}")
print(f"Run usage: {result.run_usage}")
