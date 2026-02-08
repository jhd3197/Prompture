#!/usr/bin/env python3
"""Async Agent framework example — using Prompture's AsyncAgent.

Demonstrates:
1. AsyncAgent basic usage
2. AsyncAgent with tools
3. AsyncAgent with run_stream()

Usage:
    python examples/async_agent_example.py

Requires:
    - A valid API key for a supported provider (OpenAI, Ollama, etc.)
"""

import asyncio

from prompture import AsyncAgent, StreamEventType

MODEL = "openai/gpt-4o"


async def main():
    # ── Section 1: Basic AsyncAgent ─────────────────────────────────────
    print("=" * 60)
    print("Section 1: Basic AsyncAgent")
    print("=" * 60)

    agent = AsyncAgent(MODEL, system_prompt="You are a concise assistant.")
    result = await agent.run("What is the capital of France?")

    print(f"Output: {result.output}")
    print(f"State:  {result.state}")
    print()

    # ── Section 2: AsyncAgent with tools ────────────────────────────────
    print("=" * 60)
    print("Section 2: AsyncAgent with Tools")
    print("=" * 60)

    def get_population(city: str) -> str:
        """Look up the population of a city."""
        populations = {
            "Paris": "2,161,000",
            "London": "8,982,000",
            "Tokyo": "13,960,000",
        }
        return populations.get(city, "Unknown")

    agent_with_tools = AsyncAgent(
        MODEL,
        system_prompt="Use tools to answer questions.",
        tools=[get_population],
    )

    result = await agent_with_tools.run("What is the population of Tokyo?")
    print(f"Output: {result.output}")
    print(f"Tool calls: {len(result.all_tool_calls)}")
    print()

    # ── Section 3: AsyncAgent with streaming ────────────────────────────
    print("=" * 60)
    print("Section 3: AsyncAgent with run_stream()")
    print("=" * 60)

    stream_agent = AsyncAgent(MODEL, system_prompt="Be concise.")
    stream = stream_agent.run_stream("Tell me a short joke.")

    print("Streaming:")
    async for event in stream:
        if event.event_type == StreamEventType.text_delta:
            print(event.data, end="", flush=True)
        elif event.event_type == StreamEventType.output:
            print()
            print(f"  [Done, state: {event.data.state}]")

    print(f"Result: {stream.result.output[:60] if stream.result else 'N/A'}...")


if __name__ == "__main__":
    asyncio.run(main())
