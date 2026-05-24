#!/usr/bin/env python3
"""Live agent stream — interleaved tool calling with text deltas.

This is the "Claude Code feel": the model narrates as it decides what to
do, fires a tool, narrates again, fires another tool, and finally answers
— all streamed live instead of buffered.

Compare with ``agent_example.py`` Section 10 (``run_stream``): that one
yields ``tool_call`` / ``tool_result`` events between rounds but the text
of each turn arrives in one chunk *after* the tools execute. ``run_live``
streams text deltas as the model produces them, including *before* and
*between* tool calls within a single turn.

Usage::

    OPENAI_API_KEY=sk-... python examples/agent_live_stream.py
    # or
    CLAUDE_API_KEY=sk-ant-... python examples/agent_live_stream.py --model claude/claude-haiku-4-5-20251001
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from prompture import Agent
from prompture.agents.live_events import (
    AssistantTurnStart,
    MessageStop,
    TextDelta,
    ThinkingDelta,
    ToolInputDelta,
    ToolResult,
    ToolUseStart,
    ToolUseStop,
    TurnComplete,
)

CITY_DATA: dict[str, dict[str, Any]] = {
    "Paris": {"country": "France", "population": 2_161_000, "timezone": "CET"},
    "Tokyo": {"country": "Japan", "population": 13_960_000, "timezone": "JST"},
    "London": {"country": "United Kingdom", "population": 8_982_000, "timezone": "GMT"},
    "São Paulo": {"country": "Brazil", "population": 12_396_000, "timezone": "BRT"},
}


def lookup_country(city: str) -> str:
    """Return the country a city is in."""
    info = CITY_DATA.get(city)
    return info["country"] if info else f"Unknown city: {city}"


def lookup_population(city: str) -> str:
    """Return the population of a city."""
    info = CITY_DATA.get(city)
    return str(info["population"]) if info else f"Unknown city: {city}"


def compare_populations(city_a: str, city_b: str) -> str:
    """Return which city has the larger population."""
    a, b = CITY_DATA.get(city_a), CITY_DATA.get(city_b)
    if not a or not b:
        return "Unknown city in comparison."
    if a["population"] > b["population"]:
        return f"{city_a} is larger ({a['population']:,} vs {b['population']:,})."
    if a["population"] < b["population"]:
        return f"{city_b} is larger ({b['population']:,} vs {a['population']:,})."
    return f"{city_a} and {city_b} have equal populations."


def render_event(event: Any) -> None:
    """Pretty-print one LiveEvent so the stream is visible in a terminal."""
    if isinstance(event, AssistantTurnStart):
        print(f"\n--- turn {event.turn_index} ---")
    elif isinstance(event, TextDelta):
        print(event.text, end="", flush=True)
    elif isinstance(event, ThinkingDelta):
        print(f"\033[2m{event.text}\033[0m", end="", flush=True)
    elif isinstance(event, ToolUseStart):
        print(f"\n[> calling {event.name} ({event.id[:8]})]", flush=True)
    elif isinstance(event, ToolInputDelta):
        # Show input-JSON arriving piece by piece.
        print(f"\033[2m{event.fragment}\033[0m", end="", flush=True)
    elif isinstance(event, ToolUseStop):
        print(f"\n[> {event.name} input ready: {event.input}]", flush=True)
    elif isinstance(event, ToolResult):
        out = event.output if len(event.output) < 200 else event.output[:200] + "…"
        prefix = "!" if event.is_error else "<"
        print(f"\n[{prefix} {event.name} -> {out}]", flush=True)
    elif isinstance(event, MessageStop):
        cost = event.usage.get("cost", 0.0)
        tokens = event.usage.get("total_tokens", 0)
        print(f"\n[turn done: {tokens} tokens, ${cost:.5f}, stop={event.stop_reason}]")
    elif isinstance(event, TurnComplete):
        cost = event.usage.get("cost", 0.0)
        tokens = event.usage.get("total_tokens", 0)
        print(f"\n[all done: {tokens} tokens, ${cost:.5f}]")


def main() -> int:
    parser = argparse.ArgumentParser(description="Live agent stream demo.")
    parser.add_argument("--model", default="openai/gpt-4o-mini", help="provider/model string")
    parser.add_argument(
        "--prompt",
        default=(
            "Compare Tokyo and Paris. Tell me which is bigger and what country each is in. "
            "Use your tools to verify the numbers."
        ),
    )
    args = parser.parse_args()

    agent = Agent(
        args.model,
        system_prompt=(
            "You are a careful researcher. Use tools to verify facts before answering. "
            "Narrate briefly what you are about to do before each tool call."
        ),
        tools=[lookup_country, lookup_population, compare_populations],
    )

    streamed = agent.run_live(args.prompt)
    for event in streamed:
        render_event(event)

    final = streamed.result
    if final is not None:
        print(f"\n\nFinal answer:\n{final.output_text}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
