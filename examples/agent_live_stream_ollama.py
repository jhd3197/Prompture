#!/usr/bin/env python3
"""Live agent stream against a local Ollama model.

Prompture's ``run_live`` API gives you Claude-Code-style interleaved
streaming (text → tool call → text → tool call → final answer) on a
**local Ollama model** — no proprietary cloud needed. This example
demonstrates both delivery tiers:

- **Tier 1 (default)** — native Ollama streaming + tool calls. Works
  on tool-trained models (Llama 3.1+, Mistral Nemo, Qwen 2.5, ...).
  Token deltas stream live; tool calls land at the end of the model's
  turn and are emitted as ``ToolUseStart`` / ``ToolUseStop`` events.

- **Tier 2** — prompted-tool emulation. The tool schemas are injected
  into the system prompt, and Prompture parses tool calls out of the
  token stream character-by-character. Works on **any** local model,
  even ones without function-calling training (Phi-3, base Gemma 2,
  raw Llama 3 7B, ...).

Usage::

    # Tier 1 (model has native tool support, e.g. Llama 3.1)
    python examples/agent_live_stream_ollama.py --model ollama/llama3.1:8b

    # Tier 2 (any model — forces prompted-tool emulation)
    python examples/agent_live_stream_ollama.py \\
        --model ollama/phi3:mini --prompted

Requires a local Ollama server (``ollama serve``) with the chosen model
pulled (``ollama pull llama3.1:8b``).
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
        print(f"\033[2m{event.fragment}\033[0m", end="", flush=True)
    elif isinstance(event, ToolUseStop):
        print(f"\n[> {event.name} input ready: {event.input}]", flush=True)
    elif isinstance(event, ToolResult):
        out = event.output if len(event.output) < 200 else event.output[:200] + "…"
        prefix = "!" if event.is_error else "<"
        print(f"\n[{prefix} {event.name} -> {out}]", flush=True)
    elif isinstance(event, MessageStop):
        tokens = event.usage.get("total_tokens", 0)
        print(f"\n[turn done: {tokens} tokens, stop={event.stop_reason}]")
    elif isinstance(event, TurnComplete):
        tokens = event.usage.get("total_tokens", 0)
        print(f"\n[all done: {tokens} tokens]")


def main() -> int:
    parser = argparse.ArgumentParser(description="Live agent stream against a local Ollama model.")
    parser.add_argument(
        "--model",
        default="ollama/llama3.1:8b",
        help="provider/model string (e.g. ollama/llama3.1:8b, ollama/phi3:mini)",
    )
    parser.add_argument(
        "--prompted",
        action="store_true",
        help=(
            "Force Tier 2 prompted-tool emulation (tool schemas injected into "
            "system prompt, parsed from token stream). Use for models without "
            "native function-calling training."
        ),
    )
    parser.add_argument(
        "--prompt",
        default=(
            "Compare Tokyo and Paris. Tell me which is bigger and what country each is in. "
            "Use your tools to verify the numbers."
        ),
    )
    args = parser.parse_args()

    tier = "Tier 2 (prompted emulation)" if args.prompted else "Tier 1 (native)"
    print(f"Model:  {args.model}")
    print(f"Mode:   {tier}")
    print(f"Prompt: {args.prompt}\n")

    agent = Agent(
        args.model,
        system_prompt=(
            "You are a careful researcher. Use tools to verify facts before answering. "
            "Narrate briefly what you are about to do before each tool call."
        ),
        tools=[lookup_country, lookup_population, compare_populations],
    )

    options: dict[str, Any] = {}
    if args.prompted:
        options["prompted_tools"] = True

    streamed = agent.run_live(args.prompt, options=options)
    for event in streamed:
        render_event(event)

    final = streamed.result
    if final is not None:
        print(f"\n\nFinal answer:\n{final.output_text}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
