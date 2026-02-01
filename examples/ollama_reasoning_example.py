"""
Example: Reasoning / thinking mode with Ollama (DeepSeek R1).

Demonstrates how to use a reasoning model via Ollama and access
the model's chain-of-thought through the `last_reasoning` property.

Prerequisites:
    ollama pull deepseek-r1:8b

Usage:
    python examples/ollama_reasoning_example.py
"""

import json

from prompture import Conversation

MODEL = "ollama/deepseek-r1:8b"

# ── Test 1: Simple reasoning question ────────────────────────────────────────

print("=" * 60)
print("Test 1: Reasoning with DeepSeek R1 via Ollama")
print("=" * 60)

conv = Conversation(
    MODEL,
    system_prompt="You are a helpful assistant. Think step by step.",
    options={"temperature": 0.6},
)

answer = conv.ask("What is 25 * 47? Show your work.")
print(f"\nAnswer: {answer}")
print(f"Usage:  {conv.usage_summary()}")

if conv.last_reasoning:
    print(f"\n--- Reasoning ({len(conv.last_reasoning)} chars) ---")
    # Show first 500 chars of reasoning
    preview = conv.last_reasoning[:500]
    if len(conv.last_reasoning) > 500:
        preview += "..."
    print(preview)
else:
    print("\n(No reasoning content returned)")

print()

# ── Test 2: JSON extraction with reasoning ───────────────────────────────────

print("=" * 60)
print("Test 2: Structured extraction with reasoning")
print("=" * 60)

text = (
    "The Eiffel Tower is 330 metres tall and was completed in 1889. "
    "It is located in Paris, France, on the Champ de Mars. "
    "The tower was designed by Gustave Eiffel's engineering company."
)

conv2 = Conversation(
    MODEL,
    system_prompt=(
        "Extract facts from the text into JSON with keys: "
        "name, height_m, year_completed, city, country, designer. "
        "Return ONLY valid JSON."
    ),
    options={"temperature": 0.0, "json_mode": True},
)

result = conv2.ask(text)
print(f"\nExtracted: {result}")

if conv2.last_reasoning:
    print(f"\n--- Reasoning ({len(conv2.last_reasoning)} chars) ---")
    preview = conv2.last_reasoning[:500]
    if len(conv2.last_reasoning) > 500:
        preview += "..."
    print(preview)
else:
    print("\n(No reasoning content returned)")

print(f"\nUsage: {conv2.usage_summary()}")
print()

# ── Test 3: ask_for_json with reasoning ──────────────────────────────────────

print("=" * 60)
print("Test 3: ask_for_json with last_reasoning")
print("=" * 60)

schema = {
    "type": "object",
    "properties": {
        "answer": {"type": "integer"},
        "explanation": {"type": "string"},
    },
    "required": ["answer", "explanation"],
}

conv3 = Conversation(
    MODEL,
    system_prompt="You are a math tutor. Solve the problem and explain your reasoning.",
    options={"temperature": 0.0},
)

result3 = conv3.ask_for_json(
    "If a train travels at 80 km/h for 2.5 hours, how far does it go?",
    json_schema=schema,
)

print(f"\nJSON result: {json.dumps(result3.get('json_object'), indent=2)}")
print(f"Reasoning:   {'present' if result3.get('reasoning') else 'none'}")

if result3.get("reasoning"):
    preview = result3["reasoning"][:500]
    if len(result3["reasoning"]) > 500:
        preview += "..."
    print(f"\n--- Reasoning ---\n{preview}")

print(f"\nUsage: {conv3.usage_summary()}")
print()

# ── Summary ──────────────────────────────────────────────────────────────────

print("=" * 60)
print("All tests completed.")
print("=" * 60)
