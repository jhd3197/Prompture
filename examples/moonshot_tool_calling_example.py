"""
Example: Tool calling with Moonshot AI (Kimi K2.5).

Tests both native and simulated tool calling to verify
the model correctly calls tools and returns final answers.

Setup:
    export MOONSHOT_API_KEY="your-key-here"
    # or add to .env file

Usage:
    python examples/moonshot_tool_calling_example.py
"""

import json

from prompture import Conversation, ToolRegistry

MODEL = "moonshot/kimi-k2.5"

# ── Define tools ─────────────────────────────────────────────────────────────

tools = ToolRegistry()


@tools.tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    data = {
        "London": {"temp": 14, "condition": "cloudy"},
        "Tokyo": {"temp": 28, "condition": "sunny"},
        "New York": {"temp": 22, "condition": "partly cloudy"},
    }
    info = data.get(city, {"temp": 20, "condition": "unknown"})
    return json.dumps(info)


@tools.tool
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


# ── Test 1: Native tool calling ──────────────────────────────────────────────

print("=" * 60)
print("Test 1: Native Tool Calling (kimi-k2.5)")
print("=" * 60)

conv = Conversation(
    MODEL,
    system_prompt="You are a concise assistant. Answer in one sentence.",
    tools=tools,
    max_tool_rounds=5,
)

result = conv.ask("What is the weather in Tokyo?")
print(f"Answer: {result}")
print(f"Usage:  {conv.usage_summary()}")
print(f"Turns:  {conv.usage['turns']}")
print()

# Check message history for tool calls
for i, msg in enumerate(conv.messages):
    role = msg["role"]
    content = str(msg.get("content", ""))[:80]
    has_tc = "tool_calls" in msg
    print(f"  [{i}] {role}: {content}{'  [+tool_calls]' if has_tc else ''}")
print()

# ── Test 2: Simple math tool ─────────────────────────────────────────────────

print("=" * 60)
print("Test 2: Math Tool")
print("=" * 60)

conv2 = Conversation(
    MODEL,
    system_prompt="You are a concise math assistant. Use the add tool. Answer in one sentence.",
    tools=tools,
    max_tool_rounds=5,
)

result2 = conv2.ask("What is 137 + 455?")
print(f"Answer: {result2}")
print(f"Usage:  {conv2.usage_summary()}")
print()

# ── Test 3: Simulated tool calling (forced) ──────────────────────────────────

print("=" * 60)
print("Test 3: Simulated Tool Calling (forced)")
print("=" * 60)

conv3 = Conversation(
    MODEL,
    system_prompt="You are a concise assistant. Answer in one sentence.",
    tools=tools,
    simulated_tools=True,
    max_tool_rounds=5,
)

result3 = conv3.ask("What is the weather in London?")
print(f"Answer: {result3}")
print(f"Usage:  {conv3.usage_summary()}")
print()

# Verify history uses only user/assistant roles
roles = [msg["role"] for msg in conv3.messages]
print(f"  History roles: {roles}")
assert all(r in ("user", "assistant") for r in roles), "Unexpected roles in simulated history!"
print("  All messages use user/assistant roles (simulated mode OK)")
print()

# ── Test 4: No-tool question (should skip tools) ────────────────────────────

print("=" * 60)
print("Test 4: Direct Answer (no tool needed)")
print("=" * 60)

conv4 = Conversation(
    MODEL,
    system_prompt="You are a concise assistant. Answer in one sentence.",
    tools=tools,
    max_tool_rounds=5,
)

result4 = conv4.ask("What is the capital of France?")
print(f"Answer: {result4}")
print(f"Turns:  {conv4.usage['turns']}")
print()

# ── Test 5: Reasoning content ─────────────────────────────────────────────────

print("=" * 60)
print("Test 5: Reasoning Content (last_reasoning)")
print("=" * 60)

conv5 = Conversation(
    MODEL,
    system_prompt="You are a concise assistant. Think step by step.",
    tools=tools,
    max_tool_rounds=5,
)

result5 = conv5.ask("What is the weather in Tokyo? Explain your reasoning.")
print(f"Answer:    {result5}")
print(f"Reasoning: {conv5.last_reasoning}")
if conv5.last_reasoning:
    print(f"  (reasoning length: {len(conv5.last_reasoning)} chars)")
else:
    print("  (no reasoning_content returned — model may not be a reasoning model)")
print()

# ── Summary ──────────────────────────────────────────────────────────────────

print("=" * 60)
print("All tests completed.")
print("=" * 60)
