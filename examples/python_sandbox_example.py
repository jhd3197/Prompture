#!/usr/bin/env python3
"""Sandboxed Python execution tool — let an LLM run code safely.

Demonstrates:
1. Building a ``PythonSandboxTool`` and registering it with a ToolRegistry.
2. Running an Agent that calls ``python_execute`` to solve a math problem.
3. Wiring the agent's ``on_approval_needed`` callback to ``mark_approved``
   so HIGH-risk code can proceed after user consent.
4. Granting the sandbox access to a specific directory.

Run::

    pip install prompture[sandbox]
    python examples/python_sandbox_example.py

Requires:
    - prompture[sandbox] extra (pulls in tukuy)
    - A configured LLM provider (defaults to openai/gpt-4o-mini)
"""

from __future__ import annotations

import os
import tempfile

from prompture import (
    Agent,
    AgentCallbacks,
    PythonSandboxTool,
    ToolRegistry,
)

MODEL = os.environ.get("PROMPTURE_EXAMPLE_MODEL", "openai/gpt-4o-mini")


# ── Section 1: Direct tool use ─────────────────────────────────────────────

print("=" * 60)
print("Section 1: Direct tool use (no LLM)")
print("=" * 60)

sandbox = PythonSandboxTool()
print(sandbox("import math; print(math.factorial(7))"))
print(sandbox("nums = [1, 4, 9, 16, 25]; print(sum(nums) / len(nums))"))


# ── Section 2: Agent solves math via the sandbox ───────────────────────────

print("\n" + "=" * 60)
print("Section 2: Agent + sandbox tool")
print("=" * 60)

registry = ToolRegistry()
PythonSandboxTool().register_on(registry)

agent = Agent(
    MODEL,
    system_prompt=(
        "You are a careful math assistant. When a question needs a "
        "computation, call python_execute. Show the final answer briefly."
    ),
    tools=registry,
)

result = agent.run("Compute the standard deviation of [12, 17, 19, 23, 29, 31]. Round to two decimal places.")
print("Agent answer:", result.output)


# ── Section 3: HIGH-risk approval flow ─────────────────────────────────────

print("\n" + "=" * 60)
print("Section 3: Approval-gated risky code")
print("=" * 60)

risky_sandbox = PythonSandboxTool()
risky_registry = ToolRegistry()
risky_sandbox.register_on(risky_registry)


def on_approval(tool_name: str, action: str, details: dict) -> bool:
    """Auto-approve in this example.  In production, prompt the user."""
    print(f"  [approval] {tool_name} wants to {action}")
    print(f"  [approval] reasons: {details.get('reasons')}")
    risky_sandbox.mark_approved(details["code"])
    return True


risky_agent = Agent(
    MODEL,
    system_prompt="Use python_execute when asked to inspect the OS.",
    tools=risky_registry,
    callbacks=AgentCallbacks(on_approval_needed=on_approval),
)

# Note: even after approval bypasses the AST gate, Tukuy still blocks
# ``os`` at runtime — defense in depth.  The agent will see the error
# and explain it cannot fulfil the request.
risky_result = risky_agent.run("Print the value of the HOME environment variable using Python.")
print("Agent answer:", risky_result.output)


# ── Section 4: Sandbox with read access to a specific directory ────────────

print("\n" + "=" * 60)
print("Section 4: Scoped filesystem access")
print("=" * 60)

with tempfile.TemporaryDirectory() as workspace:
    data_path = os.path.join(workspace, "scores.csv")
    with open(data_path, "w", encoding="utf-8") as f:
        f.write("name,score\nAlice,87\nBob,93\nCarol,72\n")

    fs_sandbox = PythonSandboxTool(
        allowed_read_paths=[workspace],
        auto_approve=True,  # treat the curated env as trusted
    )
    fs_registry = ToolRegistry()
    fs_sandbox.register_on(fs_registry)

    fs_agent = Agent(
        MODEL,
        system_prompt=(
            "You are a data analyst.  Use python_execute to read CSV "
            "files from the workspace and answer questions briefly."
        ),
        tools=fs_registry,
    )
    fs_result = fs_agent.run(f"Read {data_path!r} and tell me who has the highest score.")
    print("Agent answer:", fs_result.output)
