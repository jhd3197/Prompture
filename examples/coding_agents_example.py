"""
Example: Discovering and running local coding-agent CLIs.

This script demonstrates:
1. Detecting Claude Code, Codex CLI, and Gemini CLI from PATH.
2. Previewing the command Prompture would run.
3. Optionally running Codex/Claude/Gemini with Prompture's wrapper.

Run:
    python examples/coding_agents_example.py

To actually launch an agent, opt in explicitly:
    PROMPTURE_RUN_CODING_AGENT=1 python examples/coding_agents_example.py
"""

import os
import shlex

from prompture import build_coding_agent_command, get_available_coding_agents, run_coding_agent

TASK = "Summarize the repository structure and suggest one low-risk cleanup."
PREFERRED_AGENT = os.getenv("PROMPTURE_CODING_AGENT", "codex")


print("Detected coding agents:")
agents = get_available_coding_agents()
for agent in agents:
    status = "available" if agent.available else "missing"
    print(f"- {agent.id}: {status} ({agent.binary})")

available = {agent.id for agent in agents if agent.available}
if PREFERRED_AGENT not in available:
    print(f"\n{PREFERRED_AGENT!r} is not available here. Set PROMPTURE_CODING_AGENT=claude or gemini.")
    raise SystemExit(0)

print("\nCommand preview with auto-approve mode:")
command = build_coding_agent_command(
    PREFERRED_AGENT,
    TASK,
    approval_mode="auto",
)
print(shlex.join(command.argv))

if os.getenv("PROMPTURE_RUN_CODING_AGENT") != "1":
    print("\nSet PROMPTURE_RUN_CODING_AGENT=1 to run it.")
    raise SystemExit(0)

result = run_coding_agent(
    PREFERRED_AGENT,
    TASK,
    approval_mode="auto",
    timeout=600,
)

print("\nAgent output:")
print(result.output)
print("\nRun metadata:")
print(f"Agent: {result.agent}")
print(f"Exit code: {result.returncode}")
print(f"Duration: {result.duration_seconds:.2f}s")
