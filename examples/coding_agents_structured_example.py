"""Structured coding-agent output: events, cost, questions, session continuity.

Demonstrates the full feature set built on top of run_coding_agent /
astream_coding_agent:

1. JSON event stream parsing (Claude or Codex)
2. Live event streaming for UI updates
3. Cost + token capture via UsageSession
4. Clarifying-question detection
5. Session continuity (resume an answered question)

Run:
    python examples/coding_agents_structured_example.py

Opt in to actually launching the agent (otherwise everything but the live
calls is shown):
    PROMPTURE_RUN_CODING_AGENT=1 python examples/coding_agents_structured_example.py
"""

from __future__ import annotations

import asyncio
import os

from prompture import (
    UsageSession,
    astream_coding_agent,
    get_available_coding_agents,
    run_coding_agent,
)

PREFERRED_AGENT = os.getenv("PROMPTURE_CODING_AGENT", "claude")
TASK_1 = "Summarise pyproject.toml in two bullet points."
TASK_2 = "Now also list the optional dependency extras you noticed."


def _check_available() -> bool:
    agents = {a.id for a in get_available_coding_agents(verify=True) if a.available}
    if PREFERRED_AGENT not in agents:
        print(f"{PREFERRED_AGENT!r} is not installed. Available: {sorted(agents)}")
        return False
    if os.getenv("PROMPTURE_RUN_CODING_AGENT") != "1":
        print("Set PROMPTURE_RUN_CODING_AGENT=1 to actually invoke the agent.")
        return False
    return True


def demo_structured_output() -> str | None:
    """Run a single task in JSON mode; show events, cost, and any question."""
    session = UsageSession()
    result = run_coding_agent(
        PREFERRED_AGENT,
        TASK_1,
        approval_mode="auto",
        output_format="json",
        session=session,
    )

    print(f"\n=== {PREFERRED_AGENT} (one-shot, json mode) ===")
    print(f"exit={result.returncode}  duration={result.duration_seconds:.2f}s")
    if result.cost_usd is not None:
        print(f"cost=${result.cost_usd:.4f}  in={result.input_tokens}  out={result.output_tokens}")
    print(f"session_id={result.session_id}")

    for ev in result.events:
        if ev.type == "message":
            print(f"  [message] {ev.text}")
        elif ev.type == "tool_call":
            print(f"  [tool]    {ev.tool_name}({ev.tool_input})")
        elif ev.type == "question":
            print(f"  [QUESTION] {ev.text}")
            if ev.choices:
                for i, c in enumerate(ev.choices, 1):
                    print(f"       {i}. {c}")

    if q := result.asked_question:
        print(f"\nAgent asked a clarifying question:\n  {q.text}")

    print(f"\nSession summary: {session.summary()['formatted']}")
    return result.session_id


async def demo_streaming() -> None:
    """Live event stream — useful for UIs that want progress as it happens."""
    print(f"\n=== {PREFERRED_AGENT} (streaming) ===")
    async for ev in astream_coding_agent(
        PREFERRED_AGENT,
        TASK_1,
        approval_mode="auto",
    ):
        if ev.type == "message":
            print(f"  [stream] {ev.text}")
        elif ev.type == "tool_call":
            print(f"  [stream] →  {ev.tool_name}")
        elif ev.type == "done":
            cost = f"${ev.cost_usd:.4f}" if ev.cost_usd is not None else "n/a"
            print(f"  [stream] done  cost={cost}")


def demo_session_continuity(session_id: str | None) -> None:
    """Resume the previous turn to ask a follow-up question."""
    if not session_id:
        print("\nNo session_id captured — skipping continuity demo.")
        return
    print(f"\n=== {PREFERRED_AGENT} (resuming session {session_id}) ===")
    result = run_coding_agent(
        PREFERRED_AGENT,
        TASK_2,
        approval_mode="auto",
        output_format="json",
        session_id=session_id,
    )
    print(f"exit={result.returncode}  duration={result.duration_seconds:.2f}s")
    print(result.output[:500])


def main() -> None:
    if not _check_available():
        return

    session_id = demo_structured_output()
    asyncio.run(demo_streaming())
    demo_session_continuity(session_id)


if __name__ == "__main__":
    main()
