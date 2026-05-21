"""Assistant + AsyncReviewLoop end-to-end example.

Demonstrates the four pieces added to support consumers like AgentSite:

1. Assistant — bundles Persona + Skills + Tools + execution backend.
2. AsyncReviewLoop — generic developer → reviewer feedback loop.
3. pick_best_coding_agent — capability-aware CLI auto-selection.
4. extract_html_document — salvage code from raw LLM output when tool
   calls didn't fire.

The script runs three demos.  By default, only the offline (mocked)
flows execute.  Set PROMPTURE_RUN_LIVE=1 to invoke real LLM/CLI calls.

Run:
    python examples/assistant_example.py
    PROMPTURE_RUN_LIVE=1 python examples/assistant_example.py
"""

from __future__ import annotations

import asyncio
import os

from prompture import (
    Assistant,
    AsyncReviewLoop,
    Persona,
    SkillInfo,
    extract_html_document,
    pick_best_coding_agent,
)

LIVE = os.getenv("PROMPTURE_RUN_LIVE") == "1"
MODEL = os.getenv("PROMPTURE_MODEL", "openai/gpt-4o-mini")


# ---------------------------------------------------------------------------
# 1. Building a web-developer Assistant with a Skill
# ---------------------------------------------------------------------------


def build_web_developer() -> Assistant:
    persona = Persona(
        name="web_developer",
        system_prompt=(
            "You are a senior web developer building {{page_type}} pages.\n"
            "Workspace: {{workspace}}.\n"
            "Return one ```html block containing the full page."
        ),
    )

    html5_skill = SkillInfo(
        name="semantic-html5",
        description="Always prefer semantic HTML5 elements.",
        instructions=(
            "- Use <header>, <main>, <section>, <article>, <footer>.\n"
            "- Avoid generic <div> wrappers when a semantic tag fits.\n"
            "- All interactive elements must be keyboard-accessible."
        ),
    )

    return Assistant(
        name="web-developer",
        persona=persona,
        skills=[html5_skill],
        model=MODEL,
        variables={"workspace": "/srv/site"},
    )


def build_code_reviewer() -> Assistant:
    return Assistant(
        name="web-reviewer",
        persona=Persona(
            name="web_reviewer",
            system_prompt=(
                "You critique HTML pages for semantics, accessibility, and "
                "polish.  End your response with a single line:\n"
                "  SCORE: <0-10>\n"
                "Score 8+ means ship it."
            ),
        ),
        model=MODEL,
    )


# ---------------------------------------------------------------------------
# 2. Driving them with a review loop
# ---------------------------------------------------------------------------


def _approves_at_8_or_above(review) -> bool:
    """Look for `SCORE: N` in the reviewer's output and approve at >= 8."""
    for line in (review.output or "").splitlines():
        line = line.strip()
        if line.upper().startswith("SCORE:"):
            try:
                score = int(line.split(":", 1)[1].strip())
            except ValueError:
                return False
            return score >= 8
    return False


async def demo_review_loop() -> None:
    if not LIVE:
        print("[skip] demo_review_loop — set PROMPTURE_RUN_LIVE=1 to run.")
        return

    coder = build_web_developer()
    reviewer = build_code_reviewer()
    loop = AsyncReviewLoop(
        coder=coder,
        reviewer=reviewer,
        max_iters=3,
        approve_when=_approves_at_8_or_above,
    )

    result = await loop.arun(
        "Build an /about page for a small coffee shop. One hero section, a 'Our Story' paragraph, and a contact link.",
        coder_kwargs={"page_type": "about"},
    )
    print(f"\n=== Review loop done — iterations={result.iterations} approved={result.approved} ===")
    extracted = extract_html_document(result.output)
    if extracted.found:
        print(f"Salvaged HTML ({len(extracted.html)} chars).")
        print(f"  styles blocks: {len(extracted.styles)}")
        print(f"  scripts blocks: {len(extracted.scripts)}")
    else:
        print("No HTML document found in final output.")


# ---------------------------------------------------------------------------
# 3. Capability-aware CLI auto-selection
# ---------------------------------------------------------------------------


def demo_capability_picker() -> None:
    chosen = pick_best_coding_agent(
        prefer=["claude", "codex"],
        require_session_resume=True,
        verify=True,
    )
    if chosen is None:
        print("[info] No installed coding-agent CLI supports session resume on this machine.")
        return
    print(f"[info] Best CLI for session resume: {chosen.id} ({chosen.binary})")


# ---------------------------------------------------------------------------
# 4. Coding-agent-backed Assistant (delegates to the CLI we picked)
# ---------------------------------------------------------------------------


async def demo_cli_assistant() -> None:
    if not LIVE:
        print("[skip] demo_cli_assistant — set PROMPTURE_RUN_LIVE=1 to run.")
        return

    cli_dev = Assistant(
        name="cli-developer",
        persona=Persona(
            name="cli_dev",
            system_prompt="You write production-quality HTML pages.",
        ),
        coding_agent="auto",  # picks the first healthy installed CLI
        coding_agent_options={"approval_mode": "auto"},
    )
    result = await cli_dev.arun("Write /about.html for a coffee shop. Return the file contents.")
    print(f"\n=== CLI assistant ran on '{result.raw.agent}' (cost=${result.cost_usd or 0:.4f}) ===")
    print(result.output[:400])


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


async def main() -> None:
    demo_capability_picker()
    await demo_review_loop()
    await demo_cli_assistant()


if __name__ == "__main__":
    asyncio.run(main())
