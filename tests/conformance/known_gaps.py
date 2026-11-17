"""Scenarios a driver is known to fail, as ``(target id, test name)``.

Each entry is an ``xfail(strict=True)``: the suite stays green while the gap
is open, and turns red the moment it is fixed so the entry gets deleted. An
empty table is the goal.
"""

from __future__ import annotations


def _gaps(test: str, reason: str, targets: str) -> dict[tuple[str, str], str]:
    return {(target, test): reason for target in targets.split()}


KNOWN_GAPS: dict[tuple[str, str], str] = {
    **_gaps(
        "test_stream_reports_malformed_arguments",
        "streamed malformed arguments are not reported",
        "openai openai-async groq groq-async grok grok-async openrouter openrouter-async mistral mistral-async moonshot moonshot-async zai zai-async modelscope modelscope-async cachibot cachibot-async openai_compatible openai_compatible-async",
    ),
}
