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
        "test_forced_tool_choice_is_translated",
        "tool_choice ignored or sent untranslated",
        "azure-claude azure-claude-async vertex-claude vertex-claude-async bedrock bedrock-async",
    ),
    **_gaps(
        "test_image_on_a_tool_turn_is_converted",
        "image blocks are not in Anthropic's shape",
        "azure-claude azure-claude-async bedrock bedrock-async",
    ),
    **_gaps(
        "test_required_tool_choice_is_translated",
        "tool_choice dropped",
        "azure-claude azure-claude-async vertex-claude vertex-claude-async bedrock bedrock-async",
    ),
    **_gaps(
        "test_stop_reasons_use_the_shared_vocabulary",
        "stop reason not normalized",
        "azure-claude azure-claude-async vertex-claude vertex-claude-async bedrock bedrock-async",
    ),
    **_gaps(
        "test_stream_reports_malformed_arguments",
        "streamed malformed arguments are not reported",
        "openai openai-async groq groq-async grok grok-async openrouter openrouter-async mistral mistral-async moonshot moonshot-async zai zai-async modelscope modelscope-async cachibot cachibot-async openai_compatible openai_compatible-async",
    ),
    **_gaps(
        "test_thinking_is_returned_as_reasoning",
        "thinking blocks dropped",
        "azure-claude azure-claude-async vertex-claude vertex-claude-async bedrock bedrock-async",
    ),
    **_gaps(
        "test_tool_call_is_normalized",
        "stop reason not normalized / raw_stop_reason missing",
        "azure-claude azure-claude-async vertex-claude vertex-claude-async bedrock bedrock-async",
    ),
    **_gaps(
        "test_tool_history_becomes_anthropic_blocks",
        "OpenAI-shaped tool history is passed to Anthropic unconverted",
        "azure-claude azure-claude-async vertex-claude vertex-claude-async bedrock bedrock-async",
    ),
}
