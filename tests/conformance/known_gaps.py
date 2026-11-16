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
        "test_arguments_already_a_dict_are_accepted",
        "dict arguments are not accepted",
        "openai openai-async groq groq-async zai zai-async modelscope modelscope-async cachibot cachibot-async azure azure-async",
    ),
    **_gaps(
        "test_forced_tool_choice_is_translated",
        "tool_choice ignored or sent untranslated",
        "azure-claude azure-claude-async vertex-claude vertex-claude-async bedrock bedrock-async groq-async grok grok-async openrouter openrouter-async mistral mistral-async moonshot moonshot-async zai zai-async modelscope modelscope-async cachibot cachibot-async minimax minimax-async azure azure-async",
    ),
    **_gaps(
        "test_image_on_a_tool_turn_is_converted",
        "image blocks are not in Anthropic's shape",
        "azure-claude azure-claude-async bedrock bedrock-async",
    ),
    **_gaps(
        "test_malformed_arguments_are_reported",
        "malformed tool arguments silently become {}",
        "groq-async grok grok-async openrouter openrouter-async mistral mistral-async moonshot moonshot-async zai zai-async modelscope modelscope-async cachibot cachibot-async minimax minimax-async azure azure-async",
    ),
    **_gaps(
        "test_missing_tool_call_id_gets_one",
        "missing tool-call id raises or stays empty",
        "groq-async grok grok-async openrouter openrouter-async mistral mistral-async moonshot moonshot-async zai zai-async modelscope modelscope-async cachibot cachibot-async minimax minimax-async azure azure-async",
    ),
    **_gaps(
        "test_plain_answer_ends_the_turn",
        "stop reason not normalized",
        "groq-async grok grok-async openrouter openrouter-async mistral mistral-async moonshot moonshot-async zai zai-async modelscope modelscope-async cachibot cachibot-async minimax minimax-async openai_compatible openai_compatible-async azure azure-async",
    ),
    **_gaps(
        "test_plain_stream_converts_images",
        "plain stream sends images unconverted",
        "openai openai-async openrouter openrouter-async moonshot moonshot-async zai zai-async cachibot cachibot-async",
    ),
    **_gaps(
        "test_reasoning_content_is_returned",
        "reasoning_content dropped on the buffered tool path",
        "openai openai-async mistral mistral-async zai zai-async modelscope modelscope-async cachibot cachibot-async minimax minimax-async openai_compatible openai_compatible-async azure azure-async",
    ),
    **_gaps(
        "test_required_tool_choice_is_translated",
        "tool_choice dropped",
        "azure-claude azure-claude-async vertex-claude vertex-claude-async bedrock bedrock-async",
    ),
    **_gaps(
        "test_required_tool_choice_reaches_the_provider",
        "tool_choice ignored or not adapted to the provider",
        "groq-async openrouter openrouter-async mistral mistral-async cachibot cachibot-async azure azure-async",
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
        "test_stream_sends_the_same_request",
        "streamed tool request is shaped differently from the buffered one",
        "groq groq-async grok grok-async openrouter openrouter-async mistral mistral-async moonshot moonshot-async modelscope modelscope-async cachibot cachibot-async openai_compatible openai_compatible-async",
    ),
    **_gaps(
        "test_thinking_is_returned_as_reasoning",
        "thinking blocks dropped",
        "azure-claude azure-claude-async vertex-claude vertex-claude-async bedrock bedrock-async",
    ),
    **_gaps(
        "test_tool_call_is_normalized",
        "stop reason not normalized / raw_stop_reason missing",
        "claude claude-async azure-claude azure-claude-async vertex-claude vertex-claude-async bedrock bedrock-async groq-async grok grok-async openrouter openrouter-async mistral mistral-async moonshot moonshot-async zai zai-async modelscope modelscope-async cachibot cachibot-async minimax minimax-async openai_compatible openai_compatible-async azure azure-async",
    ),
    **_gaps(
        "test_tool_history_becomes_anthropic_blocks",
        "OpenAI-shaped tool history is passed to Anthropic unconverted",
        "azure-claude azure-claude-async vertex-claude vertex-claude-async bedrock bedrock-async",
    ),
    **_gaps(
        "test_top_level_one_of_schema_is_flattened",
        "raw tool schemas are not flattened at the driver boundary",
        "openai openai-async groq groq-async grok grok-async openrouter openrouter-async mistral mistral-async moonshot moonshot-async zai zai-async modelscope modelscope-async cachibot cachibot-async minimax minimax-async openai_compatible openai_compatible-async azure azure-async",
    ),
    **_gaps(
        "test_truncated_arguments_name_max_tokens",
        "truncated arguments are not reported; stop reason not normalized",
        "groq-async grok grok-async openrouter openrouter-async mistral mistral-async moonshot moonshot-async zai zai-async modelscope modelscope-async cachibot cachibot-async minimax minimax-async openai_compatible openai_compatible-async azure azure-async",
    ),
}
