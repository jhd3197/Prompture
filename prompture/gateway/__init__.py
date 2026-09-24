"""Building blocks for serving Prompture over HTTP.

Speaks the OpenAI chat-completions, OpenAI Responses and Anthropic messages dialects.
Framework-agnostic: nothing here imports FastAPI, so any server (the
built-in ``prompture serve``, a standalone gateway, your own app) can
reuse the same wire-format code.
"""

from .anthropic_format import (
    anthropic_error,
    anthropic_message,
    anthropic_sse,
    anthropic_to_driver,
    anthropic_tools_to_openai,
    estimate_input_tokens,
    live_events_for,
    new_message_id,
    stream_anthropic_events,
)
from .openai_format import (
    OPTION_FIELDS,
    SSE_DONE,
    SSE_HEADERS,
    ChatOutcome,
    arun_chat,
    astream_chat_chunks,
    chat_chunk,
    chat_completion,
    driver_options,
    error_body,
    extract_images,
    finish_reason,
    flatten_content,
    models_list,
    new_completion_id,
    run_chat,
    sse,
    stream_chat_chunks,
    to_driver_messages,
    tool_calls_to_openai,
    usage_from_meta,
)
from .responses_format import (
    new_response_id,
    response_object,
    responses_sse,
    responses_to_driver,
    responses_tools_to_openai,
    stream_responses_events,
)

__all__ = [
    "OPTION_FIELDS",
    "SSE_DONE",
    "SSE_HEADERS",
    "ChatOutcome",
    "anthropic_error",
    "anthropic_message",
    "anthropic_sse",
    "anthropic_to_driver",
    "anthropic_tools_to_openai",
    "arun_chat",
    "astream_chat_chunks",
    "chat_chunk",
    "chat_completion",
    "driver_options",
    "error_body",
    "estimate_input_tokens",
    "extract_images",
    "finish_reason",
    "flatten_content",
    "live_events_for",
    "models_list",
    "new_completion_id",
    "new_message_id",
    "new_response_id",
    "response_object",
    "responses_sse",
    "responses_to_driver",
    "responses_tools_to_openai",
    "run_chat",
    "sse",
    "stream_anthropic_events",
    "stream_chat_chunks",
    "stream_responses_events",
    "to_driver_messages",
    "tool_calls_to_openai",
    "usage_from_meta",
]
