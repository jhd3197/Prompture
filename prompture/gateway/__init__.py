"""Building blocks for serving Prompture over HTTP.

Framework-agnostic: nothing here imports FastAPI, so any server (the
built-in ``prompture serve``, a standalone gateway, your own app) can
reuse the same wire-format code.
"""

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

__all__ = [
    "OPTION_FIELDS",
    "SSE_DONE",
    "SSE_HEADERS",
    "ChatOutcome",
    "arun_chat",
    "astream_chat_chunks",
    "chat_chunk",
    "chat_completion",
    "driver_options",
    "error_body",
    "extract_images",
    "finish_reason",
    "flatten_content",
    "models_list",
    "new_completion_id",
    "run_chat",
    "sse",
    "stream_chat_chunks",
    "to_driver_messages",
    "tool_calls_to_openai",
    "usage_from_meta",
]
