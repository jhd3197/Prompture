"""prompture.aio — Async public API for Prompture.

Usage::

    from prompture.aio import extract_with_model, AsyncDriver

All functions mirror their synchronous counterparts in ``prompture.core``
but are ``async def`` and must be ``await``-ed.
"""

from ..async_conversation import AsyncConversation
from ..async_core import (
    ask_for_json,
    clean_json_text_with_ai,
    extract_and_jsonify,
    extract_from_data,
    extract_from_pandas,
    extract_with_model,
    gather_extract,
    manual_extract_and_jsonify,
    render_output,
    stepwise_extract_with_model,
)
from ..async_driver import AsyncDriver
from ..drivers import (
    AsyncAirLLMDriver,
    AsyncAzureDriver,
    AsyncClaudeDriver,
    AsyncGoogleDriver,
    AsyncGrokDriver,
    AsyncGroqDriver,
    AsyncHuggingFaceDriver,
    AsyncLMStudioDriver,
    AsyncLocalHTTPDriver,
    AsyncOllamaDriver,
    AsyncOpenAIDriver,
    AsyncOpenRouterDriver,
    get_async_driver,
    get_async_driver_for_model,
)

__all__ = [
    # Async driver classes
    "AsyncAirLLMDriver",
    "AsyncAzureDriver",
    "AsyncClaudeDriver",
    # Async conversation
    "AsyncConversation",
    # Async base class
    "AsyncDriver",
    "AsyncGoogleDriver",
    "AsyncGrokDriver",
    "AsyncGroqDriver",
    "AsyncHuggingFaceDriver",
    "AsyncLMStudioDriver",
    "AsyncLocalHTTPDriver",
    "AsyncOllamaDriver",
    "AsyncOpenAIDriver",
    "AsyncOpenRouterDriver",
    # Async core functions
    "ask_for_json",
    "clean_json_text_with_ai",
    "extract_and_jsonify",
    "extract_from_data",
    "extract_from_pandas",
    "extract_with_model",
    "gather_extract",
    # Async driver factories
    "get_async_driver",
    "get_async_driver_for_model",
    "manual_extract_and_jsonify",
    "render_output",
    "stepwise_extract_with_model",
]
