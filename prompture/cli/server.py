"""Built-in API server wrapping AsyncConversation.

Provides a FastAPI application with chat, extraction, model listing,
and an **OpenAI-compatible** ``/v1/chat/completions`` proxy endpoint.
``fastapi``, ``uvicorn``, and ``sse-starlette`` are lazy-imported so
the module is importable without them installed.

The OpenAI-compatible endpoint lets any OpenAI SDK client (Python,
Node, curl) talk to **any** Prompture-supported backend (Claude,
Gemini, Groq, Ollama, etc.) through a single unified API::

    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:9471/v1", api_key="unused")
    resp = client.chat.completions.create(
        model="claude/claude-sonnet-4-20250514",
        messages=[{"role": "user", "content": "Hello!"}],
    )

Usage::

    from prompture.server import create_app
    app = create_app(model_name="openai/gpt-4o-mini")
"""

import json
import logging
import time
import uuid
from collections import OrderedDict
from typing import Any, Optional

logger = logging.getLogger("prompture.server")


def create_app(
    model_name: str = "openai/gpt-4o-mini",
    system_prompt: Optional[str] = None,
    tools: Any = None,
    cors_origins: Optional[list[str]] = None,
    api_key: Optional[str] = None,
    max_conversations: int = 1000,
    allowed_models: Optional[list[str]] = None,
) -> Any:
    """Create and return a FastAPI application.

    Parameters:
        model_name: Default model string (``provider/model``).
        system_prompt: Optional system prompt for new conversations.
        tools: Optional :class:`~prompture.tools_schema.ToolRegistry`.
        cors_origins: CORS allowed origins.  ``["*"]`` to allow all.
        api_key: Optional Bearer token for API authentication.
            If set, all requests must include ``Authorization: Bearer <key>``.
        max_conversations: Maximum in-memory conversations before oldest
            are evicted.  Defaults to 1000.
        allowed_models: Optional allowlist of model strings.  When set,
            the OpenAI-compatible endpoint rejects models not in the list.

    Returns:
        A ``fastapi.FastAPI`` instance.
    """
    try:
        from fastapi import Depends, FastAPI, HTTPException, Request
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import JSONResponse
        from pydantic import BaseModel, Field
    except ImportError as exc:
        raise ImportError(
            "The 'serve' extra is required: pip install prompture[serve]"
        ) from exc

    from ..agents.async_conversation import AsyncConversation
    from ..agents.tools_schema import ToolRegistry

    # ---- Authentication dependency ----

    async def _verify_api_key(request: Request) -> None:
        if api_key is None:
            return
        auth = request.headers.get("Authorization", "")
        if auth != f"Bearer {api_key}":
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

    if api_key is None:
        logger.warning("Server starting without API key authentication — all requests will be accepted")

    # ---- CORS warning ----

    if cors_origins and "*" in cors_origins:
        logger.warning(
            "CORS is configured to allow all origins ('*'). "
            "This is insecure for production deployments."
        )

    # ---- Pydantic request/response models (Prompture native) ----

    class ChatRequest(BaseModel):
        message: str = Field(..., max_length=500_000)
        conversation_id: Optional[str] = Field(None, max_length=64, pattern=r"^[a-zA-Z0-9_-]+$")
        stream: bool = False
        options: Optional[dict[str, Any]] = None

    class ChatResponse(BaseModel):
        message: str
        conversation_id: str
        usage: dict[str, Any]

    class ExtractRequest(BaseModel):
        text: str = Field(..., max_length=500_000)
        schema_def: dict[str, Any] = Field(..., alias="schema")
        conversation_id: Optional[str] = Field(None, max_length=64, pattern=r"^[a-zA-Z0-9_-]+$")

        model_config = {"populate_by_name": True}

    class ExtractResponse(BaseModel):
        json_object: dict[str, Any]
        conversation_id: str
        usage: dict[str, Any]

    class ModelInfo(BaseModel):
        models: list[str]

    class ConversationHistory(BaseModel):
        conversation_id: str
        messages: list[dict[str, Any]]
        usage: dict[str, Any]

    # ---- OpenAI-compatible request/response models ----

    class OAIMessage(BaseModel):
        role: str
        content: Optional[str] = None

    class OAIChatRequest(BaseModel):
        model: Optional[str] = None
        messages: list[OAIMessage] = Field(..., max_length=200)
        temperature: Optional[float] = None
        top_p: Optional[float] = None
        max_tokens: Optional[int] = None
        stream: bool = False
        n: int = 1

    # ---- App ----

    app = FastAPI(
        title="Prompture API",
        version="0.1.0",
        dependencies=[Depends(_verify_api_key)],
    )

    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # In-memory conversation store (OrderedDict for eviction)
    _conversations: OrderedDict[str, AsyncConversation] = OrderedDict()

    tool_registry: Optional[ToolRegistry] = tools

    def _get_or_create_conversation(conv_id: Optional[str]) -> tuple[str, AsyncConversation]:
        if conv_id and conv_id in _conversations:
            # Move to end (most recently used)
            _conversations.move_to_end(conv_id)
            return conv_id, _conversations[conv_id]
        new_id = conv_id or uuid.uuid4().hex[:12]
        conv = AsyncConversation(
            model_name=model_name,
            system_prompt=system_prompt,
            tools=tool_registry,
        )
        _conversations[new_id] = conv
        # Evict oldest conversations if over the limit
        while len(_conversations) > max_conversations:
            _conversations.popitem(last=False)
        return new_id, conv

    # ---- Health endpoint ----

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    # ---- Prompture native endpoints ----

    @app.post("/v1/chat", response_model=ChatResponse)
    async def chat(chat_req: ChatRequest):
        conv_id, conv = _get_or_create_conversation(chat_req.conversation_id)

        if chat_req.stream:
            # SSE streaming
            try:
                from sse_starlette.sse import EventSourceResponse
            except ImportError:
                raise HTTPException(
                    status_code=501,
                    detail="Streaming requires sse-starlette: pip install prompture[serve]",
                ) from None

            async def event_generator():
                full_text = ""
                async for chunk in conv.ask_stream(chat_req.message, chat_req.options):
                    full_text += chunk
                    yield {"data": json.dumps({"text": chunk})}
                yield {"data": json.dumps({"text": "", "done": True, "conversation_id": conv_id, "usage": conv.usage})}

            return EventSourceResponse(event_generator())

        text = await conv.ask(chat_req.message, chat_req.options)
        return ChatResponse(message=text, conversation_id=conv_id, usage=conv.usage)

    @app.post("/v1/extract", response_model=ExtractResponse)
    async def extract(extract_req: ExtractRequest):
        conv_id, conv = _get_or_create_conversation(extract_req.conversation_id)
        result = await conv.ask_for_json(
            content=extract_req.text,
            json_schema=extract_req.schema_def,
        )
        return ExtractResponse(
            json_object=result["json_object"],
            conversation_id=conv_id,
            usage=conv.usage,
        )

    @app.get("/v1/conversations/{conversation_id}", response_model=ConversationHistory)
    async def get_conversation(conversation_id: str):
        if conversation_id not in _conversations:
            raise HTTPException(status_code=404, detail="Conversation not found")
        conv = _conversations[conversation_id]
        return ConversationHistory(
            conversation_id=conversation_id,
            messages=conv.messages,
            usage=conv.usage,
        )

    @app.delete("/v1/conversations/{conversation_id}")
    async def delete_conversation(conversation_id: str):
        if conversation_id not in _conversations:
            raise HTTPException(status_code=404, detail="Conversation not found")
        del _conversations[conversation_id]
        return {"status": "deleted", "conversation_id": conversation_id}

    # ---- OpenAI-compatible proxy endpoints ----

    @app.post("/v1/chat/completions")
    async def openai_chat_completions(req: OAIChatRequest):
        """OpenAI-compatible ``/v1/chat/completions`` proxy.

        Accepts the standard OpenAI chat format and routes the request
        through Prompture's driver system to **any** configured backend
        (Claude, Gemini, Groq, Ollama, etc.).

        The ``model`` field accepts Prompture model strings
        (``"provider/model"``).  If omitted, the server's default model
        is used.
        """
        resolved_model = req.model or model_name

        # Enforce model allowlist
        if allowed_models is not None and resolved_model not in allowed_models:
            raise HTTPException(
                status_code=403,
                detail=f"Model '{resolved_model}' is not in the allowed models list",
            )

        # Separate system prompt from messages
        sys_prompt: Optional[str] = system_prompt
        user_messages: list[dict[str, Any]] = []
        for msg in req.messages:
            if msg.role == "system":
                sys_prompt = msg.content
            else:
                user_messages.append({"role": msg.role, "content": msg.content or ""})

        # Build driver options from OpenAI params
        opts: dict[str, Any] = {}
        if req.temperature is not None:
            opts["temperature"] = req.temperature
        if req.top_p is not None:
            opts["top_p"] = req.top_p
        if req.max_tokens is not None:
            opts["max_tokens"] = req.max_tokens

        # Find the last user message to pass to AsyncConversation.ask()
        last_user_content = ""
        for msg in reversed(user_messages):
            if msg["role"] == "user":
                last_user_content = msg["content"]
                break

        if not last_user_content:
            raise HTTPException(status_code=400, detail="At least one user message is required")

        # Create a one-shot conversation for this request
        conv = AsyncConversation(
            model_name=resolved_model,
            system_prompt=sys_prompt,
            options=opts,
        )

        # Seed prior turns (everything before the final user message)
        if len(user_messages) > 1:
            for msg in user_messages[:-1]:
                conv._messages.append(msg)

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        if req.stream:
            try:
                from sse_starlette.sse import EventSourceResponse
            except ImportError:
                raise HTTPException(
                    status_code=501,
                    detail="Streaming requires sse-starlette: pip install prompture[serve]",
                ) from None

            async def oai_stream():
                full_text = ""
                async for chunk in conv.ask_stream(last_user_content, opts if opts else None):
                    full_text += chunk
                    data = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": resolved_model,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": chunk},
                            "finish_reason": None,
                        }],
                    }
                    yield {"data": json.dumps(data)}

                # Final chunk with finish_reason
                usage = conv.usage
                data = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": resolved_model,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }],
                    "usage": {
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0),
                    },
                }
                yield {"data": json.dumps(data)}
                yield {"data": "[DONE]"}

            return EventSourceResponse(oai_stream(), media_type="text/event-stream")

        # Non-streaming
        text = await conv.ask(last_user_content, opts if opts else None)
        usage = conv.usage

        return JSONResponse({
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": resolved_model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text,
                },
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
        })

    @app.get("/v1/models")
    async def list_models():
        """List available models in OpenAI-compatible format.

        Returns a response compatible with both the OpenAI SDK
        (``data`` array of model objects) and the legacy Prompture
        format (``models`` string list).
        """
        from ..infra.discovery import get_available_models

        try:
            models = get_available_models()
            model_names = [m["id"] if isinstance(m, dict) else str(m) for m in models]
        except Exception:
            model_names = [model_name]

        model_objects = []
        for name in model_names:
            model_objects.append({
                "id": name,
                "object": "model",
                "created": 0,
                "owned_by": name.split("/")[0] if "/" in name else "prompture",
            })

        return JSONResponse({
            "object": "list",
            "data": model_objects,
            # Keep legacy field for backwards compatibility
            "models": model_names,
        })

    return app
