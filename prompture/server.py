"""Built-in API server wrapping AsyncConversation.

Provides a FastAPI application with chat, extraction, and model
listing endpoints.  ``fastapi``, ``uvicorn``, and ``sse-starlette``
are lazy-imported so the module is importable without them installed.

Usage::

    from prompture.server import create_app
    app = create_app(model_name="openai/gpt-4o-mini")
"""

import json
import logging
import uuid
from typing import Any, Optional

logger = logging.getLogger("prompture.server")


def create_app(
    model_name: str = "openai/gpt-4o-mini",
    system_prompt: Optional[str] = None,
    tools: Any = None,
    cors_origins: Optional[list[str]] = None,
) -> Any:
    """Create and return a FastAPI application.

    Parameters:
        model_name: Default model string (``provider/model``).
        system_prompt: Optional system prompt for new conversations.
        tools: Optional :class:`~prompture.tools_schema.ToolRegistry`.
        cors_origins: CORS allowed origins.  ``["*"]`` to allow all.

    Returns:
        A ``fastapi.FastAPI`` instance.
    """
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel, Field
    except ImportError as exc:
        raise ImportError(
            "The 'serve' extra is required: pip install prompture[serve]"
        ) from exc

    from .async_conversation import AsyncConversation
    from .tools_schema import ToolRegistry

    # ---- Pydantic request/response models ----

    class ChatRequest(BaseModel):
        message: str
        conversation_id: Optional[str] = None
        stream: bool = False
        options: Optional[dict[str, Any]] = None

    class ChatResponse(BaseModel):
        message: str
        conversation_id: str
        usage: dict[str, Any]

    class ExtractRequest(BaseModel):
        text: str
        schema_def: dict[str, Any] = Field(..., alias="schema")
        conversation_id: Optional[str] = None

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

    # ---- App ----

    app = FastAPI(title="Prompture API", version="0.1.0")

    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # In-memory conversation store
    _conversations: dict[str, AsyncConversation] = {}

    tool_registry: Optional[ToolRegistry] = tools

    def _get_or_create_conversation(conv_id: Optional[str]) -> tuple[str, AsyncConversation]:
        if conv_id and conv_id in _conversations:
            return conv_id, _conversations[conv_id]
        new_id = conv_id or uuid.uuid4().hex[:12]
        conv = AsyncConversation(
            model_name=model_name,
            system_prompt=system_prompt,
            tools=tool_registry,
        )
        _conversations[new_id] = conv
        return new_id, conv

    # ---- Endpoints ----

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

    @app.get("/v1/models", response_model=ModelInfo)
    async def list_models():
        from .discovery import get_available_models

        try:
            models = get_available_models()
            model_names = [m["id"] if isinstance(m, dict) else str(m) for m in models]
        except Exception:
            model_names = [model_name]
        return ModelInfo(models=model_names)

    return app
