"""OpenAI-compatible API server wrapping Prompture's driver system.

Exposes a FastAPI application implementing the subset of OpenAI's API
that drop-in clients (Claude Code, Codex, Cursor, Aider, LangChain,
OpenAI SDK, …) need:

* ``POST /v1/chat/completions`` — streaming + non-streaming, with
  optional server-side tool execution (sandbox / web search).
* ``POST /v1/completions`` — legacy completions, thin wrapper around chat.
* ``GET  /v1/models`` — list models discovered by Prompture.
* ``GET  /v1/coding-agents`` — list local coding-agent CLIs.
* ``POST /v1/coding-agents/run`` — run a local coding-agent CLI; SSE-streams
  normalised events when ``stream=true``.
* ``POST /v1/embeddings`` — routes to ``get_embedding_driver_for_model``.
* ``POST /v1/images/generations`` — image generation.
* ``POST /v1/audio/speech`` / ``/v1/audio/transcriptions`` — TTS / STT.

Media generation (with async-job support):

* ``POST /v1/videos/generations`` — text/image-to-video.
* ``POST /v1/lipsync`` — image|video + audio → video.
* ``POST /v1/music`` — music generation (create / remix / extend / mashup).
* ``GET  /v1/jobs/{id}`` — poll an async media job (submit with ``poll=false`` → 202).

Plus Prompture-native endpoints:

* ``POST /v1/chat`` — message + conversation_id, multi-turn state.
* ``POST /v1/extract`` — structured JSON extraction.
* CRUD for in-memory conversations under ``/v1/conversations/...``.

``fastapi``, ``uvicorn``, and ``sse-starlette`` are lazy-imported so the
module is importable without them installed.

Single-worker constraint
~~~~~~~~~~~~~~~~~~~~~~~~

``_conversations`` and the rate-limit token buckets live in
**per-process memory**. Running with ``uvicorn --workers N`` where
``N > 1`` will partition conversations across workers: a request that
creates conversation ``abc`` on worker 1 followed by a turn that the
load-balancer routes to worker 2 returns 404. **Run with
``--workers 1`` (the default) until conversation state is moved to a
shared backend (Redis / Postgres / etc.).** A multi-worker-safe
backend is on the roadmap.

Example::

    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:9471/v1", api_key="anything")
    resp = client.chat.completions.create(
        model="claude/claude-sonnet-4-6",
        messages=[{"role": "user", "content": "Hello!"}],
    )
"""

import json
import logging
import time
import uuid
from collections import OrderedDict
from typing import Any

from .._internal.json_encoder import PromptureJSONEncoder

logger = logging.getLogger("prompture.server")


def create_app(
    model_name: str = "openai/gpt-4o-mini",
    system_prompt: str | None = None,
    tools: Any = None,
    cors_origins: list[str] | None = None,
    api_key: str | None = None,
    max_conversations: int = 1000,
    allowed_models: list[str] | None = None,
    rate_limit: int | None = None,
) -> Any:
    """Create and return a FastAPI application.

    Parameters:
        model_name: Default model string (``provider/model``).
        system_prompt: Optional system prompt injected when the client
            doesn't supply one in ``messages``.
        tools: Optional :class:`~prompture.tools_schema.ToolRegistry`.
            When set, the chat endpoint runs the agent loop and executes
            tools server-side — clients see only the final answer.
        cors_origins: CORS allowed origins.  ``["*"]`` to allow all.
        api_key: Optional Bearer token for API authentication.  When
            set, all requests must include ``Authorization: Bearer <key>``.
        max_conversations: Maximum in-memory conversations before oldest
            are evicted.  Defaults to 1000.
        allowed_models: Optional allowlist of model strings.  When set,
            requests for models outside the list return 403.
        rate_limit: Maximum requests per minute per client IP.  ``None``
            (default) disables rate limiting.

    Note:
        The returned app keeps conversations and rate-limit buckets in
        per-process memory. Running multiple uvicorn workers will
        partition state across processes — same ``conversation_id``
        can hit a worker that doesn't know about it. Use
        ``--workers 1`` until shared-state backends land.
    """
    try:
        from fastapi import Depends, FastAPI, HTTPException, Request
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import JSONResponse
        from pydantic import BaseModel, Field
    except ImportError as exc:
        raise ImportError("The 'serve' extra is required: pip install prompture[serve]") from exc

    from ..agents.async_conversation import AsyncConversation
    from ..agents.tools_schema import ToolRegistry

    # ---- Authentication dependency ----

    async def _verify_api_key(request: Request) -> None:
        if api_key is None:
            return
        # /health is always public
        if request.url.path == "/health":
            return
        auth = request.headers.get("Authorization", "")
        if auth != f"Bearer {api_key}":
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

    if api_key is None:
        logger.warning("Server starting without API key authentication — set --api-key to require Bearer auth")

    logger.info(
        "Server uses in-process state for conversations and rate-limit buckets — "
        "run with --workers 1 (the default) or conversation_id lookups will return 404 "
        "when load-balanced across processes."
    )

    # ---- Rate limiting dependency ----

    _rate_buckets: dict[str, list[float]] = {}

    async def _check_rate_limit(request: Request) -> None:
        if rate_limit is None:
            return
        client_ip = request.client.host if request.client else "unknown"
        now = time.monotonic()
        window = 60.0  # 1 minute

        bucket = _rate_buckets.setdefault(client_ip, [])
        _rate_buckets[client_ip] = bucket = [t for t in bucket if now - t < window]
        if len(bucket) >= rate_limit:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        bucket.append(now)

    if cors_origins and "*" in cors_origins:
        logger.warning("CORS allows all origins ('*') — restrict to specific hosts in production.")

    # ---- Pydantic request/response models (Prompture native) ----

    class ChatRequest(BaseModel):
        message: str = Field(..., max_length=500_000)
        conversation_id: str | None = Field(None, max_length=64, pattern=r"^[a-zA-Z0-9_-]+$")
        stream: bool = False
        options: dict[str, Any] | None = None

    class ChatResponse(BaseModel):
        message: str
        conversation_id: str
        usage: dict[str, Any]

    class ExtractRequest(BaseModel):
        text: str = Field(..., max_length=500_000)
        schema_def: dict[str, Any] = Field(..., alias="schema")
        conversation_id: str | None = Field(None, max_length=64, pattern=r"^[a-zA-Z0-9_-]+$")
        model_config = {"populate_by_name": True}

    class ExtractResponse(BaseModel):
        json_object: dict[str, Any]
        conversation_id: str
        usage: dict[str, Any]

    class ConversationHistory(BaseModel):
        conversation_id: str
        messages: list[dict[str, Any]]
        usage: dict[str, Any]

    # ---- OpenAI-compatible request/response models ----

    class OAIMessage(BaseModel):
        role: str
        content: Any | None = None  # str or list (multipart) — kept loose
        name: str | None = None
        tool_call_id: str | None = None
        tool_calls: list[dict[str, Any]] | None = None

    class OAIChatRequest(BaseModel):
        model: str | None = None
        messages: list[OAIMessage] = Field(..., max_length=500)
        temperature: float | None = None
        top_p: float | None = None
        max_tokens: int | None = None
        stream: bool = False
        n: int = 1
        tools: list[dict[str, Any]] | None = None
        tool_choice: Any | None = None
        stop: Any | None = None
        presence_penalty: float | None = None
        frequency_penalty: float | None = None
        user: str | None = None

    class OAICompletionsRequest(BaseModel):
        model: str | None = None
        prompt: Any  # str or list
        temperature: float | None = None
        top_p: float | None = None
        max_tokens: int | None = None
        stream: bool = False
        stop: Any | None = None

    class OAIEmbeddingsRequest(BaseModel):
        model: str | None = None
        input: Any  # str or list[str]
        encoding_format: str | None = None
        user: str | None = None

    class OAIImageGenRequest(BaseModel):
        model: str | None = None
        prompt: str
        n: int = 1
        size: str | None = "1024x1024"
        quality: str | None = "standard"
        style: str | None = None
        response_format: str | None = "url"  # "url" | "b64_json"
        user: str | None = None

    class OAISpeechRequest(BaseModel):
        model: str | None = None
        input: str = Field(..., max_length=4096)
        voice: str | None = None
        response_format: str | None = None  # mp3, opus, wav, ...
        speed: float | None = None

    class OAIVideoGenRequest(BaseModel):
        model: str | None = None
        prompt: str = ""
        image_url: str | None = None  # start frame → image-to-video
        duration: int | None = None
        aspect_ratio: str | None = None
        resolution: str | None = None
        poll: bool = True  # poll=false → async job (HTTP 202)

    class OAILipsyncRequest(BaseModel):
        model: str | None = None
        audio_url: str
        image_url: str | None = None  # portrait → image-based lipsync
        video_url: str | None = None  # source clip → video-based lipsync
        prompt: str | None = None
        resolution: str | None = None
        poll: bool = True

    class OAIMusicRequest(BaseModel):
        model: str | None = None
        prompt: str = ""
        instrumental: bool | None = None
        operation: str | None = None  # create | remix | extend | mashup
        audio_url: str | None = None  # source for remix/extend
        duration: int | None = None
        style: str | None = None
        poll: bool = True

    # ---- App ----

    app = FastAPI(
        title="Prompture API",
        version="0.2.0",
        dependencies=[Depends(_verify_api_key), Depends(_check_rate_limit)],
    )

    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # In-memory conversation store
    _conversations: OrderedDict[str, AsyncConversation] = OrderedDict()

    # In-memory async media job store (job_id -> serialized JobHandle record)
    _media_jobs: OrderedDict[str, dict[str, Any]] = OrderedDict()
    _MEDIA_JOBS_MAX = 1000

    tool_registry: ToolRegistry | None = tools

    # ---- Helpers ----

    def _check_model_allowed(model: str) -> None:
        if allowed_models is not None and model not in allowed_models:
            raise HTTPException(
                status_code=403,
                detail=f"Model '{model}' is not in the allowed models list",
            )

    def _store_media_job(handle: Any) -> str:
        """Persist an in-flight media JobHandle and return its server job id."""
        job_id = "job_" + uuid.uuid4().hex[:16]
        _media_jobs[job_id] = {
            "handle": handle.to_dict(),
            "created": int(time.time()),
            "model": f"{handle.provider}/{handle.model}",
            "modality": handle.modality,
        }
        while len(_media_jobs) > _MEDIA_JOBS_MAX:
            _media_jobs.popitem(last=False)
        return job_id

    def _media_result_or_job(result: dict[str, Any], key: str, model: str) -> Any:
        """Return a 200 result (assets) or a 202 job envelope for a pending submit."""
        meta = result.get("meta", {})
        jh = meta.get("job_handle")
        if jh:
            from ..jobs import JobHandle

            job_id = _store_media_job(JobHandle.from_dict(jh))
            return JSONResponse(
                {"id": job_id, "object": "media.job", "status": "pending", "model": meta.get("model_name", model)},
                status_code=202,
            )
        items = result.get(key, []) or []
        data = [{"url": getattr(c, "url", None)} for c in items if getattr(c, "url", None)]
        return JSONResponse(
            {
                "created": int(time.time()),
                "data": data,
                "model": meta.get("model_name", model),
                "cost": meta.get("cost", 0.0),
            }
        )

    def _get_or_create_conversation(conv_id: str | None) -> tuple[str, AsyncConversation]:
        if conv_id and conv_id in _conversations:
            _conversations.move_to_end(conv_id)
            return conv_id, _conversations[conv_id]
        new_id = conv_id or uuid.uuid4().hex[:12]
        conv = AsyncConversation(
            model_name=model_name,
            system_prompt=system_prompt,
            tools=tool_registry,
        )
        _conversations[new_id] = conv
        while len(_conversations) > max_conversations:
            _conversations.popitem(last=False)
        return new_id, conv

    def _seed_conversation_from_messages(
        conv: AsyncConversation,
        messages: list[OAIMessage],
    ) -> tuple[str, list[Any]]:
        """Replay OpenAI ``messages[]`` into *conv*.

        Returns ``(last_user_content, last_user_images)``.  The
        conversation's internal message list is populated for everything
        EXCEPT the final user message, which is returned so the caller
        can pass it (with any extracted images) to ``conv.ask()``.
        """
        # Find the index of the final user message.
        final_user_idx: int | None = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].role == "user":
                final_user_idx = i
                break
        if final_user_idx is None:
            raise HTTPException(status_code=400, detail="At least one user message is required")

        final_msg = messages[final_user_idx]
        last_user_content = _flatten_content(final_msg.content)
        last_user_images = _extract_images(final_msg.content)
        last_system: str | None = None

        for i, msg in enumerate(messages):
            if i == final_user_idx:
                continue
            if msg.role == "system":
                # Override the conversation's system prompt.
                last_system = _flatten_content(msg.content)
                continue
            entry: dict[str, Any] = {"role": msg.role, "content": _flatten_content(msg.content)}
            if msg.tool_calls:
                entry["tool_calls"] = msg.tool_calls
            if msg.tool_call_id:
                entry["tool_call_id"] = msg.tool_call_id
            if msg.name:
                entry["name"] = msg.name
            conv._messages.append(entry)

        if last_system is not None:
            conv._system_prompt = last_system

        return last_user_content, last_user_images

    def _flatten_content(content: Any) -> str:
        """Reduce OpenAI multipart content arrays to a single text string.

        Drops image_url parts here — they're extracted separately by
        :func:`_extract_images` and forwarded via ``options["images"]``.
        """
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text" and isinstance(part.get("text"), str):
                        parts.append(part["text"])
                elif isinstance(part, str):
                    parts.append(part)
            return "\n".join(parts)
        return str(content)

    def _extract_images(content: Any) -> list[Any]:
        """Pull image_url parts out of an OpenAI multipart message.

        Returns a list of Prompture :class:`ImageInput`-compatible
        values (URL strings or data URIs).  Vision-capable drivers
        accept these directly via ``options["images"]``.
        """
        if not isinstance(content, list):
            return []
        images: list[Any] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") != "image_url":
                continue
            url_field = part.get("image_url")
            url = url_field.get("url") if isinstance(url_field, dict) else url_field
            if isinstance(url, str) and url:
                images.append(url)
        return images

    def _build_options(req: OAIChatRequest) -> dict[str, Any]:
        opts: dict[str, Any] = {}
        if req.temperature is not None:
            opts["temperature"] = req.temperature
        if req.top_p is not None:
            opts["top_p"] = req.top_p
        if req.max_tokens is not None:
            opts["max_tokens"] = req.max_tokens
        if req.stop is not None:
            opts["stop"] = req.stop
        if req.presence_penalty is not None:
            opts["presence_penalty"] = req.presence_penalty
        if req.frequency_penalty is not None:
            opts["frequency_penalty"] = req.frequency_penalty
        return opts

    # ---- Health endpoint ----

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    # ---- Prompture-native endpoints (unchanged from v0.1) ----

    @app.post("/v1/chat", response_model=ChatResponse)
    async def chat(chat_req: ChatRequest) -> Any:
        conv_id, conv = _get_or_create_conversation(chat_req.conversation_id)
        if chat_req.stream:
            try:
                from sse_starlette.sse import EventSourceResponse
            except ImportError:
                raise HTTPException(
                    status_code=501,
                    detail="Streaming requires sse-starlette: pip install prompture[serve]",
                ) from None

            async def event_generator() -> Any:
                async for chunk in conv.ask_stream(chat_req.message, chat_req.options):
                    yield {"data": json.dumps({"text": chunk})}
                yield {"data": json.dumps({"text": "", "done": True, "conversation_id": conv_id, "usage": conv.usage})}

            return EventSourceResponse(event_generator())

        text = await conv.ask(chat_req.message, chat_req.options)
        return ChatResponse(message=text, conversation_id=conv_id, usage=conv.usage)

    @app.post("/v1/extract", response_model=ExtractResponse)
    async def extract(extract_req: ExtractRequest) -> ExtractResponse:
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
    async def get_conversation(conversation_id: str) -> ConversationHistory:
        if conversation_id not in _conversations:
            raise HTTPException(status_code=404, detail="Conversation not found")
        conv = _conversations[conversation_id]
        return ConversationHistory(
            conversation_id=conversation_id,
            messages=conv.messages,
            usage=conv.usage,
        )

    @app.delete("/v1/conversations/{conversation_id}")
    async def delete_conversation(conversation_id: str) -> dict[str, str]:
        if conversation_id not in _conversations:
            raise HTTPException(status_code=404, detail="Conversation not found")
        del _conversations[conversation_id]
        return {"status": "deleted", "conversation_id": conversation_id}

    # ---- OpenAI-compatible: /v1/chat/completions ----

    @app.post("/v1/chat/completions")
    async def openai_chat_completions(req: OAIChatRequest) -> Any:
        """OpenAI Chat Completions — routes to any Prompture-supported backend.

        Streaming and non-streaming are supported.  When the server has
        a configured ``tools`` registry (e.g. via ``--sandbox`` /
        ``--web-search``), tool calls execute on the server and the
        client only sees the final assistant message.

        Client-supplied ``tools[]`` in the request body are forwarded
        to the underlying driver where the driver supports it; tool
        calls returned by the model are surfaced in the response so the
        client can execute them locally and continue the conversation.
        """
        resolved_model = req.model or model_name
        _check_model_allowed(resolved_model)

        opts = _build_options(req)
        # Forward client-supplied tools to the driver (best-effort —
        # driver decides how to use them).  Note: when server-side
        # tools are also configured, both sets coexist; the agent
        # registry handles its own tools, the driver gets the rest.
        if req.tools:
            opts["tools"] = req.tools
        if req.tool_choice is not None:
            opts["tool_choice"] = req.tool_choice

        conv = AsyncConversation(
            model_name=resolved_model,
            system_prompt=system_prompt,
            options=opts,
            tools=tool_registry,
        )
        last_user_content, last_user_images = _seed_conversation_from_messages(conv, req.messages)

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

            async def oai_stream() -> Any:
                # First chunk includes role.
                first_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": resolved_model,
                    "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                }
                yield {"data": json.dumps(first_chunk)}

                async for chunk in conv.ask_stream(
                    last_user_content,
                    opts if opts else None,
                    images=last_user_images or None,
                ):
                    data = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": resolved_model,
                        "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}],
                    }
                    yield {"data": json.dumps(data)}

                usage = conv.usage
                final = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": resolved_model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0),
                    },
                }
                yield {"data": json.dumps(final)}
                yield {"data": "[DONE]"}

            return EventSourceResponse(oai_stream(), media_type="text/event-stream")

        # Non-streaming
        text = await conv.ask(
            last_user_content,
            opts if opts else None,
            images=last_user_images or None,
        )
        usage = conv.usage

        # Surface tool_calls from the last assistant message if the
        # model emitted them without server-side execution.
        assistant_tool_calls: list[dict[str, Any]] | None = None
        finish_reason = "stop"
        if conv._messages:
            last_msg = conv._messages[-1]
            if last_msg.get("role") == "assistant" and last_msg.get("tool_calls"):
                assistant_tool_calls = last_msg["tool_calls"]
                finish_reason = "tool_calls"

        message: dict[str, Any] = {"role": "assistant", "content": text}
        if assistant_tool_calls:
            message["tool_calls"] = assistant_tool_calls

        return JSONResponse(
            {
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": resolved_model,
                "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
                "usage": {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                },
            }
        )

    # ---- OpenAI-compatible: /v1/completions (legacy) ----

    @app.post("/v1/completions")
    async def openai_completions(req: OAICompletionsRequest) -> Any:
        """Legacy completions — thin wrapper around chat."""
        resolved_model = req.model or model_name
        _check_model_allowed(resolved_model)

        prompt_text = req.prompt if isinstance(req.prompt, str) else "\n".join(req.prompt)

        opts: dict[str, Any] = {}
        if req.temperature is not None:
            opts["temperature"] = req.temperature
        if req.top_p is not None:
            opts["top_p"] = req.top_p
        if req.max_tokens is not None:
            opts["max_tokens"] = req.max_tokens
        if req.stop is not None:
            opts["stop"] = req.stop

        conv = AsyncConversation(
            model_name=resolved_model,
            system_prompt=system_prompt,
            options=opts,
        )

        completion_id = f"cmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        if req.stream:
            try:
                from sse_starlette.sse import EventSourceResponse
            except ImportError:
                raise HTTPException(
                    status_code=501,
                    detail="Streaming requires sse-starlette: pip install prompture[serve]",
                ) from None

            async def oai_stream() -> Any:
                async for chunk in conv.ask_stream(prompt_text, opts if opts else None):
                    data = {
                        "id": completion_id,
                        "object": "text_completion",
                        "created": created,
                        "model": resolved_model,
                        "choices": [{"index": 0, "text": chunk, "finish_reason": None, "logprobs": None}],
                    }
                    yield {"data": json.dumps(data)}
                yield {"data": "[DONE]"}

            return EventSourceResponse(oai_stream(), media_type="text/event-stream")

        text = await conv.ask(prompt_text, opts if opts else None)
        usage = conv.usage
        return JSONResponse(
            {
                "id": completion_id,
                "object": "text_completion",
                "created": created,
                "model": resolved_model,
                "choices": [{"index": 0, "text": text, "finish_reason": "stop", "logprobs": None}],
                "usage": {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                },
            }
        )

    # ---- OpenAI-compatible: /v1/embeddings ----

    @app.post("/v1/embeddings")
    async def openai_embeddings(req: OAIEmbeddingsRequest) -> Any:
        """Embeddings — routes to ``get_async_embedding_driver_for_model``."""
        resolved_model = req.model or model_name
        _check_model_allowed(resolved_model)

        from ..drivers.embedding_registry import get_async_embedding_driver_for_model

        try:
            driver = get_async_embedding_driver_for_model(resolved_model)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Unknown embedding model: {exc}") from exc

        inputs = req.input if isinstance(req.input, list) else [req.input]
        if not all(isinstance(x, str) for x in inputs):
            raise HTTPException(status_code=400, detail="input must be string or list[str]")

        result = await driver.embed(inputs, {})
        meta = result.get("meta", {})

        return JSONResponse(
            {
                "object": "list",
                "data": [
                    {"object": "embedding", "embedding": vec, "index": i} for i, vec in enumerate(result["embeddings"])
                ],
                "model": meta.get("model_name", resolved_model),
                "usage": {
                    "prompt_tokens": meta.get("total_tokens", 0),
                    "total_tokens": meta.get("total_tokens", 0),
                },
            }
        )

    # ---- OpenAI-compatible: /v1/images/generations ----

    @app.post("/v1/images/generations")
    async def openai_images_generations(req: OAIImageGenRequest) -> Any:
        """Image generation — routes to ``get_async_img_gen_driver_for_model``."""
        resolved_model = req.model or model_name
        _check_model_allowed(resolved_model)

        from ..drivers.img_gen_registry import get_async_img_gen_driver_for_model

        try:
            driver = get_async_img_gen_driver_for_model(resolved_model)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Unknown image model: {exc}") from exc

        opts: dict[str, Any] = {"n": req.n}
        if req.size:
            opts["size"] = req.size
        if req.quality:
            opts["quality"] = req.quality
        if req.style:
            opts["style"] = req.style

        result = await driver.generate_image(req.prompt, opts)
        images = result.get("images", [])
        meta = result.get("meta", {})
        want_b64 = req.response_format == "b64_json"

        data: list[dict[str, Any]] = []
        for img in images:
            entry: dict[str, Any] = {}
            url = getattr(img, "url", None)
            b64 = getattr(img, "data", None)
            if want_b64 and b64:
                entry["b64_json"] = b64
            elif url:
                entry["url"] = url
            elif b64:
                # No URL available — fall back to a data URI so the
                # client still receives something usable.
                media_type = getattr(img, "media_type", "image/png")
                entry["url"] = f"data:{media_type};base64,{b64}"
            if meta.get("revised_prompt"):
                entry["revised_prompt"] = meta["revised_prompt"]
            data.append(entry)

        return JSONResponse(
            {
                "created": int(time.time()),
                "data": data,
                "model": meta.get("model_name", resolved_model),
            }
        )

    # ---- OpenAI-compatible: /v1/audio/speech (TTS) ----

    @app.post("/v1/audio/speech")
    async def openai_audio_speech(req: OAISpeechRequest) -> Any:
        """Text-to-speech — routes to ``get_async_tts_driver_for_model``."""
        from fastapi.responses import Response

        resolved_model = req.model or model_name
        _check_model_allowed(resolved_model)

        from ..drivers.audio_registry import get_async_tts_driver_for_model

        try:
            driver = get_async_tts_driver_for_model(resolved_model)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Unknown TTS model: {exc}") from exc

        opts: dict[str, Any] = {}
        if req.voice:
            opts["voice"] = req.voice
        if req.response_format:
            opts["format"] = req.response_format
        if req.speed is not None:
            opts["speed"] = req.speed

        result = await driver.synthesize(req.input, opts)
        audio_bytes = result.get("audio", b"")
        media_type = result.get("media_type", "audio/mpeg")

        if not isinstance(audio_bytes, (bytes, bytearray)):
            raise HTTPException(status_code=502, detail="Driver returned non-bytes audio payload")

        return Response(content=audio_bytes, media_type=media_type)

    # ---- OpenAI-compatible: /v1/audio/transcriptions (STT) ----

    from fastapi import File, Form, UploadFile

    @app.post("/v1/audio/transcriptions")
    async def openai_audio_transcriptions(
        file: UploadFile = File(...),  # noqa: B008 — FastAPI dependency idiom
        model: str = Form(default=""),
        language: str | None = Form(default=None),
        prompt: str | None = Form(default=None),
        response_format: str | None = Form(default=None),
    ) -> Any:
        """Speech-to-text — routes to ``get_async_stt_driver_for_model``.

        Accepts ``multipart/form-data`` with the standard OpenAI fields:
        ``file`` (audio binary), ``model``, ``language``, ``prompt``,
        ``response_format`` (``"json"`` or ``"text"``).  Requires
        ``python-multipart`` (pulled in by ``prompture[serve]``).
        """
        resolved_model = model or model_name
        _check_model_allowed(resolved_model)

        from ..drivers.audio_registry import get_async_stt_driver_for_model

        try:
            driver = get_async_stt_driver_for_model(resolved_model)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Unknown STT model: {exc}") from exc

        audio_bytes = await file.read()
        opts: dict[str, Any] = {}
        if language:
            opts["language"] = language
        if prompt:
            opts["prompt"] = prompt

        result = await driver.transcribe(audio_bytes, opts)
        if response_format == "text":
            from fastapi.responses import PlainTextResponse

            return PlainTextResponse(result.get("text", ""))
        return JSONResponse(
            {
                "text": result.get("text", ""),
                "language": result.get("language"),
                "segments": result.get("segments", []),
            }
        )

    # ---- Media generation: video / lipsync (+ async jobs) ----

    @app.post("/v1/videos/generations")
    async def media_videos_generations(req: OAIVideoGenRequest) -> Any:
        """Video generation (text/image-to-video).

        With ``poll=true`` (default) blocks until the clip is ready and returns
        ``{data: [{url}], cost}``. With ``poll=false`` submits the job and returns
        HTTP 202 ``{id, status: "pending"}`` — poll ``GET /v1/jobs/{id}``.
        """
        resolved_model = req.model or model_name
        _check_model_allowed(resolved_model)

        from ..drivers.video_gen_registry import get_async_video_gen_driver_for_model

        try:
            driver = get_async_video_gen_driver_for_model(resolved_model)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Unknown video model: {exc}") from exc

        opts: dict[str, Any] = {"poll": req.poll}
        if req.image_url:
            opts["image_url"] = req.image_url
        if req.duration is not None:
            opts["duration"] = req.duration
        if req.aspect_ratio:
            opts["aspect_ratio"] = req.aspect_ratio
        if req.resolution:
            opts["resolution"] = req.resolution

        result = await driver.generate_video(req.prompt, opts)
        return _media_result_or_job(result, "videos", resolved_model)

    @app.post("/v1/lipsync")
    async def media_lipsync(req: OAILipsyncRequest) -> Any:
        """Lipsync (image|video + audio → video). ``poll=false`` returns a 202 job."""
        resolved_model = req.model or model_name
        _check_model_allowed(resolved_model)

        from ..drivers.lipsync_registry import get_async_lipsync_driver_for_model

        try:
            driver = get_async_lipsync_driver_for_model(resolved_model)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Unknown lipsync model: {exc}") from exc

        opts: dict[str, Any] = {"poll": req.poll, "audio_url": req.audio_url}
        if req.image_url:
            opts["image_url"] = req.image_url
        if req.video_url:
            opts["video_url"] = req.video_url
        if req.prompt:
            opts["prompt"] = req.prompt
        if req.resolution:
            opts["resolution"] = req.resolution

        result = await driver.generate_lipsync(None, opts)
        return _media_result_or_job(result, "videos", resolved_model)

    @app.post("/v1/music")
    async def media_music(req: OAIMusicRequest) -> Any:
        """Music generation (create / remix / extend / mashup). ``poll=false`` → 202 job."""
        resolved_model = req.model or model_name
        _check_model_allowed(resolved_model)

        from ..drivers.music_registry import get_async_music_driver_for_model

        try:
            driver = get_async_music_driver_for_model(resolved_model)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Unknown music model: {exc}") from exc

        opts: dict[str, Any] = {"poll": req.poll}
        if req.instrumental is not None:
            opts["instrumental"] = req.instrumental
        if req.operation:
            opts["operation"] = req.operation
        if req.audio_url:
            opts["audio_url"] = req.audio_url
        if req.duration is not None:
            opts["duration"] = req.duration
        if req.style:
            opts["style"] = req.style

        result = await driver.generate_music(req.prompt, opts)
        return _media_result_or_job(result, "audio", resolved_model)

    @app.get("/v1/jobs/{job_id}")
    async def media_job_status(job_id: str) -> Any:
        """Fetch the current state of an async media job submitted with ``poll=false``."""
        rec = _media_jobs.get(job_id)
        if rec is None:
            raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}")

        from starlette.concurrency import run_in_threadpool

        from ..jobs import JobHandle, resume

        handle = JobHandle.from_dict(rec["handle"])
        try:
            # resume() is sync (blocking httpx) — run off the event loop.
            result = await run_in_threadpool(resume, handle, poll=False)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Job fetch failed: {exc}") from exc

        rec["status"] = result.status
        return JSONResponse(
            {
                "id": job_id,
                "object": "media.job",
                "status": result.status,
                "done": result.done,
                "model": rec["model"],
                "created": rec["created"],
                "data": [{"url": a.url} for a in result.assets if a.url],
            }
        )

    # ---- OpenAI-compatible: /v1/models ----

    @app.get("/v1/coding-agents")
    async def list_coding_agents(verify: bool = True) -> Any:
        """List available local coding-agent CLIs."""
        from dataclasses import asdict

        from ..infra.discovery import get_available_coding_agents

        agents = get_available_coding_agents(verify=verify)
        agent_objects: list[dict[str, Any]] = []
        for agent in agents:
            item = asdict(agent)
            item["object"] = "coding_agent"
            agent_objects.append(item)

        return JSONResponse(
            {
                "object": "list",
                "data": agent_objects,
                # Legacy/simple field for small app integrations.
                "coding_agents": [agent["id"] for agent in agent_objects if agent["available"]],
            }
        )

    class CodingAgentRunRequest(BaseModel):
        agent: str
        task: str
        cwd: str | None = None
        approval_mode: str = "default"
        model: str | None = None
        extra_args: list[str] | None = None
        timeout: int = 600
        stream: bool = False
        output_format: str = "text"
        session_id: str | None = None

    @app.post("/v1/coding-agents/run")
    async def run_coding_agent_endpoint(req: CodingAgentRunRequest, _: None = Depends(_verify_api_key)) -> Any:
        """Run a local coding-agent CLI and return its output.

        When ``stream=true`` the response is an SSE stream of normalised
        :class:`CodingAgentEvent` objects (requires ``output_format='json'``,
        which is forced on); when ``stream=false`` the agent runs to
        completion and the JSON body mirrors :class:`CodingAgentRunResult`.
        """
        from dataclasses import asdict

        from ..infra.coding_agents import arun_coding_agent, astream_coding_agent

        if req.stream:
            try:
                from sse_starlette.sse import EventSourceResponse
            except ImportError as exc:
                raise HTTPException(
                    status_code=501,
                    detail="Streaming requires sse-starlette: pip install prompture[serve]",
                ) from exc

            async def event_generator() -> Any:
                try:
                    async for event in astream_coding_agent(
                        req.agent,
                        req.task,
                        cwd=req.cwd,
                        approval_mode=req.approval_mode,  # type: ignore[arg-type]
                        model=req.model,
                        extra_args=req.extra_args,
                        session_id=req.session_id,
                    ):
                        yield {"data": json.dumps(asdict(event), cls=PromptureJSONEncoder)}
                except ValueError as exc:
                    yield {"data": json.dumps({"type": "error", "error": str(exc)})}
                yield {"data": json.dumps({"type": "stream_end"})}

            return EventSourceResponse(event_generator(), media_type="text/event-stream")

        try:
            result = await arun_coding_agent(
                req.agent,
                req.task,
                cwd=req.cwd,
                approval_mode=req.approval_mode,
                model=req.model,
                extra_args=req.extra_args,
                timeout=req.timeout,
                output_format=req.output_format,
                session_id=req.session_id,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        payload = asdict(result)
        # tuples → lists; dataclass events are already dicts via asdict
        payload["events"] = list(payload.get("events") or [])
        payload["object"] = "coding_agent_run"
        return JSONResponse(payload)

    @app.get("/v1/models")
    async def list_models() -> Any:
        """List available models in OpenAI-compatible format."""
        from ..infra.discovery import get_available_models

        try:
            models = get_available_models()
            model_names = [m["id"] if isinstance(m, dict) else str(m) for m in models]
        except Exception:
            model_names = [model_name]

        if allowed_models is not None:
            allowlist = set(allowed_models)
            model_names = [m for m in model_names if m in allowlist]

        model_objects = [
            {
                "id": name,
                "object": "model",
                "created": 0,
                "owned_by": name.split("/")[0] if "/" in name else "prompture",
            }
            for name in model_names
        ]

        return JSONResponse(
            {
                "object": "list",
                "data": model_objects,
                # Legacy field kept for backwards compatibility.
                "models": model_names,
            }
        )

    return app
