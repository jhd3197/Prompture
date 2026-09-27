"""Gemini ``generateContent`` wire format ↔ Prompture drivers.

Speaks the request and response shapes of Google's ``generateContent`` /
``streamGenerateContent`` (the Gemini API, and the Code Assist API that
Gemini CLI uses when signed in with Google, which wraps the same request in
``{"model", "project", "request": {...}}`` and each reply in
``{"response": {...}}``). Callers unwrap and wrap; this module handles the
inner objects.

Function calls are matched to their responses by id when Gemini sends one,
else by name, in order.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Iterable, Iterator
from typing import Any

from .openai_format import ChatOutcome


def _lower_types(schema: Any) -> Any:
    """Gemini's OpenAPI-style schemas spell types in capitals (``"OBJECT"``); JSON Schema doesn't."""
    if isinstance(schema, dict):
        out = {k: _lower_types(v) for k, v in schema.items()}
        if isinstance(out.get("type"), str):
            out["type"] = out["type"].lower()
        return out
    if isinstance(schema, list):
        return [_lower_types(v) for v in schema]
    return schema


def gemini_tools_to_openai(tools: Iterable[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Gemini function declarations → OpenAI function tools (built-in tools like search are skipped)."""
    out = []
    for group in tools or []:
        for fn in (group or {}).get("functionDeclarations") or (group or {}).get("function_declarations") or []:
            params = fn.get("parametersJsonSchema") or fn.get("parameters") or {"type": "object", "properties": {}}
            out.append(
                {
                    "type": "function",
                    "function": {
                        "name": fn.get("name", ""),
                        "description": fn.get("description", ""),
                        "parameters": _lower_types(params),
                    },
                }
            )
    return out


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return "\n".join(str(p.get("text", "")) for p in content.get("parts") or [] if isinstance(p, dict))
    return ""


def gemini_to_driver(request: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """A ``generateContent`` request → ``(messages, tools, options)``."""
    messages: list[dict[str, Any]] = []
    system = _text_of(request.get("systemInstruction") or request.get("system_instruction"))
    if system:
        messages.append({"role": "system", "content": system})
    pending: dict[str, list[str]] = {}  # function name → ids of calls not answered yet
    for content in request.get("contents") or []:
        if not isinstance(content, dict):
            continue
        role = "assistant" if content.get("role") == "model" else "user"
        texts: list[str] = []
        images: list[dict[str, Any]] = []
        calls: list[dict[str, Any]] = []
        results: list[dict[str, Any]] = []
        for part in content.get("parts") or []:
            if not isinstance(part, dict) or part.get("thought"):
                continue
            if isinstance(part.get("text"), str):
                texts.append(part["text"])
            elif isinstance(part.get("inlineData"), dict) and part["inlineData"].get("data"):
                from ..media.image import make_image

                data = part["inlineData"]
                uri = f"data:{data.get('mimeType', 'image/png')};base64,{data['data']}"
                images.append({"type": "image", "source": make_image(uri)})
            elif isinstance(part.get("functionCall"), dict):
                call = part["functionCall"]
                name = str(call.get("name") or "")
                cid = str(call.get("id") or f"call_{uuid.uuid4().hex[:16]}")
                pending.setdefault(name, []).append(cid)
                calls.append(
                    {
                        "id": cid,
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(call.get("args") or {})},
                    }
                )
            elif isinstance(part.get("functionResponse"), dict):
                resp = part["functionResponse"]
                name = str(resp.get("name") or "")
                queue = pending.get(name) or []
                cid = str(resp.get("id") or (queue.pop(0) if queue else f"call_{name}"))
                if resp.get("id") and cid in queue:
                    queue.remove(cid)
                payload = resp.get("response")
                results.append({"role": "tool", "tool_call_id": cid, "content": _response_text(payload)})
        text = "\n".join(t for t in texts if t)
        messages.extend(results)
        if calls:
            messages.append({"role": "assistant", "content": text, "tool_calls": calls})
        elif images:
            messages.append({"role": role, "content": ([{"type": "text", "text": text}] if text else []) + images})
        elif text:
            messages.append({"role": role, "content": text})

    options: dict[str, Any] = {}
    config = request.get("generationConfig") or request.get("generation_config") or {}
    for src, dst in (("temperature", "temperature"), ("topP", "top_p"), ("maxOutputTokens", "max_tokens")):
        if config.get(src) is not None:
            options[dst] = config[src]
    if config.get("stopSequences"):
        options["stop"] = config["stopSequences"]
    schema = config.get("responseJsonSchema") or config.get("responseSchema")
    if isinstance(schema, dict):
        options["json_mode"] = True
        options["json_schema"] = _lower_types(schema)
    return messages, gemini_tools_to_openai(request.get("tools")), options


def _response_text(payload: Any) -> str:
    """A function response as the text a tool message carries."""
    if isinstance(payload, dict):
        output = payload.get("output")
        if set(payload) <= {"output"} and isinstance(output, str):
            return output
        if "error" in payload and set(payload) <= {"error"}:
            return f"[tool error] {payload['error']}"
    return json.dumps(payload, default=str) if not isinstance(payload, str) else payload


def gemini_usage(meta: dict[str, Any] | None) -> dict[str, Any]:
    meta = meta or {}
    prompt = int(meta.get("prompt_tokens", 0) or 0)
    output = int(meta.get("completion_tokens", 0) or 0)
    usage = {"promptTokenCount": prompt, "candidatesTokenCount": output, "totalTokenCount": prompt + output}
    if meta.get("cached_prompt_tokens"):
        usage["cachedContentTokenCount"] = int(meta["cached_prompt_tokens"])
    return usage


_FINISH = {
    "end_turn": "STOP",
    "stop": "STOP",
    "tool_use": "STOP",
    "tool_calls": "STOP",
    "max_tokens": "MAX_TOKENS",
    "length": "MAX_TOKENS",
}


def _call_part(tc: dict[str, Any]) -> dict[str, Any]:
    fn = tc.get("function") or {}
    args = tc.get("arguments", fn.get("arguments"))
    if isinstance(args, str):
        try:
            args = json.loads(args or "{}")
        except ValueError:
            args = {"_raw": args}
    return {"functionCall": {"name": tc.get("name") or fn.get("name", ""), "args": args or {}, "id": tc.get("id")}}


def gemini_response(outcome: ChatOutcome, *, model: str, response_id: str | None = None) -> dict[str, Any]:
    """A complete ``generateContent`` response from a chat outcome."""
    parts: list[dict[str, Any]] = []
    if outcome.text:
        parts.append({"text": outcome.text})
    parts += [_call_part(tc) for tc in outcome.tool_calls]
    return {
        "candidates": [
            {
                "content": {"role": "model", "parts": parts},
                "finishReason": _FINISH.get((outcome.stop_reason or "stop").lower(), "STOP"),
                "index": 0,
            }
        ],
        "usageMetadata": gemini_usage(outcome.meta),
        "modelVersion": model,
        "responseId": response_id or f"resp_{uuid.uuid4().hex[:24]}",
    }


def stream_gemini_events(
    events: Iterable[Any],
    *,
    model: str,
    response_id: str | None = None,
    on_complete: Callable[[ChatOutcome], None] | None = None,
) -> Iterator[dict[str, Any]]:
    """Driver stream events → ``streamGenerateContent`` chunks (inner responses, unwrapped).

    A failure is yielded as ``{"error": {...}}`` rather than raised, like the
    other dialects' streams, so a caller can tell a failed start apart.
    """
    rid = response_id or f"resp_{uuid.uuid4().hex[:24]}"
    outcome = ChatOutcome()

    def chunk(parts: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
        return {
            "candidates": [{"content": {"role": "model", "parts": parts}, "index": 0, **extra}],
            "modelVersion": model,
            "responseId": rid,
        }

    try:
        for ev in events:
            if isinstance(ev, dict):
                if ev.get("type") == "delta" and ev.get("text"):
                    outcome.text += ev["text"]
                    yield chunk([{"text": ev["text"]}])
                elif ev.get("type") == "done":
                    outcome.meta = ev.get("meta") or {}
                    outcome.stop_reason = ev.get("stop_reason") or outcome.meta.get("stop_reason")
                continue
            kind = getattr(ev, "event_type", None)
            if kind == "text_delta" and ev.text:
                outcome.text += ev.text
                yield chunk([{"text": ev.text}])
            elif kind == "tool_use_stop":
                call = {"id": ev.id, "name": ev.name, "arguments": ev.input or {}}
                outcome.tool_calls.append(call)
                yield chunk([_call_part(call)])
            elif kind == "message_stop":
                outcome.stop_reason = ev.stop_reason
                outcome.meta = dict(ev.usage or {})
    except Exception as exc:
        outcome.error = exc
        if on_complete:
            on_complete(outcome)
        yield {"error": {"code": 500, "message": str(exc), "status": "INTERNAL"}}
        return
    final = chunk([], finishReason=_FINISH.get((outcome.stop_reason or "stop").lower(), "STOP"))
    final["usageMetadata"] = gemini_usage(outcome.meta)
    yield final
    if on_complete:
        on_complete(outcome)
