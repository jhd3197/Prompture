# Integrating & Extending Prompture

This guide covers two things:

1. **[Integrating Prompture into your project](#integrating-prompture-into-your-project)** — common patterns for FastAPI, SSE streaming, structured extraction, and error handling.
2. **[Extending Prompture with custom providers](#extending-prompture)** — the plugin architecture and how to publish your own provider.

---

## Integrating Prompture into Your Project

### FastAPI + AsyncAgent with Tools

The most common integration pattern — an AI chat endpoint with database-backed tools:

```python
from fastapi import APIRouter, Depends
from prompture import AsyncAgent, ToolRegistry, ProviderEnvironment, BudgetExceededError

router = APIRouter()

def build_tools(db) -> ToolRegistry:
    registry = ToolRegistry()

    @registry.tool
    async def search_records(query: str) -> str:
        """Search the database for matching records."""
        results = await db.execute(...)
        return format_results(results)

    return registry

@router.post("/chat")
async def chat(message: str, db=Depends(get_db)):
    env = ProviderEnvironment(openai_api_key=get_api_key_from_db(db))

    agent = AsyncAgent(
        "openai/gpt-4o",
        env=env,
        tools=build_tools(db),
        system_prompt="You are a helpful assistant with database access.",
        max_cost=0.25,
        budget_policy="hard_stop",
    )

    try:
        result = await agent.run(message)
        return {"reply": result.output_text, "usage": result.usage}
    except BudgetExceededError:
        return {"error": "Cost limit exceeded"}, 429
```

### SSE Streaming Endpoint

Stream responses via Server-Sent Events:

```python
from fastapi.responses import StreamingResponse
from prompture import AsyncAgent, StreamEventType

@router.post("/chat/stream")
async def chat_stream(message: str):
    agent = AsyncAgent("claude/claude-sonnet-4-6", env=env, system_prompt="...")

    async def event_stream():
        async for event in agent.run_stream(message):
            match event.event_type:
                case StreamEventType.text_delta:
                    yield f"data: {json.dumps({'type': 'text', 'content': event.data})}\n\n"
                case StreamEventType.tool_call:
                    yield f"data: {json.dumps({'type': 'tool_call', 'name': event.data['name']})}\n\n"
                case StreamEventType.output:
                    yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
```

### Structured Extraction in Endpoints

Use `AsyncConversation.ask_for_json()` for one-shot structured data extraction:

```python
from prompture import AsyncConversation

@router.get("/insights")
async def get_insights():
    conv = AsyncConversation("openai/gpt-4o", system_prompt="You analyze data.")
    result = await conv.ask_for_json(
        f"Analyze this data and produce insights:\n\n{context}",
        {"type": "object", "properties": {
            "insights": {"type": "array", "items": {"type": "object", ...}},
            "summary": {"type": "string"},
        }},
    )
    return result["json_object"]
```

### Error Handling

Key exceptions to catch in production:

```python
from prompture import BudgetExceededError, DriverError, ExtractionError, ValidationError

try:
    result = await agent.run(message)
except BudgetExceededError:
    # Cost or token limit exceeded — return 429
    pass
except DriverError:
    # Provider API error (auth, rate limit, network) — return 502
    pass
except ExtractionError:
    # JSON parsing/validation failed — return 422
    pass
except ValidationError:
    # Schema validation failed — return 422
    pass
```

---

## Extending Prompture

Prompture's provider registry is plugin-based. Every built-in provider
(OpenAI, Claude, Google, etc.) is contributed by a `ProviderPlugin`
instance registered in `prompture.plugins.builtins`. Third-party packages
can register their own providers via the `prompture.providers` Python
entry-point group — no fork required.

### Plugin Architecture

At import time, `prompture` discovers plugins from two sources:

1. **Built-in plugins** — loaded from `prompture.plugins.builtins` directly.
2. **External plugins** — discovered through the `prompture.providers`
   entry-point group via `importlib.metadata.entry_points()`.

Each plugin returns one or more `ProviderDescriptor` instances. Prompture
then wires them up to the LLM, audio, image, video, embedding, rerank,
and moderation driver registries.

### Writing a Plugin

Create a Python file that subclasses `ProviderPlugin`:

```python
# my_package/plugin.py
from prompture.plugins import ProviderPlugin
from prompture.drivers.provider_descriptors import (
    ProviderDescriptor,
    DriverSpec,
)


class MyProviderPlugin(ProviderPlugin):
    name = "my_provider"
    version = "0.1.0"

    def descriptors(self):
        return [
            ProviderDescriptor(
                name="my_provider",
                llm_sync=DriverSpec(
                    cls_path="my_package.driver.MyDriver",
                    kwarg_map={"api_key": "my_provider_api_key"},
                    default_model="my-model-1",
                ),
                display_name="My Provider",
                is_configured_check="my_provider_api_key",
            ),
        ]
```

Then declare the entry point in your package's `pyproject.toml`:

```toml
[project.entry-points."prompture.providers"]
my_provider = "my_package.plugin:MyProviderPlugin"
```

Once `pip install`-ed alongside Prompture, your provider becomes
available automatically:

```python
from prompture import get_driver_for_model

driver = get_driver_for_model("my_provider/my-model-1")
```
