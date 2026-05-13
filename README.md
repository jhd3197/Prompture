<p align="center">
  <h1 align="center">Prompture</h1>
  <p align="center">Structured JSON extraction from any LLM. Schema-enforced, Pydantic-native, multi-provider.</p>
</p>

<p align="center">
  <a href="https://pypi.org/project/prompture/"><img src="https://badge.fury.io/py/prompture.svg" alt="PyPI version"></a>
  <a href="https://pypi.org/project/prompture/"><img src="https://img.shields.io/pypi/pyversions/prompture.svg" alt="Python versions"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
  <a href="https://pepy.tech/project/prompture"><img src="https://static.pepy.tech/badge/prompture" alt="Downloads"></a>
  <a href="https://github.com/jhd3197/prompture"><img src="https://img.shields.io/github/stars/jhd3197/prompture?style=social" alt="GitHub stars"></a>
</p>

---

**Prompture** is a Python library that turns LLM responses into validated, structured data. Define a schema or Pydantic model, point it at any provider, and get typed output back — with token tracking, cost calculation, and automatic JSON repair built in.

```python
from pydantic import BaseModel
from prompture import extract_with_model

class Person(BaseModel):
    name: str
    age: int
    profession: str

person = extract_with_model(Person, "Maria is 32, a developer in NYC.", model_name="openai/gpt-4")
print(person.name)  # Maria
```

## Key Features

- **Structured output** — JSON schema enforcement and direct Pydantic model population
- **23+ providers** — OpenAI, Claude, Google, Groq, Grok, Azure, Ollama, LM Studio, OpenRouter, HuggingFace, Moonshot, ModelScope, Z.ai, Vertex AI, AirLLM, CachiBot, Runway, MiniMax/Hailuo, Kling AI, Fal.ai, Mistral AI, DeepSeek, generic OpenAI-compatible (Fireworks, Together, Cerebras, SambaNova, Perplexity, NVIDIA, DeepInfra, SiliconFlow), and generic HTTP
- **Multi-modal** — Drivers for embeddings, image generation (DALL-E, Imagen, Grok, Stability, Runway), video generation (Grok Imagine Video, Runway text/image/video → video), text-to-speech (OpenAI, ElevenLabs, Runway), sound effects, voice dubbing / isolation / conversion (Runway), and speech-to-text (Whisper, ElevenLabs)
- **Multi-model fallback** — Try a list of models in sequence with per-attempt cost, token, and capability accounting
- **Strategy cascade** — Auto-selects between provider-native JSON mode, tool-call extraction, and prompted repair so extraction works on any model
- **TOON input conversion** — 45-60% token savings when sending structured data via [Token-Oriented Object Notation](https://github.com/jhd3197/python-toon)
- **Stepwise extraction** — Per-field prompts with smart type coercion (shorthand numbers, multilingual booleans, dates)
- **Field registry** — 50+ predefined extraction fields with template variables and Pydantic integration
- **Conversations** — Stateful multi-turn sessions with sync and async support
- **Tool use** — Function calling and streaming across supported providers, with automatic prompt-based simulation for models without native tool support
- **Caching** — Built-in response cache with memory, SQLite, and Redis backends
- **Plugin system** — Register custom drivers via entry points
- **Usage tracking** — Token counts and cost calculation on every call
- **Auto-repair** — Optional second LLM pass to fix malformed JSON
- **Batch testing** — Spec-driven suites to compare models side by side

## Built With Prompture

Projects powered by Prompture at their core:

- **[CachiBot](https://github.com/jhd3197/CachiBot)** — AI-powered bot built on Prompture's structured extraction and multi-provider driver system
- **[AgentSite](https://github.com/jhd3197/AgentSite)** — Agent-driven web platform using Prompture for LLM orchestration and structured output

## Installation

```bash
pip install prompture
```

Optional extras:

```bash
pip install prompture[redis]     # Redis cache backend
pip install prompture[serve]     # FastAPI server mode
pip install prompture[airllm]    # AirLLM local inference
```

## Configuration

Set API keys for the providers you use. Prompture reads from environment variables or a `.env` file:

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
GROQ_API_KEY=...
GROK_API_KEY=...
# optional xAI-compatible alias for Grok APIs
XAI_API_KEY=...
OPENROUTER_API_KEY=...
AZURE_OPENAI_ENDPOINT=...
AZURE_OPENAI_API_KEY=...
```

Local providers (Ollama, LM Studio) work out of the box with no keys required.

### Runtime API Keys (No Environment Variables)

Pass API keys at runtime via `ProviderEnvironment` — useful for multi-tenant apps, web backends, or anywhere you don't want to set `os.environ`:

```python
from prompture import AsyncAgent, ProviderEnvironment

env = ProviderEnvironment(
    openai_api_key="sk-...",
    claude_api_key="sk-ant-...",
)

agent = AsyncAgent("openai/gpt-4o", env=env)
result = await agent.run("Hello!")
```

Works on `Agent`, `AsyncAgent`, `Conversation`, and `AsyncConversation`.

## Providers

Model strings use `"provider/model"` format. The provider prefix routes to the correct driver automatically.

| Provider | Example Model | Cost |
|---|---|---|
| `openai` | `openai/gpt-4` | Automatic |
| `claude` | `claude/claude-3` | Automatic |
| `google` | `google/gemini-1.5-pro` | Automatic |
| `google_vertexai` | `google_vertexai/gemini-1.5-pro` | Automatic |
| `groq` | `groq/llama2-70b-4096` | Automatic |
| `grok` | `grok/grok-4-fast-reasoning` | Automatic |
| `azure` | `azure/deployed-name` | Automatic |
| `openrouter` | `openrouter/anthropic/claude-2` | Automatic |
| `moonshot` | `moonshot/kimi-k2` | Automatic |
| `modelscope` | `modelscope/Qwen2.5-72B-Instruct` | Automatic |
| `zai` | `zai/glm-4` | Automatic |
| `cachibot` | `cachibot/openai/gpt-4o-mini` | Automatic |
| `ollama` | `ollama/llama3.1:8b` | Free (local) |
| `lmstudio` | `lmstudio/local-model` | Free (local) |
| `huggingface` | `hf/model-name` | Free (local) |
| `airllm` | `airllm/Qwen2-7B` | Free (local) |
| `local_http` | `local_http/self-hosted` | Free |
| `runway` | `runway/gen4.5` (video), `runway/gpt_image_2` (image), `runway/eleven_multilingual_v2` (TTS) | Automatic |
| `minimax` | `minimax/MiniMax-Text-01` (LLM), `minimax/MiniMax-Hailuo-2.3` (video) | Automatic |
| `kling` | `kling/kling-v2-1` (image + video) | Automatic |
| `fal` | `fal/fal-ai/flux/dev` (image), `fal/fal-ai/kling-video/v2.6/pro/image-to-video` (video) | Automatic |
| `mistral` | `mistral/mistral-large-latest` | Automatic |
| `deepseek` | `deepseek/deepseek-chat`, `deepseek/deepseek-reasoner` | Automatic |
| `openai_compatible` | `openai_compatible/<profile>/<model>` — 8 curated profiles: `fireworks`, `together`, `cerebras`, `sambanova`, `perplexity`, `nvidia`, `deepinfra`, `siliconflow` (or pass an explicit `endpoint=` for anything else) | Automatic where pricing is known |

Aliases (`anthropic`, `gemini`, `chatgpt`, `xai`, `lm_studio`, `zhipu`, `hf`, `dalle`, `runwayml`, `hailuo`, `mistralai`) route to their canonical providers.

## Multi-Modal

Beyond text LLMs, Prompture exposes drivers for adjacent modalities under the same `provider/model` routing:

- **Embeddings** — OpenAI (`text-embedding-3-*`) and Ollama (`nomic-embed-text`)
- **Image generation** — OpenAI DALL-E + GPT image, Google Imagen, Grok, Stability AI, Runway (`gen4_image`, `gen4_image_turbo`, `gpt_image_2`, `gemini_image3_pro`, `gemini_2.5_flash`), Kling AI, Fal.ai
- **Video generation** — Grok Imagine Video; Runway text/image/video → video (`gen4.5`, `gen4_turbo`, `gen3a_turbo`, `gen4_aleph`, `veo3`, `veo3.1`, `veo3.1_fast`); MiniMax / Hailuo; Kling AI; Fal.ai
- **Text-to-speech** — OpenAI (`tts-1`), ElevenLabs, Runway (`eleven_multilingual_v2`)
- **Sound effects** — Runway (`eleven_text_to_sound_v2`)
- **Audio transforms** — Runway voice dubbing, voice isolation, speech-to-speech (`RunwayAudioTransformDriver`)
- **Speech-to-text** — OpenAI Whisper and ElevenLabs

```python
from prompture.drivers.img_gen_registry import get_img_gen_driver_for_model

driver = get_img_gen_driver_for_model("openai/dall-e-3")
result = driver.generate_image(
    "a cat on a surfboard at sunset",
    {"size": "1024x1024", "quality": "hd"},
)
print(result["meta"]["cost"], result["meta"]["image_count"])
```

Video generation uses the same provider/model routing. Set `GROK_API_KEY` or `XAI_API_KEY`, then request a Grok video model:

```python
from prompture import get_video_gen_driver_for_model

driver = get_video_gen_driver_for_model("grok/grok-imagine-video")
result = driver.generate_video(
    "wide shot of a crystal-powered rocket launching from red desert dunes",
    {"duration": 8, "aspect_ratio": "16:9", "resolution": "720p"},
)

video = result["videos"][0]
print(video.url)
print(result["meta"]["request_id"], result["meta"]["cost"])
```

For local smoke tests without waiting on the render, pass `{"poll": False}` to get the provider request ID. The async factory is available as `get_async_video_gen_driver_for_model()`.

Runnable example: `python examples/grok_video_generation_example.py`.

### Runway

Runway is a single API surface covering image, video, and audio. One key (`RUNWAY_API_KEY`, or `RUNWAYML_API_SECRET`) unlocks all of it:

```python
from prompture.drivers.img_gen_registry import get_img_gen_driver_for_model
from prompture.drivers.video_gen_registry import get_video_gen_driver_for_model
from prompture.drivers.audio_registry import get_tts_driver_for_model
from prompture.drivers import RunwayAudioTransformDriver

# Image — text_to_image, optionally with reference images
img = get_img_gen_driver_for_model("runway/gpt_image_2").generate_image(
    "A cinematic wide shot of a neon-lit Tokyo alleyway at night in the rain",
    {"ratio": "1920:1080", "quality": "high"},
)

# Video — one driver, three modes (auto-detected from inputs)
vid = get_video_gen_driver_for_model("runway/gen4.5").generate_video(
    "wide cinematic shot of a rocket launching from desert dunes",
    {"ratio": "1280:720", "duration": 5},          # text_to_video
)
# Pass `image=...` → image_to_video; `video=...` → video_to_video (gen4_aleph).

# Speech and sound effects
tts = get_tts_driver_for_model("runway/eleven_multilingual_v2").synthesize(
    "Hello from Runway via Prompture.", {"voice": "Maya"},
)
sfx = get_tts_driver_for_model("runway/eleven_text_to_sound_v2").synthesize(
    "Heavy tropical rain on a metal roof", {"duration": 5},
)

# Voice transforms (audio in → audio out, not a registered modality)
dub = RunwayAudioTransformDriver().dub("https://.../speech.mp3", target_lang="es")
```

Inspect any model's capabilities (operations, endpoints, cost) as data — no need to instantiate the driver:

```python
from prompture.drivers import get_runway_model_info, get_runway_models_by_op

get_runway_model_info("gen4.5")
# {'modality': 'video',
#  'operations': ['text_to_video', 'image_to_video'],
#  'endpoints':  ['/v1/text_to_video', '/v1/image_to_video'],
#  'cost': '$0.12 per second'}

get_runway_models_by_op("text_to_video")
# ['gen4.5', 'veo3', 'veo3.1', 'veo3.1_fast']
```

Runnable examples:
- `python examples/runway_image_generation_example.py`
- `python examples/runway_video_generation_example.py`
- `python examples/runway_audio_example.py`

## Usage

### One-Shot Pydantic Extraction

Single LLM call, returns a validated Pydantic instance:

```python
from typing import List, Optional
from pydantic import BaseModel
from prompture import extract_with_model

class Person(BaseModel):
    name: str
    age: int
    profession: str
    city: str
    hobbies: List[str]
    education: Optional[str] = None

person = extract_with_model(
    Person,
    "Maria is 32, a software developer in New York. She loves hiking and photography.",
    model_name="openai/gpt-4"
)
print(person.model_dump())
```

### Stepwise Extraction

One LLM call per field. Higher accuracy, per-field error recovery:

```python
from prompture import stepwise_extract_with_model

result = stepwise_extract_with_model(
    Person,
    "Maria is 32, a software developer in New York. She loves hiking and photography.",
    model_name="openai/gpt-4"
)
print(result["model"].model_dump())
print(result["usage"])  # per-field and total token usage
```

| Aspect | `extract_with_model` | `stepwise_extract_with_model` |
|---|---|---|
| LLM calls | 1 | N (one per field) |
| Speed / cost | Faster, cheaper | Slower, higher |
| Accuracy | Good global coherence | Higher per-field accuracy |
| Error handling | All-or-nothing | Per-field recovery |

### JSON Schema Extraction

For raw JSON output with full control:

```python
from prompture import ask_for_json

schema = {
    "type": "object",
    "required": ["name", "age"],
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"}
    }
}

result = ask_for_json(
    content_prompt="Extract the person's info from: John is 28 and lives in Miami.",
    json_schema=schema,
    model_name="openai/gpt-4"
)
print(result["json_object"])  # {"name": "John", "age": 28}
print(result["usage"])        # token counts and cost
```

### Strategy Cascade

Prompture picks how to obtain structured JSON based on each model's capabilities. The cascade is `provider_native` (built-in JSON mode / schema enforcement) → `tool_call` (encode the schema as a function definition and read it back from the tool call) → `prompted_repair` (prompt for JSON, repair malformed output via AI cleanup). Pass `strategy="auto"` (default) to let Prompture select per model, or pin a specific strategy via the `StructuredOutputStrategy` enum or its string value. The strategy used is recorded in the response so you can see which path each call took.

### Multi-Model Fallback

Try a list of models in priority order, with full per-attempt accounting — every model tried (success, failure, or skipped) is recorded with its cost, tokens, duration, capabilities, and strategy. The first success wins; if all fail, an optional `fallback` Pydantic instance is returned instead of raising.

```python
from prompture import extract_with_models

result = extract_with_models(
    Person,
    "Maria is 32, a software developer in NYC.",
    models=[
        "openai/gpt-4o-mini",        # try first
        "claude/claude-3-5-haiku",   # fallback
        "ollama/llama3.1:8b",        # last resort, free
    ],
    fallback=Person(name="unknown", age=0, profession="unknown"),
)

print(result["selected_model"])     # winning model string
print(result["model"])              # validated Pydantic instance
print(result["total_cost"])         # cumulative cost across all attempts
print(result["total_attempts"])     # number of models actually called

for attempt in result["attempts"]:
    print(
        attempt["model"],
        attempt["status"],          # "success" | "failed" | "skipped"
        attempt["strategy"],        # "single" | "stepwise"
        attempt["cost"],
        attempt["prompt_tokens"],
        attempt["completion_tokens"],
        attempt["duration_ms"],
        attempt["capabilities"],    # {"json_mode": bool, "json_schema": bool}
    )
```

If every model fails and no `fallback` is provided, an `ExtractionError` is raised with the full `attempts` list, `total_cost`, and `total_tokens` attached as attributes.

### TOON Input — Token Savings

Analyze structured data with automatic TOON conversion for 45-60% fewer tokens:

```python
from prompture import extract_from_data

products = [
    {"id": 1, "name": "Laptop", "price": 999.99, "rating": 4.5},
    {"id": 2, "name": "Book", "price": 19.99, "rating": 4.2},
    {"id": 3, "name": "Headphones", "price": 149.99, "rating": 4.7},
]

result = extract_from_data(
    data=products,
    question="What is the average price and highest rated product?",
    json_schema={
        "type": "object",
        "properties": {
            "average_price": {"type": "number"},
            "highest_rated": {"type": "string"}
        }
    },
    model_name="openai/gpt-4"
)

print(result["json_object"])
# {"average_price": 389.99, "highest_rated": "Headphones"}

print(f"Token savings: {result['token_savings']['percentage_saved']}%")
```

Works with Pandas DataFrames via `extract_from_pandas()`.

### Field Definitions

Use the built-in field registry for consistent extraction across models:

```python
from pydantic import BaseModel
from prompture import field_from_registry, stepwise_extract_with_model

class Person(BaseModel):
    name: str = field_from_registry("name")
    age: int = field_from_registry("age")
    email: str = field_from_registry("email")
    occupation: str = field_from_registry("occupation")

result = stepwise_extract_with_model(
    Person,
    "John Smith, 25, software engineer at TechCorp, john@example.com",
    model_name="openai/gpt-4"
)
```

Register custom fields with template variables:

```python
from prompture import register_field

register_field("document_date", {
    "type": "str",
    "description": "Document creation date",
    "instructions": "Use {{current_date}} if not specified",
    "default": "{{current_date}}",
    "nullable": False
})
```

### Conversations

Stateful multi-turn sessions:

```python
from prompture import Conversation

conv = Conversation(model_name="openai/gpt-4")
conv.add_message("system", "You are a helpful assistant.")
response = conv.send("What is the capital of France?")
follow_up = conv.send("What about Germany?")  # retains context
```

### Tool Use

Register Python functions as tools the LLM can call during a conversation:

```python
from prompture import Conversation, ToolRegistry

registry = ToolRegistry()

@registry.tool
def get_weather(city: str, units: str = "celsius") -> str:
    """Get the current weather for a city."""
    return f"Weather in {city}: 22 {units}"

conv = Conversation("openai/gpt-4", tools=registry)
result = conv.ask("What's the weather in London?")
```

For models without native function calling (Ollama, LM Studio, etc.), Prompture automatically simulates tool use by describing tools in the prompt and parsing structured JSON responses:

```python
# Auto-detect: uses native tool calling if available, simulation otherwise
conv = Conversation("ollama/llama3.1:8b", tools=registry, simulated_tools="auto")

# Force simulation even on capable models
conv = Conversation("openai/gpt-4", tools=registry, simulated_tools=True)

# Disable tool use entirely
conv = Conversation("openai/gpt-4", tools=registry, simulated_tools=False)
```

The simulation loop describes tools in the system prompt, asks the model to respond with JSON (`tool_call` or `final_answer`), executes tools, and feeds results back — all transparent to the caller.

### Budget Control

Set cost and token limits with policy-based enforcement:

```python
from prompture import AsyncAgent

agent = AsyncAgent(
    "openai/gpt-4o",
    max_cost=0.50,
    budget_policy="hard_stop",       # accepts strings or BudgetPolicy enum
    fallback_models=["openai/gpt-4o-mini"],
)
```

Policies: `"hard_stop"` (raise `BudgetExceededError` on exceed), `"warn_and_continue"` (log and proceed), `"degrade"` (auto-switch to cheaper model at 80% budget).

### Provider Utilities

Extract provider info from model strings:

```python
from prompture import provider_for_model, parse_model_string

provider_for_model("claude/claude-sonnet-4-6")                  # "claude"
provider_for_model("claude/claude-sonnet-4-6", canonical=True)  # "anthropic"
parse_model_string("openai/gpt-4o")                             # ("openai", "gpt-4o")
```

### Model Discovery

Auto-detect available models from configured providers:

```python
from prompture import get_available_models

models = get_available_models()
for model in models:
    print(model)  # "openai/gpt-4", "ollama/llama3:latest", ...
```

For non-LLM modalities, use the matching helper:

```python
from prompture.infra.discovery import (
    get_available_image_gen_models,
    get_available_video_gen_models,
    get_available_audio_models,
)

get_available_image_gen_models()        # ['runway/gpt_image_2', 'openai/dall-e-3', ...]
get_available_video_gen_models()        # ['runway/gen4.5', 'runway/gen4_aleph', ...]
get_available_audio_models(modality="tts")  # ['runway/eleven_multilingual_v2', ...]
```

### Logging and Debugging

```python
import logging
from prompture import configure_logging

configure_logging(logging.DEBUG)
```

### Response Shape

All extraction functions return a consistent structure:

```python
{
    "json_string": str,       # raw JSON text
    "json_object": dict,      # parsed result
    "usage": {
        "prompt_tokens": int,
        "completion_tokens": int,
        "total_tokens": int,
        "cost": float,
        "model_name": str
    }
}
```

## CLI

```bash
prompture run <spec-file>
```

Run spec-driven extraction suites for cross-model comparison.

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

## Development

```bash
# Install with dev dependencies
pip install -e ".[test,dev]"

# Run tests
pytest

# Run integration tests (requires live LLM access)
pytest --run-integration

# Lint and format
ruff check .
ruff format .
```

## Contributing

PRs welcome. Please add tests for new functionality and examples under `examples/` for new drivers or patterns.

## License

[MIT](https://opensource.org/licenses/MIT)
