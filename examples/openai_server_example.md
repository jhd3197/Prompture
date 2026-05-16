# OpenAI-Compatible Server — usage examples

Prompture's `serve` command exposes an OpenAI-shaped API
(`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`,
`/v1/models`) backed by Prompture's driver registry.  Point any
OpenAI SDK or OpenAI-compatible client at it and it will route to
whichever Prompture-supported provider you ask for.

## Start the server

```bash
pip install prompture[serve]
prompture serve --model openai/gpt-4o-mini --api-key sk-prompt-local
```

Add `--sandbox` to register a server-side sandboxed `python_execute`
tool, or `--web-search` to register the search tool — the LLM uses
them transparently; clients only see the final assistant message.

```bash
prompture serve \
  --model claude/claude-sonnet-4-6 \
  --api-key sk-prompt-local \
  --sandbox \
  --web-search
```

## Use it from the OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:9471/v1",
    api_key="sk-prompt-local",   # whatever you passed to --api-key
)

resp = client.chat.completions.create(
    model="claude/claude-sonnet-4-6",   # any Prompture model string
    messages=[
        {"role": "user", "content": "Hello!"},
    ],
)
print(resp.choices[0].message.content)
```

Streaming works too:

```python
for chunk in client.chat.completions.create(
    model="ollama/llama3.1:8b",
    messages=[{"role": "user", "content": "Tell me a joke."}],
    stream=True,
):
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
```

## Use it from Claude Code, Codex, Cursor, Aider, …

Most coding agents accept `OPENAI_BASE_URL` + `OPENAI_API_KEY`:

```bash
export OPENAI_BASE_URL=http://localhost:9471/v1
export OPENAI_API_KEY=sk-prompt-local
claude    # or `codex`, `aider`, etc.
```

Anything you type now routes through Prompture and can target any
configured provider (Ollama, OpenRouter, Bedrock, Groq, Z.ai, etc.).

## Embeddings

```python
resp = client.embeddings.create(
    model="openai/text-embedding-3-small",
    input=["hello world", "another doc"],
)
print(resp.data[0].embedding[:5])
```

## List available models

```python
print([m.id for m in client.models.list().data])
```

Use `--allow-models openai/gpt-4o-mini,ollama/llama3.1:8b` to restrict
the surface area to specific routes.

## With curl

```bash
curl http://localhost:9471/v1/chat/completions \
  -H "Authorization: Bearer sk-prompt-local" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ollama/llama3.1:8b",
    "messages": [{"role":"user","content":"Hello"}]
  }'
```

## Notes

- **Server-side tools vs. client-supplied `tools[]`**: when the server
  is started with `--sandbox` / `--web-search`, those tools execute
  on the server inside Prompture's agent loop — the client only sees
  the final answer.  Any `tools[]` array in the client's request body
  is forwarded to the driver as schema; if the model returns
  `tool_calls`, they appear in the response so the client can execute
  locally and reply with `role="tool"` messages.
- **Bearer auth** is opt-in via `--api-key`. The `/health` endpoint
  is always public so load balancers can probe it.
- **CORS**: pass `--cors-origins "*"` for browser apps. Lock down to
  specific hosts in production.
- **Rate limit**: `--rate-limit 60` caps each client IP to 60
  requests per minute.
