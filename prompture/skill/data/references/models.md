# Calling models

Model strings are `provider/model`: `openai/gpt-4o-mini`,
`claude/claude-haiku-4-5`, `google/gemini-2.5-flash`,
`groq/llama-3.3-70b-versatile`, `ollama/llama3.1:8b`,
`openrouter/<vendor>/<model>`.

## Structured output

```python
from pydantic import BaseModel
from prompture import extract_with_model

class Person(BaseModel):
    name: str
    age: int

p = extract_with_model(Person, "Maria is 32.", model_name="openai/gpt-4o-mini")
```

`ask_for_json(...)` / `extract_and_jsonify(...)` take a JSON schema instead.

## Chat, tools and agents

```python
from prompture import Agent, Conversation

conv = Conversation(model_name="openai/gpt-4o-mini")
conv.ask("Hi")

agent = Agent("openai/gpt-4o-mini", tools=["web:all", my_function])
print(agent.run("...").output)
```

## Failover across models

```python
from prompture import resilient
driver = resilient("openai/gpt-4o", "claude/claude-sonnet-4-5", "ollama/llama3.1:8b")
resp = driver.generate_messages([{"role": "user", "content": "hi"}], {})
resp["meta"]["route"]   # served_by, fallback, attempts — announce it
```

## What's available here

```bash
prompture doctor --only providers [--live]   # key present, SDK importable, (live) models list
```

```python
from prompture import get_available_models
get_available_models()
```

Every response carries `prompt_tokens`, `completion_tokens`, `total_tokens`
and `cost` in its metadata.

## Keys

Prefer `prompture configure OPENAI_API_KEY` (hidden prompt, live-validated,
stored owner-only in `~/.prompture/credentials.yaml`) or `prompture setup`.
Env vars and `.env` always win over the store. Profiles:
`--profile work` / `PROMPTURE_PROFILE`. Never print key values.
