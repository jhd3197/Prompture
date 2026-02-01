---
name: add-tool
description: Add function-calling tools to Prompture agents. Covers ToolDefinition, ToolRegistry, tool_from_function, decorator patterns, serialization formats, and Conversation integration. Use when adding callable tools for LLM function calling.
metadata:
  author: prompture
  version: "1.0"
---

# Add a Tool for Function Calling

Creates callable tools that LLMs can invoke during conversations via Prompture's tool use system.

## Before Starting

Ask the user for:
- **Tool function name** and what it does
- **Parameters** — names, types, and which are required vs optional
- **Return type** — what the tool returns (string result fed back to the LLM)
- **Registration method** — decorator (`@registry.tool`) or explicit (`registry.register()`)
- **Target driver** — which provider will use the tools (must have `supports_tool_use = True`)

## Key Concepts

### ToolDefinition

A dataclass describing a single callable tool:

```python
from prompture.tools_schema import ToolDefinition

td = ToolDefinition(
    name="get_weather",
    description="Get current weather for a city.",
    parameters={
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"},
            "units": {"type": "string", "description": "Temperature units"},
        },
        "required": ["city"],
    },
    function=get_weather_fn,
)
```

### tool_from_function — Auto-generate from type hints

```python
from prompture.tools_schema import tool_from_function

def get_weather(city: str, units: str = "celsius") -> str:
    """Get the current weather for a city."""
    return f"Weather in {city}: 22 {units}"

td = tool_from_function(get_weather)
# Automatically extracts: name, description (from docstring), parameters (from type hints)
```

**Type mapping:** `str` -> `"string"`, `int` -> `"integer"`, `float` -> `"number"`, `bool` -> `"boolean"`, `list` -> `"array"`, `dict` -> `"object"`. `Optional[X]` is unwrapped. `list[X]` becomes `{"type": "array", "items": {...}}`.

**Required vs optional:** Parameters without defaults are `required`. Parameters with defaults are optional.

### ToolRegistry — Managing tool collections

```python
from prompture import ToolRegistry

registry = ToolRegistry()

# Method 1: Decorator
@registry.tool
def search(query: str) -> str:
    """Search the knowledge base."""
    return f"Results for: {query}"

# Method 2: Explicit registration
def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

registry.register(calculate)

# Method 3: With name/description override
registry.register(some_func, name="custom_name", description="Custom description")

# Method 4: Add pre-built ToolDefinition
registry.add(td)
```

### Serialization Formats

```python
# For OpenAI, Groq, Grok, Azure, Ollama (OpenAI-compatible format)
tools_list = registry.to_openai_format()
# Returns: [{"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}]

# For Anthropic/Claude
tools_list = registry.to_anthropic_format()
# Returns: [{"name": ..., "description": ..., "input_schema": ...}]

# Single tool
td.to_openai_format()
td.to_anthropic_format()
```

### Conversation Integration

```python
from prompture import Conversation, ToolRegistry

registry = ToolRegistry()

@registry.tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Sunny in {city}"

# Pass tools at construction
conv = Conversation(driver=driver, tools=registry)
result = conv.ask("What's the weather in Paris?")

# Or register after construction
conv = Conversation(driver=driver)
conv.register_tool(get_weather)
```

The conversation automatically:
1. Sends tools to the LLM via `generate_messages_with_tools()`
2. Parses `tool_calls` from the response
3. Executes the tool function with the provided arguments
4. Sends the result back to the LLM
5. Repeats until the LLM returns a final text response (or `max_tool_rounds` is hit)

### Execution and Error Handling

```python
# Direct execution
result = registry.execute("get_weather", {"city": "Paris"})

# Missing tool raises KeyError
registry.execute("nonexistent", {})  # KeyError: "Tool not registered: 'nonexistent'"

# Tool errors during conversation are caught and sent back as error strings
# The LLM sees: "Error: <exception message>" and can respond accordingly
```

## Implementation Steps

### 1. Define the tool function

```python
def my_tool(param1: str, param2: int = 10) -> str:
    """Clear, concise description — this becomes the LLM's tool description.

    The first line of the docstring is used as the description.
    """
    # Implementation
    return json.dumps({"result": "value"})
```

**Rules:**
- Always include type hints — they generate the JSON Schema
- Always include a docstring — first line becomes the tool description
- Return a string — this is what the LLM sees as the tool result
- For complex return data, return `json.dumps(result)`

### 2. Register in a ToolRegistry

```python
registry = ToolRegistry()
registry.register(my_tool)
```

### 3. Use with a Conversation

```python
conv = Conversation(
    driver=driver,          # Must have supports_tool_use = True
    tools=registry,
    max_tool_rounds=3,      # Safety limit (default: 3)
)
```

### 4. Write tests

```python
def test_my_tool_schema():
    td = tool_from_function(my_tool)
    assert td.name == "my_tool"
    assert "param1" in td.parameters["properties"]
    assert td.parameters["required"] == ["param1"]

def test_my_tool_execution():
    reg = ToolRegistry()
    reg.register(my_tool)
    result = reg.execute("my_tool", {"param1": "test"})
    assert "result" in result
```

## Drivers with Tool Use Support

The following drivers support `generate_messages_with_tools()`:
- **OpenAI** (`openai_driver.py`, `async_openai_driver.py`)
- **Claude** (`claude_driver.py`, `async_claude_driver.py`)
- **Groq** (`groq_driver.py`, `async_groq_driver.py`)
- **Grok/xAI** (`grok_driver.py`, `async_grok_driver.py`)
- **Azure** (`azure_driver.py`, `async_azure_driver.py`)
- **Ollama** (`ollama_driver.py`, `async_ollama_driver.py`)
- **Moonshot** (`moonshot_driver.py`, `async_moonshot_driver.py`)
- **OpenRouter** (`openrouter_driver.py`, `async_openrouter_driver.py`)

Check `driver.supports_tool_use` before using tools with a driver.

## Best Practices

- **Keep tools focused** — one function per action, not a kitchen-sink tool
- **Use descriptive docstrings** — the LLM relies on the description to decide when to call tools
- **Type hints are schema** — always annotate parameters; they generate the JSON Schema the LLM sees
- **Return strings** — tool results are injected into the conversation as text
- **Handle errors gracefully** — exceptions are caught and sent to the LLM as `"Error: ..."` messages
- **Set `max_tool_rounds`** — prevent infinite tool-calling loops
- **Test schemas** — use `tool_from_function()` + assertions to verify the generated schema matches expectations
