# Quick Start

## Basic Setup

Make sure you have your API keys configured in a `.env` file:

```bash
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

## Your First Extraction

```python
from prompture import extract_and_jsonify

fields = {
    "name": "string",
    "age": "integer",
    "occupation": "string"
}

text = "Sarah Johnson is a 32-year-old software engineer at TechCorp."

result = extract_and_jsonify(
    prompt=text,
    fields=fields,
    model_name="openai/gpt-3.5-turbo"
)

print(result)
# {"name": "Sarah Johnson", "age": 32, "occupation": "software engineer"}
```

## Using Pydantic Models (Recommended)

```python
from pydantic import BaseModel
from typing import Optional
from prompture import field_from_registry, extract_with_model

class Person(BaseModel):
    name: str = field_from_registry("name")
    age: int = field_from_registry("age")
    email: Optional[str] = field_from_registry("email")
    occupation: Optional[str] = field_from_registry("occupation")

text = "Dr. Alice Smith, 45, is a cardiologist. Email: alice@hospital.com"

result = extract_with_model(
    model_class=Person,
    prompt=text,
    model_name="openai/gpt-4"
)

print(f"Name: {result.name}, Age: {result.age}")
```

## Custom Field Definitions

```python
from prompture import register_field, field_from_registry, extract_with_model
from pydantic import BaseModel
from typing import List, Optional

register_field("skills", {
    "type": "list",
    "description": "List of professional skills and competencies",
    "instructions": "Extract skills as a list of strings",
    "default": [],
    "nullable": True
})

class Professional(BaseModel):
    name: str = field_from_registry("name")
    skills: Optional[List[str]] = field_from_registry("skills")
    occupation: Optional[str] = field_from_registry("occupation")

text = """
Michael Chen has 8 years of experience as a data scientist.
His skills include Python, machine learning, SQL, and data visualization.
"""

result = extract_with_model(
    model_class=Professional,
    prompt=text,
    model_name="openai/gpt-4"
)

print(f"Skills: {', '.join(result.skills)}")
```

## Different LLM Providers

Simply change the `model_name` parameter:

```python
# OpenAI
result = extract_and_jsonify(text, fields, model_name="openai/gpt-4")

# Anthropic Claude
result = extract_and_jsonify(text, fields, model_name="anthropic/claude-3-haiku-20240307")

# Google Gemini
result = extract_and_jsonify(text, fields, model_name="google/gemini-pro")

# Groq (fast inference)
result = extract_and_jsonify(text, fields, model_name="groq/llama2-70b-4096")

# Local models via Ollama
result = extract_and_jsonify(text, fields, model_name="ollama/llama2")
```

## Configuration Tips

- **Model Selection**: Use `gpt-3.5-turbo` for fast/cheap, `gpt-4` for complex tasks, local models for privacy.
- **Field Definitions**: Use built-in fields when possible. Register custom fields for domain-specific data.
- **Error Handling**: Always wrap extraction calls in try-catch blocks.
