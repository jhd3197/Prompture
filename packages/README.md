# llm-to-json

Tiny wrapper around `prompture` with a minimal, easy-to-use API for converting LLM output (or raw text) into JSON according to a schema.

Install:
```bash
pip install llm-to-json
```

Usage:

```python
from llm_to_json import from_llm_text
schema = {"name": "string", "age": "int"}
print(from_llm_text("Name: Juan Age: 30", schema))
```

For full docs and advanced features, see the main project: Prompture — https://github.com/jhd3197/prompture