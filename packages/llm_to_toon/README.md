# llm-to-toon

Tiny wrapper around `prompture` that returns [TOON](https://github.com/jmorganca/python-toon)
(Token-Oriented Object Notation) instead of JSON. Under the hood it uses
`prompture.extract_and_jsonify(..., output_format="toon")` and converts the result
into the ultra-compact TOON text automatically.

Install:
```bash
pip install llm-to-toon
```

Usage:

```python
from llm_to_toon import from_llm_text

schema = {"name": "string", "age": "int"}
toon_text = from_llm_text("Name: Juan Age: 30", schema)
print(toon_text)
```

By default the helper spins up the local Ollama driver (`gemma:latest`). Pass your
own Prompture driver if you want to call OpenAI, Azure, Groq, etc. For the full
Prompture feature-set see the main project: https://github.com/jhd3197/prompture
