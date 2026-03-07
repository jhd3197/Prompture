# Installation

## Requirements

- Python 3.9 or higher
- pip (Python package installer)

## Installing from PyPI

```bash
pip install prompture
```

### Optional extras

```bash
pip install prompture[redis]     # Redis cache backend
pip install prompture[serve]     # FastAPI server mode
pip install prompture[airllm]    # AirLLM local inference
pip install prompture[toon]      # TOON input conversion
pip install prompture[ingest]    # Document ingestion (PDF, DOCX, etc.)
pip install prompture[all]       # Everything
```

## Installing from Source

```bash
git clone https://github.com/jhd3197/prompture.git
cd prompture
pip install -e .
```

## Environment Configuration

Create a `.env` file in your project directory:

```bash
# OpenAI
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Google AI
GOOGLE_API_KEY=your_google_api_key_here

# Groq
GROQ_API_KEY=your_groq_api_key_here

# Azure OpenAI
AZURE_OPENAI_API_KEY=your_azure_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/

# Local models
OLLAMA_BASE_URL=http://localhost:11434
```

## Verification

```python
import prompture
print(f"Prompture version: {prompture.__version__}")

from prompture import extract_and_jsonify, field_from_registry
print("Prompture installed successfully!")
```

## Development Installation

```bash
git clone https://github.com/jhd3197/prompture.git
cd prompture
pip install -e ".[dev]"
pytest tests/
```
