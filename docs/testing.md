# Testing Prompture with Different AI Providers

This guide explains how to test Prompture with various AI providers using our environment-variable based testing approach.

## Table of Contents
- [Overview](#overview)
- [General Testing Setup](#general-testing-setup)
- [Provider-Specific Configuration](#provider-specific-configuration)
  - [OpenAI](#openai)
  - [Ollama](#ollama)
  - [Claude](#claude)
  - [Azure](#azure)
  - [Mock Provider](#mock-provider)
- [Advanced Testing Topics](#advanced-testing-topics)
  - [Skipping Tests](#skipping-tests)
  - [Using Mock Fallbacks](#using-mock-fallbacks)
  - [Environment Variable Overrides](#environment-variable-overrides)
  - [Testing with Different Models](#testing-with-different-models)

## Overview

Prompture uses an environment-variable based approach for testing with different AI providers. The primary control variable is `AI_PROVIDER`, which determines which provider to use for tests.

## General Testing Setup

1. Copy the example environment file:
   ```bash
   cp .env.copy .env
   ```

2. Set the AI provider in your .env file:
   ```
   AI_PROVIDER=openai  # or ollama, claude, azure, mock
   ```

3. Run tests:
   ```bash
   pytest tests/
   ```

## Provider-Specific Configuration

### OpenAI

Required Environment Variables:
```env
AI_PROVIDER=openai
OPENAI_API_KEY=your-api-key
OPENAI_MODEL=gpt-4-turbo-preview  # optional, defaults to gpt-4-turbo-preview
```

Example Commands:
```bash
# Run all tests with OpenAI
pytest tests/

# Run specific test file
pytest tests/test_drivers.py

# Run with output
pytest -v tests/
```

Troubleshooting:
- Ensure your API key has sufficient credits
- Check model availability for your account
- Verify network connectivity to OpenAI's API

### Ollama

Required Environment Variables:
```env
AI_PROVIDER=ollama
OLLAMA_ENDPOINT=http://localhost:11434/api/generate
OLLAMA_MODEL=gemma:latest  # or any other installed model
```

Example Commands:
```bash
# Start Ollama server first
ollama serve

# In another terminal, run tests
pytest tests/
```

Troubleshooting:
- Ensure Ollama server is running
- Verify model is downloaded (`ollama list`)
- Check server endpoint accessibility

### Claude

Required Environment Variables:
```env
AI_PROVIDER=claude
CLAUDE_API_KEY=your-api-key
CLAUDE_MODEL_NAME=claude-3-sonnet-20240229  # optional
```

Example Commands:
```bash
# Run all tests with Claude
pytest tests/

# Run with detailed output
pytest -v tests/
```

Troubleshooting:
- Verify API key validity
- Check model availability
- Ensure sufficient API credits

### Azure

Required Environment Variables:
```env
AI_PROVIDER=azure
AZURE_API_KEY=your-api-key
AZURE_API_ENDPOINT=your-endpoint
AZURE_DEPLOYMENT_ID=your-deployment
AZURE_API_VERSION=2023-07-01-preview  # optional
```

Example Commands:
```bash
# Run all tests
pytest tests/

# Run specific test class
pytest tests/test_drivers.py::TestAzureDriver
```

Troubleshooting:
- Verify all Azure credentials
- Check deployment status
- Ensure endpoint is accessible

### Mock Provider

The mock provider is useful for testing without API access:

```env
AI_PROVIDER=mock
```

Example Commands:
```bash
# Run with mock provider
pytest tests/

# Run specific test
pytest tests/test_drivers.py::TestMockDriver
```

## Advanced Testing Topics

### Skipping Tests

To skip integration tests:
```bash
pytest -m "not integration" tests/
```

### Using Mock Fallbacks

Tests automatically fall back to mock provider if:
- No AI_PROVIDER is set
- Required credentials are missing
- Network/API issues occur

### Environment Variable Overrides

Override from command line:
```bash
AI_PROVIDER=openai OPENAI_MODEL=gpt-4 pytest tests/
```

### Testing with Different Models

Each provider supports model configuration:

```env
# OpenAI
OPENAI_MODEL=gpt-4-turbo-preview

# Ollama
OLLAMA_MODEL=llama2

# Claude
CLAUDE_MODEL_NAME=claude-3-sonnet-20240229

# Azure
AZURE_DEPLOYMENT_ID=your-model-deployment
```

You can override models per test:
```python
@pytest.mark.integration
def test_with_specific_model(integration_driver):
    result = integration_driver.generate("test", {"model": "specific-model"})