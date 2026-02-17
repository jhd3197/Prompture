# Contributing to Prompture

Thanks for your interest in contributing!

## Development Setup

```bash
git clone https://github.com/jhd3197/prompture.git
cd prompture
pip install -e ".[dev,all]"
```

## Running Tests

```bash
# Unit tests only (no API keys required)
pytest

# Integration tests (requires provider API keys in .env)
pytest --run-integration
```

## Linting

```bash
ruff check .
ruff format .
```

## Submitting Changes

1. Fork the repository and create a feature branch from `dev`.
2. Add tests for any new functionality.
3. Ensure `pytest` and `ruff check .` pass.
4. Open a pull request against the `dev` branch with a clear description of your changes.
