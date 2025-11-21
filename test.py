#!/usr/bin/env python3
import argparse
import os
import sys
from typing import Dict, List, Tuple
from dotenv import load_dotenv
from pathlib import Path
import re


# Add src directory to path for prompture package imports
sys.path.append('src')

# Load environment variables from .env file
env_path = Path('.env')
load_dotenv(env_path)
print(f"\nEnvironment Configuration:")
print(f"Current directory: {os.getcwd()}")
print(f".env file path: {env_path.absolute()}")
print(f"Loading environment from: {env_path}\n")

VALID_PROVIDERS = [
    'openai', 
    'ollama',
    'claude',
    'azure',
    'lmstudio',
    'hugging',
    'local_http'
]

YELLOW = "\033[33m"
RESET = "\033[0m"

PROVIDER_REQUIREMENTS: Dict[str, List[str]] = {
    'openai': ['OPENAI_API_KEY'],
    'ollama': ['OLLAMA_ENDPOINT'],
    'claude': ['CLAUDE_API_KEY'],
    'azure': ['AZURE_API_KEY', 'AZURE_API_ENDPOINT', 'AZURE_DEPLOYMENT_ID'],
    'lmstudio': ['LMSTUDIO_ENDPOINT'],
    'hugging': ['HF_ENDPOINT', 'HF_TOKEN'],
    'local_http': ['LOCAL_HTTP_ENDPOINT']
}

def get_provider_from_model(model: str) -> str:
    """Extract provider from model string.
    
    Format: provider/model:tag (e.g. ollama/gpt-oss:20b)
    """
    provider = model.split('/', 1)[0]
    return provider.lower()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Prompture Test Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Environment Variables:
  TEST_SKIP_NO_CREDENTIALS If true, skips integration tests when credentials missing

Provider-Specific Environment Variables:
  OpenAI:   OPENAI_API_KEY
  Ollama:   OLLAMA_ENDPOINT
  Claude:   CLAUDE_API_KEY
  Azure:    AZURE_API_KEY, AZURE_API_ENDPOINT, AZURE_DEPLOYMENT_ID
  HuggingFace: HF_ENDPOINT, HF_TOKEN
  Local HTTP: LOCAL_HTTP_ENDPOINT

Examples:
  # Run all tests
  python test.py

  # Skip integration tests when credentials missing
  TEST_SKIP_NO_CREDENTIALS=true python test.py
        '''
    )
    
    parser.add_argument(
        '--skip-no-creds',
        action='store_true',
        help='Skip integration tests if credentials missing (overrides TEST_SKIP_NO_CREDENTIALS)'
    )
    
    parser.add_argument(
        'pytest_args',
        nargs='*',
        help='Additional arguments to pass to pytest'
    )
    
    return parser.parse_args()

def validate_provider_credentials(provider: str) -> bool:
    """Check if all required credentials for a provider exist."""
    required_vars = PROVIDER_REQUIREMENTS.get(provider, [])
    return all(os.getenv(var) for var in required_vars)

def configure_test_environment(args: argparse.Namespace) -> None:
    """Configure the test environment."""
    from tests.conftest import DEFAULT_MODEL
    
    provider = get_provider_from_model(DEFAULT_MODEL)
    if provider not in VALID_PROVIDERS:
        print(f"Error: Invalid provider '{provider}' in DEFAULT_MODEL. Must be one of: {', '.join(VALID_PROVIDERS)}")
        sys.exit(1)
    
    # Print diagnostic information
    print("\nTest Configuration:")
    print(f"Using DEFAULT_MODEL: {DEFAULT_MODEL}")
    print("Environment Variables:")
    for var in PROVIDER_REQUIREMENTS.get(provider, []):
        value = os.getenv(var)
        masked_value = '***' if value else 'Not Set'
        print(f"  {var}: {masked_value}")
    print()
    server_hints = {
        "ollama": ("Ollama", os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")),
        "lmstudio": ("LM Studio", os.getenv("LMSTUDIO_ENDPOINT", "http://localhost:1234")),
    }
    if provider in server_hints:
        server_name, endpoint = server_hints[provider]
        print(
            f"{YELLOW}⚠️  Make sure your {server_name} server is running and reachable at {endpoint}.\n"
            "   Integration tests will fail if the local inference server is stopped." 
            f"{RESET}"
        )
        print()

    # Check credentials
    has_creds = validate_provider_credentials(provider)
    
    # Handle missing credentials
    if not has_creds:
        skip_no_creds = (
            args.skip_no_creds or
            os.getenv('TEST_SKIP_NO_CREDENTIALS', 'true').lower() == 'true'
        )
        
        missing_vars = [
            var for var in PROVIDER_REQUIREMENTS[provider]
            if not os.getenv(var)
        ]
        print(f"Warning: Missing required credentials for {provider}: {', '.join(missing_vars)}")
        
        if skip_no_creds:
            print("Skipping integration tests due to missing credentials")
            os.environ['TEST_SKIP_NO_CREDENTIALS'] = 'true'
        else:
            print("Error: Provider credentials missing and skip tests not enabled")
            sys.exit(1)


def read_default_model_from_conftest() -> str:
    path = Path('tests') / 'conftest.py'
    text = path.read_text(encoding='utf-8')
    m = re.search(r"DEFAULT_MODEL\s*=\s*['\"]([^'\"]+)['\"]", text)
    if not m:
        raise RuntimeError("Couldn't locate DEFAULT_MODEL in tests/conftest.py")
    return m.group(1)

def configure_test_environment_from_model(model: str, args):
    provider = get_provider_from_model(model)
    if provider not in VALID_PROVIDERS:
        print(f"Error: Invalid provider '{provider}' in DEFAULT_MODEL.")
        sys.exit(1)
    # print diagnostics and credential checks same as before, but take `model` param

def main() -> int:
    args = parse_args()

    # Import pytest only now, inside main, before any test module import
    import pytest

    try:
        DEFAULT_MODEL = read_default_model_from_conftest()
        configure_test_environment_from_model(DEFAULT_MODEL, args)

        # safe to run pytest.main() now because pytest is loaded and its import hooks installed
        return pytest.main(args.pytest_args)

    except Exception as e:
        print(f"Error running tests: {e}")
        return 1
if __name__ == '__main__':
    sys.exit(main())
