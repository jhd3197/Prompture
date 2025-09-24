from .mock_driver import MockDriver
from .openai_driver import OpenAIDriver
from .local_http_driver import LocalHTTPDriver
from .ollama_driver import OllamaDriver
from .claude_driver import ClaudeDriver
from .azure_driver import AzureDriver
from ..settings import settings

def get_driver(provider_name: str = None):
    """
    Factory to get a driver instance based on the provider name.
    
    The provider name can come from:
    - Explicit argument (provider_name)
    - Environment variable (AI_PROVIDER) via settings.ai_provider
    
    Case-insensitive: "Ollama", "ollama", "OLLAMA" are treated the same.
    """
    provider = (provider_name or settings.ai_provider or "").strip().lower()

    if provider == "mock":
        return MockDriver()
    if provider == "openai":
        return OpenAIDriver(api_key=settings.openai_api_key, model=settings.openai_model)
    if provider == "local_http":
        return LocalHTTPDriver(endpoint=settings.hf_endpoint)
    if provider == "ollama":
        return OllamaDriver(endpoint=settings.ollama_endpoint, model=settings.ollama_model)
    if provider == "claude":
        return ClaudeDriver(api_key=settings.claude_api_key, model=settings.claude_model)
    if provider == "azure":
        return AzureDriver(
            api_key=settings.azure_api_key,
            endpoint=settings.azure_api_endpoint,
            deployment_id=settings.azure_deployment_id
        )

    raise ValueError(f"Unknown provider: {provider_name}")

__all__ = [
    "MockDriver",
    "OpenAIDriver",
    "LocalHTTPDriver",
    "OllamaDriver",
    "ClaudeDriver",
    "AzureDriver",
    "get_driver",
]
