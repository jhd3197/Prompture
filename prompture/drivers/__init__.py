from .openai_driver import OpenAIDriver
from .local_http_driver import LocalHTTPDriver
from .ollama_driver import OllamaDriver
from .claude_driver import ClaudeDriver
from .azure_driver import AzureDriver
from .lmstudio_driver import LMStudioDriver
from ..settings import settings


# Central registry: maps provider → factory function
DRIVER_REGISTRY = {
    "openai": lambda model=None: OpenAIDriver(
        api_key=settings.openai_api_key,
        model=model or settings.openai_model
    ),
    "ollama": lambda model=None: OllamaDriver(
        endpoint=settings.ollama_endpoint,
        model=model or settings.ollama_model
    ),
    "claude": lambda model=None: ClaudeDriver(
        api_key=settings.claude_api_key,
        model=model or settings.claude_model
    ),
    "lmstudio": lambda model=None: LMStudioDriver(
        endpoint=settings.lmstudio_endpoint,
        model=model or settings.lmstudio_model
    ),
    "azure": lambda model=None: AzureDriver(
        api_key=settings.azure_api_key,
        endpoint=settings.azure_api_endpoint,
        deployment_id=settings.azure_deployment_id
    ),
    "local_http": lambda model=None: LocalHTTPDriver(
        endpoint=settings.local_http_endpoint,
        model=model
    ),
}


def get_driver(provider_name: str = None):
    """
    Factory to get a driver instance based on the provider name (legacy style).
    Uses default model from settings if not overridden.
    """
    provider = (provider_name or settings.ai_provider or "ollama").strip().lower()
    if provider not in DRIVER_REGISTRY:
        raise ValueError(f"Unknown provider: {provider_name}")
    return DRIVER_REGISTRY[provider]()  # use default model from settings


def get_driver_for_model(model_str: str):
    """
    Factory to get a driver instance based on a full model string.
    Format: provider/model_id
    Example: "openai/gpt-4-turbo-preview"
    """
    try:
        provider, model_id = model_str.split("/", 1)
    except ValueError:
        raise ValueError(
            f"Invalid model string '{model_str}'. Expected format 'provider/model'."
        )

    provider = provider.lower()
    if provider not in DRIVER_REGISTRY:
        raise ValueError(f"Unsupported provider '{provider}' in model string '{model_str}'.")

    return DRIVER_REGISTRY[provider](model_id)


__all__ = [
    "OpenAIDriver",
    "LocalHTTPDriver",
    "OllamaDriver",
    "ClaudeDriver",
    "LMStudioDriver",
    "AzureDriver",
    "get_driver",
    "get_driver_for_model",
]
