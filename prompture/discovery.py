"""Discovery module for auto-detecting available models."""

import logging
import os

import requests

from .drivers import (
    AzureDriver,
    ClaudeDriver,
    GoogleDriver,
    GrokDriver,
    GroqDriver,
    LMStudioDriver,
    LocalHTTPDriver,
    OllamaDriver,
    OpenAIDriver,
    OpenRouterDriver,
)
from .settings import settings

logger = logging.getLogger(__name__)


def get_available_models() -> list[str]:
    """
    Auto-detects all available models based on configured drivers and environment variables.

    Iterates through supported providers and checks if they are configured (e.g. API key present).
    For static drivers, returns models from their MODEL_PRICING keys.
    For dynamic drivers (like Ollama), attempts to fetch available models from the endpoint.

    Returns:
        A list of unique model strings in the format "provider/model_id".
    """
    available_models: set[str] = set()
    configured_providers: set[str] = set()

    # Map of provider name to driver class
    # We need to map the registry keys to the actual classes to check MODEL_PRICING
    # and instantiate for dynamic checks if needed.
    provider_classes = {
        "openai": OpenAIDriver,
        "azure": AzureDriver,
        "claude": ClaudeDriver,
        "google": GoogleDriver,
        "groq": GroqDriver,
        "openrouter": OpenRouterDriver,
        "grok": GrokDriver,
        "ollama": OllamaDriver,
        "lmstudio": LMStudioDriver,
        "local_http": LocalHTTPDriver,
    }

    for provider, driver_cls in provider_classes.items():
        try:
            # 1. Check if the provider is configured (has API key or endpoint)
            # We can check this by looking at the settings or env vars that the driver uses.
            # A simple way is to try to instantiate it with defaults, but that might fail if keys are missing.
            # Instead, let's check the specific requirements for each known provider.

            is_configured = False

            if provider == "openai":
                if settings.openai_api_key or os.getenv("OPENAI_API_KEY"):
                    is_configured = True
            elif provider == "azure":
                if (
                    (settings.azure_api_key or os.getenv("AZURE_API_KEY"))
                    and (settings.azure_api_endpoint or os.getenv("AZURE_API_ENDPOINT"))
                    and (settings.azure_deployment_id or os.getenv("AZURE_DEPLOYMENT_ID"))
                ):
                    is_configured = True
            elif provider == "claude":
                if settings.claude_api_key or os.getenv("CLAUDE_API_KEY"):
                    is_configured = True
            elif provider == "google":
                if settings.google_api_key or os.getenv("GOOGLE_API_KEY"):
                    is_configured = True
            elif provider == "groq":
                if settings.groq_api_key or os.getenv("GROQ_API_KEY"):
                    is_configured = True
            elif provider == "openrouter":
                if settings.openrouter_api_key or os.getenv("OPENROUTER_API_KEY"):
                    is_configured = True
            elif provider == "grok":
                if settings.grok_api_key or os.getenv("GROK_API_KEY"):
                    is_configured = True
            elif provider == "ollama":
                # Ollama is always considered "configured" as it defaults to localhost
                # We will check connectivity later
                is_configured = True
            elif provider == "lmstudio":
                # LM Studio is similar to Ollama, defaults to localhost
                is_configured = True
            elif provider == "local_http" and (settings.local_http_endpoint or os.getenv("LOCAL_HTTP_ENDPOINT")):
                is_configured = True

            if not is_configured:
                continue

            configured_providers.add(provider)

            # 2. Static Detection: Get models from MODEL_PRICING
            if hasattr(driver_cls, "MODEL_PRICING"):
                pricing = driver_cls.MODEL_PRICING
                for model_id in pricing:
                    # Skip "default" or generic keys if they exist
                    if model_id == "default":
                        continue

                    # For Azure, the model_id in pricing is usually the base model name,
                    # but the user needs to use the deployment ID.
                    # However, our Azure driver implementation uses the deployment_id from init
                    # as the "model" for the request, but expects the user to pass a model name
                    # that maps to pricing?
                    # Looking at AzureDriver:
                    # kwargs = {"model": self.deployment_id, ...}
                    # model = options.get("model", self.model) -> used for pricing lookup
                    # So we should list the keys in MODEL_PRICING as available "models"
                    # even though for Azure specifically it's a bit weird because of deployment IDs.
                    # But for general discovery, listing supported models is correct.

                    available_models.add(f"{provider}/{model_id}")

            # 3. Dynamic Detection: Specific logic for Ollama
            if provider == "ollama":
                try:
                    endpoint = settings.ollama_endpoint or os.getenv(
                        "OLLAMA_ENDPOINT", "http://localhost:11434/api/generate"
                    )
                    # We need the base URL for tags, usually http://localhost:11434/api/tags
                    # The configured endpoint might be .../api/generate or .../api/chat
                    base_url = endpoint.split("/api/")[0]
                    tags_url = f"{base_url}/api/tags"

                    resp = requests.get(tags_url, timeout=2)
                    if resp.status_code == 200:
                        data = resp.json()
                        models = data.get("models", [])
                        for model in models:
                            name = model.get("name")
                            if name:
                                # Ollama model names often include tags like "llama3:latest"
                                # We can keep them as is.
                                available_models.add(f"ollama/{name}")
                except Exception as e:
                    logger.debug(f"Failed to fetch Ollama models: {e}")

            # Future: Add dynamic detection for LM Studio if they have an endpoint for listing models

        except Exception as e:
            logger.warning(f"Error detecting models for provider {provider}: {e}")
            continue

    # Enrich with live model list from models.dev cache
    from .model_rates import PROVIDER_MAP, get_all_provider_models

    for prompture_name, api_name in PROVIDER_MAP.items():
        if prompture_name in configured_providers:
            for model_id in get_all_provider_models(api_name):
                available_models.add(f"{prompture_name}/{model_id}")

    return sorted(list(available_models))
