"""Per-consumer provider environment for isolated API keys and endpoints."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProviderEnvironment:
    """Bundle of provider keys/endpoints for one consumer (e.g. one bot).

    Fields default to ``None``, meaning "use the global settings singleton".
    Pass explicit values to override specific providers per-consumer.
    """

    openai_api_key: str | None = None
    claude_api_key: str | None = None
    google_api_key: str | None = None
    groq_api_key: str | None = None
    grok_api_key: str | None = None
    openrouter_api_key: str | None = None
    moonshot_api_key: str | None = None
    moonshot_endpoint: str | None = None
    zhipu_api_key: str | None = None
    zhipu_endpoint: str | None = None
    modelscope_api_key: str | None = None
    modelscope_endpoint: str | None = None
    azure_api_key: str | None = None
    azure_api_endpoint: str | None = None
    azure_deployment_id: str | None = None
    ollama_endpoint: str | None = None
    lmstudio_endpoint: str | None = None
    lmstudio_api_key: str | None = None
    cachibot_api_key: str | None = None
    cachibot_endpoint: str | None = None
    stability_api_key: str | None = None
    elevenlabs_api_key: str | None = None
    hf_endpoint: str | None = None
    hf_token: str | None = None

    def resolve(self, env_attr: str, settings_attr: str | None = None) -> str | None:
        """Return the env value if set, else fall back to global settings.

        Args:
            env_attr: Attribute name on this ProviderEnvironment.
            settings_attr: Attribute name on the global settings singleton.
                If ``None``, uses *env_attr* for both.

        Returns:
            The resolved value, or ``None`` if neither is set.
        """
        from .settings import settings

        val = getattr(self, env_attr, None)
        if val is not None:
            return val  # type: ignore[no-any-return]
        return getattr(settings, settings_attr or env_attr, None)
