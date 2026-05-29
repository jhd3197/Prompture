"""Display-only branding metadata for known providers.

This is intentionally a flat data module rather than a field on
:class:`ProviderDescriptor` so plugins don't have to know or care about
display concerns — they can register a driver without ever touching
this file, and an unknown provider just gets a ``None`` brand.

Logo serving
------------
We don't ship binary assets. Each known provider gets an ``icon_slug``
that resolves at https://cdn.simpleicons.org/{slug} (or
https://cdn.simpleicons.org/{slug}/{hex} for tinted variants). Consumers
that prefer to self-host should look the slug up against the upstream
simple-icons package.

Schema
------
Every provider listed here has:

- ``display_name`` — human label, e.g. ``"OpenAI"`` vs the registry's ``"openai"``.
- ``icon_slug``    — slug at https://simpleicons.org (or ``None`` if no good match).
- ``brand_color``  — official-ish hex without the leading ``#``.
- ``is_local``     — true for providers that run on the operator's machine
                     (Ollama, LM Studio, local_http) — UIs surface this as
                     a "local" badge.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderBrand:
    display_name: str
    icon_slug: str | None
    brand_color: str | None
    is_local: bool = False


# Hand-curated. Keys must match the canonical provider name registered
# in :mod:`prompture.drivers.provider_descriptors`.
PROVIDER_BRANDS: dict[str, ProviderBrand] = {
    # ── frontier model labs ──
    "openai": ProviderBrand("OpenAI", "openai", "74AA9C"),
    "anthropic": ProviderBrand("Anthropic", "anthropic", "D97757"),
    "claude": ProviderBrand("Anthropic Claude", "anthropic", "D97757"),
    "google": ProviderBrand("Google Gemini", "googlegemini", "8E75B2"),
    "google-gemini": ProviderBrand("Google Gemini", "googlegemini", "8E75B2"),

    # ── inference platforms ──
    "groq": ProviderBrand("Groq", "groq", "F55036"),
    "openrouter": ProviderBrand("OpenRouter", "openrouter", "6566F1"),
    "huggingface": ProviderBrand("Hugging Face", "huggingface", "FFD21E"),
    "modelscope": ProviderBrand("ModelScope", None, "2F8AC4"),

    # ── enterprise / cloud ──
    "azure": ProviderBrand("Azure OpenAI", "microsoftazure", "0078D4"),
    "azure-openai": ProviderBrand("Azure OpenAI", "microsoftazure", "0078D4"),
    "bedrock": ProviderBrand("AWS Bedrock", "amazonaws", "FF9900"),
    "vertex": ProviderBrand("Vertex AI", "googlecloud", "4285F4"),

    # ── specialised ──
    "grok": ProviderBrand("xAI Grok", "x", "000000"),
    "xai": ProviderBrand("xAI Grok", "x", "000000"),
    "moonshot": ProviderBrand("Moonshot Kimi", None, "1E40AF"),
    "kimi": ProviderBrand("Moonshot Kimi", None, "1E40AF"),
    "zai": ProviderBrand("Z.AI", None, "10B981"),
    "z-ai": ProviderBrand("Z.AI", None, "10B981"),
    "cohere": ProviderBrand("Cohere", "cohere", "39594D"),
    "mistral": ProviderBrand("Mistral", None, "FF7000"),

    # ── multimodal ──
    "runway": ProviderBrand("Runway", "runway", "000000"),
    "runwayml": ProviderBrand("Runway", "runway", "000000"),
    "elevenlabs": ProviderBrand("ElevenLabs", "elevenlabs", "000000"),
    "stability": ProviderBrand("Stability AI", "stabilityai", "8A2BE2"),

    # ── local ──
    "ollama": ProviderBrand("Ollama", "ollama", "000000", is_local=True),
    "lmstudio": ProviderBrand("LM Studio", None, "6366F1", is_local=True),
    "lm-studio": ProviderBrand("LM Studio", None, "6366F1", is_local=True),
    "local_http": ProviderBrand("Local HTTP", None, "64748B", is_local=True),
    "airllm": ProviderBrand("AirLLM", None, "64748B", is_local=True),

    # ── prompture's own helpers ──
    "cachibot": ProviderBrand("CachiBot Router", None, "10B981"),
}


def get_provider_brand(name: str) -> ProviderBrand | None:
    """Return brand info for ``name`` (case-insensitive) or ``None`` if unknown.

    Unknown providers should fall back to a generic visual treatment
    (initials avatar etc.) rather than 404.
    """
    if not name:
        return None
    return PROVIDER_BRANDS.get(name.strip().lower())


def icon_url(brand: ProviderBrand | None) -> str | None:
    """CDN URL for the brand icon, or ``None`` if no slug is set."""
    if brand is None or brand.icon_slug is None:
        return None
    return f"https://cdn.simpleicons.org/{brand.icon_slug}"
