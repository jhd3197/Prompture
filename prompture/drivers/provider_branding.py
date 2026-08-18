"""Display-only branding metadata for known providers.

This is intentionally a flat data module rather than a field on
:class:`ProviderDescriptor` so plugins don't have to know or care about
display concerns — they can register a driver without ever touching
this file, and an unknown provider just gets a ``None`` brand.

Logo serving
------------
We don't ship binary assets. :func:`icon_url` returns the best available
URL using a two-tier strategy:

1. **simple-icons CDN** (``https://cdn.simpleicons.org/{slug}``) — used
   when an ``icon_slug`` is set. Best quality but coverage is shaky:
   simple-icons has dropped several major brand logos over time
   (OpenAI, Groq, Azure, AWS, Cohere, Stability, Runway as of late 2025)
   due to trademark policy disputes.
2. **Google favicon service**
   (``https://www.google.com/s2/favicons?sz=64&domain={domain}``) — used
   when only a ``domain`` is set. Works for nearly any brand with a
   website. Quality varies (sometimes the brand mark, sometimes a
   wordmark crop).

Set both for resilience: if a CDN entry is later removed, the favicon
fallback keeps the UI looking right.

Schema
------
- ``display_name`` — human label, e.g. ``"OpenAI"`` vs the registry's ``"openai"``.
- ``icon_slug``    — slug at https://simpleicons.org (or ``None`` if no good match).
- ``brand_color``  — official-ish hex without the leading ``#``.
- ``is_local``     — true for providers that run on the operator's machine.
- ``domain``       — bare domain (no protocol) used for the favicon fallback.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderBrand:
    display_name: str
    icon_slug: str | None
    brand_color: str | None
    is_local: bool = False
    domain: str | None = None


# Hand-curated. Keys must match the canonical provider name registered
# in :mod:`prompture.drivers.provider_descriptors` (lowercased on lookup).
#
# Slug coverage is verified against the live simple-icons CDN — entries
# that 404 are listed as ``icon_slug=None`` so the favicon path picks
# them up instead. Update by re-running ``curl`` against the slugs you
# add.
PROVIDER_BRANDS: dict[str, ProviderBrand] = {
    # ── frontier model labs ──
    "openai": ProviderBrand(
        "OpenAI", icon_slug=None, brand_color="000000", domain="openai.com",
    ),
    "anthropic": ProviderBrand(
        "Anthropic", icon_slug="anthropic", brand_color="D97757",
        domain="anthropic.com",
    ),
    "claude": ProviderBrand(
        "Anthropic Claude", icon_slug="anthropic", brand_color="D97757",
        domain="anthropic.com",
    ),
    "google": ProviderBrand(
        "Google Gemini", icon_slug="googlegemini", brand_color="8E75B2",
        domain="gemini.google.com",
    ),
    "google-gemini": ProviderBrand(
        "Google Gemini", icon_slug="googlegemini", brand_color="8E75B2",
        domain="gemini.google.com",
    ),
    "google_vertexai": ProviderBrand(
        "Vertex AI", icon_slug="googlecloud", brand_color="4285F4",
        domain="cloud.google.com",
    ),
    "vertex": ProviderBrand(
        "Vertex AI", icon_slug="googlecloud", brand_color="4285F4",
        domain="cloud.google.com",
    ),

    # ── inference platforms ──
    "groq": ProviderBrand(
        "Groq", icon_slug=None, brand_color="F55036", domain="groq.com",
    ),
    "openrouter": ProviderBrand(
        "OpenRouter", icon_slug="openrouter", brand_color="6566F1",
        domain="openrouter.ai",
    ),
    "huggingface": ProviderBrand(
        "Hugging Face", icon_slug="huggingface", brand_color="FFD21E",
        domain="huggingface.co",
    ),
    "modelscope": ProviderBrand(
        "ModelScope", icon_slug=None, brand_color="2F8AC4",
        domain="modelscope.cn",
    ),

    # ── enterprise / cloud ──
    "azure": ProviderBrand(
        "Azure OpenAI", icon_slug=None, brand_color="0078D4",
        domain="azure.microsoft.com",
    ),
    "azure-openai": ProviderBrand(
        "Azure OpenAI", icon_slug=None, brand_color="0078D4",
        domain="azure.microsoft.com",
    ),
    "bedrock": ProviderBrand(
        "AWS Bedrock", icon_slug=None, brand_color="FF9900",
        domain="aws.amazon.com",
    ),

    # ── specialised ──
    "grok": ProviderBrand(
        "xAI Grok", icon_slug="x", brand_color="000000", domain="x.ai",
    ),
    "xai": ProviderBrand(
        "xAI Grok", icon_slug="x", brand_color="000000", domain="x.ai",
    ),
    "moonshot": ProviderBrand(
        "Moonshot Kimi", icon_slug=None, brand_color="1E40AF",
        domain="moonshot.ai",
    ),
    "kimi": ProviderBrand(
        "Moonshot Kimi", icon_slug=None, brand_color="1E40AF",
        domain="moonshot.ai",
    ),
    "zai": ProviderBrand(
        "Z.AI", icon_slug=None, brand_color="10B981", domain="z.ai",
    ),
    "z-ai": ProviderBrand(
        "Z.AI", icon_slug=None, brand_color="10B981", domain="z.ai",
    ),
    "cohere": ProviderBrand(
        "Cohere", icon_slug=None, brand_color="39594D", domain="cohere.com",
    ),
    "mistral": ProviderBrand(
        "Mistral", icon_slug=None, brand_color="FF7000", domain="mistral.ai",
    ),

    # ── multimodal ──
    "runway": ProviderBrand(
        "Runway", icon_slug=None, brand_color="000000", domain="runwayml.com",
    ),
    "runwayml": ProviderBrand(
        "Runway", icon_slug=None, brand_color="000000", domain="runwayml.com",
    ),
    "elevenlabs": ProviderBrand(
        "ElevenLabs", icon_slug="elevenlabs", brand_color="000000",
        domain="elevenlabs.io",
    ),
    "stability": ProviderBrand(
        "Stability AI", icon_slug=None, brand_color="8A2BE2",
        domain="stability.ai",
    ),

    # ── local ──
    "ollama": ProviderBrand(
        "Ollama", icon_slug="ollama", brand_color="000000",
        is_local=True, domain="ollama.com",
    ),
    "lmstudio": ProviderBrand(
        "LM Studio", icon_slug=None, brand_color="6366F1",
        is_local=True, domain="lmstudio.ai",
    ),
    "lm-studio": ProviderBrand(
        "LM Studio", icon_slug=None, brand_color="6366F1",
        is_local=True, domain="lmstudio.ai",
    ),
    "local_http": ProviderBrand(
        "Local HTTP", icon_slug=None, brand_color="64748B", is_local=True,
    ),
    "airllm": ProviderBrand(
        "AirLLM", icon_slug=None, brand_color="64748B", is_local=True,
    ),
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
    """Best-available logo URL for the brand, or ``None`` if no source is set.

    Prefers simple-icons (vector, scalable) over favicons (raster, often
    cropped). Both, then ``None``, in that order — so callers can swap
    sources transparently when simple-icons coverage drops.
    """
    if brand is None:
        return None
    if brand.icon_slug:
        return f"https://cdn.simpleicons.org/{brand.icon_slug}"
    if brand.domain:
        return f"https://www.google.com/s2/favicons?sz=64&domain={brand.domain}"
    return None
