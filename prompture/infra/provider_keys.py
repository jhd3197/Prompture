"""Provider API keys: which providers take one, and a cheap live check of a key.

Shared by ``prompture setup`` and by apps that let people paste a key (no ``click`` needed).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

__all__ = ["KeyCheck", "ProviderKey", "provider_key_catalogue", "validate_key"]


@dataclass(frozen=True)
class ProviderKey:
    """A provider whose access is a single API key."""

    provider: str
    display: str
    env_var: str
    model_setting: str | None = None
    llm: bool = False


def provider_key_catalogue() -> list[ProviderKey]:
    """Key-based providers from the provider registry, LLM providers first."""
    from ..drivers.provider_descriptors import PROVIDER_DESCRIPTORS
    from .settings import Settings

    fields = Settings.model_fields
    entries: list[ProviderKey] = []
    for desc in PROVIDER_DESCRIPTORS:
        check = desc.is_configured_check
        if desc.alias_for or not check or check not in fields:
            continue
        if not check.endswith(("_key", "_token")):
            continue
        model_setting = None
        if desc.llm_sync and desc.llm_sync.default_model in fields:
            model_setting = desc.llm_sync.default_model
        entries.append(
            ProviderKey(
                provider=desc.name,
                display=desc.display_name or desc.name,
                env_var=check.upper(),
                model_setting=model_setting,
                llm=desc.llm_sync is not None,
            )
        )
    entries.sort(key=lambda e: not e.llm)
    return entries


@dataclass(frozen=True)
class KeyCheck:
    """Outcome of a live key check.

    ``status``: ``valid`` (provider accepted it), ``invalid`` (provider rejected
    it), ``unverified`` (the check could not complete — network, rate limit,
    server error) or ``unchecked`` (no cheap check exists for this provider).
    """

    status: Literal["valid", "invalid", "unverified", "unchecked"]
    detail: str = ""


# provider -> (url, auth style). Each is the cheapest authenticated read the
# provider offers (a models / key-info listing).
_KEY_CHECKS: dict[str, tuple[str, str]] = {
    "openai": ("https://api.openai.com/v1/models", "bearer"),
    "claude": ("https://api.anthropic.com/v1/models?limit=1", "anthropic"),
    "google": ("https://generativelanguage.googleapis.com/v1beta/models?pageSize=1", "goog"),
    "groq": ("https://api.groq.com/openai/v1/models", "bearer"),
    "grok": ("https://api.x.ai/v1/models", "bearer"),
    "openrouter": ("https://openrouter.ai/api/v1/key", "bearer"),
    "moonshot": ("https://api.moonshot.ai/v1/models", "bearer"),
    "mistral": ("https://api.mistral.ai/v1/models", "bearer"),
    "deepseek": ("https://api.deepseek.com/models", "bearer"),
    "cohere": ("https://api.cohere.com/v1/models?page_size=1", "bearer"),
    "elevenlabs": ("https://api.elevenlabs.io/v1/models", "xi"),
    "deepgram": ("https://api.deepgram.com/v1/projects", "token"),
}


def _auth_headers(style: str, key: str) -> dict[str, str]:
    if style == "anthropic":
        return {"x-api-key": key, "anthropic-version": "2023-06-01"}
    if style == "goog":
        return {"x-goog-api-key": key}
    if style == "xi":
        return {"xi-api-key": key}
    if style == "token":
        return {"Authorization": f"Token {key}"}
    return {"Authorization": f"Bearer {key}"}


def validate_key(provider: str, key: str, *, timeout: float = 10.0) -> KeyCheck:
    """Check *key* against *provider* with one cheap authenticated GET.

    Honors ``PROMPTURE_<PROVIDER>_PROXY`` / ``PROMPTURE_PROXY``. Never includes
    the key in the returned detail.
    """
    spec = _KEY_CHECKS.get(provider)
    if spec is None:
        return KeyCheck("unchecked", "no live check for this provider; saved as entered")
    url, style = spec
    import requests

    from ..capabilities.http import proxies_for

    try:
        resp = requests.get(
            url,
            headers=_auth_headers(style, key),
            timeout=timeout,
            allow_redirects=False,
            proxies=proxies_for(provider),
        )
    except requests.Timeout:
        return KeyCheck("unverified", "the provider did not answer in time")
    except requests.RequestException as exc:
        return KeyCheck("unverified", f"network error ({type(exc).__name__})")
    code = resp.status_code
    if 200 <= code < 300:
        return KeyCheck("valid", "accepted by the provider")
    if code in (401, 403):
        return KeyCheck("invalid", f"rejected by the provider (HTTP {code})")
    if code == 400 and provider == "google":
        return KeyCheck("invalid", "rejected by the provider (HTTP 400)")
    if code == 429:
        return KeyCheck("unverified", "rate limited (HTTP 429); try again later")
    return KeyCheck("unverified", f"unexpected response (HTTP {code})")
