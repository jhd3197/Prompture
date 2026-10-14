"""Which local model servers are running right now, and the models each one serves.

Discovery treats Ollama and LM Studio as always configured, because they need no key. That
says nothing about whether one is up, so an app that wants to show "Ollama: running, 3 models"
or "LM Studio: not running" asks here. Each probe is one quick GET to the server's model list.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from .settings import settings

__all__ = ["LocalServer", "probe_local_servers"]


@dataclass(frozen=True)
class LocalServer:
    """One local server: ``provider`` is the model-string prefix (``"ollama"``)."""

    provider: str
    name: str
    url: str
    running: bool
    models: tuple[str, ...] = field(default_factory=tuple)
    detail: str = ""


def _origin(endpoint: str) -> str:
    parts = urlsplit(endpoint)
    return f"{parts.scheme}://{parts.netloc}" if parts.scheme and parts.netloc else endpoint.rstrip("/")


def _ollama(timeout: float) -> LocalServer:
    origin = _origin(settings.ollama_endpoint or "http://localhost:11434")
    return _probe(
        "ollama",
        "Ollama",
        origin,
        f"{origin}/api/tags",
        timeout,
        lambda body: [m.get("name") for m in body.get("models", [])],
    )


def _lmstudio(timeout: float) -> LocalServer:
    origin = _origin(settings.lmstudio_endpoint or "http://127.0.0.1:1234")
    headers = {"Authorization": f"Bearer {settings.lmstudio_api_key}"} if settings.lmstudio_api_key else {}
    return _probe(
        "lmstudio",
        "LM Studio",
        origin,
        f"{origin}/v1/models",
        timeout,
        lambda body: [m.get("id") for m in body.get("data", [])],
        headers,
    )


def _probe(provider, name, origin, url, timeout, names, headers=None) -> LocalServer:  # type: ignore[no-untyped-def]
    import requests

    try:
        response = requests.get(url, timeout=timeout, headers=headers or {})
    except requests.RequestException as exc:
        return LocalServer(provider, name, origin, False, detail=f"not running ({type(exc).__name__})")
    if response.status_code != 200:
        return LocalServer(provider, name, origin, False, detail=f"answered HTTP {response.status_code}")
    try:
        found = [n for n in names(response.json()) if isinstance(n, str) and n]
    except (ValueError, AttributeError, TypeError):
        return LocalServer(provider, name, origin, False, detail="answered with something that is not a model list")
    return LocalServer(provider, name, origin, True, tuple(found))


_PROBES = {"ollama": _ollama, "lmstudio": _lmstudio}


def probe_local_servers(*, timeout: float = 0.8, only: tuple[str, ...] | None = None) -> list[LocalServer]:
    """Ask each local server (Ollama, LM Studio) for its models, all at once.

    Never raises: a server that is down, slow or answers oddly comes back ``running=False``
    with the reason in ``detail``. ``only`` limits the probe to those providers.
    """
    wanted = [p for p in _PROBES if only is None or p in only]
    with ThreadPoolExecutor(max_workers=len(wanted) or 1) as pool:
        return list(pool.map(lambda p: _PROBES[p](timeout), wanted))
