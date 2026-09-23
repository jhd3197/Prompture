"""Kev decision driver — self-hosted Jev-compatible decision models.

Kev (Apache 2.0) is a family of small decision models built on Qwen3.5 that
serves the same ``/v1/systemone`` endpoint locally::

    uv run --extra serve python -m kev.serve --run jaredpalmer/kev-4b --port 8009

Point the driver at it with ``KEV_BASE_URL`` (default ``http://127.0.0.1:8009``).
No API key is required; set ``KEV_API_KEY`` only if you put the server behind
an authenticating proxy.
"""

from __future__ import annotations

from .systemone_compatible_driver import SystemOneCompatibleDriver


class KevDecisionDriver(SystemOneCompatibleDriver):
    """Self-hosted Kev decision driver.

    Default model: ``kev-latest`` — whichever checkpoint the local server was
    started with.  Pass a concrete id (``kev-4b``) when you run several.
    """

    PROVIDER = "kev"
    DEFAULT_BASE_URL = "http://127.0.0.1:8009"
    DEFAULT_MODEL = "kev-latest"
    ENV_KEY = "KEV_API_KEY"  # nosec B105 — env var name, not a secret
    ENV_BASE_URL = "KEV_BASE_URL"
    REQUIRES_API_KEY = False

    KNOWN_MODELS: tuple[str, ...] = (
        "kev-latest",
        "kev-0.8b",
        "kev-4b",
        "kev-9b",
    )
