"""TypeSafe decision driver — the hosted Jev "System One" model.

Uses ``POST https://api.typesafe.ai/v1/systemone``.  Requires ``TYPESAFE_API_KEY``
(create one at https://console.typesafe.ai/).

Jev bills input tokens only; output tokens are free.  A typical decision costs
a small fraction of a cent, which is what makes it viable as a gate in front of
a full LLM call.
"""

from __future__ import annotations

from .systemone_compatible_driver import SystemOneCompatibleDriver


class TypeSafeDecisionDriver(SystemOneCompatibleDriver):
    """Hosted TypeSafe Jev decision driver.

    Default model: ``jev-latest`` (an alias — the response's ``model`` field
    reports the versioned id that actually answered).
    """

    PROVIDER = "typesafe"
    DEFAULT_BASE_URL = "https://api.typesafe.ai"
    DEFAULT_MODEL = "jev-latest"
    ENV_KEY = "TYPESAFE_API_KEY"  # nosec B105 — env var name, not a secret
    ENV_BASE_URL = "TYPESAFE_BASE_URL"
    REQUIRES_API_KEY = True

    KNOWN_MODELS: tuple[str, ...] = (
        "jev-latest",
        "jev-preview",
        "jev-1.13.0",
    )
