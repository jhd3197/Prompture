"""Capability foundation shared by web tools, media understanding, MCP and doctor.

* :mod:`.backends` — :class:`BackendChain` ordered failover with route records.
* :mod:`.probe` — :func:`probe_command` real binary health probes.
* :mod:`.health` — :class:`HealthStatus` and the capability registry doctor walks.
* :mod:`.url_safety` — :func:`normalize_public_http_url` public-URL guard.
* :mod:`.http` — :func:`safe_get` redirect-checked, size-capped GET.
* :mod:`.challenge` — :func:`is_challenge_page` interstitial detection and
  :func:`is_js_shell_page` for pages that are only a "needs JavaScript" shell.
"""

from .backends import Backend, BackendChain, BaseBackend, ChainResult, order_backends, parse_override
from .challenge import is_challenge_page, is_js_shell_page
from .errors import (
    AllBackendsFailedError,
    BackendUnavailableError,
    CapabilityError,
    ChallengePageError,
    HTTPStatusError,
    ResponseTooLargeError,
    UnsafeURLError,
)
from .health import (
    Capability,
    HealthStatus,
    check_capabilities,
    list_capabilities,
    register_capability,
    unregister_capability,
    worst_status,
)
from .http import SafeResponse, proxies_for, resolve_proxy, safe_get
from .probe import ProbeResult, cached_probe, clear_probe_cache, probe_command
from .url_safety import is_public_http_url, is_public_ip, normalize_public_http_url

__all__ = [
    "AllBackendsFailedError",
    "Backend",
    "BackendChain",
    "BackendUnavailableError",
    "BaseBackend",
    "Capability",
    "CapabilityError",
    "ChainResult",
    "ChallengePageError",
    "HTTPStatusError",
    "HealthStatus",
    "ProbeResult",
    "ResponseTooLargeError",
    "SafeResponse",
    "UnsafeURLError",
    "cached_probe",
    "check_capabilities",
    "clear_probe_cache",
    "is_challenge_page",
    "is_js_shell_page",
    "is_public_http_url",
    "is_public_ip",
    "list_capabilities",
    "normalize_public_http_url",
    "order_backends",
    "parse_override",
    "probe_command",
    "proxies_for",
    "register_capability",
    "resolve_proxy",
    "safe_get",
    "unregister_capability",
    "worst_status",
]
