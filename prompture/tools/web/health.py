"""Doctor rows for the web tools.

Importing this module registers capability checks (category ``tools``):

* ``web_search`` — the search backend chain (active backend + inactive ones).
* ``web_fetch`` — the fetch backend chain.
* ``read_url:<reader>`` — one row per registered reader.
* ``search_platform:<platform>`` — one row per platform.

Offline checks only inspect configuration, installed packages and binaries;
``live=True`` runs one cheap real request per backend.
"""

from __future__ import annotations

from ...capabilities.health import HealthStatus, register_capability
from .fetch import fetch_chain
from .platform import PLATFORMS, platform_chain
from .readers import get_reader, list_readers
from .search import search_chain


def check_web_search(live: bool = False) -> HealthStatus:
    return search_chain().check(live, category="tools")


def check_web_fetch(live: bool = False) -> HealthStatus:
    return fetch_chain().check(live, category="tools")


def _reader_check(name: str):  # type: ignore[no-untyped-def]
    def check(live: bool = False) -> HealthStatus:
        reader = get_reader(name)
        if reader is None:
            return HealthStatus(f"read_url:{name}", "skipped", category="tools", message="reader not registered")
        row = reader.check(live)
        row.name = f"read_url:{name}"
        return row

    return check


def _platform_check(name: str):  # type: ignore[no-untyped-def]
    def check(live: bool = False) -> HealthStatus:
        return platform_chain(name).check(live, category="tools")

    return check


def check_web_cache(live: bool = False) -> HealthStatus:
    """Where search / fetch / reader results are cached, and how many are stored."""
    from . import cache as web_cache

    info = web_cache.cache_info()
    if info["mode"] == "off":
        return HealthStatus(
            "web_cache",
            "skipped",
            category="tools",
            message="web cache disabled (PROMPTURE_WEB_CACHE=off)",
            details=info,
        )
    entries = info["entries"]
    where = info["path"] or "this process only"
    count = f", {entries} entries" if entries is not None else ""
    return HealthStatus(
        "web_cache",
        "ok",
        category="tools",
        active_backend=info["mode"],
        message=f"{where}{count}",
        details=info,
    )


def register_web_capabilities() -> None:
    """(Re-)register every web tool health row."""
    register_capability("web_cache", "tools", check_web_cache, description="Local cache for web results")
    register_capability("web_search", "tools", check_web_search, description="Web search backend chain")
    register_capability("web_fetch", "tools", check_web_fetch, description="URL → Markdown fetch chain")
    for reader in list_readers():
        register_capability(
            f"read_url:{reader.name}",
            "tools",
            _reader_check(reader.name),
            description=getattr(reader, "description", "") or f"{reader.name} reader",
        )
    for platform in PLATFORMS:
        register_capability(
            f"search_platform:{platform}",
            "tools",
            _platform_check(platform),
            description=f"{platform} platform search",
        )


register_web_capabilities()
