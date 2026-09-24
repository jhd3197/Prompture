"""Provider account balances and spend from documented account endpoints.

Rate-limit headers (:mod:`.rate_limits`) say how much of a *window* is left.
This module answers the slower question: how much money or credit is left on
an account or key, and how much of the current period has been spent.

Each provider is an :class:`AccountSource` in a pluggable registry shaped like
the pricing-source registry. Built-in sources use only vendor-documented
endpoints and only credentials already configured for Prompture:

====================  ================================  =========================
Source                Endpoint                          Credential
====================  ================================  =========================
``openrouter``        ``GET /api/v1/key``               ``OPENROUTER_API_KEY``
``deepseek``          ``GET /user/balance``             ``DEEPSEEK_API_KEY``
``moonshot``          ``GET /v1/users/me/balance``      ``MOONSHOT_API_KEY``
``openai_billing``    Organization costs (month to date)  ``OPENAI_ADMIN_KEY``
``anthropic_billing`` Organization costs (month to date)  ``ANTHROPIC_ADMIN_KEY``
====================  ================================  =========================

Nothing here runs on import, and no source is called unless it is configured.
Lookups never raise into the caller: a failing source yields a snapshot with
``error`` set, so a dashboard can show *why* a number is missing instead of
inventing one.

Example::

    from prompture.infra.accounts import get_account_snapshots

    for name, snap in get_account_snapshots().items():
        print(name, snap.balance, snap.currency, snap.error)
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol, runtime_checkable

import httpx

logger = logging.getLogger("prompture.accounts")

__all__ = [
    "AccountSnapshot",
    "AccountSource",
    "BillingSpendSource",
    "DeepSeekBalanceSource",
    "MoonshotBalanceSource",
    "OpenRouterKeySource",
    "aget_account_snapshots",
    "clear_account_cache",
    "get_account_snapshot",
    "get_account_snapshots",
    "get_account_sources",
    "register_account_source",
    "unregister_account_source",
]

_USER_AGENT = "Prompture (https://github.com/jhd3197/prompture)"


@dataclass(frozen=True)
class AccountSnapshot:
    """What one account source reported at one moment.

    Attributes:
        source: Registry name of the source that produced this snapshot.
        provider: Prompture provider name (``"openrouter"``, ``"deepseek"``…).
        currency: ISO currency of the money fields, when the provider says.
        balance: Funds or credit still spendable.
        spent: Amount spent during ``period``.
        limit: Spending cap that ``spent`` counts against, if any.
        period: What ``spent`` covers (``"all_time"``, ``"month"``, ``"day"``…).
        available: The provider's own "can this account make calls" flag.
        details: Provider-specific extras (breakdowns, per-currency balances).
        error: Why the fetch failed; all numbers are ``None`` when set.
    """

    source: str
    provider: str
    currency: str | None = None
    balance: float | None = None
    spent: float | None = None
    limit: float | None = None
    period: str | None = None
    available: bool | None = None
    details: dict[str, Any] = field(default_factory=dict)
    observed_at: float = field(default_factory=time.time)
    error: str | None = None

    @property
    def fraction_remaining(self) -> float | None:
        """``(limit - spent) / limit`` when a cap is known, clamped to ``[0, 1]``."""
        if self.limit is None or self.spent is None or self.limit <= 0:
            return None
        return max(0.0, min(1.0, (self.limit - self.spent) / self.limit))

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "provider": self.provider,
            "currency": self.currency,
            "balance": self.balance,
            "spent": self.spent,
            "limit": self.limit,
            "period": self.period,
            "available": self.available,
            "fraction_remaining": self.fraction_remaining,
            "details": self.details,
            "observed_at": self.observed_at,
            "error": self.error,
        }


@runtime_checkable
class AccountSource(Protocol):
    """A way to read one provider account's balance or spend."""

    name: str
    provider: str

    def is_configured(self) -> bool:
        """True when the credential this source needs is available."""
        ...

    def fetch(self, client: httpx.Client) -> AccountSnapshot:
        """Call the provider. May raise; the registry turns errors into snapshots."""
        ...


def _num(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _setting(name: str) -> str | None:
    value = os.environ.get(name.upper())
    if value:
        return value
    try:
        from .settings import settings

        configured = getattr(settings, name.lower(), None)
    except Exception:  # pragma: no cover - settings import failures are reported elsewhere
        return None
    if configured is None:
        return None
    if hasattr(configured, "get_secret_value"):
        configured = configured.get_secret_value()
    return str(configured) or None


class _BearerJSONSource:
    """Shared plumbing for sources that GET one JSON document with a bearer key."""

    name = ""
    provider = ""
    key_setting = ""
    url = ""

    def __init__(self, api_key: str | None = None, url: str | None = None) -> None:
        self._api_key = api_key
        if url:
            self.url = url

    @property
    def api_key(self) -> str | None:
        return self._api_key or _setting(self.key_setting)

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def _get_json(self, client: httpx.Client) -> Any:
        response = client.get(
            self.url,
            headers={"Authorization": f"Bearer {self.api_key}", "User-Agent": _USER_AGENT},
            follow_redirects=False,
        )
        if response.status_code != 200:
            # Status only: response bodies can echo request details.
            raise RuntimeError(f"{self.name} returned HTTP {response.status_code}")
        return response.json()


class OpenRouterKeySource(_BearerJSONSource):
    """Credit cap and usage of the configured OpenRouter key (``GET /api/v1/key``)."""

    name = "openrouter"
    provider = "openrouter"
    key_setting = "openrouter_api_key"
    url = "https://openrouter.ai/api/v1/key"

    def fetch(self, client: httpx.Client) -> AccountSnapshot:
        body = self._get_json(client)
        data = body.get("data", body) if isinstance(body, dict) else {}
        details = {
            k: data[k]
            for k in ("label", "is_free_tier", "usage_daily", "usage_weekly", "usage_monthly", "limit_reset")
            if k in data
        }
        return AccountSnapshot(
            source=self.name,
            provider=self.provider,
            currency="USD",
            balance=_num(data.get("limit_remaining")),
            spent=_num(data.get("usage")),
            limit=_num(data.get("limit")),
            period="all_time",
            details=details,
        )


class DeepSeekBalanceSource(_BearerJSONSource):
    """Account balance (``GET /user/balance``); prefers the USD entry when several currencies exist."""

    name = "deepseek"
    provider = "deepseek"
    key_setting = "deepseek_api_key"
    url = "https://api.deepseek.com/user/balance"

    def fetch(self, client: httpx.Client) -> AccountSnapshot:
        body = self._get_json(client)
        infos = [i for i in body.get("balance_infos") or [] if isinstance(i, dict)]
        chosen = next((i for i in infos if i.get("currency") == "USD"), infos[0] if infos else {})
        return AccountSnapshot(
            source=self.name,
            provider=self.provider,
            currency=chosen.get("currency"),
            balance=_num(chosen.get("total_balance")),
            available=body.get("is_available") if isinstance(body.get("is_available"), bool) else None,
            details={
                "granted_balance": _num(chosen.get("granted_balance")),
                "topped_up_balance": _num(chosen.get("topped_up_balance")),
                "balances": infos,
            },
        )


class MoonshotBalanceSource(_BearerJSONSource):
    """Account balance in USD (``GET /v1/users/me/balance``)."""

    name = "moonshot"
    provider = "moonshot"
    key_setting = "moonshot_api_key"
    url = "https://api.moonshot.ai/v1/users/me/balance"

    def __init__(self, api_key: str | None = None, url: str | None = None) -> None:
        if url is None:
            endpoint = _setting("moonshot_endpoint")
            if endpoint:
                url = endpoint.rstrip("/") + "/users/me/balance"
        super().__init__(api_key, url)

    def fetch(self, client: httpx.Client) -> AccountSnapshot:
        body = self._get_json(client)
        data = body.get("data") if isinstance(body, dict) else None
        data = data if isinstance(data, dict) else {}
        balance = _num(data.get("available_balance"))
        return AccountSnapshot(
            source=self.name,
            provider=self.provider,
            currency="USD",
            balance=balance,
            available=None if balance is None else balance > 0,
            details={
                "voucher_balance": _num(data.get("voucher_balance")),
                "cash_balance": _num(data.get("cash_balance")),
            },
        )


class BillingSpendSource:
    """Month-to-date organization spend from the OpenAI / Anthropic cost reports.

    Wraps :class:`~prompture.infra.billing.BillingClient`, so it needs an admin
    key (``OPENAI_ADMIN_KEY`` / ``ANTHROPIC_ADMIN_KEY``), not an ordinary API key.
    """

    def __init__(self, provider: str, admin_key: str | None = None) -> None:
        if provider not in ("openai", "anthropic"):
            raise ValueError("BillingSpendSource supports openai and anthropic")
        self.provider = "claude" if provider == "anthropic" else provider
        self.billing_provider = provider
        self.name = f"{provider}_billing"
        self._admin_key = admin_key

    def is_configured(self) -> bool:
        return bool(self._admin_key or _setting(f"{self.billing_provider}_admin_key"))

    def fetch(self, client: httpx.Client) -> AccountSnapshot:
        from .billing import BillingClient

        now = datetime.now(timezone.utc)
        start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        end = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        billing = BillingClient(self.billing_provider, admin_key=self._admin_key, client=client)  # type: ignore[arg-type]
        report = billing.get_costs(start, end)
        totals = report.totals
        currency = "USD" if "USD" in totals else next(iter(totals), None)
        return AccountSnapshot(
            source=self.name,
            provider=self.provider,
            currency=currency.upper() if currency else None,
            spent=float(totals[currency]) if currency else 0.0,
            period="month",
            details={"complete": report.complete, "warnings": list(report.warnings)},
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_sources: dict[str, AccountSource] = {}
_cache: dict[str, AccountSnapshot] = {}
_initialized = False


def _ensure_initialized() -> None:
    global _initialized
    with _lock:
        if _initialized:
            return
        _initialized = True
        for source in (
            OpenRouterKeySource(),
            DeepSeekBalanceSource(),
            MoonshotBalanceSource(),
            BillingSpendSource("openai"),
            BillingSpendSource("anthropic"),
        ):
            _sources.setdefault(source.name, source)


def register_account_source(source: AccountSource, *, replace: bool = False) -> None:
    """Add a source. Raises ``ValueError`` if the name is taken unless *replace* is set."""
    _ensure_initialized()
    with _lock:
        if source.name in _sources and not replace:
            raise ValueError(f"Account source '{source.name}' is already registered")
        _sources[source.name] = source
        _cache.pop(source.name, None)


def unregister_account_source(name: str) -> bool:
    _ensure_initialized()
    with _lock:
        _cache.pop(name, None)
        return _sources.pop(name, None) is not None


def get_account_sources() -> list[AccountSource]:
    _ensure_initialized()
    with _lock:
        return list(_sources.values())


def clear_account_cache() -> None:
    with _lock:
        _cache.clear()


def _fetch(source: AccountSource, client: httpx.Client) -> AccountSnapshot:
    try:
        return source.fetch(client)
    except Exception as exc:
        logger.debug("Account source %s failed: %s", source.name, exc)
        return AccountSnapshot(source=source.name, provider=source.provider, error=str(exc)[:200])


def get_account_snapshots(
    names: Iterable[str] | None = None,
    *,
    max_age: float = 60.0,
    client: httpx.Client | None = None,
    timeout: float = 15.0,
) -> dict[str, AccountSnapshot]:
    """Snapshots for every configured source (or just *names*), keyed by source name.

    Results younger than *max_age* seconds are served from memory, so a
    dashboard polling this does not hit provider endpoints on every refresh.
    Pass ``max_age=0`` to force a fetch. Unconfigured sources are left out.
    """
    wanted = set(names) if names is not None else None
    sources = [s for s in get_account_sources() if (wanted is None or s.name in wanted) and s.is_configured()]
    now = time.time()
    out: dict[str, AccountSnapshot] = {}
    stale: list[AccountSource] = []
    with _lock:
        for source in sources:
            cached = _cache.get(source.name)
            if cached is not None and now - cached.observed_at < max_age:
                out[source.name] = cached
            else:
                stale.append(source)
    if stale:
        owned = client is None
        http = client or httpx.Client(timeout=timeout)
        try:
            for source in stale:
                snapshot = _fetch(source, http)
                out[source.name] = snapshot
                with _lock:
                    _cache[source.name] = snapshot
        finally:
            if owned:
                http.close()
    return out


def get_account_snapshot(name: str, **kwargs: Any) -> AccountSnapshot | None:
    """One source's snapshot, or ``None`` if it is unknown or not configured."""
    return get_account_snapshots([name], **kwargs).get(name)


async def aget_account_snapshots(names: Iterable[str] | None = None, **kwargs: Any) -> dict[str, AccountSnapshot]:
    """Async wrapper; runs the blocking fetches in a worker thread."""
    return await asyncio.to_thread(get_account_snapshots, names, **kwargs)
