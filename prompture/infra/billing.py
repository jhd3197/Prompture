"""Optional organization billing reports, kept separate from per-call estimates.

Reports are snapshots of provider aggregates, not invoices or per-request charges.
No network access occurs until a retrieval method is called. Only explicit admin
credentials (or OPENAI_ADMIN_KEY / ANTHROPIC_ADMIN_KEY) are used. Async helpers
offload the synchronous HTTP client to a worker thread.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Literal

import httpx

Provider = Literal["openai", "anthropic"]
ReportKind = Literal["usage", "costs"]
_WIDTHS = {"1m": 60, "1h": 3600, "1d": 86400}
_FILTERS = {
    ("openai", "usage"): {"project_ids", "user_ids", "api_key_ids", "models", "batch", "service_tiers"},
    ("openai", "costs"): {"project_ids", "api_key_ids", "line_items"},
    ("anthropic", "usage"): {
        "account_ids",
        "api_key_ids",
        "context_window",
        "inference_geos",
        "models",
        "service_account_ids",
        "service_tiers",
        "speeds",
        "workspace_ids",
    },
    ("anthropic", "costs"): set(),
}
_GROUPS = {
    ("openai", "usage"): {"project_id", "user_id", "api_key_id", "model", "batch", "service_tier"},
    ("openai", "costs"): {"project_id", "line_item", "api_key_id"},
    ("anthropic", "usage"): {
        "account_id",
        "api_key_id",
        "context_window",
        "inference_geo",
        "model",
        "service_account_id",
        "service_tier",
        "speed",
        "workspace_id",
    },
    ("anthropic", "costs"): {"description", "workspace_id"},
}


def _utc(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Report bounds must be timezone-aware datetimes")
    return value.astimezone(timezone.utc)


def _money(value: Any) -> Decimal:
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise ValueError("Invalid monetary amount in billing report") from None
    if not amount.is_finite():
        raise ValueError("Billing amounts must be finite")
    return amount


@dataclass(frozen=True)
class BillingScope:
    """A declared organization and exact filters, shared by both sides of comparison.

    ``organization_id`` identifies the account associated with the admin key; the
    caller must supply the correct identifier. It is not inferred from a secret.
    ``exclusions`` records provider coverage limits, such as Priority Tier costs.
    """

    provider: Provider
    organization_id: str | None = None
    filters: tuple[tuple[str, tuple[str, ...]], ...] = ()
    exclusions: tuple[str, ...] = ()


@dataclass(frozen=True)
class BillingRow:
    start: datetime
    end: datetime
    dimensions: dict[str, Any]
    raw: dict[str, Any]
    amount: Decimal | None = None
    currency: str | None = None


@dataclass
class BillingReport:
    """Provider aggregate snapshot; ``complete`` means all pages were fetched.

    Completion does not guarantee that delayed provider records have arrived or
    that the provider covers all billed services. Inspect ``warnings`` and scope.
    """

    kind: ReportKind
    scope: BillingScope
    start: datetime
    end: datetime
    group_by: tuple[str, ...] = ()
    rows: list[BillingRow] = field(default_factory=list)
    raw_pages: list[dict[str, Any]] = field(default_factory=list, repr=False)
    complete: bool = False
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = "provider_reported"

    @property
    def totals(self) -> dict[str, Decimal]:
        """Return observed currency totals; partial reports remain explicitly partial."""
        totals: dict[str, Decimal] = {}
        for row in self.rows:
            if row.amount is not None and row.currency is not None:
                totals[row.currency] = totals.get(row.currency, Decimal(0)) + row.amount
        return totals


@dataclass(frozen=True)
class LocalCostSummary:
    """Caller-attested local estimate for exactly this scope and half-open interval.

    Include all activity for the declared scope, or mark ``complete=False``.
    Unknown-price events must be counted in ``unknown_cost_events``. Do not copy
    the provider's exclusions here unless local activity was actually filtered.
    """

    scope: BillingScope
    start: datetime
    end: datetime
    amount: Decimal
    currency: str = "USD"
    complete: bool = True
    unknown_cost_events: int = 0


@dataclass(frozen=True)
class CostReconciliation:
    comparable: bool
    estimated_cost: Decimal
    reported_cost: Decimal | None
    difference: Decimal | None
    currency: str
    reasons: tuple[str, ...] = ()


def reconcile_costs(local: LocalCostSummary, reported: BillingReport) -> CostReconciliation:
    """Compare equivalent aggregates without adding, persisting, or allocating them.

    ``difference`` is provider-reported minus locally estimated cost. A difference
    may reflect billing delays, coverage, discounts, or activity outside Prompture.
    """
    reasons = []
    currency = local.currency.upper()
    amount = _money(local.amount)
    if reported.kind != "costs" or reported.source != "provider_reported":
        reasons.append("A provider-reported costs report is required")
    if not local.scope.organization_id or not reported.scope.organization_id:
        reasons.append("A known organization identity is required")
    if local.scope != reported.scope:
        reasons.append("Provider, organization, filters, or exclusions differ")
    if (_utc(local.start), _utc(local.end)) != (reported.start, reported.end):
        reasons.append("Time ranges differ")
    if not local.complete or not reported.complete or reported.errors:
        reasons.append("One or both reports are incomplete")
    if local.unknown_cost_events:
        reasons.append("Local events have unknown pricing")
    totals = reported.totals
    if any(key != currency for key in totals) or (not totals and currency != "USD"):
        reasons.append("Currencies differ")
    billed = totals.get(currency, Decimal(0)) if not reasons else None
    return CostReconciliation(
        not reasons, amount, billed, billed - amount if billed is not None else None, currency, tuple(reasons)
    )


class BillingAPIError(RuntimeError):
    """Retrieval failed; ``report`` retains any pages successfully fetched."""

    def __init__(self, message: str, report: BillingReport, status_code: int | None = None):
        super().__init__(message)
        self.report = report
        self.status_code = status_code


class BillingClient:
    """Read-only OpenAI/Anthropic administrative usage and cost API adapter.

    Use as a context manager, or call ``close()``. An injected httpx.Client remains
    owned by its caller. Endpoints are fixed to the official providers. Requests
    never follow redirects and error messages omit response bodies and secrets.
    Bounds must align to the chosen UTC bucket width to prevent misleading
    comparisons when provider endpoints snap bounds to bucket boundaries.
    """

    def __init__(
        self,
        provider: Provider,
        *,
        admin_key: str | None = None,
        organization_id: str | None = None,
        client: httpx.Client | None = None,
        timeout: float = 30.0,
        max_pages: int = 10000,
    ):
        if provider not in {"openai", "anthropic"}:
            raise ValueError("Billing provider must be openai or anthropic")
        if max_pages < 1:
            raise ValueError("max_pages must be positive")
        env = "OPENAI_ADMIN_KEY" if provider == "openai" else "ANTHROPIC_ADMIN_KEY"
        self._admin_key = admin_key if admin_key is not None else os.environ.get(env)
        if admin_key is None and env not in os.environ:
            from .settings import settings

            configured_key = getattr(settings, env.lower())
            if configured_key is not None:
                self._admin_key = configured_key.get_secret_value()
        if not self._admin_key:
            raise ValueError(f"An explicit admin_key or {env} is required")
        self.provider = provider
        self.organization_id = organization_id
        self.max_pages = max_pages
        self._client = client or httpx.Client(timeout=timeout)
        self._owns_client = client is None

    def __enter__(self) -> BillingClient:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def get_usage(self, start: datetime, end: datetime, **kwargs: Any) -> BillingReport:
        """Get all usage pages, with optional group_by, filters, and bucket_width."""
        return self._get("usage", start, end, **kwargs)

    def get_costs(self, start: datetime, end: datetime, **kwargs: Any) -> BillingReport:
        """Get all daily cost pages. Anthropic costs do not support API filters."""
        return self._get("costs", start, end, **kwargs)

    async def aget_usage(self, start: datetime, end: datetime, **kwargs: Any) -> BillingReport:
        return await asyncio.to_thread(self.get_usage, start, end, **kwargs)

    async def aget_costs(self, start: datetime, end: datetime, **kwargs: Any) -> BillingReport:
        return await asyncio.to_thread(self.get_costs, start, end, **kwargs)

    def report(
        self,
        start: datetime,
        end: datetime,
        *,
        usage_options: Mapping[str, Any] | None = None,
        cost_options: Mapping[str, Any] | None = None,
        allow_partial: bool = False,
    ) -> dict[str, BillingReport]:
        """Retrieve separate usage/cost snapshots, optionally retaining denied endpoints."""
        return {
            "usage": self.get_usage(start, end, **{**(usage_options or {}), "allow_partial": allow_partial}),
            "costs": self.get_costs(start, end, **{**(cost_options or {}), "allow_partial": allow_partial}),
        }

    async def areport(self, start: datetime, end: datetime, **kwargs: Any) -> dict[str, BillingReport]:
        return await asyncio.to_thread(self.report, start, end, **kwargs)

    def _get(
        self,
        kind: ReportKind,
        start: datetime,
        end: datetime,
        *,
        group_by: Sequence[str] = (),
        filters: Mapping[str, Any] | None = None,
        bucket_width: str = "1d",
        allow_partial: bool = False,
    ) -> BillingReport:
        start, end = _utc(start), _utc(end)
        width = _WIDTHS.get(bucket_width)
        if width is None or (kind == "costs" and bucket_width != "1d"):
            raise ValueError("Usage supports 1m, 1h, 1d; costs supports only 1d")
        if start >= end or any(value.timestamp() % width for value in (start, end)):
            raise ValueError("Bounds must increase and align to the UTC bucket width")
        if isinstance(group_by, str) or set(group_by) - _GROUPS[self.provider, kind]:
            raise ValueError("Unsupported group_by for this provider and report kind")
        filters = dict(filters or {})
        if set(filters) - _FILTERS[self.provider, kind]:
            raise ValueError("Unsupported filters for this provider and report kind")
        canonical_filters = []
        params: list[tuple[str, Any]] = [("bucket_width", bucket_width)]
        for name, value in sorted(filters.items()):
            values = value if isinstance(value, (list, tuple)) else [value]
            if not values or any(item is None or isinstance(item, (dict, list, tuple)) for item in values):
                raise ValueError("Filters must contain nonempty scalar values")
            normalized = tuple(sorted({str(item).lower() if isinstance(item, bool) else str(item) for item in values}))
            canonical_filters.append((name, normalized))
            key = name + "[]" if self.provider == "anthropic" else name
            params.extend((key, item) for item in normalized)
        groups = tuple(sorted(set(group_by)))
        exclusions = ("priority_tier_costs",) if (self.provider, kind) == ("anthropic", "costs") else ()
        result = BillingReport(
            kind,
            BillingScope(self.provider, self.organization_id, tuple(canonical_filters), exclusions),
            start,
            end,
            groups,
        )
        result.warnings.append("Provider reporting may be delayed; this snapshot is not a finalized invoice")
        if exclusions:
            result.warnings.append("Anthropic's cost endpoint excludes Priority Tier costs")
        headers = {"User-Agent": "Prompture (https://github.com/jhd3197/prompture)"}
        if self.provider == "openai":
            endpoint = "usage/completions" if kind == "usage" else "costs"
            url = f"https://api.openai.com/v1/organization/{endpoint}"
            headers["Authorization"] = f"Bearer {self._admin_key}"
            params.extend([("start_time", int(start.timestamp())), ("end_time", int(end.timestamp()))])
            params.extend(("group_by", name) for name in groups)
        else:
            endpoint = "usage_report/messages" if kind == "usage" else "cost_report"
            url = f"https://api.anthropic.com/v1/organizations/{endpoint}"
            headers.update({"x-api-key": self._admin_key, "anthropic-version": "2023-06-01"})
            if "speed" in groups or "speeds" in filters:
                headers["anthropic-beta"] = "fast-mode-2026-02-01"
            params.extend([("starting_at", start.isoformat()), ("ending_at", end.isoformat())])
            params.extend(("group_by[]", name) for name in groups)
        cursor = None
        seen: set[str] = set()
        try:
            for _ in range(self.max_pages):
                query = params + ([("page", cursor)] if cursor else [])
                response = self._client.get(url, params=query, headers=headers, follow_redirects=False)
                if not response.is_success:
                    message = f"{self.provider} {kind} report failed (HTTP {response.status_code})"
                    raise BillingAPIError(message, result, response.status_code)
                page = response.json(parse_float=Decimal)
                if not isinstance(page, dict) or not isinstance(page.get("data"), list):
                    raise ValueError("Malformed billing report page")
                # Parse atomically so malformed pages never contribute partial amounts.
                rows = self._parse_rows(page["data"], result)
                result.raw_pages.append(page)
                result.rows.extend(rows)
                if page.get("has_more") is False:
                    result.complete = True
                    return result
                cursor = page.get("next_page")
                if page.get("has_more") is not True or not isinstance(cursor, str) or not cursor or cursor in seen:
                    raise ValueError("Missing or repeated billing pagination cursor")
                seen.add(cursor)
            raise ValueError("Billing pagination exceeded max_pages")
        except (httpx.HTTPError, ValueError, KeyError, TypeError, OverflowError, BillingAPIError) as exc:
            # Avoid leaking provider response bodies, URLs, or authentication headers.
            message = (
                str(exc)
                if isinstance(exc, BillingAPIError)
                else f"{self.provider} {kind} report retrieval failed ({type(exc).__name__})"
            )
            result.errors.append(message)
            if allow_partial:
                return result
            if isinstance(exc, BillingAPIError):
                raise
            raise BillingAPIError(message, result) from None

    def _parse_rows(self, buckets: list[Any], report: BillingReport) -> list[BillingRow]:
        rows = []
        last_end = report.rows[-1].end if report.rows else None
        for bucket in buckets:
            if not isinstance(bucket, dict):
                raise ValueError("Malformed billing bucket")
            if self.provider == "openai":
                start = datetime.fromtimestamp(bucket["start_time"], timezone.utc)
                end = datetime.fromtimestamp(bucket["end_time"], timezone.utc)
            else:
                if not isinstance(bucket.get("starting_at"), str) or not isinstance(bucket.get("ending_at"), str):
                    raise ValueError("Malformed billing bucket times")
                start = _utc(datetime.fromisoformat(bucket["starting_at"].replace("Z", "+00:00")))
                end = _utc(datetime.fromisoformat(bucket["ending_at"].replace("Z", "+00:00")))
            if start < report.start or end > report.end or start >= end or (last_end and start < last_end):
                raise ValueError("Provider returned overlapping or out-of-range buckets")
            last_end = end
            results = bucket["results"]
            if not isinstance(results, list):
                raise ValueError("Malformed billing bucket results")
            for raw in results:
                if not isinstance(raw, dict):
                    raise ValueError("Malformed billing row")
                amount = currency = None
                if report.kind == "costs":
                    if self.provider == "openai":
                        amount = _money(raw["amount"]["value"])
                        currency = raw["amount"]["currency"]
                    else:
                        amount = _money(raw["amount"]) / 100
                        currency = raw["currency"]
                    if not isinstance(currency, str) or len(currency) != 3:
                        raise ValueError("Missing or invalid currency")
                    currency = currency.upper()
                dimensions = {key: raw.get(key) for key in report.group_by}
                rows.append(BillingRow(start, end, dimensions, dict(raw), amount, currency))
        return rows
