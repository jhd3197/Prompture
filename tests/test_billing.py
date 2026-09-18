"""Offline billing API contract and reconciliation regression tests."""

import asyncio
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import httpx
import pytest

from prompture.infra.billing import BillingAPIError, BillingClient, LocalCostSummary, reconcile_costs

START = datetime(2026, 1, 1, tzinfo=timezone.utc)
END = START + timedelta(days=2)


def page(provider, rows, *, start=START, end=END, more=False, cursor=None):
    bucket = {"results": rows}
    if provider == "openai":
        bucket.update(start_time=int(start.timestamp()), end_time=int(end.timestamp()))
    else:
        bucket.update(starting_at=start.isoformat(), ending_at=end.isoformat())
    return {"data": [bucket], "has_more": more, "next_page": cursor}


def client(provider, handler, **kwargs):
    return BillingClient(
        provider,
        admin_key="secret-admin",
        organization_id="org-1",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        **kwargs,
    )


class TestBillingRetrieval:
    """Exercise real HTTP request encoding against mocked provider responses."""

    def test_openai_pagination_preserves_scope_and_currency(self):
        calls = []

        def handler(request):
            calls.append(request)
            assert request.headers["authorization"] == "Bearer secret-admin"
            assert request.url.path == "/v1/organization/costs"
            assert request.url.params.get_list("project_ids") == ["p1", "p2"]
            assert request.url.params.get_list("group_by") == ["line_item", "project_id"]
            assert request.url.params["end_time"] == str(int(END.timestamp()))
            first = len(calls) == 1
            if not first:
                assert request.url.params["page"] == "next"
            return httpx.Response(
                200,
                json=page(
                    "openai",
                    [{"amount": {"value": 0.123456, "currency": "usd"}, "project_id": "p1", "line_item": "tokens"}],
                    start=START if first else START + timedelta(days=1),
                    end=START + timedelta(days=1) if first else END,
                    more=first,
                    cursor="next" if first else None,
                ),
            )

        report = client("openai", handler).get_costs(
            START,
            END,
            filters={"project_ids": ["p2", "p1"]},
            group_by=["project_id", "line_item"],
        )
        assert report.complete
        assert report.totals == {"USD": Decimal("0.246912")}
        assert report.scope.filters == (("project_ids", ("p1", "p2")),)
        assert len(report.raw_pages) == 2
        assert report.rows[0].dimensions == {"line_item": "tokens", "project_id": "p1"}

    def test_anthropic_cent_conversion_and_priority_exclusion(self):
        def handler(request):
            assert request.url.path == "/v1/organizations/cost_report"
            assert request.headers["x-api-key"] == "secret-admin"
            assert request.headers["anthropic-version"] == "2023-06-01"
            assert request.url.params.get_list("group_by[]") == ["description", "workspace_id"]
            return httpx.Response(
                200,
                json=page(
                    "anthropic",
                    [
                        {
                            "amount": "123.78912",
                            "currency": "USD",
                            "workspace_id": None,
                            "description": "Web Search",
                            "cost_type": "web_search",
                        }
                    ],
                ),
            )

        report = client("anthropic", handler).get_costs(START, END, group_by=["description", "workspace_id"])
        assert report.totals == {"USD": Decimal("1.2378912")}
        assert report.scope.exclusions == ("priority_tier_costs",)
        assert report.rows[0].raw["cost_type"] == "web_search"
        assert report.rows[0].dimensions["workspace_id"] is None
        assert any("Priority" in warning for warning in report.warnings)

    def test_anthropic_usage_preserves_nested_cache_and_speed_beta(self):
        raw = {"model": "claude-model", "cache_creation": {"ephemeral_1h_input_tokens": 17}, "output_tokens": 3}

        def handler(request):
            assert request.url.path == "/v1/organizations/usage_report/messages"
            assert request.headers["anthropic-beta"] == "fast-mode-2026-02-01"
            assert request.url.params.get_list("models[]") == ["claude-model"]
            assert request.url.params.get_list("speeds[]") == ["fast"]
            return httpx.Response(200, json=page("anthropic", [raw]))

        report = client("anthropic", handler).get_usage(
            START,
            END,
            group_by=["model"],
            filters={"models": ["claude-model"], "speeds": ["fast"]},
        )
        assert report.rows[0].raw == raw
        assert report.totals == {}

    def test_permissions_can_return_partial_pair(self):
        def handler(request):
            if request.url.path.endswith("costs"):
                return httpx.Response(403, text="secret-admin must never appear in errors")
            return httpx.Response(200, json=page("openai", [{"input_tokens": 19}]))

        reports = client("openai", handler).report(START, END, allow_partial=True)
        assert reports["usage"].complete
        assert not reports["costs"].complete
        assert reports["costs"].errors == ["openai costs report failed (HTTP 403)"]
        assert reports["costs"].totals == {}

    def test_error_retains_completed_pages_without_exposing_body(self):
        calls = []

        def handler(request):
            calls.append(request)
            if len(calls) == 2:
                return httpx.Response(429, text="secret-admin")
            return httpx.Response(
                200,
                json=page(
                    "openai",
                    [{"amount": {"value": 1, "currency": "usd"}}],
                    end=START + timedelta(days=1),
                    more=True,
                    cursor="next",
                ),
            )

        with pytest.raises(BillingAPIError) as caught:
            client("openai", handler).get_costs(START, END)
        assert caught.value.status_code == 429
        assert caught.value.report.totals == {"USD": Decimal(1)}
        assert not caught.value.report.complete
        assert "secret-admin" not in str(caught.value)

    @pytest.mark.parametrize(
        "payload",
        [
            {"data": [], "has_more": True, "next_page": None},
            {"data": [], "has_more": "false"},
            {"data": "bad", "has_more": False},
        ],
    )
    def test_malformed_pagination_is_never_complete(self, payload):
        report = client("openai", lambda _: httpx.Response(200, json=payload)).get_costs(START, END, allow_partial=True)
        assert not report.complete
        assert report.errors

    def test_repeated_cursor_and_page_cap(self):
        payload = {"data": [], "has_more": True, "next_page": "same"}
        for max_pages in (1, 3):
            report = client("openai", lambda _: httpx.Response(200, json=payload), max_pages=max_pages).get_costs(
                START,
                END,
                allow_partial=True,
            )
            assert not report.complete
            assert report.errors

    def test_duplicate_buckets_are_not_double_counted(self):
        payload = page("openai", [{"amount": {"value": 1, "currency": "usd"}}], more=True, cursor="a")
        report = client("openai", lambda _: httpx.Response(200, json=payload)).get_costs(START, END, allow_partial=True)
        assert report.totals == {"USD": Decimal(1)}
        assert not report.complete

    @pytest.mark.parametrize("amount", [None, "NaN", "Infinity", "invalid"])
    def test_unknown_or_nonfinite_cost_never_becomes_zero(self, amount):
        payload = page("openai", [{"amount": {"value": amount, "currency": "usd"}}])
        with pytest.raises(BillingAPIError):
            client("openai", lambda _: httpx.Response(200, json=payload)).get_costs(START, END)

    def test_async_methods(self):
        api = client("openai", lambda _: httpx.Response(200, json=page("openai", [])))

        async def run():
            assert (await api.aget_costs(START, END)).complete
            assert (await api.aget_usage(START, END)).complete
            assert (await api.areport(START, END))["costs"].complete

        asyncio.run(run())

    def test_no_redirect_following(self):
        calls = []

        def handler(request):
            calls.append(request)
            return httpx.Response(302, headers={"location": "https://other.invalid"})

        with pytest.raises(BillingAPIError):
            client("openai", handler).get_costs(START, END)
        assert len(calls) == 1

    def test_decimal_json_amounts_never_pass_through_float(self):
        payload = (
            '{"data":[{"start_time":1767225600,"end_time":1767398400,"results":'
            '[{"amount":{"value":0.123456789012345678901234567,"currency":"usd"}}]}],"has_more":false}'
        )
        report = client("openai", lambda _: httpx.Response(200, text=payload)).get_costs(START, END)
        assert report.totals["USD"] == Decimal("0.123456789012345678901234567")

    @pytest.mark.parametrize("currency", [None, 123, "", "US"])
    def test_malformed_currency_returns_explicit_failure(self, currency):
        payload = page("openai", [{"amount": {"value": 1, "currency": currency}}])
        report = client("openai", lambda _: httpx.Response(200, json=payload)).get_costs(START, END, allow_partial=True)
        assert not report.complete
        assert report.errors

    def test_out_of_range_bucket_is_not_allocated_to_requested_interval(self):
        payload = page("openai", [{"amount": {"value": 9, "currency": "usd"}}], start=START - timedelta(days=1))
        report = client("openai", lambda _: httpx.Response(200, json=payload)).get_costs(START, END, allow_partial=True)
        assert not report.complete
        assert report.totals == {}


class TestBillingValidation:
    """Prevent silent credential fallback and misleading scopes."""

    @pytest.mark.parametrize(
        "provider,env,inference_env",
        [
            ("openai", "OPENAI_ADMIN_KEY", "OPENAI_API_KEY"),
            ("anthropic", "ANTHROPIC_ADMIN_KEY", "ANTHROPIC_API_KEY"),
        ],
    )
    def test_admin_credentials_only(self, monkeypatch, provider, env, inference_env):
        from prompture.infra.settings import settings

        monkeypatch.delenv(env, raising=False)
        monkeypatch.setattr(settings, env.lower(), None)
        monkeypatch.setenv(inference_env, "inference-secret")
        with pytest.raises(ValueError, match=env):
            BillingClient(provider)
        monkeypatch.setenv(env, "admin-secret")
        with BillingClient(provider) as api:
            assert "admin-secret" not in repr(api)

    @pytest.mark.parametrize("provider,env", [("openai", "OPENAI_ADMIN_KEY"), ("anthropic", "ANTHROPIC_ADMIN_KEY")])
    def test_dotenv_admin_credentials_are_used_and_remain_secret(self, monkeypatch, tmp_path, provider, env):
        import importlib

        from prompture.infra.settings import Settings

        monkeypatch.delenv(env, raising=False)
        dotenv = tmp_path / ".env"
        dotenv.write_text(f"{env}=dotenv-admin-secret\n", encoding="utf-8")
        configured = Settings(_env_file=dotenv)
        monkeypatch.setattr(importlib.import_module("prompture.infra.settings"), "settings", configured)
        assert "dotenv-admin-secret" not in repr(configured)
        assert "dotenv-admin-secret" not in configured.model_dump_json()
        requests = []

        def handler(request):
            requests.append(request)
            return httpx.Response(200, json=page(provider, []))

        transport = httpx.Client(transport=httpx.MockTransport(handler))
        api = BillingClient(provider, client=transport)
        assert "dotenv-admin-secret" not in repr(api)
        api.get_costs(START, END)
        header = "authorization" if provider == "openai" else "x-api-key"
        expected = "Bearer dotenv-admin-secret" if provider == "openai" else "dotenv-admin-secret"
        assert requests[0].headers[header] == expected
        # An explicitly empty shell value disables the .env credential.
        monkeypatch.setenv(env, "")
        with pytest.raises(ValueError, match=env):
            BillingClient(provider)

    @pytest.mark.parametrize(
        "start,end,kwargs",
        [
            (START.replace(tzinfo=None), END, {}),
            (END, START, {}),
            (START + timedelta(seconds=1), END, {}),
            (START, END, {"bucket_width": "1h"}),
            (START, END, {"filters": {"page": "injected"}}),
            (START, END, {"group_by": "project_id"}),
            (START, END, {"group_by": ["model"]}),
            (START, END, {"filters": {"project_ids": []}}),
        ],
    )
    def test_invalid_queries_fail_before_network(self, start, end, kwargs):
        def handler(_):
            pytest.fail("Invalid query must not make HTTP requests")

        with pytest.raises(ValueError):
            client("openai", handler).get_costs(start, end, **kwargs)

    def test_anthropic_cost_filters_cannot_silently_change_scope(self):
        with pytest.raises(ValueError, match="Unsupported filters"):
            client("anthropic", lambda _: pytest.fail()).get_costs(START, END, filters={"workspace_ids": ["w1"]})


class TestCostReconciliation:
    """Only compare caller-attested equivalent scope, coverage, and currency."""

    def setup_method(self):
        payload = page("openai", [{"amount": {"value": "1.20", "currency": "usd"}}])
        self.report = client("openai", lambda _: httpx.Response(200, json=payload)).get_costs(START, END)
        self.local = LocalCostSummary(self.report.scope, START, END, Decimal("1.10"))

    def test_difference_is_idempotent_and_does_not_mutate_totals(self):
        first = reconcile_costs(self.local, self.report)
        assert first == reconcile_costs(self.local, self.report)
        assert first.comparable
        assert first.difference == Decimal("0.10")
        assert first.reported_cost == Decimal("1.20")
        assert self.report.totals == {"USD": Decimal("1.20")}
        assert self.local.amount == Decimal("1.10")

    @pytest.mark.parametrize(
        "change",
        [
            {"complete": False},
            {"unknown_cost_events": 1},
            {"currency": "EUR"},
            {"end": END + timedelta(days=1)},
        ],
    )
    def test_incompatible_local_data_produces_no_difference(self, change):
        result = reconcile_costs(replace(self.local, **change), self.report)
        assert not result.comparable
        assert result.difference is None
        assert result.reasons

    def test_account_and_coverage_must_match(self):
        for scope in [
            replace(self.local.scope, organization_id=None),
            replace(self.local.scope, organization_id="other"),
            replace(self.local.scope, exclusions=("priority_tier_costs",)),
            replace(self.local.scope, filters=(("project_ids", ("p1",)),)),
        ]:
            assert not reconcile_costs(replace(self.local, scope=scope), self.report).comparable

    def test_provider_partial_or_usage_report_cannot_be_reconciled(self):
        assert not reconcile_costs(self.local, replace(self.report, complete=False)).comparable
        assert not reconcile_costs(self.local, replace(self.report, kind="usage")).comparable

    def test_empty_completed_usd_report_is_zero(self):
        result = reconcile_costs(self.local, replace(self.report, rows=[]))
        assert result.reported_cost == Decimal(0)
        assert result.difference == Decimal("-1.10")
