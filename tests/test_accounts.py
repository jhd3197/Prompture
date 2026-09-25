"""Provider account balance / spend sources."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from prompture.infra import accounts
from prompture.infra.accounts import (
    AccountSnapshot,
    BillingSpendSource,
    DeepSeekBalanceSource,
    MoonshotBalanceSource,
    OpenRouterKeySource,
    aget_account_snapshots,
    clear_account_cache,
    get_account_snapshot,
    get_account_snapshots,
    register_account_source,
    unregister_account_source,
)


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _json(body: Any, status: int = 200):
    def handler(request: httpx.Request) -> httpx.Response:
        handler.requests.append(request)
        return httpx.Response(status, json=body)

    handler.requests = []
    return handler


@pytest.fixture
def isolated_registry(monkeypatch):
    """Empty registry and cache, so no configured .env key reaches a real endpoint."""
    monkeypatch.setattr(accounts, "_sources", {})
    monkeypatch.setattr(accounts, "_cache", {})
    monkeypatch.setattr(accounts, "_initialized", True)


class FakeSource:
    def __init__(self, name: str = "fake", configured: bool = True, fail: bool = False) -> None:
        self.name = name
        self.provider = name
        self.configured = configured
        self.fail = fail
        self.calls = 0

    def is_configured(self) -> bool:
        return self.configured

    def fetch(self, client: httpx.Client) -> AccountSnapshot:
        self.calls += 1
        if self.fail:
            raise RuntimeError("fake returned HTTP 401")
        return AccountSnapshot(source=self.name, provider=self.provider, currency="USD", balance=12.5)


class TestOpenRouter:
    def test_key_limits_and_usage(self):
        handler = _json(
            {
                "data": {
                    "label": "sk-or-v1-abc...",
                    "limit": 20,
                    "limit_remaining": 7.25,
                    "usage": 12.75,
                    "usage_daily": 1.5,
                    "is_free_tier": False,
                }
            }
        )
        snap = OpenRouterKeySource(api_key="sk-or-test").fetch(_client(handler))
        assert (snap.balance, snap.spent, snap.limit, snap.currency, snap.period) == (
            7.25,
            12.75,
            20.0,
            "USD",
            "all_time",
        )
        assert snap.fraction_remaining == pytest.approx(7.25 / 20)
        assert snap.details["usage_daily"] == 1.5
        request = handler.requests[0]
        assert str(request.url) == "https://openrouter.ai/api/v1/key"
        assert request.headers["authorization"] == "Bearer sk-or-test"

    def test_unlimited_key_has_no_fraction(self):
        snap = OpenRouterKeySource(api_key="k").fetch(
            _client(_json({"data": {"limit": None, "limit_remaining": None, "usage": 3}}))
        )
        assert snap.limit is None
        assert snap.fraction_remaining is None


class TestDeepSeek:
    def test_prefers_usd_balance(self):
        body = {
            "is_available": True,
            "balance_infos": [
                {
                    "currency": "CNY",
                    "total_balance": "110.00",
                    "granted_balance": "10.00",
                    "topped_up_balance": "100.00",
                },
                {"currency": "USD", "total_balance": "4.20", "granted_balance": "0.20", "topped_up_balance": "4.00"},
            ],
        }
        handler = _json(body)
        snap = DeepSeekBalanceSource(api_key="sk-ds").fetch(_client(handler))
        assert (snap.currency, snap.balance, snap.available) == ("USD", 4.2, True)
        assert snap.details["granted_balance"] == 0.2
        assert len(snap.details["balances"]) == 2
        assert str(handler.requests[0].url) == "https://api.deepseek.com/user/balance"

    def test_single_currency(self):
        body = {"is_available": False, "balance_infos": [{"currency": "CNY", "total_balance": "0.00"}]}
        snap = DeepSeekBalanceSource(api_key="k").fetch(_client(_json(body)))
        assert (snap.currency, snap.balance, snap.available) == ("CNY", 0.0, False)


class TestMoonshot:
    def test_available_balance(self):
        body = {
            "code": 0,
            "data": {"available_balance": 49.5, "voucher_balance": 46.5, "cash_balance": 3.0},
            "scode": "0x0",
            "status": True,
        }
        handler = _json(body)
        snap = MoonshotBalanceSource(api_key="sk-ms", url="https://api.moonshot.ai/v1/users/me/balance").fetch(
            _client(handler)
        )
        assert (snap.balance, snap.currency, snap.available) == (49.5, "USD", True)
        assert snap.details == {"voucher_balance": 46.5, "cash_balance": 3.0}

    def test_endpoint_setting_moves_the_url(self, monkeypatch):
        monkeypatch.setenv("MOONSHOT_ENDPOINT", "https://api.moonshot.cn/v1/")
        assert MoonshotBalanceSource(api_key="k").url == "https://api.moonshot.cn/v1/users/me/balance"


class TestBillingSpend:
    def test_month_to_date_openai_costs(self):
        def handler(request: httpx.Request) -> httpx.Response:
            params = dict(request.url.params)
            page = {
                "data": [
                    {
                        "start_time": int(params["start_time"]),
                        "end_time": int(params["end_time"]),
                        "results": [
                            {"amount": {"value": 1.25, "currency": "usd"}},
                            {"amount": {"value": 0.5, "currency": "usd"}},
                        ],
                    }
                ],
                "has_more": False,
            }
            return httpx.Response(200, content=json.dumps(page))

        snap = BillingSpendSource("openai", admin_key="sk-admin").fetch(_client(handler))
        assert (snap.provider, snap.currency, snap.spent, snap.period) == ("openai", "USD", 1.75, "month")
        assert snap.details["complete"] is True

    def test_anthropic_maps_to_claude_provider(self):
        source = BillingSpendSource("anthropic", admin_key="sk-ant-admin")
        assert (source.name, source.provider) == ("anthropic_billing", "claude")

    def test_rejects_other_providers(self):
        with pytest.raises(ValueError):
            BillingSpendSource("groq")


class TestRegistry:
    def test_only_configured_sources_are_fetched(self, isolated_registry):
        on, off = FakeSource("on"), FakeSource("off", configured=False)
        register_account_source(on)
        register_account_source(off)
        snaps = get_account_snapshots(client=_client(_json({})))
        assert list(snaps) == ["on"]
        assert snaps["on"].balance == 12.5
        assert off.calls == 0

    def test_results_are_cached_until_max_age(self, isolated_registry):
        source = FakeSource()
        register_account_source(source)
        client = _client(_json({}))
        get_account_snapshots(client=client)
        get_account_snapshots(client=client)
        assert source.calls == 1
        get_account_snapshots(client=client, max_age=0)
        assert source.calls == 2
        clear_account_cache()
        get_account_snapshots(client=client)
        assert source.calls == 3

    def test_failures_become_error_snapshots(self, isolated_registry):
        register_account_source(FakeSource("broken", fail=True))
        snap = get_account_snapshot("broken", client=_client(_json({})))
        assert snap is not None
        assert snap.error == "fake returned HTTP 401"
        assert snap.balance is None
        assert snap.to_dict()["error"] == "fake returned HTTP 401"

    def test_http_errors_report_status_only(self, isolated_registry):
        register_account_source(OpenRouterKeySource(api_key="k"))
        snap = get_account_snapshot("openrouter", client=_client(_json({"error": "secret detail"}, status=401)))
        assert snap is not None
        assert snap.error == "openrouter returned HTTP 401"

    def test_duplicate_names_need_replace(self, isolated_registry):
        register_account_source(FakeSource())
        with pytest.raises(ValueError):
            register_account_source(FakeSource())
        register_account_source(FakeSource(), replace=True)
        assert unregister_account_source("fake") is True
        assert unregister_account_source("fake") is False

    def test_unknown_name_is_none(self, isolated_registry):
        assert get_account_snapshot("nope") is None

    def test_async_wrapper(self, isolated_registry):
        register_account_source(FakeSource())
        snaps = asyncio.run(aget_account_snapshots(client=_client(_json({}))))
        assert snaps["fake"].balance == 12.5

    def test_builtin_sources_register_once(self, monkeypatch):
        monkeypatch.setattr(accounts, "_sources", {})
        monkeypatch.setattr(accounts, "_initialized", False)
        names = [s.name for s in accounts.get_account_sources()]
        assert names == ["openrouter", "deepseek", "moonshot", "openai_billing", "anthropic_billing"]
