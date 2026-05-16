"""Tests for ``prompture.tools.web_search.WebSearchTool``.

All four backends are exercised through a mocked ``requests.Session``.
A live Tavily test is included under ``@pytest.mark.integration``.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest
import requests

from prompture import SearchResult, ToolRegistry, WebSearchTool, web_search_tool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_session(payload: dict) -> MagicMock:
    """A MagicMock session whose .get/.post both return *payload*.json()."""
    sess = MagicMock(spec=requests.Session)
    response = MagicMock()
    response.json.return_value = payload
    response.raise_for_status = MagicMock()
    sess.post.return_value = response
    sess.get.return_value = response
    return sess


# ---------------------------------------------------------------------------
# ToolDefinition shape
# ---------------------------------------------------------------------------


def test_to_tool_definition_shape():
    tool = WebSearchTool(provider="tavily", api_key="fake")
    td = tool.to_tool_definition()
    assert td.name == "web_search"
    assert td.parameters["required"] == ["query"]
    assert td.parameters["properties"]["query"]["type"] == "string"


def test_web_search_tool_shortcut():
    td = web_search_tool(provider="tavily", api_key="fake")
    assert td.name == "web_search"
    assert callable(td.function)


def test_custom_tool_name_and_description():
    tool = WebSearchTool(
        provider="tavily",
        api_key="fake",
        tool_name="search",
        tool_description="Custom desc.",
    )
    td = tool.to_tool_definition()
    assert td.name == "search"
    assert td.description == "Custom desc."


# ---------------------------------------------------------------------------
# Tavily
# ---------------------------------------------------------------------------


def test_tavily_parses_results_and_answer():
    session = make_session(
        {
            "answer": "Paris is the capital of France.",
            "results": [
                {
                    "title": "Wiki: Paris",
                    "url": "https://en.wikipedia.org/wiki/Paris",
                    "content": "Paris is the capital and most populous city of France.",
                    "score": 0.95,
                    "published_date": "2024-01-01",
                },
                {
                    "title": "France Tourism",
                    "url": "https://tourism.example",
                    "content": "Visit Paris…",
                    "score": 0.8,
                },
            ],
        }
    )
    tool = WebSearchTool(provider="tavily", api_key="fake", session=session)
    results = tool.search("capital of France")

    assert len(results) == 2
    assert isinstance(results[0], SearchResult)
    assert results[0].title == "Wiki: Paris"
    assert results[0].url == "https://en.wikipedia.org/wiki/Paris"
    assert results[0].score == 0.95
    assert results[0].extra["published_date"] == "2024-01-01"

    # Markdown output includes the synthesised answer
    md = tool.search_markdown("capital of France")
    assert "**Answer:** Paris is the capital of France." in md
    assert "Wiki: Paris" in md
    assert "<https://en.wikipedia.org/wiki/Paris>" in md


def test_tavily_sends_correct_payload():
    session = make_session({"results": []})
    tool = WebSearchTool(
        provider="tavily",
        api_key="my-secret",
        max_results=7,
        session=session,
    )
    tool.search("LangChain release")
    session.post.assert_called_once()
    args, kwargs = session.post.call_args
    assert args[0] == "https://api.tavily.com/search"
    assert kwargs["json"]["api_key"] == "my-secret"
    assert kwargs["json"]["query"] == "LangChain release"
    assert kwargs["json"]["max_results"] == 7


# ---------------------------------------------------------------------------
# Serper
# ---------------------------------------------------------------------------


def test_serper_parses_organic_results():
    session = make_session(
        {
            "organic": [
                {
                    "title": "Hit One",
                    "link": "https://one.example",
                    "snippet": "First hit snippet.",
                    "position": 1,
                },
                {
                    "title": "Hit Two",
                    "link": "https://two.example",
                    "snippet": "Second hit.",
                    "position": 2,
                },
            ],
        }
    )
    tool = WebSearchTool(provider="serper", api_key="k", session=session)
    results = tool.search("anything")

    assert [r.title for r in results] == ["Hit One", "Hit Two"]
    assert results[0].extra["position"] == 1

    session.post.assert_called_once()
    args, kwargs = session.post.call_args
    assert args[0] == "https://google.serper.dev/search"
    assert kwargs["headers"]["X-API-KEY"] == "k"


# ---------------------------------------------------------------------------
# Brave
# ---------------------------------------------------------------------------


def test_brave_parses_web_results():
    session = make_session(
        {
            "web": {
                "results": [
                    {
                        "title": "Brave Hit",
                        "url": "https://brave.example",
                        "description": "Result description.",
                        "age": "2 days ago",
                    }
                ]
            }
        }
    )
    tool = WebSearchTool(provider="brave", api_key="k", session=session)
    results = tool.search("topic")

    assert results[0].title == "Brave Hit"
    assert results[0].snippet == "Result description."
    assert results[0].extra["age"] == "2 days ago"

    session.get.assert_called_once()
    args, kwargs = session.get.call_args
    assert args[0] == "https://api.search.brave.com/res/v1/web/search"
    assert kwargs["headers"]["X-Subscription-Token"] == "k"


# ---------------------------------------------------------------------------
# SearXNG
# ---------------------------------------------------------------------------


def test_searxng_parses_results():
    session = make_session(
        {
            "results": [
                {
                    "title": "Sx Hit",
                    "url": "https://sx.example",
                    "content": "Snippet text.",
                    "score": 0.5,
                    "engine": "bing",
                }
            ]
        }
    )
    tool = WebSearchTool(
        provider="searxng",
        endpoint="https://searx.example",
        session=session,
    )
    results = tool.search("query")

    assert results[0].title == "Sx Hit"
    assert results[0].score == 0.5
    assert results[0].extra["engine"] == "bing"

    args, kwargs = session.get.call_args
    assert args[0] == "https://searx.example/search"
    assert kwargs["params"]["format"] == "json"


def test_searxng_requires_endpoint():
    with pytest.raises(RuntimeError, match="SEARXNG_ENDPOINT"):
        WebSearchTool(provider="searxng", endpoint=None)


# ---------------------------------------------------------------------------
# Auto-detect & validation
# ---------------------------------------------------------------------------


def test_unknown_provider_rejected():
    with pytest.raises(ValueError, match="Unknown provider"):
        WebSearchTool(provider="bing", api_key="k")


def test_provider_requires_api_key(monkeypatch):
    monkeypatch.setattr(
        "prompture.tools.web_search.settings",
        type(
            "S",
            (),
            {
                "tavily_api_key": None,
                "serper_api_key": None,
                "brave_search_api_key": None,
                "searxng_endpoint": None,
            },
        ),
    )
    with pytest.raises(RuntimeError, match="TAVILY_API_KEY"):
        WebSearchTool(provider="tavily")


def test_auto_detect_picks_first_configured(monkeypatch):
    monkeypatch.setattr(
        "prompture.tools.web_search.settings",
        type(
            "S",
            (),
            {
                "tavily_api_key": None,
                "serper_api_key": "ks",
                "brave_search_api_key": "kb",
                "searxng_endpoint": None,
            },
        ),
    )
    # No explicit provider → should pick Serper (first configured).
    tool = WebSearchTool()
    assert tool.provider == "serper"


def test_auto_detect_raises_when_nothing_configured(monkeypatch):
    monkeypatch.setattr(
        "prompture.tools.web_search.settings",
        type(
            "S",
            (),
            {
                "tavily_api_key": None,
                "serper_api_key": None,
                "brave_search_api_key": None,
                "searxng_endpoint": None,
            },
        ),
    )
    with pytest.raises(RuntimeError, match="No web-search provider"):
        WebSearchTool()


# ---------------------------------------------------------------------------
# Registry integration & error handling
# ---------------------------------------------------------------------------


def test_register_on_tool_registry():
    tool = WebSearchTool(
        provider="tavily",
        api_key="fake",
        session=make_session({"results": []}),
    )
    reg = ToolRegistry()
    td = tool.register_on(reg)
    assert td.name in reg
    assert reg.get("web_search") is td


def test_registry_execute_returns_markdown():
    session = make_session(
        {
            "results": [
                {
                    "title": "Hit",
                    "url": "https://example.com",
                    "content": "Body.",
                }
            ]
        }
    )
    tool = WebSearchTool(provider="tavily", api_key="fake", session=session)
    reg = ToolRegistry()
    tool.register_on(reg)
    out = reg.execute("web_search", {"query": "anything"})
    assert "Hit" in out
    assert "<https://example.com>" in out


def test_no_results_returns_friendly_message():
    tool = WebSearchTool(
        provider="tavily",
        api_key="fake",
        session=make_session({"results": []}),
    )
    out = tool("nothing matches this")
    assert 'No results for "nothing matches this"' in out


def test_network_error_returned_as_string():
    sess = MagicMock(spec=requests.Session)
    sess.post.side_effect = requests.ConnectionError("boom")
    tool = WebSearchTool(provider="tavily", api_key="fake", session=sess)
    out = tool.to_tool_definition().function(query="anything")
    assert out.startswith("Error:")
    assert "ConnectionError" in out


# ---------------------------------------------------------------------------
# Live integration (Tavily only)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_tavily_live_smoke():
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        pytest.skip("TAVILY_API_KEY not set")
    tool = WebSearchTool(provider="tavily", api_key=api_key, max_results=3)
    results = tool.search("OpenAI GPT release")
    assert results
    assert all(r.url.startswith(("http://", "https://")) for r in results)
