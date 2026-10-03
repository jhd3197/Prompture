# Web search and research

## Fallback chain

`web_search` tries keyed providers first and the keyless one last:

`tavily` ▸ `exa` ▸ `serper` ▸ `brave` ▸ `jina` ▸ `searxng` ▸ `exa_mcp` (keyless)

Unconfigured providers are skipped. Auth / quota / rate-limit errors move to
the next provider immediately; timeouts and 5xx retry once, then move on.
`PROMPTURE_SEARCH_PROVIDERS="brave,exa_mcp"` puts those first (others keep
their place after them); `providers=[...]` restricts to exactly those.

| Provider | Key |
|---|---|
| tavily | `TAVILY_API_KEY` |
| exa | `EXA_API_KEY` |
| serper | `SERPER_API_KEY` |
| brave | `BRAVE_SEARCH_API_KEY` |
| jina | `JINA_API_KEY` |
| searxng | `SEARXNG_ENDPOINT` (self-hosted, no key) |
| exa_mcp | none |

## Python

```python
from prompture.tools.web import web_search

resp = web_search("rust 2024 edition changes", max_results=5,
                  include_domains=["blog.rust-lang.org"], recency_days=365)
for r in resp.results:
    print(r.title, r.url)
print(resp.served_by, resp.route["fallback"])   # announce this
print(resp.to_markdown())                        # ends with "served by <backend>"
```

Platform search (no web index needed):

```python
from prompture.tools.web import search_platform
search_platform("github", "vector database", kind="repositories")   # repositories|issues|code
search_platform("hackernews", "sqlite")
search_platform("arxiv", "ti:mixture of experts")
search_platform("youtube", "pycon keynote")                         # needs a healthy yt-dlp
```

Agent tools: `tools=["web:search"]` (or `web:all`). `WebSearchTool()` keeps
working as before; `provider="tavily"` pins one provider.

## Cited research

```bash
prompture research "What changed in Python 3.13?" --depth quick|standard|deep \
    [--json] [--model provider/model] [--max-fetches N] [--max-cost USD] [--timeout S]
```

```python
from prompture.research import ResearchAgent, ResearchBudget
report = ResearchAgent("openai/gpt-4o-mini", depth="quick",
                       budget=ResearchBudget(max_fetches=6, max_cost=0.05)).run("question")
print(report.to_markdown())    # answer with [n] citations, conflicts, gaps, sources, budget
```

The report only cites pages it actually opened. Model defaults to
`PROMPTURE_RESEARCH_MODEL`, then the first configured provider's cheap model.
Other agents can call it as a tool: `tools=[research_tool(depth="quick")]`.

## Cache

Results are cached locally (`~/.prompture/cache/web_cache.db`). Volatile
searches (weather, prices, scores, news, "today", `recency_days <= 1`) live 10
minutes; others live hours. A cached result has `route["cached"] is True` and
`route["cache_age_s"]` — say so when it matters ("from a search 5 min ago").
For something that must be live right now, pass `use_cache=False`.

## When search fails

1. `prompture doctor --only tools --json` → look at the `web_search` row.
2. All keyed providers failing → the keyless `exa_mcp` should still answer.
3. Everything failing → network or proxy problem: set `PROMPTURE_PROXY`.
