# Finance, news and places packs

Domain packs are curated tool bundles: `Agent(model, tools=["pack:finance"])`.
Tools without their key are left out; `prompture doctor --only tools` shows
each `pack:*` row with the exact env var to set.

| Pack | Tools | Keys |
|---|---|---|
| `finance` | `stock_quote`, `stock_news`, `stock_search` | `FINNHUB_API_KEY` |
| | `crypto_price`, `crypto_search`, `crypto_trending` | none (CoinGecko) |
| `news` | `news_headlines`, `news_search` | `NEWSAPI_API_KEY` |
| | `read_feed` | none |
| `places` | `maps_geocode`, `maps_reverse_geocode`, `maps_places_search` | `GOOGLE_MAPS_API_KEY` |
| | `geocode` | `OPENCAGE_API_KEY` |
| | `country_info`, `country_search` | none |

```python
from prompture import Agent
from prompture.tools import resolve_pack_tools

agent = Agent("openai/gpt-4o-mini", tools=["pack:finance", "pack:news"])
print(agent.run("How did NVDA trade today and why?").output)

[t.name for t in resolve_pack_tools("finance")]   # only the live ones
```

## Fallback chains

- Quotes: `stock_quote` (Finnhub) ▸ `web_search` for the ticker.
- Company news: `stock_news` ▸ `news_search` ▸ `web_search` with `recency_days=7`.
- Crypto: `crypto_price` (keyless).
- Headlines: `news_headlines` ▸ `read_feed` on a known RSS feed ▸ `web_search`.
- Geocoding: `maps_geocode` ▸ `geocode` (OpenCage) ▸ `country_info` for country-level facts.

State the source and timestamp of any quote; prices may be delayed. Never
present financial data as advice.
