---
name: prompture
description: Use when you need to search the web, read a URL (pages, PDFs, YouTube, GitHub, Hacker News, arXiv, Wikipedia, RSS, podcasts), transcribe or summarize video/audio, research a question with citations, call or compare LLMs from Python, extract structured JSON, mount MCP servers, use finance/news/dev/places tool packs, or check which of these work on this machine. Covers the `prompture` CLI and Python library.
---

# Prompture

Prompture gives agents working capabilities — search, read, watch, listen,
call models and external tools — and tells you exactly what works on this
machine and how to fix what doesn't. Web search and URL reading have keyless
defaults; keys only raise their limits or quality. Model calls, transcription,
packs and MCP servers need the provider, key, package or binary that doctor
names. Where a capability has more than one permitted backend, it fails over
and reports which one served the request and whether a fallback happened.
Transcription never moves audio to a second provider without explicit consent.

Install: `pip install prompture` (extras: `prompture[web]`, `prompture[media]`,
`prompture[mcp]`, `prompture[all]`).

## Standing rules

1. **Check before multi-backend work.** Run `prompture doctor --json` once at
   the start of a session that will search, fetch, transcribe or use MCP. Read
   `checks[].status` and `checks[].active_backend`; don't guess what's
   configured. `--only tools|media|mcp|providers|binaries` narrows it.
2. **Announce the backend.** Results carry `served_by` / `route`. Tell the user
   which backend answered (e.g. "searched via exa_mcp", "transcribed with
   openai"), and say so when a fallback happened (`route["fallback"]`).
3. **Follow the documented fallback chain on failure.** Each reference page
   lists the chain. Use it in order — don't improvise scrapers, don't hit
   private/internal URLs, don't drive a browser or reuse cookies.
4. **Fix with the exact hint.** When doctor shows a non-ok row, quote its
   `fix_hint` (env var to set, `pip install prompture[x]`, binary to install).
   Never install system packages or binaries yourself; offer the command.
5. **Ask before moving data.** Audio/documents go to a second provider only
   with explicit consent (`allow_provider_fallback=True`). Never echo API keys;
   use `prompture configure KEY` (hidden prompt) to store them.
6. **Cite what you opened.** When answering from the web, cite URLs you
   actually fetched/read, not just search snippets.
7. **Updates.** After a substantial task, if `prompture check-update --json`
   reports `update_available`, mention it once (new version + one-line
   highlight). Never interrupt work for it and never repeat it for the same
   version (`prompture.infra.updates.should_announce` / `mark_announced`).

## Routing table

| Task | Start with | Reference |
|---|---|---|
| Web search, current events | `web_search(...)` / `tools=["web:search"]` | [references/search.md](references/search.md) |
| Read a page, PDF, YouTube, GitHub, HN, arXiv, Wikipedia, feed, podcast | `read_url(url)` / `web_fetch(url)` | [references/web.md](references/web.md) |
| Transcribe / summarize video or audio | `prompture transcribe <url>` | [references/media.md](references/media.md) |
| Call an LLM, structured output, failover, cost | `extract_with_model`, `Agent`, `resilient` | [references/models.md](references/models.md) |
| Mount an MCP server as tools | `prompture mcp add --preset exa` → `tools=["mcp:exa"]` | [references/mcp.md](references/mcp.md) |
| Quotes, company/crypto news, headlines, geocoding | `tools=["pack:finance"]` etc. | [references/finance.md](references/finance.md) |
| GitHub, HN, arXiv, PyPI, npm, `gh`, `yt-dlp` | `tools=["pack:dev"]`, `tools=["cli:gh"]` | [references/dev.md](references/dev.md) |
| Multi-source cited research | `prompture research "question"` | [references/search.md](references/search.md) |
| What works here / how to fix | `prompture doctor` | this file |

## Quick commands

```bash
prompture doctor [--live] [--json] [--only tools]   # health, active backend, exact fix
prompture research "question" --depth quick --json   # cited multi-source report
prompture transcribe <url|file> --summary            # transcript + key points
prompture mcp add --preset exa && prompture mcp test exa
prompture setup                                      # wizard: keys (validated), default model
prompture configure OPENAI_API_KEY                   # store one key (prompted, hidden)
prompture check-update                               # newer version + highlights
prompture skill install --target project             # put this skill in ./.claude/skills
```

## Python in one block

```python
from prompture import Agent
from prompture.tools.web import web_search, read_url

print(web_search("latest CPython release", max_results=3).to_markdown())
print(read_url("https://youtu.be/<id>").content[:2000])

agent = Agent("openai/gpt-4o-mini", tools=["web:all", "pack:dev"],
              system_prompt="Cite every fact with a URL you opened.")
print(agent.run("What changed in the latest pandas release?").output)
```

Tool specs accepted by agents: `web:all|search|fetch|read|platform|media`,
`mcp:<server>` (or `mcp:<server>/tool1,tool2`, `mcp:all`),
`pack:finance|news|dev|places|all`, `cli:gh|yt-dlp|all|<your tool>`.
