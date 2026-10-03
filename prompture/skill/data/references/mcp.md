# MCP servers

Mount named MCP servers as agent tools: `Agent(model, tools=["mcp:<name>"])`.
Needs `pip install "prompture[mcp]"`.

## Add, test, use

```bash
prompture mcp add --preset exa                 # keyless web search over HTTP
prompture mcp add --preset fetch               # needs uvx; memory/filesystem need npx
prompture mcp add search --url https://host/mcp --header 'Authorization=Bearer ${SEARCH_TOKEN}'
prompture mcp add local --command python --arg -m --arg my_server
prompture mcp list [--json] [--presets]
prompture mcp test exa [--json]                # initialize + list tools
prompture mcp remove|enable|disable <name>
```

Presets: `exa`, `deepwiki`, `github` (needs `GITHUB_TOKEN`), `fetch`, `time`,
`git` (uvx), `memory`, `filesystem`, `sequential-thinking` (npx).

```python
from prompture import Agent
agent = Agent("openai/gpt-4o-mini", tools=["mcp:exa"])
agent = Agent("openai/gpt-4o-mini", tools=["mcp:github/search_repositories,get_file_contents"])
```

Tools are named `<server>__<tool>`. `mcp:all` mounts every enabled server.
A preset name works without registering it first (`mcp:exa`). Sessions start
lazily and stay pooled; each call lands in the usage ledger.

## Registry and secrets

- `~/.prompture/mcp.json` (user) and `./.prompture/mcp.json` (project, wins
  by name; `--project` writes there).
- Secrets are stored **only** as `${ENV}` references and resolved at connect
  time. Literal secret-looking values are refused. Never paste a token into
  `--env` / `--header`; set the env var and reference it (single-quote the
  argument so your shell doesn't expand it).
- `prompture mcp import --from claude-desktop|claude-code|cursor|vscode|windsurf`
  reads that editor's config only after the user confirms; inline secrets are
  converted to `${ENV}` references and the needed env vars are listed.

## When a server fails

`prompture doctor --only mcp` (offline: config, env vars, launcher probe) and
`prompture doctor --only mcp --live` (initialize + list tools).
