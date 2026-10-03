# Installing Prompture — a guide for coding agents

This page is written for an AI coding agent. A user can hand it to you with
one line:

> Install Prompture for me by following https://github.com/jhd3197/prompture/blob/main/docs/agent-install.md

Follow the steps in order. Ask the user before anything that installs
optional pieces, stores a key, or touches files outside the project.

## 1. Install the package

Use the project's own environment if it has one (a venv, `uv`, `poetry`);
otherwise the user's Python.

```bash
pip install -U prompture
prompture --help
```

If `prompture` isn't on PATH afterwards, use `python -m prompture`
and tell the user their scripts directory isn't on PATH.

## 2. See what already works

```bash
prompture doctor --json
```

This is offline: no network calls, no writes. Read it before suggesting
anything:

- `summary.worst` — `ok` means nothing is broken.
- `checks[]` — one row per provider, tool, media piece, MCP server and binary,
  with `status`, `active_backend`, `message` and `fix_hint`.

Web search, page reading and the URL readers already work with zero keys.
Tell the user that, in one sentence.

## 3. Offer optional extras as a menu

Show only the ones that matter for what the user does, and install only what
they pick:

| Pick | Command | Unlocks |
|---|---|---|
| Web extras | `pip install "prompture[web]"` | RSS/Atom feeds, YouTube transcripts without keys, better article extraction |
| Media | `pip install "prompture[media]"` | Video/podcast audio download for transcription (`yt-dlp`) |
| MCP | `pip install "prompture[mcp]"` | Mount MCP servers as tools |
| Provider SDKs | `pip install "prompture[openai]"` (or `anthropic`, `google`, `groq`) | Calling those models |
| Everything | `pip install "prompture[all]"` | All of the above |

System binaries (`ffmpeg`, `gh`, `node`) are never installed by Prompture. If
doctor marks one `missing` or `broken` and the user wants that capability,
show them the `fix_hint` and let them install it.

## 4. Configure only what the user chooses

Never ask the user to paste a key into the chat. Have them run one of these
themselves (keys are prompted with hidden input and checked live):

```bash
prompture setup                         # wizard: providers, keys, default model, proxy
prompture configure OPENAI_API_KEY      # one key, prompted
prompture configure --list              # names + masked values
```

Keys go to `~/.prompture/credentials.yaml` (owner-only). Environment
variables and `.env` always take precedence. Use `--profile work` to keep
separate sets. Every mutating command accepts `--dry-run`.

On a server or in CI, doctor and setup detect it; if a provider blocks
datacenter IPs, suggest `prompture configure PROMPTURE_PROXY http://host:port`.

## 5. Optional: MCP servers

```bash
prompture mcp add --preset exa      # keyless web search server
prompture mcp test exa
```

Only run `prompture mcp import --from <editor>` if the user asks to reuse an
editor's MCP config; it shows what it found and asks before saving.

## 6. Install the skill so future sessions know Prompture

```bash
prompture skill install --target project --dry-run   # show what would be written
prompture skill install --target project             # ./.claude/skills/prompture
# or --target claude for every project (~/.claude/skills/prompture)
```

## 7. Verify and report

```bash
prompture doctor
```

Report back in three short lines: what works now, what the user chose to
skip, and the exact command for anything still off.
