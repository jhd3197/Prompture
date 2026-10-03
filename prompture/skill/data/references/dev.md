# Developer sources: GitHub, HN, arXiv, packages, `gh`, `yt-dlp`

## `pack:dev`

`Agent(model, tools=["pack:dev"])` adds `github_search`, `github_read`,
`hackernews_search`, `hackernews_read`, `arxiv_search`, `arxiv_read`,
`pypi_package`, `npm_package`. No keys required; `GITHUB_TOKEN` raises GitHub
rate limits and enables discussions and code search.

## Readers and platform search directly

```python
from prompture.tools.web import read_url, search_platform
read_url("https://github.com/owner/repo")                  # README + metadata
read_url("https://github.com/owner/repo/issues/123")       # issue + comments
read_url("https://arxiv.org/abs/2401.00001")
search_platform("github", "rate limiter", kind="code")     # repositories|issues|code
```

GitHub chain: REST API (`GITHUB_TOKEN` optional) ▸ `gh` CLI.

## Wrapped CLIs (`cli:`)

`tools=["cli:gh"]` exposes read-only GitHub CLI tools: `gh_repo_view`,
`gh_repo_readme`, `gh_issue_list`, `gh_issue_view`, `gh_pr_list`,
`gh_pr_view`, `gh_search_repos`, `gh_search_issues`, `gh_search_prs`,
`gh_search_code`, `gh_api_get` (GET only). `tools=["cli:yt-dlp"]`: metadata,
subtitles, `ytsearch`.

They run without a shell, only declared subcommands are allowed, values can't
inject flags, and every run has a timeout and an output cap. A tool is only
active when its binary passes a real `--version` probe — a stale `yt-dlp`
shim shows as `broken` in `prompture doctor --only binaries`.

## Your own CLI tools

Declare them in `.prompture/tools.yaml` (project) or `~/.prompture/tools.yaml`
— a program, its allowed read-only commands and their arguments — then mount
with `tools=["cli:<name>"]`. Use `${VAR}` in `env:` to pass secrets without
writing them to the file. `prompture skill show --file references/dev.md`
prints this page; see `prompture.tools.cli.config` for the full schema.
