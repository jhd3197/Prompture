"""Shipped CLI tool definitions: ``gh`` (GitHub read + search) and ``yt-dlp``.

Both are read-only by construction — only the commands below are allowed.

``gh`` allowlist: ``repo view``, ``issue list|view``, ``pr list|view``,
``search repos|issues|prs|code`` and ``api``. ``gh api`` is forced to
``--method GET`` and never receives ``-f``/``-F``/``--input``/``-H``, so it
cannot write; the endpoint must be a relative REST path (no URLs, no
``graphql``, no ``..``).

``yt-dlp`` allowlist: metadata (``--dump-json --skip-download``), subtitle
listing, subtitle download into a private temporary directory, and
``ytsearchN:`` search. ``--ignore-config`` keeps user config files (which
may contain ``--exec`` or cookie options) out of agent runs.
"""

from __future__ import annotations

import re
from typing import Any

from .adapter import CLIArg, CLICommand, CLITool

# owner/name
_REPO_PATTERN = r"[A-Za-z0-9_.-]{1,100}/[A-Za-z0-9_.-]{1,100}"
_LABEL_PATTERN = r"[^\x00-\x1f]{1,100}"
_LANG_PATTERN = r"[A-Za-z0-9+#._ -]{1,40}"
_HTTP_URL_PATTERN = r"https?://[^\s\x00-\x1f]{1,2040}"
# REST path with optional query; {owner}/{repo} placeholders allowed.
_API_ENDPOINT_PATTERN = r"/?[A-Za-z0-9._~{}/%-]{1,400}(\?[A-Za-z0-9._~%=&,+:-]{0,400})?"

_ISSUE_FIELDS = "number,title,state,author,labels,createdAt,updatedAt,url"
_ISSUE_VIEW_FIELDS = (
    "number,title,state,author,body,labels,assignees,milestone,createdAt,updatedAt,closedAt,url,comments"
)
_PR_FIELDS = "number,title,state,author,headRefName,baseRefName,isDraft,labels,createdAt,updatedAt,url"
_PR_VIEW_FIELDS = (
    "number,title,state,author,body,headRefName,baseRefName,isDraft,labels,reviewDecision,"
    "additions,deletions,changedFiles,files,createdAt,updatedAt,mergedAt,closedAt,url,comments"
)
_REPO_FIELDS = (
    "nameWithOwner,description,url,homepageUrl,stargazerCount,forkCount,primaryLanguage,licenseInfo,"
    "repositoryTopics,createdAt,updatedAt,pushedAt,isArchived,isFork,defaultBranchRef,latestRelease"
)
_SEARCH_REPO_FIELDS = "fullName,description,url,stargazersCount,forksCount,language,license,updatedAt,isArchived"
_SEARCH_ISSUE_FIELDS = "number,title,state,repository,author,labels,commentsCount,createdAt,updatedAt,url"
_SEARCH_CODE_FIELDS = "path,repository,url,textMatches"

_GH_ENV = {"GH_PROMPT_DISABLED": "1", "GH_NO_UPDATE_NOTIFIER": "1", "NO_COLOR": "1", "CLICOLOR": "0"}


def _repo_arg(required: bool = False) -> CLIArg:
    return CLIArg(
        "repo",
        description="Repository as OWNER/NAME.",
        flag="--repo",
        pattern=_REPO_PATTERN,
        required=required,
    )


def _limit_arg(default: int = 20, maximum: int = 100) -> CLIArg:
    return CLIArg(
        "limit",
        type="integer",
        description=f"Maximum results (1-{maximum}).",
        flag="--limit",
        default=default,
        minimum=1,
        maximum=maximum,
    )


def _query_arg(description: str) -> CLIArg:
    # gh search treats a leading '-' as an exclusion qualifier (``-label:bug``);
    # it is safe because the query follows ``--``.
    return CLIArg("query", description=description, required=True, allow_dash=True, max_length=500)


def _validate_api(params: dict[str, Any]) -> str | None:
    endpoint = str(params.get("endpoint") or "")
    path = endpoint.split("?", 1)[0].strip("/")
    if path.lower() == "graphql" or path.lower().startswith("graphql/"):
        return "gh api: the GraphQL endpoint is not allowed (it is POST-only); use a REST path."
    if ".." in path.split("/"):
        return "gh api: '..' is not allowed in the endpoint."
    return None


def gh_tool() -> CLITool:
    """GitHub CLI wrapper: repo/issue/PR read and search (read-only)."""
    return CLITool(
        name="gh",
        command="gh",
        description="GitHub CLI (read-only): repositories, issues, pull requests, search and REST GET.",
        version_args=("--version",),
        live_check_args=("auth", "status"),
        timeout=45,
        max_output_bytes=200_000,
        env=dict(_GH_ENV),
        install_hint="Install the GitHub CLI: https://cli.github.com/ and run `gh auth login`.",
        commands=[
            CLICommand(
                "repo_view",
                ("repo", "view"),
                "Show a GitHub repository's metadata (stars, language, license, topics, latest release).",
                args=[
                    CLIArg(
                        "repo",
                        description="Repository as OWNER/NAME (defaults to the current directory's repo).",
                        pattern=_REPO_PATTERN,
                    )
                ],
                fixed_args=("--json", _REPO_FIELDS),
                output="json",
                end_of_options=True,
            ),
            CLICommand(
                "repo_readme",
                ("repo", "view"),
                "Show a GitHub repository's description and README as text.",
                args=[CLIArg("repo", description="Repository as OWNER/NAME.", required=True, pattern=_REPO_PATTERN)],
                end_of_options=True,
            ),
            CLICommand(
                "issue_list",
                ("issue", "list"),
                "List issues in a GitHub repository.",
                args=[
                    _repo_arg(required=True),
                    CLIArg("state", description="Issue state.", flag="--state", enum=["open", "closed", "all"]),
                    CLIArg("label", description="Only issues with this label.", flag="--label", pattern=_LABEL_PATTERN),
                    CLIArg(
                        "search", description="GitHub search qualifiers to filter by.", flag="--search", max_length=300
                    ),
                    _limit_arg(),
                ],
                fixed_args=("--json", _ISSUE_FIELDS),
                output="json",
            ),
            CLICommand(
                "issue_view",
                ("issue", "view"),
                "Show one GitHub issue with its body and comments.",
                args=[
                    CLIArg("number", type="integer", description="Issue number.", required=True, minimum=1),
                    _repo_arg(required=True),
                ],
                fixed_args=("--json", _ISSUE_VIEW_FIELDS),
                output="json",
                end_of_options=True,
            ),
            CLICommand(
                "pr_list",
                ("pr", "list"),
                "List pull requests in a GitHub repository.",
                args=[
                    _repo_arg(required=True),
                    CLIArg("state", description="PR state.", flag="--state", enum=["open", "closed", "merged", "all"]),
                    CLIArg("label", description="Only PRs with this label.", flag="--label", pattern=_LABEL_PATTERN),
                    CLIArg(
                        "search", description="GitHub search qualifiers to filter by.", flag="--search", max_length=300
                    ),
                    _limit_arg(),
                ],
                fixed_args=("--json", _PR_FIELDS),
                output="json",
            ),
            CLICommand(
                "pr_view",
                ("pr", "view"),
                "Show one GitHub pull request with body, changed files and comments.",
                args=[
                    CLIArg("number", type="integer", description="Pull request number.", required=True, minimum=1),
                    _repo_arg(required=True),
                ],
                fixed_args=("--json", _PR_VIEW_FIELDS),
                output="json",
                end_of_options=True,
            ),
            CLICommand(
                "search_repos",
                ("search", "repos"),
                "Search GitHub repositories. Supports qualifiers like `language:python stars:>100`.",
                args=[
                    _query_arg("Search query with optional GitHub qualifiers."),
                    CLIArg("language", description="Filter by language.", flag="--language", pattern=_LANG_PATTERN),
                    CLIArg(
                        "sort",
                        description="Sort field.",
                        flag="--sort",
                        enum=["stars", "forks", "updated", "help-wanted-issues"],
                    ),
                    _limit_arg(default=10),
                ],
                fixed_args=("--json", _SEARCH_REPO_FIELDS),
                output="json",
                end_of_options=True,
            ),
            CLICommand(
                "search_issues",
                ("search", "issues"),
                "Search GitHub issues across repositories (prefix a qualifier with '-' to exclude it).",
                args=[
                    _query_arg("Search query with optional qualifiers (`is:open label:bug`)."),
                    _repo_arg(),
                    CLIArg("state", description="Issue state.", flag="--state", enum=["open", "closed"]),
                    _limit_arg(default=10),
                ],
                fixed_args=("--json", _SEARCH_ISSUE_FIELDS),
                output="json",
                end_of_options=True,
            ),
            CLICommand(
                "search_prs",
                ("search", "prs"),
                "Search GitHub pull requests across repositories.",
                args=[
                    _query_arg("Search query with optional qualifiers."),
                    _repo_arg(),
                    CLIArg("state", description="PR state.", flag="--state", enum=["open", "closed"]),
                    _limit_arg(default=10),
                ],
                fixed_args=("--json", _SEARCH_ISSUE_FIELDS),
                output="json",
                end_of_options=True,
            ),
            CLICommand(
                "search_code",
                ("search", "code"),
                "Search code on GitHub (requires `gh auth login`).",
                args=[
                    _query_arg("Code search query."),
                    _repo_arg(),
                    CLIArg("language", description="Filter by language.", flag="--language", pattern=_LANG_PATTERN),
                    _limit_arg(default=10),
                ],
                fixed_args=("--json", _SEARCH_CODE_FIELDS),
                output="json",
                end_of_options=True,
            ),
            CLICommand(
                "api_get",
                ("api",),
                "GET a GitHub REST API path (read-only), e.g. `repos/OWNER/NAME/releases/latest`.",
                args=[
                    CLIArg(
                        "endpoint",
                        description="REST path relative to https://api.github.com, optional query string.",
                        required=True,
                        pattern=_API_ENDPOINT_PATTERN,
                    ),
                    CLIArg(
                        "jq", description="Optional jq filter applied to the response.", flag="--jq", max_length=500
                    ),
                ],
                fixed_args=("--method", "GET"),
                output="json",
                end_of_options=True,
                validate=_validate_api,
            ),
        ],
    )


# ---------------------------------------------------------------------------
# yt-dlp
# ---------------------------------------------------------------------------

_YTDLP_BASE = ("--ignore-config", "--no-warnings", "--no-progress")

_VIDEO_KEYS = (
    "id",
    "title",
    "fulltitle",
    "uploader",
    "channel",
    "channel_url",
    "upload_date",
    "duration",
    "view_count",
    "like_count",
    "comment_count",
    "description",
    "tags",
    "categories",
    "chapters",
    "language",
    "webpage_url",
    "extractor",
    "is_live",
    "availability",
)
_SEARCH_KEYS = ("id", "title", "url", "channel", "uploader", "duration", "view_count", "description")

_VTT_TIMING = re.compile(r"^\d{1,2}:?\d{2}:\d{2}[.,]\d{3}\s+-->\s+")
_VTT_TAG = re.compile(r"<[^>]+>")


def vtt_to_text(vtt: str) -> str:
    """Flatten WebVTT/SRT subtitles to plain text, dropping cues' rolling duplicates."""
    lines: list[str] = []
    last = ""
    for raw in vtt.splitlines():
        line = raw.strip()
        if not line or line.startswith(("WEBVTT", "Kind:", "Language:", "NOTE", "STYLE")) or line.isdigit():
            continue
        if _VTT_TIMING.match(line) or "-->" in line:
            continue
        line = _VTT_TAG.sub("", line).replace("&nbsp;", " ").replace("&amp;", "&").strip()
        if line and line != last:
            lines.append(line)
            last = line
    return "\n".join(lines)


def _subtitles_postprocess(text: str, params: dict[str, Any]) -> str:
    if not params.get("plain_text", True):
        return text
    sections = re.split(r"(?m)^## ", text)
    out: list[str] = []
    for section in sections:
        if not section.strip():
            continue
        header, _, body = section.partition("\n")
        out.append(f"## {header}\n{vtt_to_text(body)}")
    return "\n\n".join(out)


def _url_arg() -> CLIArg:
    return CLIArg("url", description="Video or audio page URL (http/https).", required=True, pattern=_HTTP_URL_PATTERN)


def yt_dlp_tool() -> CLITool:
    """yt-dlp wrapper: metadata, subtitle listing/download and YouTube search (no media download)."""
    return CLITool(
        name="yt-dlp",
        command="yt-dlp",
        prefix="ytdlp",
        description="yt-dlp (read-only): video metadata, subtitles and YouTube search; never downloads media.",
        version_args=("--version",),
        timeout=90,
        max_output_bytes=400_000,
        install_hint="Install yt-dlp: `pip install -U yt-dlp` (or your package manager).",
        commands=[
            CLICommand(
                "metadata",
                (),
                "Get metadata for a video URL (title, channel, duration, views, description, chapters).",
                args=[_url_arg()],
                fixed_args=(*_YTDLP_BASE, "--dump-json", "--skip-download", "--no-playlist"),
                output="json",
                json_keys=_VIDEO_KEYS,
                end_of_options=True,
            ),
            CLICommand(
                "list_subs",
                (),
                "List the subtitle and auto-caption languages available for a video URL.",
                args=[_url_arg()],
                fixed_args=(*_YTDLP_BASE, "--list-subs", "--skip-download", "--no-playlist"),
                end_of_options=True,
            ),
            CLICommand(
                "subtitles",
                (),
                "Fetch a video's subtitles (or auto-captions) as text.",
                args=[
                    _url_arg(),
                    CLIArg(
                        "languages",
                        description="Subtitle languages, comma separated (yt-dlp syntax, e.g. `en.*,es`).",
                        flag="--sub-langs",
                        default="en.*,en",
                        pattern=r"[A-Za-z0-9.*,_-]{1,60}",
                    ),
                    CLIArg(
                        "plain_text",
                        type="boolean",
                        description="Strip timestamps and markup (default true).",
                        default=True,
                        emit=False,
                    ),
                ],
                fixed_args=(
                    *_YTDLP_BASE,
                    "--skip-download",
                    "--no-playlist",
                    "--write-subs",
                    "--write-auto-subs",
                    "--sub-format",
                    "vtt/srt/best",
                    "--paths",
                    "{tmpdir}",
                    "--output",
                    "%(id)s.%(ext)s",
                ),
                temp_dir=True,
                collect="*",
                postprocess=_subtitles_postprocess,
                end_of_options=True,
            ),
            CLICommand(
                "search",
                (),
                "Search YouTube and list matching videos (title, URL, channel, duration, views).",
                args=[
                    CLIArg(
                        "query",
                        description="Search terms.",
                        required=True,
                        max_length=300,
                        template="ytsearch{max_results}:{value}",
                    ),
                    CLIArg(
                        "max_results",
                        type="integer",
                        description="Number of results (1-25).",
                        default=5,
                        minimum=1,
                        maximum=25,
                        emit=False,
                    ),
                ],
                fixed_args=(*_YTDLP_BASE, "--dump-json", "--flat-playlist", "--skip-download"),
                output="jsonl",
                json_keys=_SEARCH_KEYS,
                end_of_options=True,
            ),
        ],
    )


def builtin_cli_tools() -> dict[str, CLITool]:
    """Fresh instances of the shipped CLI tools, keyed by name."""
    return {t.name: t for t in (gh_tool(), yt_dlp_tool())}
