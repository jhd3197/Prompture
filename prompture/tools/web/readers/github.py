"""GitHub reader: repositories, files, directories, issues, pull requests, discussions, releases.

Chain: ``rest`` (api.github.com; ``GITHUB_TOKEN`` optional, raises the rate
limit and enables discussions) ▸ ``gh`` (the GitHub CLI, using its own login).
Read-only — nothing is ever written to GitHub.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, urlencode, urlsplit

from ....capabilities.errors import BackendUnavailableError
from ....capabilities.http import proxies_for
from .. import _common
from .._common import config_value, default_session, host_of, raise_for_status, run_command
from .base import BaseReader, ReadResult, StepBackend, clip, http_get

API = "https://api.github.com/"
_RESERVED_OWNERS = frozenset(
    {
        "settings",
        "orgs",
        "marketplace",
        "topics",
        "explore",
        "features",
        "sponsors",
        "login",
        "logout",
        "about",
        "pricing",
        "enterprise",
        "collections",
        "trending",
        "search",
        "notifications",
        "new",
        "apps",
        "pulls",
        "issues",
        "codespaces",
        "site",
        "security",
        "customer-stories",
        "readme",
    }
)
_LANG_BY_EXT = {
    "py": "python",
    "js": "javascript",
    "ts": "typescript",
    "tsx": "tsx",
    "jsx": "jsx",
    "rs": "rust",
    "go": "go",
    "java": "java",
    "rb": "ruby",
    "c": "c",
    "h": "c",
    "cpp": "cpp",
    "cs": "csharp",
    "sh": "bash",
    "yml": "yaml",
    "yaml": "yaml",
    "json": "json",
    "toml": "toml",
    "html": "html",
    "css": "css",
    "sql": "sql",
    "kt": "kotlin",
    "swift": "swift",
    "php": "php",
}

DISCUSSION_QUERY = """
query($owner: String!, $name: String!, $number: Int!) {
  repository(owner: $owner, name: $name) {
    discussion(number: $number) {
      title body createdAt url
      author { login }
      category { name }
      answer { body author { login } }
      comments(first: 30) { nodes { body createdAt author { login } } }
    }
  }
}
""".strip()


@dataclass
class GitHubTarget:
    """Parsed github.com URL."""

    kind: str  # repo | file | tree | issue | pr | discussion | releases
    owner: str
    repo: str
    ref: str | None = None
    path: str = ""
    number: int | None = None

    @property
    def slug(self) -> str:
        return f"{self.owner}/{self.repo}"


def parse_github_url(url: str) -> GitHubTarget | None:
    """Parse a github.com URL into a :class:`GitHubTarget` (``None`` when unsupported)."""
    if host_of(url).removeprefix("www.") != "github.com":
        return None
    try:
        segs = [s for s in urlsplit(url).path.split("/") if s]
    except ValueError:
        return None
    if len(segs) < 2 or segs[0].lower() in _RESERVED_OWNERS:
        return None
    owner, repo = segs[0], segs[1].removesuffix(".git")
    if len(segs) == 2:
        return GitHubTarget("repo", owner, repo)
    section = segs[2]
    if section == "blob" and len(segs) >= 5:
        return GitHubTarget("file", owner, repo, ref=segs[3], path="/".join(segs[4:]))
    if section == "tree" and len(segs) >= 4:
        return GitHubTarget("tree", owner, repo, ref=segs[3], path="/".join(segs[4:]))
    if section in ("issues", "pull", "discussions") and len(segs) >= 4 and segs[3].isdigit():
        kind = {"issues": "issue", "pull": "pr", "discussions": "discussion"}[section]
        return GitHubTarget(kind, owner, repo, number=int(segs[3]))
    if section == "releases":
        return GitHubTarget("releases", owner, repo)
    return None


def github_token() -> str | None:
    return config_value("github_token", "GITHUB_TOKEN") or config_value(None, "GH_TOKEN")


# ---------------------------------------------------------------------------
# Transports
# ---------------------------------------------------------------------------


class RestTransport:
    """api.github.com over HTTPS (keyless works at 60 requests/hour)."""

    name = "rest"

    def __init__(self, session: Any = None, timeout: float = 20.0) -> None:
        self.session = session
        self.timeout = timeout
        self.token = github_token()

    def _headers(self, raw: bool) -> dict[str, str]:
        h = {
            "Accept": "application/vnd.github.raw" if raw else "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    def get(self, path: str, *, raw: bool = False, params: dict[str, Any] | None = None) -> Any:
        resp = http_get(
            API + path.lstrip("/"),
            session=self.session,
            params=params,
            headers=self._headers(raw),
            timeout=self.timeout,
            backend="github",
        )
        return resp.text if raw else resp.json()

    def graphql(self, query: str, variables: dict[str, Any]) -> Any:
        if not self.token:
            raise BackendUnavailableError("GitHub GraphQL needs GITHUB_TOKEN (or the gh CLI)")
        sess = self.session or default_session()
        resp = sess.post(
            API + "graphql",
            json={"query": query, "variables": variables},
            headers=self._headers(False),
            timeout=self.timeout,
            proxies=proxies_for("github"),
        )
        raise_for_status(resp, "github")
        data = resp.json()
        if data.get("errors"):
            raise BackendUnavailableError(f"GitHub GraphQL error: {data['errors'][0].get('message')}")
        return data.get("data") or {}


class GhTransport:
    """The ``gh`` CLI (uses the user's existing ``gh auth login``)."""

    name = "gh"

    def __init__(self, timeout: float = 30.0) -> None:
        self.timeout = timeout

    def get(self, path: str, *, raw: bool = False, params: dict[str, Any] | None = None) -> Any:
        target = path.lstrip("/")
        if params:
            target += ("&" if "?" in target else "?") + urlencode(params)
        argv = [
            "gh",
            "api",
            target,
            "-H",
            "Accept: " + ("application/vnd.github.raw" if raw else "application/vnd.github+json"),
        ]
        out = run_command(argv, timeout=self.timeout)
        return out if raw else json.loads(out or "null")

    def graphql(self, query: str, variables: dict[str, Any]) -> Any:
        argv = ["gh", "api", "graphql", "-f", f"query={query}"]
        for k, v in variables.items():
            argv += ["-F" if isinstance(v, int) else "-f", f"{k}={v}"]
        data = json.loads(run_command(argv, timeout=self.timeout) or "{}")
        if data.get("errors"):
            raise BackendUnavailableError(f"GitHub GraphQL error: {data['errors'][0].get('message')}")
        return data.get("data") or {}


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _user(obj: Any) -> str:
    return (obj or {}).get("login") or "ghost"


def _comments_md(comments: list[dict[str, Any]], limit: int) -> str:
    parts = []
    for c in comments[:limit]:
        author = _user(c.get("user") or c.get("author"))
        when = (c.get("created_at") or c.get("createdAt") or "")[:10]
        parts.append(f"### @{author} — {when}\n\n{(c.get('body') or '').strip()}")
    if len(comments) > limit:
        parts.append(f"_{len(comments) - limit} more comments not shown._")
    return "\n\n".join(parts)


def read_target(transport: Any, target: GitHubTarget, url: str, *, max_comments: int = 30) -> ReadResult:
    """Render *target* using *transport* (``RestTransport`` or ``GhTransport``)."""
    repo_path = f"repos/{quote(target.owner)}/{quote(target.repo)}"
    meta: dict[str, Any] = {"repo": target.slug, "kind": target.kind}

    if target.kind == "repo":
        info = transport.get(repo_path)
        try:
            readme = transport.get(f"{repo_path}/readme", raw=True)
        except Exception:
            readme = ""
        facts = [
            f"**{info.get('full_name', target.slug)}** — {info.get('description') or 'no description'}",
            f"- Stars: {info.get('stargazers_count', 0)} · Forks: {info.get('forks_count', 0)} · "
            f"Open issues: {info.get('open_issues_count', 0)}",
            f"- Language: {info.get('language') or 'n/a'} · License: {((info.get('license') or {}).get('spdx_id')) or 'n/a'}",
            f"- Default branch: {info.get('default_branch', 'main')} · Last push: {(info.get('pushed_at') or '')[:10]}",
        ]
        if info.get("topics"):
            facts.append("- Topics: " + ", ".join(info["topics"]))
        if info.get("homepage"):
            facts.append(f"- Homepage: <{info['homepage']}>")
        if info.get("archived"):
            facts.append("- **Archived**")
        content = "\n".join(facts)
        if readme:
            content += "\n\n## README\n\n" + readme.strip()
        meta.update(
            {
                "stars": info.get("stargazers_count"),
                "forks": info.get("forks_count"),
                "language": info.get("language"),
                "default_branch": info.get("default_branch"),
                "topics": info.get("topics") or [],
            }
        )
        return ReadResult(url, info.get("full_name") or target.slug, content, "github", "repo", meta)

    if target.kind == "file":
        text = transport.get(f"{repo_path}/contents/{quote(target.path)}", raw=True, params={"ref": target.ref})
        ext = target.path.rsplit(".", 1)[-1].lower() if "." in target.path else ""
        if ext in ("md", "markdown", "rst", "txt"):
            content = text
        else:
            content = f"```{_LANG_BY_EXT.get(ext, '')}\n{text.rstrip()}\n```"
        meta.update({"path": target.path, "ref": target.ref})
        return ReadResult(url, f"{target.slug}: {target.path}", content, "github", "file", meta)

    if target.kind == "tree":
        listing = transport.get(f"{repo_path}/contents/{quote(target.path)}", params={"ref": target.ref})
        if isinstance(listing, dict):
            listing = [listing]
        rows = []
        for item in sorted(listing, key=lambda i: (i.get("type") != "dir", i.get("name", ""))):
            if item.get("type") == "dir":
                rows.append(f"- {item.get('name')}/")
            else:
                rows.append(f"- {item.get('name')} ({item.get('size', 0)} bytes)")
        meta.update({"path": target.path, "ref": target.ref, "entries": len(rows)})
        return ReadResult(url, f"{target.slug}/{target.path}".rstrip("/"), "\n".join(rows), "github", "tree", meta)

    if target.kind in ("issue", "pr"):
        issue = transport.get(f"{repo_path}/issues/{target.number}")
        comments = transport.get(f"{repo_path}/issues/{target.number}/comments", params={"per_page": max_comments})
        head = [
            f"**{'Pull request' if target.kind == 'pr' else 'Issue'} #{target.number}** · {issue.get('state')} · "
            f"opened by @{_user(issue.get('user'))} on {(issue.get('created_at') or '')[:10]}",
        ]
        labels = [lbl.get("name") for lbl in issue.get("labels") or [] if isinstance(lbl, dict)]
        if labels:
            head.append("Labels: " + ", ".join(labels))
        if target.kind == "pr":
            pr = transport.get(f"{repo_path}/pulls/{target.number}")
            head.append(
                f"{pr.get('head', {}).get('label')} → {pr.get('base', {}).get('label')} · "
                f"+{pr.get('additions', 0)} −{pr.get('deletions', 0)} in {pr.get('changed_files', 0)} files · "
                f"{'merged' if pr.get('merged') else 'not merged'}"
            )
            meta.update(
                {"merged": pr.get("merged"), "additions": pr.get("additions"), "deletions": pr.get("deletions")}
            )
        body = (issue.get("body") or "").strip() or "_No description._"
        content = "\n".join(head) + "\n\n" + body
        if comments:
            content += "\n\n## Comments\n\n" + _comments_md(list(comments), max_comments)
        meta.update(
            {"number": target.number, "state": issue.get("state"), "comments": issue.get("comments"), "labels": labels}
        )
        return ReadResult(url, issue.get("title") or f"#{target.number}", content, "github", target.kind, meta)

    if target.kind == "discussion":
        data = transport.graphql(
            DISCUSSION_QUERY, {"owner": target.owner, "name": target.repo, "number": int(target.number or 0)}
        )
        disc = ((data.get("repository") or {}).get("discussion")) or None
        if not disc:
            raise BackendUnavailableError(f"discussion #{target.number} not found")
        content = (
            f"**Discussion #{target.number}** · {(disc.get('category') or {}).get('name', '')} · "
            f"started by @{_user(disc.get('author'))} on {(disc.get('createdAt') or '')[:10]}\n\n{(disc.get('body') or '').strip()}"
        )
        if disc.get("answer"):
            ans = disc["answer"]
            content += f"\n\n## Accepted answer (@{_user(ans.get('author'))})\n\n{(ans.get('body') or '').strip()}"
        nodes = ((disc.get("comments") or {}).get("nodes")) or []
        if nodes:
            content += "\n\n## Comments\n\n" + _comments_md(nodes, max_comments)
        meta.update({"number": target.number, "comments": len(nodes)})
        return ReadResult(
            url, disc.get("title") or f"Discussion #{target.number}", content, "github", "discussion", meta
        )

    if target.kind == "releases":
        releases = transport.get(f"{repo_path}/releases", params={"per_page": 5})
        parts = []
        for rel in releases or []:
            parts.append(
                f"## {rel.get('name') or rel.get('tag_name')} ({(rel.get('published_at') or '')[:10]})\n\n"
                f"{clip((rel.get('body') or '').strip(), 3000)}"
            )
        meta["releases"] = [r.get("tag_name") for r in releases or []]
        return ReadResult(
            url, f"{target.slug} releases", "\n\n".join(parts) or "_No releases._", "github", "releases", meta
        )

    raise BackendUnavailableError(f"unsupported GitHub URL kind {target.kind!r}")


class GitHubReader(BaseReader):
    """github.com repos, files, trees, issues, PRs, discussions and releases."""

    name = "github"
    description = "GitHub repo/file/issue/PR/discussion → Markdown"

    def can_handle(self, url: str) -> bool:
        return parse_github_url(url) is not None

    def steps(self) -> list[StepBackend]:
        return [
            StepBackend("rest", self._via_rest, live=lambda: RestTransport().get("rate_limit")),
            StepBackend(
                "gh",
                self._via_gh,
                available=lambda: _common.binary_ok("gh"),
                requires=("gh",),
                hint=lambda: _common.binary_hint("gh"),
            ),
        ]

    def check(self, live: bool = False) -> Any:
        status = super().check(live)
        if status.ok and not github_token():
            status.fix_hint = "Optional: set GITHUB_TOKEN for higher rate limits and discussions"
        return status

    def _target(self, url: str) -> GitHubTarget:
        target = parse_github_url(url)
        if target is None:
            raise BackendUnavailableError(f"not a supported GitHub URL: {url}")
        return target

    def _via_rest(self, url: str, *, session: Any = None, max_comments: int = 30, **_: Any) -> ReadResult:
        return read_target(RestTransport(session=session), self._target(url), url, max_comments=max_comments)

    def _via_gh(self, url: str, *, max_comments: int = 30, **_: Any) -> ReadResult:
        return read_target(GhTransport(), self._target(url), url, max_comments=max_comments)
