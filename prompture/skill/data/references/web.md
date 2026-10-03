# Reading URLs

## `read_url` — routed readers

`read_url(url)` picks the first reader that handles the URL; anything else
goes to `web_fetch`.

| Reader | Handles | Chain |
|---|---|---|
| youtube | watch / youtu.be / shorts | `transcript_api` ▸ `yt_dlp` subtitles ▸ `transcription` ▸ `web_fetch` |
| github | repo / file / issue / PR / discussion | REST (`GITHUB_TOKEN` optional) ▸ `gh` CLI |
| hackernews | item pages | Firebase API |
| arxiv | abs / pdf | arXiv API ▸ `web_fetch` |
| wikipedia | `/wiki/*` | REST summary + HTML |
| podcasts | episode pages / audio enclosures | feed enclosure ▸ transcription |
| feeds | RSS / Atom, pages advertising a feed | feedparser ▸ stdlib XML |

Reorder one reader's steps with `PROMPTURE_READER_<NAME>_BACKENDS`, e.g.
`PROMPTURE_READER_YOUTUBE_BACKENDS=yt_dlp,transcript_api`. Force a reader with
`read_url(url, reader="feeds")`.

```python
from prompture.tools.web import read_url
r = read_url("https://news.ycombinator.com/item?id=1")
print(r.reader, r.kind, r.title)
print(r.content[:3000])
print(r.route)          # which step served it
```

## `web_fetch` — any page as Markdown

Chain: `jina_reader` (r.jina.ai, keyless; handles PDFs and JS pages,
`JINA_API_KEY` raises limits) ▸ `direct` (safe GET + HTML→Markdown; uses
`trafilatura` when installed). Reorder with `PROMPTURE_FETCH_BACKENDS`.

```python
from prompture.tools.web import web_fetch
page = web_fetch("https://example.com/long-article", max_chars=20000)
print(page.content)
if page.truncated:
    more = web_fetch(page.url, start=page.next_start)   # paging
```

Long pages end with `[truncated — call again with start=N]`. Results are
cached for 10 minutes. `compress=True` trims boilerplate for agent loops.

## Safety (don't work around these)

- Only public `http(s)` URLs. Private, loopback, link-local and cloud-metadata
  addresses are refused, including via redirects. `PROMPTURE_WEB_ALLOW_PRIVATE=1`
  exists for the user's own local services — only set it when the user asks.
- Bot-challenge / captcha pages count as a failure and move to the next
  backend. Never try to solve or bypass them; never log in with cookies.
- Size caps and timeouts apply to every fetch.

`pip install "prompture[web]"` adds feedparser, youtube-transcript-api and
trafilatura.
