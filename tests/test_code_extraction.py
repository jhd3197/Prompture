"""Tests for the code-salvage extractors (fenced blocks + raw HTML)."""

from __future__ import annotations

from prompture import (
    ExtractedHtml,
    FencedBlock,
    extract_fenced_blocks,
    extract_html_document,
)

# ---------------------------------------------------------------------------
# extract_fenced_blocks
# ---------------------------------------------------------------------------


def test_fenced_blocks_basic():
    text = "Some prose\n\n```html\n<h1>Hi</h1>\n```\n\nMore."
    blocks = extract_fenced_blocks(text)
    assert blocks == [FencedBlock(language="html", content="<h1>Hi</h1>")]


def test_fenced_blocks_multiple_languages():
    text = "```html\n<p>x</p>\n```\n```css\np { color: red; }\n```\n```js\nconsole.log(1);\n```"
    blocks = extract_fenced_blocks(text)
    assert [b.language for b in blocks] == ["html", "css", "js"]


def test_fenced_blocks_language_filter():
    text = "```bash\nls\n```\n```html\n<p/>\n```\n```python\nx=1\n```"
    blocks = extract_fenced_blocks(text, languages=["html", "css", "js"])
    assert [b.language for b in blocks] == ["html"]


def test_fenced_blocks_no_language_tag():
    text = "```\nsome code\n```"
    blocks = extract_fenced_blocks(text)
    assert blocks == [FencedBlock(language="", content="some code")]


def test_fenced_blocks_empty_block_dropped():
    text = "```html\n\n```\n```html\n<p/>\n```"
    blocks = extract_fenced_blocks(text)
    assert blocks == [FencedBlock(language="html", content="<p/>")]


def test_fenced_blocks_returns_empty_when_no_matches():
    assert extract_fenced_blocks("no fences here") == []


def test_fenced_blocks_case_insensitive_filter():
    text = "```HTML\n<p/>\n```"
    blocks = extract_fenced_blocks(text, languages=["HTML"])
    # language tag is normalised to lowercase
    assert blocks == [FencedBlock(language="html", content="<p/>")]


# ---------------------------------------------------------------------------
# extract_html_document
# ---------------------------------------------------------------------------


def test_html_document_simple():
    text = "<!DOCTYPE html><html><body><p>hi</p></body></html>"
    result = extract_html_document(text)
    assert result.found
    assert result.html.startswith("<!DOCTYPE html>")


def test_html_document_with_inline_style_and_script():
    text = (
        "<!DOCTYPE html>"
        "<html><head><style>body { color: red; }</style></head>"
        "<body><script>alert(1);</script></body></html>"
    )
    result = extract_html_document(text)
    assert result.styles == ("body { color: red; }",)
    assert result.scripts == ("alert(1);",)


def test_html_document_strips_outer_markdown_fence():
    text = "```html\n<!DOCTYPE html><html></html>\n```"
    result = extract_html_document(text)
    assert result.found
    assert result.html == "<!DOCTYPE html><html></html>"


def test_html_document_finds_bare_html_tag():
    text = "prose <html><body/></html> prose"
    result = extract_html_document(text)
    assert result.found
    assert result.html == "<html><body/></html>"


def test_html_document_no_match_returns_not_found():
    result = extract_html_document("just some text, no html")
    assert isinstance(result, ExtractedHtml)
    assert not result.found
    assert result.html == ""
    assert result.styles == ()
    assert result.scripts == ()


def test_html_document_empty_input():
    result = extract_html_document("")
    assert not result.found


def test_html_document_skips_empty_script_tags():
    text = (
        "<!DOCTYPE html><html><body>"
        "<script src='x.js'></script>"  # empty body
        "<script>doThing();</script>"
        "</body></html>"
    )
    result = extract_html_document(text)
    assert result.scripts == ("doThing();",)


def test_html_document_multiple_styles_collected():
    text = "<!DOCTYPE html><html><style>a { color: red; }</style><style>b { color: blue; }</style></html>"
    result = extract_html_document(text)
    assert result.styles == ("a { color: red; }", "b { color: blue; }")
