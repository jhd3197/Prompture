"""Unit tests for Phase 10 RAG document loaders."""

from __future__ import annotations

import asyncio
import builtins
import json
import sys
from io import BytesIO
from pathlib import Path

import pytest

from prompture.rag import (
    CSVLoader,
    Document,
    JSONLoader,
    MarkdownLoader,
    PDFLoader,
    TextLoader,
    get_async_loader_for_path,
    get_loader,
    get_loader_for_path,
)
from prompture.rag.loaders.docx_loader import DOCXLoader
from prompture.rag.loaders.epub_loader import EPUBLoader
from prompture.rag.loaders.html_loader import HTMLLoader
from prompture.rag.loaders.xlsx_loader import XLSXLoader

# ── TextLoader ───────────────────────────────────────────────────────────────


def test_text_loader_path(tmp_path: Path) -> None:
    p = tmp_path / "hello.txt"
    p.write_text("hello world", encoding="utf-8")
    docs = TextLoader().load(p)
    assert len(docs) == 1
    assert docs[0].content == "hello world"
    assert docs[0].metadata["type"] == "text"
    assert docs[0].metadata["source"] == str(p)


def test_text_loader_bytes() -> None:
    docs = TextLoader().load(b"abc")
    assert docs[0].content == "abc"
    assert docs[0].metadata["source"] == "<bytes>"


def test_text_loader_len_dunder() -> None:
    doc = Document(content="hello")
    assert len(doc) == 5


# ── PDFLoader ────────────────────────────────────────────────────────────────


def _make_pdf_bytes() -> bytes:
    pypdf = pytest.importorskip("pypdf")
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.add_blank_page(width=72, height=72)
    buf = BytesIO()
    writer.write(buf)
    return buf.getvalue()


def test_pdf_loader_pages_mode() -> None:
    pytest.importorskip("pypdf")
    pdf_bytes = _make_pdf_bytes()
    docs = PDFLoader().load(pdf_bytes)
    assert len(docs) == 2
    assert docs[0].metadata["page"] == 1
    assert docs[0].metadata["total_pages"] == 2
    assert docs[0].metadata["type"] == "pdf"


def test_pdf_loader_single_mode() -> None:
    pytest.importorskip("pypdf")
    pdf_bytes = _make_pdf_bytes()
    docs = PDFLoader().load(pdf_bytes, mode="single")
    assert len(docs) == 1
    assert docs[0].metadata["total_pages"] == 2


def test_pdf_loader_missing_dep_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # Hide pypdf so the lazy import fails.
    real_import = builtins.__import__

    def _block(name, *a, **kw):
        if name == "pypdf" or name.startswith("pypdf."):
            raise ImportError("blocked")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", _block)
    # Also evict any cached module reference.
    monkeypatch.delitem(sys.modules, "pypdf", raising=False)
    with pytest.raises(ImportError, match=r"prompture\[rag-pdf\]"):
        PDFLoader().load(b"%PDF-1.4\n")


# ── DOCXLoader ───────────────────────────────────────────────────────────────


def test_docx_loader(tmp_path: Path) -> None:
    docx = pytest.importorskip("docx")
    p = tmp_path / "sample.docx"
    d = docx.Document()
    d.add_paragraph("First paragraph.")
    d.add_paragraph("Second paragraph.")
    d.save(str(p))
    docs = DOCXLoader().load(p)
    assert len(docs) == 1
    assert "First paragraph." in docs[0].content
    assert "Second paragraph." in docs[0].content
    assert docs[0].metadata["paragraph_count"] == 2
    assert docs[0].metadata["type"] == "docx"


def test_docx_loader_missing_dep(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def _block(name, *a, **kw):
        if name == "docx":
            raise ImportError("blocked")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", _block)
    monkeypatch.delitem(sys.modules, "docx", raising=False)
    with pytest.raises(ImportError, match=r"prompture\[rag-docx\]"):
        DOCXLoader().load(b"PK\x03\x04")


# ── HTMLLoader ───────────────────────────────────────────────────────────────


def test_html_loader_text_mode(tmp_path: Path) -> None:
    pytest.importorskip("bs4")
    html = (
        "<html><head><title>Hi</title></head><body>"
        "<script>alert('x')</script>"
        "<style>.a{}</style>"
        "<p>Hello <b>World</b></p>"
        "</body></html>"
    )
    p = tmp_path / "page.html"
    p.write_text(html, encoding="utf-8")
    docs = HTMLLoader().load(p)
    assert len(docs) == 1
    assert "alert" not in docs[0].content
    assert ".a{}" not in docs[0].content
    assert "Hello" in docs[0].content
    assert docs[0].metadata["title"] == "Hi"
    assert docs[0].metadata["type"] == "html"


def test_html_loader_markdown_mode(tmp_path: Path) -> None:
    pytest.importorskip("bs4")
    pytest.importorskip("markdownify")
    html = "<html><body><h1>Title</h1><p>body</p></body></html>"
    p = tmp_path / "page.html"
    p.write_text(html, encoding="utf-8")
    docs = HTMLLoader().load(p, mode="markdown")
    assert "# Title" in docs[0].content or "Title" in docs[0].content
    assert "body" in docs[0].content


# ── MarkdownLoader ───────────────────────────────────────────────────────────


def test_markdown_loader_raw(tmp_path: Path) -> None:
    md = "# H1\n\nintro\n\n## Sub\n\nbody"
    p = tmp_path / "x.md"
    p.write_text(md, encoding="utf-8")
    docs = MarkdownLoader().load(p)
    assert len(docs) == 1
    assert docs[0].content == md
    assert docs[0].metadata["type"] == "markdown"


def test_markdown_loader_by_heading(tmp_path: Path) -> None:
    md = "preamble\n\n# Title\n\nA body\n\n## Sub\n\nmore"
    p = tmp_path / "y.md"
    p.write_text(md, encoding="utf-8")
    docs = MarkdownLoader().load(p, mode="by_heading")
    headings = [d.metadata["heading"] for d in docs]
    assert "Title" in headings
    assert "Sub" in headings
    # Preamble emitted as its own doc with heading=None.
    assert any(d.metadata.get("heading") is None for d in docs)


# ── JSONLoader ───────────────────────────────────────────────────────────────


def test_json_loader_single(tmp_path: Path) -> None:
    p = tmp_path / "a.json"
    p.write_text(json.dumps({"k": 1}), encoding="utf-8")
    docs = JSONLoader().load(p)
    assert len(docs) == 1
    assert '"k": 1' in docs[0].content
    assert docs[0].metadata["type"] == "json"


def test_jsonl_loader(tmp_path: Path) -> None:
    p = tmp_path / "a.jsonl"
    p.write_text('{"a":1}\n{"b":2}\n', encoding="utf-8")
    docs = JSONLoader().load(p)
    assert len(docs) == 2
    assert docs[0].metadata["line_number"] == 1
    assert docs[1].metadata["line_number"] == 2
    assert docs[0].metadata["type"] == "jsonl"


def test_json_loader_dotted_path(tmp_path: Path) -> None:
    p = tmp_path / "items.json"
    p.write_text(json.dumps({"items": [{"text": "a"}, {"text": "b"}]}), encoding="utf-8")
    docs = JSONLoader().load(p, jq_schema="items[].text")
    assert len(docs) == 2
    assert docs[0].content == "a"
    assert docs[1].content == "b"


# ── CSVLoader ────────────────────────────────────────────────────────────────


def test_csv_loader_rows_mode(tmp_path: Path) -> None:
    csv_text = "name,age\nAlice,30\nBob,25\nCarol,40\n"
    p = tmp_path / "people.csv"
    p.write_text(csv_text, encoding="utf-8")
    docs = CSVLoader().load(p)
    assert len(docs) == 3
    assert "name: Alice" in docs[0].content
    assert "age: 30" in docs[0].content
    assert docs[2].metadata["row_number"] == 3


def test_csv_loader_source_column(tmp_path: Path) -> None:
    csv_text = "id,text\n1,foo\n2,bar\n"
    p = tmp_path / "x.csv"
    p.write_text(csv_text, encoding="utf-8")
    docs = CSVLoader().load(p, source_column="id")
    assert docs[0].metadata["source_column_value"] == "1"


# ── EPUBLoader ───────────────────────────────────────────────────────────────


def test_epub_loader(tmp_path: Path) -> None:
    ebooklib = pytest.importorskip("ebooklib")
    pytest.importorskip("bs4")
    from ebooklib import epub

    book = epub.EpubBook()
    book.set_identifier("id-1")
    book.set_title("Test")
    book.set_language("en")
    c1 = epub.EpubHtml(title="Ch1", file_name="c1.xhtml", lang="en")
    c1.content = "<html><body><h1>Ch1</h1><p>alpha</p></body></html>"
    book.add_item(c1)
    book.toc = (c1,)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", c1]
    p = tmp_path / "x.epub"
    epub.write_epub(str(p), book)

    docs = EPUBLoader().load(p)
    # At least the one HTML chapter we added shows up; nav may or may not.
    assert any(d.metadata.get("type") == "epub" for d in docs)
    assert any("alpha" in d.content for d in docs)
    _ = ebooklib  # silence unused


# ── XLSXLoader ───────────────────────────────────────────────────────────────


def test_xlsx_loader_sheets_mode(tmp_path: Path) -> None:
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    ws1 = wb.active
    ws1.title = "S1"
    ws1.append(["a", "b"])
    ws1.append([1, 2])
    ws2 = wb.create_sheet("S2")
    ws2.append(["x"])
    ws2.append([9])
    p = tmp_path / "data.xlsx"
    wb.save(str(p))
    docs = XLSXLoader().load(p)
    assert len(docs) == 2
    sheet_names = [d.metadata["sheet_name"] for d in docs]
    assert sheet_names == ["S1", "S2"]


# ── Registry ─────────────────────────────────────────────────────────────────


def test_get_loader_by_name() -> None:
    assert isinstance(get_loader("text"), TextLoader)


def test_get_loader_for_path_pdf() -> None:
    assert isinstance(get_loader_for_path("foo.pdf"), PDFLoader)


def test_get_loader_for_path_unknown_raises() -> None:
    with pytest.raises(KeyError, match=r"\.unknown|Known extensions"):
        get_loader_for_path("foo.unknown")


def test_get_loader_for_path_no_extension_raises() -> None:
    with pytest.raises(KeyError, match=r"no file extension"):
        get_loader_for_path("foo")


# ── Async smoke ──────────────────────────────────────────────────────────────


def test_async_loader_smoke(tmp_path: Path) -> None:
    p = tmp_path / "hello.txt"
    p.write_text("hi", encoding="utf-8")

    async def _run() -> list[Document]:
        loader = get_async_loader_for_path(p)
        return await loader.load(p)

    docs = asyncio.run(_run())
    assert len(docs) == 1
    assert docs[0].content == "hi"
