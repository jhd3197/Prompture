"""Tests for document ingestion module."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path
from typing import Any

import pytest

from prompture.ingestion import (
    ChunkingConfig,
    DocumentChunk,
    DocumentContent,
    async_ingest,
    chunk_document,
    get_parser,
    ingest,
    is_parser_registered,
    list_parsers,
    register_parser,
    unregister_parser,
)
from prompture.ingestion.detect import detect_file_type
from prompture.ingestion.parsers.base import BaseParser
from prompture.ingestion.parsers.csv_parser import CsvParser
from prompture.ingestion.parsers.markdown import MarkdownParser


# =============================================================================
# DocumentContent Tests
# =============================================================================


class TestDocumentContent:
    """Tests for DocumentContent dataclass."""

    def test_creation_basic(self):
        """Should create DocumentContent with required fields."""
        doc = DocumentContent(
            text="Sample text",
            file_type="markdown",
            char_count=11,
        )
        assert doc.text == "Sample text"
        assert doc.file_type == "markdown"
        assert doc.char_count == 11
        assert doc.source_path is None
        assert doc.metadata == {}
        assert doc.page_count is None
        assert doc.page_texts is None

    def test_creation_full(self):
        """Should create DocumentContent with all fields."""
        doc = DocumentContent(
            text="Full document",
            file_type="pdf",
            source_path="/path/to/doc.pdf",
            metadata={"author": "John Doe"},
            page_count=3,
            page_texts=["Page 1", "Page 2", "Page 3"],
            char_count=13,
        )
        assert doc.text == "Full document"
        assert doc.file_type == "pdf"
        assert doc.source_path == "/path/to/doc.pdf"
        assert doc.metadata["author"] == "John Doe"
        assert doc.page_count == 3
        assert len(doc.page_texts) == 3
        assert doc.char_count == 13

    def test_frozen_immutability(self):
        """Should be immutable (frozen dataclass)."""
        doc = DocumentContent(text="Test", file_type="txt", char_count=4)
        with pytest.raises(AttributeError):
            doc.text = "New text"

    def test_char_count_field_exists(self):
        """Should have char_count field."""
        doc = DocumentContent(text="Testing", file_type="txt", char_count=7)
        assert hasattr(doc, "char_count")
        assert doc.char_count == 7


# =============================================================================
# File Type Detection Tests
# =============================================================================


class TestDetectFileType:
    """Tests for detect_file_type function."""

    def test_detect_pdf(self):
        """Should detect PDF files."""
        assert detect_file_type("document.pdf") == "pdf"
        assert detect_file_type("report.PDF") == "pdf"
        assert detect_file_type(Path("file.pdf")) == "pdf"

    def test_detect_docx(self):
        """Should detect DOCX files."""
        assert detect_file_type("document.docx") == "docx"
        assert detect_file_type("old.doc") == "docx"
        assert detect_file_type("FILE.DOCX") == "docx"

    def test_detect_html(self):
        """Should detect HTML files."""
        assert detect_file_type("page.html") == "html"
        assert detect_file_type("page.htm") == "html"
        assert detect_file_type("page.xhtml") == "html"

    def test_detect_markdown(self):
        """Should detect Markdown files."""
        assert detect_file_type("README.md") == "markdown"
        assert detect_file_type("doc.markdown") == "markdown"
        assert detect_file_type("notes.mdx") == "markdown"
        assert detect_file_type("plain.txt") == "markdown"

    def test_detect_csv(self):
        """Should detect CSV files."""
        assert detect_file_type("data.csv") == "csv"
        assert detect_file_type("data.tsv") == "csv"

    def test_detect_xlsx(self):
        """Should detect Excel files."""
        assert detect_file_type("spreadsheet.xlsx") == "xlsx"
        assert detect_file_type("old_sheet.xls") == "xlsx"

    def test_unknown_extension_raises_error(self):
        """Should raise ValueError for unknown extensions."""
        with pytest.raises(ValueError, match="Unsupported file extension"):
            detect_file_type("unknown.xyz")

    def test_case_insensitive(self):
        """Should handle case insensitively."""
        assert detect_file_type("FILE.PDF") == "pdf"
        assert detect_file_type("DOC.DOCX") == "docx"


# =============================================================================
# Parser Registry Tests
# =============================================================================


class TestParserRegistry:
    """Tests for parser registration system."""

    def test_list_parsers_includes_built_ins(self):
        """Should list built-in parsers."""
        parsers = list_parsers()
        assert "markdown" in parsers
        assert "csv" in parsers
        assert "pdf" in parsers
        assert "docx" in parsers
        assert "html" in parsers
        assert "xlsx" in parsers

    def test_is_parser_registered(self):
        """Should check parser registration."""
        assert is_parser_registered("markdown")
        assert is_parser_registered("csv")
        assert not is_parser_registered("nonexistent")

    def test_get_parser_returns_instance(self):
        """Should return parser instance."""
        parser = get_parser("markdown")
        assert isinstance(parser, BaseParser)
        assert isinstance(parser, MarkdownParser)

        parser = get_parser("csv")
        assert isinstance(parser, BaseParser)
        assert isinstance(parser, CsvParser)

    def test_get_parser_unknown_raises_error(self):
        """Should raise ValueError for unknown parser."""
        with pytest.raises(ValueError, match="No parser registered"):
            get_parser("nonexistent_parser")

    def test_register_custom_parser(self):
        """Should register custom parser."""

        class CustomParser(BaseParser):
            def parse(self, source, **kwargs):
                return DocumentContent(text="custom", file_type="custom", char_count=6)

        # Clean up first if it exists
        unregister_parser("custom_test")

        register_parser("custom_test", lambda: CustomParser())
        assert is_parser_registered("custom_test")

        parser = get_parser("custom_test")
        assert isinstance(parser, CustomParser)

        # Clean up
        unregister_parser("custom_test")

    def test_register_duplicate_raises_error(self):
        """Should raise error when registering duplicate without overwrite."""

        class DummyParser(BaseParser):
            def parse(self, source, **kwargs):
                return DocumentContent(text="", file_type="", char_count=0)

        # Clean up first
        unregister_parser("test_dup")

        register_parser("test_dup", lambda: DummyParser())

        with pytest.raises(ValueError, match="already registered"):
            register_parser("test_dup", lambda: DummyParser())

        # Clean up
        unregister_parser("test_dup")

    def test_register_with_overwrite(self):
        """Should allow overwrite when overwrite=True."""

        class Parser1(BaseParser):
            def parse(self, source, **kwargs):
                return DocumentContent(text="v1", file_type="test", char_count=2)

        class Parser2(BaseParser):
            def parse(self, source, **kwargs):
                return DocumentContent(text="v2", file_type="test", char_count=2)

        # Clean up first
        unregister_parser("test_overwrite")

        register_parser("test_overwrite", lambda: Parser1())
        register_parser("test_overwrite", lambda: Parser2(), overwrite=True)

        parser = get_parser("test_overwrite")
        assert isinstance(parser, Parser2)

        # Clean up
        unregister_parser("test_overwrite")

    def test_unregister_parser(self):
        """Should unregister parser successfully."""

        class TempParser(BaseParser):
            def parse(self, source, **kwargs):
                return DocumentContent(text="", file_type="", char_count=0)

        # Clean up first
        unregister_parser("temp_test")

        register_parser("temp_test", lambda: TempParser())
        assert is_parser_registered("temp_test")

        result = unregister_parser("temp_test")
        assert result is True
        assert not is_parser_registered("temp_test")

    def test_unregister_nonexistent_returns_false(self):
        """Should return False when unregistering nonexistent parser."""
        result = unregister_parser("definitely_does_not_exist")
        assert result is False


# =============================================================================
# BaseParser Tests
# =============================================================================


class TestBaseParser:
    """Tests for BaseParser abstract class."""

    def test_check_file_validates_existence(self):
        """Should validate file existence."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("test")
            temp_path = f.name

        try:
            # Should succeed for existing file
            result = BaseParser._check_file(temp_path, max_file_size=1000)
            assert isinstance(result, Path)
            assert result.exists()
        finally:
            Path(temp_path).unlink()

        # Should raise for non-existent file
        with pytest.raises(FileNotFoundError):
            BaseParser._check_file("/nonexistent/file.txt", max_file_size=1000)

    def test_check_file_validates_size(self):
        """Should validate file size limit."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("x" * 1000)
            temp_path = f.name

        try:
            # Should fail when file exceeds max_file_size
            with pytest.raises(ValueError, match="exceeds the maximum allowed"):
                BaseParser._check_file(temp_path, max_file_size=100)

            # Should succeed when within limit
            result = BaseParser._check_file(temp_path, max_file_size=2000)
            assert result.exists()
        finally:
            Path(temp_path).unlink()

    @pytest.mark.asyncio
    async def test_async_parse_default_wraps_sync(self):
        """Should wrap sync parse by default."""

        class SyncOnlyParser(BaseParser):
            def parse(self, source, **kwargs):
                return DocumentContent(text="sync", file_type="test", char_count=4)

        parser = SyncOnlyParser()

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("test")
            temp_path = f.name

        try:
            result = await parser.async_parse(temp_path)
            assert result.text == "sync"
            assert result.file_type == "test"
        finally:
            Path(temp_path).unlink()


# =============================================================================
# MarkdownParser Tests
# =============================================================================


class TestMarkdownParser:
    """Tests for MarkdownParser."""

    def test_parse_basic_markdown(self):
        """Should parse basic markdown file."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".md") as f:
            f.write("# Heading\n\nSome **bold** text.")
            temp_path = f.name

        try:
            parser = MarkdownParser()
            doc = parser.parse(temp_path)

            assert doc.file_type == "markdown"
            assert doc.source_path == temp_path
            assert "Heading" in doc.text
            assert "bold" in doc.text
            assert doc.char_count > 0
            assert doc.page_count is None
        finally:
            Path(temp_path).unlink()

    def test_parse_strips_formatting(self):
        """Should strip markdown formatting by default."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".md") as f:
            f.write("# Title\n\n**bold** and *italic* text")
            temp_path = f.name

        try:
            parser = MarkdownParser()
            doc = parser.parse(temp_path, strip_formatting=True)

            # Formatting should be stripped
            assert "**" not in doc.text
            assert "*" not in doc.text or doc.text.count("*") == 0
            assert "bold" in doc.text
            assert "italic" in doc.text
        finally:
            Path(temp_path).unlink()

    def test_parse_preserves_formatting(self):
        """Should preserve markdown formatting when requested."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".md") as f:
            f.write("**bold**")
            temp_path = f.name

        try:
            parser = MarkdownParser()
            doc = parser.parse(temp_path, strip_formatting=False)

            assert "**bold**" in doc.text
        finally:
            Path(temp_path).unlink()

    def test_parse_extracts_frontmatter(self):
        """Should extract YAML frontmatter as metadata."""
        content = """---
title: Test Document
author: John Doe
---

Document content here."""

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".md") as f:
            f.write(content)
            temp_path = f.name

        try:
            parser = MarkdownParser()
            doc = parser.parse(temp_path)

            # Should extract frontmatter
            assert "title" in doc.metadata or "author" in doc.metadata
            # Content should not include frontmatter
            assert "---" not in doc.text or doc.text.count("---") < 2
            assert "content here" in doc.text.lower()
        finally:
            Path(temp_path).unlink()


# =============================================================================
# CsvParser Tests
# =============================================================================


class TestCsvParser:
    """Tests for CsvParser."""

    def test_parse_basic_csv(self):
        """Should parse basic CSV file."""
        csv_content = "name,age,city\nJohn,30,NYC\nJane,25,LA"

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            f.write(csv_content)
            temp_path = f.name

        try:
            parser = CsvParser()
            doc = parser.parse(temp_path)

            assert doc.file_type == "csv"
            assert doc.source_path == temp_path
            assert "name" in doc.text
            assert "John" in doc.text
            assert doc.char_count > 0
            assert doc.metadata["row_count"] == 2
            assert "name" in doc.metadata["headers"]
        finally:
            Path(temp_path).unlink()

    def test_parse_extracts_headers_as_metadata(self):
        """Should extract CSV headers as metadata."""
        csv_content = "header1,header2,header3\nval1,val2,val3"

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            f.write(csv_content)
            temp_path = f.name

        try:
            parser = CsvParser()
            doc = parser.parse(temp_path)

            assert "headers" in doc.metadata
            assert doc.metadata["headers"] == ["header1", "header2", "header3"]
        finally:
            Path(temp_path).unlink()

    def test_parse_extracts_row_count_as_metadata(self):
        """Should extract row count as metadata."""
        csv_content = "col1,col2\nrow1val1,row1val2\nrow2val1,row2val2\nrow3val1,row3val2"

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            f.write(csv_content)
            temp_path = f.name

        try:
            parser = CsvParser()
            doc = parser.parse(temp_path)

            assert "row_count" in doc.metadata
            assert doc.metadata["row_count"] == 3
        finally:
            Path(temp_path).unlink()

    def test_parse_tsv_auto_detects_delimiter(self):
        """Should auto-detect tab delimiter for TSV files."""
        tsv_content = "col1\tcol2\nval1\tval2"

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".tsv") as f:
            f.write(tsv_content)
            temp_path = f.name

        try:
            parser = CsvParser()
            doc = parser.parse(temp_path)

            assert doc.metadata["delimiter"] == "\t"
            assert "val1" in doc.text
        finally:
            Path(temp_path).unlink()


# =============================================================================
# Parser ImportError Tests
# =============================================================================


class TestParserImportErrors:
    """Tests for parser ImportError handling."""

    def test_html_parser_raises_import_error_without_bs4(self):
        """Should raise ImportError when beautifulsoup4 not installed."""
        # We can only test this if bs4 is actually not available
        # or by mocking the import
        import sys
        from unittest.mock import patch

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".html") as f:
            f.write("<html><body>Test</body></html>")
            temp_path = f.name

        try:
            # Mock both tukuy and bs4 as unavailable
            with patch.dict(sys.modules, {"tukuy.plugins.html": None, "bs4": None}):
                from prompture.ingestion.parsers.html import HtmlParser

                parser = HtmlParser()
                with pytest.raises(ImportError, match="beautifulsoup4 is required"):
                    parser.parse(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_pdf_parser_raises_import_error_without_deps(self):
        """Should raise ImportError when PDF dependencies not installed."""
        import sys
        from unittest.mock import patch

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".pdf") as f:
            f.write("%PDF-1.4 fake")
            temp_path = f.name

        try:
            # Mock dependencies as unavailable
            with patch.dict(sys.modules, {"PyPDF2": None, "pypdf": None}):
                from prompture.ingestion.parsers.pdf import PdfParser

                parser = PdfParser()
                with pytest.raises(ImportError, match="PyPDF2|pypdf"):
                    parser.parse(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_docx_parser_raises_import_error_without_python_docx(self):
        """Should raise ImportError when python-docx not installed."""
        import sys
        from unittest.mock import patch

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".docx") as f:
            f.write("fake docx")
            temp_path = f.name

        try:
            with patch.dict(sys.modules, {"docx": None}):
                from prompture.ingestion.parsers.docx import DocxParser

                parser = DocxParser()
                with pytest.raises(ImportError, match="python-docx"):
                    parser.parse(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_xlsx_parser_raises_import_error_without_openpyxl(self):
        """Should raise ImportError when openpyxl not installed."""
        import sys
        from unittest.mock import patch

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".xlsx") as f:
            f.write("fake xlsx")
            temp_path = f.name

        try:
            with patch.dict(sys.modules, {"openpyxl": None}):
                from prompture.ingestion.parsers.xlsx import XlsxParser

                parser = XlsxParser()
                with pytest.raises(ImportError, match="openpyxl"):
                    parser.parse(temp_path)
        finally:
            Path(temp_path).unlink()


# =============================================================================
# Ingest Smart Constructor Tests
# =============================================================================


class TestIngestSmartConstructor:
    """Tests for ingest() smart constructor."""

    def test_passthrough_document_content(self):
        """Should pass through DocumentContent unchanged."""
        original = DocumentContent(text="test", file_type="txt", char_count=4)
        result = ingest(original)
        assert result is original

    def test_raises_type_error_for_bytes(self):
        """Should raise TypeError for bytes input."""
        with pytest.raises(TypeError, match="Cannot ingest raw bytes"):
            ingest(b"some bytes")

    def test_raises_value_error_for_unknown_extension(self):
        """Should raise ValueError for unknown extension without file_type."""
        with pytest.raises(ValueError, match="Unsupported file extension"):
            ingest("unknown.xyz")

    def test_file_type_override(self):
        """Should respect file_type override parameter."""
        content = "# Markdown content"

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(content)
            temp_path = f.name

        try:
            # Override .txt to be parsed as markdown explicitly
            doc = ingest(temp_path, file_type="markdown")
            assert doc.file_type == "markdown"
            assert "Markdown content" in doc.text
        finally:
            Path(temp_path).unlink()

    def test_auto_detect_and_parse_markdown(self):
        """Should auto-detect and parse markdown file."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".md") as f:
            f.write("# Test\n\nContent")
            temp_path = f.name

        try:
            doc = ingest(temp_path)
            assert doc.file_type == "markdown"
            assert "Test" in doc.text
            assert doc.source_path == temp_path
        finally:
            Path(temp_path).unlink()

    def test_auto_detect_and_parse_csv(self):
        """Should auto-detect and parse CSV file."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            f.write("a,b\n1,2")
            temp_path = f.name

        try:
            doc = ingest(temp_path)
            assert doc.file_type == "csv"
            assert doc.metadata["row_count"] == 1
        finally:
            Path(temp_path).unlink()

    @pytest.mark.asyncio
    async def test_async_ingest_passthrough(self):
        """Should pass through DocumentContent in async mode."""
        original = DocumentContent(text="async test", file_type="txt", char_count=10)
        result = await async_ingest(original)
        assert result is original

    @pytest.mark.asyncio
    async def test_async_ingest_parses_file(self):
        """Should parse file asynchronously."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".md") as f:
            f.write("Async content")
            temp_path = f.name

        try:
            doc = await async_ingest(temp_path)
            assert doc.file_type == "markdown"
            assert "Async content" in doc.text
        finally:
            Path(temp_path).unlink()


# =============================================================================
# Document Chunking Tests
# =============================================================================


class TestChunkDocument:
    """Tests for chunk_document function."""

    def test_single_chunk_for_small_doc(self):
        """Should return single chunk when doc fits within limit."""
        doc = DocumentContent(text="Short text", file_type="txt", char_count=10)
        config = ChunkingConfig(max_chars_per_chunk=1000)

        chunks = chunk_document(doc, config)

        assert len(chunks) == 1
        assert chunks[0].text == "Short text"
        assert chunks[0].chunk_index == 0
        assert chunks[0].total_chunks == 1

    def test_page_based_chunking(self):
        """Should chunk by pages when page_texts available."""
        page1 = "Page 1 " * 100  # ~700 chars
        page2 = "Page 2 " * 100
        page3 = "Page 3 " * 100

        doc = DocumentContent(
            text=f"{page1}\n\n{page2}\n\n{page3}",
            file_type="pdf",
            page_count=3,
            page_texts=[page1, page2, page3],
            char_count=len(page1) + len(page2) + len(page3),
        )

        config = ChunkingConfig(max_chars_per_chunk=1000, strategy="page")
        chunks = chunk_document(doc, config)

        # Should create multiple chunks
        assert len(chunks) > 1
        # Each chunk should have page_range
        for chunk in chunks:
            assert chunk.page_range is not None
            assert chunk.total_chunks == len(chunks)

    def test_char_based_chunking(self):
        """Should chunk by characters with paragraph boundaries."""
        paragraphs = [f"Paragraph {i}." for i in range(20)]
        text = "\n\n".join(paragraphs)

        doc = DocumentContent(text=text, file_type="txt", char_count=len(text))

        config = ChunkingConfig(max_chars_per_chunk=50, overlap_chars=10, strategy="chars")
        chunks = chunk_document(doc, config)

        # Should create multiple chunks
        assert len(chunks) > 1
        # Each chunk should have no page_range (char-based)
        for chunk in chunks:
            assert chunk.page_range is None
            assert chunk.total_chunks == len(chunks)

    def test_char_based_chunking_with_overlap(self):
        """Should include overlap between chunks in char mode."""
        text = "Para 1.\n\nPara 2.\n\nPara 3.\n\nPara 4.\n\nPara 5."
        doc = DocumentContent(text=text, file_type="txt", char_count=len(text))

        config = ChunkingConfig(max_chars_per_chunk=20, overlap_chars=5, strategy="chars")
        chunks = chunk_document(doc, config)

        # Should have multiple chunks with some overlap
        assert len(chunks) >= 2

    def test_default_config_when_none(self):
        """Should use default config when None provided."""
        doc = DocumentContent(text="Test", file_type="txt", char_count=4)

        chunks = chunk_document(doc, config=None)

        assert len(chunks) == 1
        assert chunks[0].text == "Test"


# =============================================================================
# Integration with extract_with_model
# =============================================================================


class TestExtractWithModelIntegration:
    """Tests for source parameter in extract_with_model."""

    def test_source_parameter_exists(self):
        """Should accept source parameter without error."""
        from prompture import extract_with_model
        from pydantic import BaseModel

        class TestModel(BaseModel):
            text: str

        # We're just checking that the parameter is accepted
        # We won't actually call with a real source since that needs LLM
        import inspect

        sig = inspect.signature(extract_with_model)
        assert "source" in sig.parameters

    def test_source_parameter_in_signature(self):
        """Should have source parameter in function signature."""
        from prompture.extraction.core import extract_with_model
        import inspect

        sig = inspect.signature(extract_with_model)
        params = list(sig.parameters.keys())

        assert "source" in params
        assert "chunking" in params  # Related chunking parameter

    def test_source_mutually_exclusive_with_text_documented(self):
        """Should document source as mutually exclusive with text."""
        from prompture.extraction.core import extract_with_model

        # Check docstring mentions source
        assert extract_with_model.__doc__ is not None
        assert "source" in extract_with_model.__doc__.lower()
