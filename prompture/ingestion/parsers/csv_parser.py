"""CSV parser using stdlib csv with optional TOON encoding."""

from __future__ import annotations

import csv
import io
import json
import logging
from pathlib import Path
from typing import Any

from ..document import DocumentContent
from .base import BaseParser

logger = logging.getLogger("prompture.ingestion.parsers.csv")


class CsvParser(BaseParser):
    """Parse CSV/TSV files using Python's stdlib ``csv`` module.

    Converts tabular data to a JSON-lines or TOON representation
    suitable for LLM consumption.  No external dependencies required.
    """

    extensions = [".csv", ".tsv"]
    mime_types = ["text/csv", "text/tab-separated-values"]

    def parse(
        self,
        source: str | Path,
        *,
        pages: list[int] | None = None,
        max_file_size: int = 50 * 1024 * 1024,
        delimiter: str | None = None,
        max_rows: int | None = None,
        output_format: str = "json",
        **kwargs: Any,
    ) -> DocumentContent:
        p = self._check_file(source, max_file_size)
        path_str = str(p)

        # Auto-detect delimiter from extension
        if delimiter is None:
            delimiter = "\t" if p.suffix.lower() == ".tsv" else ","

        raw = p.read_text(encoding="utf-8", errors="replace")
        reader = csv.DictReader(io.StringIO(raw), delimiter=delimiter)
        headers = reader.fieldnames or []

        rows: list[dict[str, str]] = []
        for i, row in enumerate(reader):
            if max_rows is not None and i >= max_rows:
                break
            rows.append(dict(row))

        metadata: dict[str, Any] = {
            "headers": headers,
            "row_count": len(rows),
            "delimiter": delimiter,
        }

        # Encode as TOON if requested and available
        if output_format == "toon":
            text = self._to_toon(rows, headers)
        else:
            text = self._to_json_text(rows, headers)

        return DocumentContent(
            text=text,
            file_type="csv",
            source_path=path_str,
            metadata=metadata,
            page_count=None,
            page_texts=None,
            char_count=len(text),
        )

    # ------------------------------------------------------------------
    # Output formatters
    # ------------------------------------------------------------------

    @staticmethod
    def _to_json_text(rows: list[dict[str, str]], headers: list[str]) -> str:
        """Render rows as a compact JSON array string."""
        return json.dumps(rows, ensure_ascii=False, default=str)

    @staticmethod
    def _to_toon(rows: list[dict[str, str]], headers: list[str]) -> str:
        """Try encoding as TOON for reduced token usage."""
        try:
            import toon

            return toon.dumps(rows)
        except ImportError:
            logger.debug("python-toon not available, falling back to JSON for CSV output")
            return json.dumps(rows, ensure_ascii=False, default=str)
        except Exception:
            return json.dumps(rows, ensure_ascii=False, default=str)
