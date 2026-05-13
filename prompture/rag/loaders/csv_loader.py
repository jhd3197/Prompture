"""CSV document loader (stdlib)."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

from ..documents import Document, DocumentLoader, LoaderSource


class CSVLoader(DocumentLoader):
    """Load a CSV file into one Document per row or one Document for the whole file."""

    supported_extensions = (".csv",)

    def load(self, source: LoaderSource, **options: Any) -> list[Document]:
        mode = options.get("mode", "rows")  # "rows" | "single"
        encoding = options.get("encoding", "utf-8")
        source_column = options.get("source_column")
        delimiter = options.get("delimiter", ",")

        if isinstance(source, bytes):
            text = source.decode(encoding, errors="replace")
            source_label = "<bytes>"
        else:
            path = Path(source)
            # ``Path.read_text`` only gained ``newline=`` in 3.10; open
            # manually for 3.9 compatibility.
            with open(path, encoding=encoding, newline="") as fh:
                text = fh.read()
            source_label = str(path)

        if mode == "single":
            return [
                Document(
                    content=text,
                    metadata={"source": source_label, "type": "csv"},
                )
            ]

        reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
        docs: list[Document] = []
        for i, row in enumerate(reader, start=1):
            lines = [f"{k}: {v}" for k, v in row.items() if k is not None]
            content = "\n".join(lines)
            metadata: dict = {
                "source": source_label,
                "row_number": i,
                "type": "csv",
            }
            if source_column and source_column in row:
                metadata["source_column_value"] = row[source_column]
            docs.append(Document(content=content, metadata=metadata))
        return docs
