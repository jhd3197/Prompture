"""JSON / JSON-Lines document loader."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..documents import Document, DocumentLoader, LoaderSource

# Token forms supported by the dotted-path extractor.
# Examples:
#   "items[].text"       → iterate list at items, take .text of each
#   "results[].content"  → same shape
#   "a.b.c"              → simple drill-down to a single value
#   "data[0].name"       → indexed access to one element
_TOKEN_RE = re.compile(r"([^.\[\]]+)|\[(\d*)\]")


def _walk_path(data: Any, path: str) -> list[Any]:
    """Tiny dotted-path extractor.

    Returns a list of matched values.  An empty ``[]`` index expands to all
    elements of the current list; a numeric ``[n]`` index selects a single
    element.  A bare key (``a.b.c``) drills into mappings.
    """
    if not path:
        return [data]
    # Strip any leading "."
    path = path.lstrip(".")
    current: list[Any] = [data]
    for match in _TOKEN_RE.finditer(path):
        key, idx = match.group(1), match.group(2)
        nxt: list[Any] = []
        for node in current:
            if key is not None:
                if isinstance(node, dict) and key in node:
                    nxt.append(node[key])
            else:
                # ``idx`` may be "" (expand all) or a digit string.
                if isinstance(node, list):
                    if idx == "":
                        nxt.extend(node)
                    elif idx.isdigit():
                        i = int(idx)
                        if 0 <= i < len(node):
                            nxt.append(node[i])
        current = nxt
    return current


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2)


class JSONLoader(DocumentLoader):
    """Load ``.json`` or ``.jsonl`` files into ``Document`` objects.

    Options:
        jq_schema: Dotted-path expression (``"items[].text"``) used to extract
            a list of values from the parsed JSON.  Each value becomes its
            own ``Document``.  Ignored for ``.jsonl`` (each line is already
            a separate Document).
    """

    supported_extensions = (".json", ".jsonl")

    def load(self, source: LoaderSource, **options: Any) -> list[Document]:
        encoding = options.get("encoding", "utf-8")
        jq_schema = options.get("jq_schema")

        if isinstance(source, bytes):
            raw = source.decode(encoding, errors="replace")
            source_label = "<bytes>"
            # When given raw bytes, default to plain JSON behavior.
            ext = options.get("ext", ".json")
        else:
            path = Path(source)
            raw = path.read_text(encoding=encoding)
            source_label = str(path)
            ext = path.suffix.lower()

        if ext == ".jsonl":
            docs: list[Document] = []
            for i, line in enumerate(raw.splitlines(), start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError:
                    parsed = line
                docs.append(
                    Document(
                        content=_stringify(parsed),
                        metadata={
                            "source": source_label,
                            "line_number": i,
                            "type": "jsonl",
                        },
                    )
                )
            return docs

        # .json (or anything else): single-document mode unless jq_schema set.
        parsed = json.loads(raw)
        if jq_schema:
            values = _walk_path(parsed, jq_schema)
            return [
                Document(
                    content=_stringify(v),
                    metadata={
                        "source": source_label,
                        "type": "json",
                        "jq_schema": jq_schema,
                        "index": i,
                    },
                )
                for i, v in enumerate(values)
            ]

        return [
            Document(
                content=json.dumps(parsed, ensure_ascii=False, indent=2),
                metadata={"source": source_label, "type": "json"},
            )
        ]
