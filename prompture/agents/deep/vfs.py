"""Virtual filesystem tool factories for :class:`DeepAgent`.

The virtual filesystem is just a ``dict[str, str]`` carried on the
shared :class:`DeepAgentState`. Each tool closes over that dict and
mutates it directly. Paths are flat strings (treated as keys); leading
slashes are normalised away.
"""

from __future__ import annotations

import fnmatch
import re

from ..deep_prompts import (
    EDIT_FILE_DESC,
    GLOB_DESC,
    GREP_DESC,
    LS_DESC,
    READ_FILE_DESC,
    WRITE_FILE_DESC,
)
from ..deep_state import DeepAgentState
from ..tools_schema import ToolDefinition

_DEFAULT_READ_LIMIT = 2000


def _normalise_path(path: str) -> str:
    """Strip leading slashes and whitespace. Reject empty paths."""
    if not isinstance(path, str):
        raise ValueError("file_path must be a string")
    cleaned = path.strip().lstrip("/")
    if not cleaned:
        raise ValueError("file_path must not be empty")
    return cleaned


def _format_with_line_numbers(content: str, start_line: int) -> str:
    """Prefix each line with its 1-indexed line number, tab-separated."""
    lines = content.splitlines()
    if not lines and not content:
        return ""
    formatted = []
    for offset, line in enumerate(lines):
        line_no = start_line + offset
        formatted.append(f"{line_no:6d}\t{line}")
    return "\n".join(formatted)


def make_vfs_tools(state: DeepAgentState) -> list[ToolDefinition]:
    """Return the six VFS tool definitions, all closing over *state*."""

    def read_file(file_path: str, offset: int = 0, limit: int = _DEFAULT_READ_LIMIT) -> str:
        """Read a file with optional pagination."""
        try:
            path = _normalise_path(file_path)
        except ValueError as exc:
            return f"Error: {exc}"
        if path not in state.files:
            return f"Error: file not found: {file_path!r}"
        content = state.files[path]
        if not content:
            return "(empty file)"
        try:
            offset_i = max(0, int(offset))
            limit_i = max(1, int(limit))
        except (TypeError, ValueError):
            offset_i = 0
            limit_i = _DEFAULT_READ_LIMIT
        lines = content.splitlines()
        end = min(offset_i + limit_i, len(lines))
        if offset_i >= len(lines):
            return f"(no more content; file has {len(lines)} line(s))"
        slice_lines = lines[offset_i:end]
        slice_content = "\n".join(slice_lines)
        result = _format_with_line_numbers(slice_content, offset_i + 1)
        if end < len(lines):
            result += f"\n\n[truncated at line {end} of {len(lines)} — call again with offset={end}]"
        return result

    def write_file(file_path: str, content: str) -> str:
        """Create or overwrite a file."""
        try:
            path = _normalise_path(file_path)
        except ValueError as exc:
            return f"Error: {exc}"
        if not isinstance(content, str):
            content = str(content)
        existed = path in state.files
        state.files[path] = content
        verb = "Overwrote" if existed else "Wrote"
        return f"{verb} {len(content)} char(s) to {path}"

    def edit_file(
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> str:
        """Replace ``old_string`` with ``new_string`` in an existing file."""
        try:
            path = _normalise_path(file_path)
        except ValueError as exc:
            return f"Error: {exc}"
        if path not in state.files:
            return f"Error: file not found: {file_path!r}"
        if not isinstance(old_string, str) or not isinstance(new_string, str):
            return "Error: old_string and new_string must both be strings"
        original = state.files[path]
        count = original.count(old_string)
        if count == 0:
            return f"Error: old_string not found in {path}"
        if count > 1 and not replace_all:
            return (
                f"Error: old_string appears {count} times in {path}. "
                "Provide more surrounding context to make it unique, or "
                "pass replace_all=true."
            )
        if replace_all:
            state.files[path] = original.replace(old_string, new_string)
            return f"Replaced {count} occurrence(s) in {path}"
        state.files[path] = original.replace(old_string, new_string, 1)
        return f"Replaced 1 occurrence in {path}"

    def ls(path: str = "") -> str:
        """List files, optionally filtered by path prefix."""
        prefix = ""
        if isinstance(path, str) and path.strip():
            try:
                prefix = _normalise_path(path)
            except ValueError as exc:
                return f"Error: {exc}"
        if not state.files:
            return "(no files)"
        matches = [p for p in sorted(state.files) if p.startswith(prefix)] if prefix else sorted(state.files)
        if not matches:
            return f"(no files matching prefix {prefix!r})"
        return "\n".join(matches)

    def glob_tool(pattern: str) -> str:
        """Find files by fnmatch-style pattern."""
        if not isinstance(pattern, str) or not pattern.strip():
            return "Error: pattern must be a non-empty string"
        matches = sorted(p for p in state.files if fnmatch.fnmatchcase(p, pattern))
        if not matches:
            return f"(no files matching {pattern!r})"
        return "\n".join(matches)

    def grep_tool(
        pattern: str,
        glob: str = "*",
        output_mode: str = "files_with_matches",
    ) -> str:
        """Regex search across file contents."""
        if not isinstance(pattern, str) or pattern == "":
            return "Error: pattern must be a non-empty string"
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            return f"Error: invalid regex: {exc}"
        glob_pat = glob if isinstance(glob, str) and glob else "*"
        mode = output_mode if output_mode in {"files_with_matches", "content", "count"} else "files_with_matches"

        files_to_search = sorted(p for p in state.files if fnmatch.fnmatchcase(p, glob_pat))
        if not files_to_search:
            return f"(no files matching glob {glob_pat!r})"

        if mode == "files_with_matches":
            hits = [p for p in files_to_search if regex.search(state.files[p])]
            return "\n".join(hits) if hits else "(no matches)"

        if mode == "count":
            counts = []
            for p in files_to_search:
                n = len(regex.findall(state.files[p]))
                if n:
                    counts.append(f"{p}: {n}")
            return "\n".join(counts) if counts else "(no matches)"

        # content mode — file:line:match style
        lines_out: list[str] = []
        for p in files_to_search:
            for line_no, line in enumerate(state.files[p].splitlines(), start=1):
                if regex.search(line):
                    lines_out.append(f"{p}:{line_no}:{line}")
        return "\n".join(lines_out) if lines_out else "(no matches)"

    return [
        ToolDefinition(
            name="read_file",
            description=READ_FILE_DESC,
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path of the file to read."},
                    "offset": {
                        "type": "integer",
                        "description": "Line number to start reading from (0-indexed). Default 0.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of lines to read. Default 2000.",
                    },
                },
                "required": ["file_path"],
            },
            function=read_file,
        ),
        ToolDefinition(
            name="write_file",
            description=WRITE_FILE_DESC,
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path of the file."},
                    "content": {"type": "string", "description": "Full file content."},
                },
                "required": ["file_path", "content"],
            },
            function=write_file,
        ),
        ToolDefinition(
            name="edit_file",
            description=EDIT_FILE_DESC,
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path of the file."},
                    "old_string": {"type": "string", "description": "Substring to replace."},
                    "new_string": {"type": "string", "description": "Replacement substring."},
                    "replace_all": {
                        "type": "boolean",
                        "description": "When true, replace every occurrence. Default false.",
                    },
                },
                "required": ["file_path", "old_string", "new_string"],
            },
            function=edit_file,
        ),
        ToolDefinition(
            name="ls",
            description=LS_DESC,
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Optional path prefix to filter by.",
                    }
                },
            },
            function=ls,
        ),
        ToolDefinition(
            name="glob",
            description=GLOB_DESC,
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "fnmatch glob pattern."},
                },
                "required": ["pattern"],
            },
            function=glob_tool,
        ),
        ToolDefinition(
            name="grep",
            description=GREP_DESC,
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Python regex."},
                    "glob": {
                        "type": "string",
                        "description": "fnmatch pattern to restrict files. Default '*'.",
                    },
                    "output_mode": {
                        "type": "string",
                        "enum": ["files_with_matches", "content", "count"],
                        "description": "Result format. Default 'files_with_matches'.",
                    },
                },
                "required": ["pattern"],
            },
            function=grep_tool,
        ),
    ]


def make_vfs_tools_dict(state: DeepAgentState) -> dict[str, ToolDefinition]:
    """Helper returning the VFS tools as ``{name: ToolDefinition}``."""
    return {td.name: td for td in make_vfs_tools(state)}


VFS_TOOL_NAMES = {"read_file", "write_file", "edit_file", "ls", "glob", "grep"}
"""Reserved tool names provided by the VFS. Used to detect collisions."""
