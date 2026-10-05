"""Keep tool schemas and tool results from flooding the context window.

Three loop-level helpers shared by :class:`Conversation` and
:class:`AsyncConversation`:

* :func:`shape_tool_results` — results that are too large, alone or as a
  parallel batch, are moved into an :class:`ArtifactStore` and replaced by a
  head/tail preview plus a handle.  The model reads them back with the
  ``read_tool_result`` tool instead of losing everything past a cut-off.
* :class:`ToolLoader` — when the tool schemas themselves are too large, only a
  ``search_tools`` / ``load_tools`` pair is sent; the schemas the model loads
  stay loaded for the rest of the conversation.
* :func:`repair_tool_pairs` — every assistant tool call gets a result and no
  result is left without its call, so an interrupted run does not leave a
  history that providers reject.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from ..execution.context import ArtifactStore, ToolCatalog, count_tokens
from .tools_schema import ToolDefinition, ToolRegistry, tool_from_function

logger = logging.getLogger("prompture.agents.tool_context")

__all__ = [
    "LOAD_TOOLS_NAME",
    "READ_TOOL_RESULT_NAME",
    "SEARCH_TOOLS_NAME",
    "ToolContextMixin",
    "ToolLoader",
    "ToolResultPolicy",
    "make_read_tool_result",
    "repair_tool_pairs",
    "shape_tool_results",
]

READ_TOOL_RESULT_NAME = "read_tool_result"
SEARCH_TOOLS_NAME = "search_tools"
LOAD_TOOLS_NAME = "load_tools"

INTERRUPTED_TOOL_RESULT = (
    "Error: this tool call was not executed because the run was interrupted. Call it again if you still need it."
)


# ---------------------------------------------------------------------------
# Large tool results
# ---------------------------------------------------------------------------


@dataclass
class ToolResultPolicy:
    """When a tool result stays inline and when it is moved out of context.

    Attributes:
        per_call_tokens: A single result above this is offloaded.
        batch_tokens: When one round's results together exceed this, the
            largest are offloaded first until the round fits.
        max_chars: Optional hard character ceiling per result (the legacy
            ``max_tool_result_length``); exceeding it also offloads.
        preview_chars: Characters kept from each end of an offloaded result.
        offload: ``False`` restores plain head truncation with no handle.
        error_chars: Error results are never offloaded, only cut to this.
    """

    per_call_tokens: int = 4000
    batch_tokens: int = 12000
    max_chars: int | None = None
    preview_chars: int = 300
    offload: bool = True
    error_chars: int = 2000

    def __post_init__(self) -> None:
        if self.per_call_tokens > self.batch_tokens:
            raise ValueError(f"per_call_tokens ({self.per_call_tokens}) must be <= batch_tokens ({self.batch_tokens})")

    @property
    def truncate_chars(self) -> int:
        """Inline length used when offloading is off."""
        return self.max_chars if self.max_chars is not None else self.per_call_tokens * 4


def _preview(text: str, chars: int) -> str:
    if len(text) <= chars * 2:
        return text
    return f"{text[:chars]}\n...\n{text[-chars:]}"


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n\n[... result truncated ({len(text):,} chars total)]"


def shape_tool_results(
    batch: Sequence[tuple[str, str, bool]],
    *,
    policy: ToolResultPolicy,
    store: ArtifactStore,
    scope: str,
) -> tuple[list[str], int]:
    """Decide what each result in one round of tool calls looks like in history.

    Args:
        batch: ``(tool_name, result_text, is_error)`` for every call in the round.
        policy: Thresholds.
        store: Where offloaded results go.
        scope: Artifact scope; only this conversation can read them back.

    Returns:
        ``(contents, offloaded_count)`` with ``contents`` in batch order.
    """
    contents = [text for _name, text, _err in batch]
    over_chars = [policy.max_chars is not None and len(text) > policy.max_chars for text in contents]
    # Tokens never outnumber characters, so a short round skips the tokenizer entirely.
    if sum(len(text) for text in contents) < policy.per_call_tokens and not any(over_chars):
        return contents, 0
    tokens = [count_tokens(text) for text in contents]
    total = sum(tokens)

    blocked: set[int] = set()
    for i, (_name, _text, is_error) in enumerate(batch):
        if is_error:
            if len(contents[i]) > policy.error_chars:
                contents[i] = _truncate(contents[i], policy.error_chars)
                shortened = count_tokens(contents[i])
                total -= tokens[i] - shortened
                tokens[i] = shortened
            continue
        if tokens[i] > policy.per_call_tokens or over_chars[i]:
            blocked.add(i)

    remaining = total - sum(tokens[i] for i in blocked)
    for i in sorted(range(len(batch)), key=lambda j: -tokens[j]):
        if remaining <= policy.batch_tokens:
            break
        if i in blocked or batch[i][2]:
            continue
        blocked.add(i)
        remaining -= tokens[i]

    # A tight char ceiling also bounds the preview, so the stub stays well under the limit.
    preview_chars = min(policy.preview_chars, policy.truncate_chars // 4)
    offloaded = 0
    for i in sorted(blocked):
        name, text, _err = batch[i]
        if not policy.offload:
            contents[i] = _truncate(text, policy.truncate_chars)
            continue
        ref = store.put(text, scope=scope, kind="tool_result", source=name)
        offloaded += 1
        contents[i] = (
            f"[Result of {name} was too large to keep in context: {ref.size_chars:,} chars "
            f"(~{ref.approx_tokens:,} tokens). It is stored as {ref.handle}]\n"
            f"Read what you need with {READ_TOOL_RESULT_NAME}(handle, query=...) for matching sections "
            f"or offset=... for a page, or call {name} again with narrower arguments.\n\n"
            f"Preview (first and last {preview_chars} chars):\n{_preview(text, preview_chars)}"
        )
        logger.debug("Offloaded %s result (%d chars) to %s", name, ref.size_chars, ref.handle)
    return contents, offloaded


def make_read_tool_result(store: ArtifactStore, scope: str, *, page_chars: int = 8000) -> ToolDefinition:
    """The tool the model uses to read an offloaded result back."""

    def read_tool_result(handle: str, query: str = "", offset: int = 0, max_chars: int = 4000) -> str:
        """Read part of a tool result that was too large to keep in context.

        Args:
            handle: The artifact:// handle given in place of the result.
            query: Keywords; returns the sections around the best matches.
            offset: Character offset to read from when no query is given.
            max_chars: How much to return (capped).
        """
        limit = max(200, min(int(max_chars), page_chars))
        try:
            if query:
                return store.excerpt(handle, scope=scope, query=query, max_chars=limit)
            text = store.text(handle, scope=scope)
        except KeyError:
            return f"Error: no stored result {handle!r}. Stored results do not survive a reload; call the tool again."
        except PermissionError as exc:
            return f"Error: {exc}"
        start = max(0, int(offset))
        end = min(len(text), start + limit)
        footer = f"\n[chars {start:,}-{end:,} of {len(text):,}"
        footer += f"; next offset={end}]" if end < len(text) else "; end of result]"
        return text[start:end] + footer

    return tool_from_function(read_tool_result)


# ---------------------------------------------------------------------------
# Deferred tool loading
# ---------------------------------------------------------------------------


class ToolLoader:
    """Sends tool schemas on demand once a registry is too large to send whole.

    Args:
        defer: ``True`` always defers, ``False`` never does, ``"auto"`` defers
            once the full schemas would cost more than ``auto_threshold_tokens``.
        auto_threshold_tokens: Schema size that switches ``"auto"`` on.
        preload: Tools whose schemas are always sent, deferred or not.
    """

    def __init__(
        self,
        *,
        defer: bool | Literal["auto"] = "auto",
        auto_threshold_tokens: int = 8000,
        preload: Iterable[str] = (),
    ) -> None:
        self.defer = defer
        self.auto_threshold_tokens = auto_threshold_tokens
        self.loaded: set[str] = set(preload)
        self._size_cache: tuple[tuple[str, ...], int] | None = None

    def schema_tokens(self, registry: ToolRegistry) -> int:
        key = tuple(registry.names)
        if self._size_cache is None or self._size_cache[0] != key:
            self._size_cache = (key, count_tokens(json.dumps(registry.to_openai_format())))
        return self._size_cache[1]

    def is_deferring(self, registry: ToolRegistry) -> bool:
        if not registry or self.defer is False:
            return False
        if self.defer is True:
            return True
        return self.schema_tokens(registry) > self.auto_threshold_tokens

    def mark_used(self, name: str, registry: ToolRegistry) -> None:
        """A tool called by name without loading it is loaded from then on."""
        if name in registry:
            self.loaded.add(name)

    def visible(self, registry: ToolRegistry) -> ToolRegistry:
        """The user tools whose schemas go to the model this round."""
        if not self.is_deferring(registry):
            return registry
        return registry.subset([n for n in registry.names if n in self.loaded])

    def meta_tools(self, registry_getter: Callable[[], ToolRegistry]) -> list[ToolDefinition]:
        registry = registry_getter()
        deferred = [n for n in registry.names if n not in self.loaded]
        listing = ", ".join(deferred)
        if len(listing) > 1500:
            listing = listing[:1500].rsplit(", ", 1)[0] + f", ... ({len(deferred)} total)"

        def search_tools(query: str, limit: int = 8) -> str:
            reg = registry_getter()
            catalog = ToolCatalog(reg)
            hits = catalog.search(query, limit=max(1, int(limit)), min_relevance=0.01)
            if not hits:
                needle = query.casefold().strip()
                hits = [s for s in catalog.summaries() if needle and needle in s.name.casefold()][: int(limit)]
            if not hits:
                return f"No tools matched {query!r}. Try other keywords."
            lines = [
                f"- {s.name}{' (loaded)' if s.name in self.loaded else ''}: {s.description or '(no description)'}"
                for s in hits
            ]
            return "\n".join(lines) + f"\n\nCall {LOAD_TOOLS_NAME} with the names you need."

        def load_tools(names: list[str]) -> str:
            reg = registry_getter()
            found = [n for n in names if n in reg]
            missing = [n for n in names if n not in reg]
            self.loaded.update(found)
            parts = []
            if found:
                parts.append(f"Loaded {', '.join(found)}. Their full schemas are available from your next step on.")
                parts.extend(reg.get(n).to_prompt_format() for n in found)  # type: ignore[union-attr]
            if missing:
                parts.append(f"Unknown tools: {', '.join(missing)}. Use {SEARCH_TOOLS_NAME} to find exact names.")
            return "\n\n".join(parts)

        return [
            tool_from_function(
                search_tools,
                description=(
                    f"Search tools that are available but whose schemas are not loaded yet. "
                    f"Deferred tools: {listing}. Returns names and one-line descriptions."
                ),
                max_description_chars=None,
            ),
            tool_from_function(
                load_tools,
                description=(
                    "Load the full schemas of the named tools so you can call them. "
                    "Loaded tools stay available for the rest of the conversation."
                ),
            ),
        ]


# ---------------------------------------------------------------------------
# Conversation wiring
# ---------------------------------------------------------------------------


class ToolContextMixin:
    """The tool-context state and helpers shared by the sync and async conversations."""

    _tools: ToolRegistry
    _system_prompt: str | None
    _conversation_id: str

    def _init_tool_context(
        self,
        *,
        max_tool_result_length: int | None,
        tool_result_policy: ToolResultPolicy | None,
        defer_tools: bool | Literal["auto"],
        preload_tools: Iterable[str] | None,
    ) -> None:
        # An explicit max_tool_result_length=None keeps the old "never shorten" meaning.
        if tool_result_policy is None and max_tool_result_length is not None:
            tool_result_policy = ToolResultPolicy(max_chars=max_tool_result_length)
        self._tool_result_policy = tool_result_policy
        self._artifacts = ArtifactStore()
        self._has_offloaded = False
        self._tool_loader = ToolLoader(defer=defer_tools, preload=preload_tools or ())
        self._system_tools = ToolRegistry()

    def _reset_tool_context(self) -> None:
        self._artifacts.clear()
        self._has_offloaded = False

    def _active_tools(self) -> ToolRegistry:
        """Tools whose schemas the model sees this round.

        User tools, minus the ones still deferred, plus the built-in helpers
        this conversation currently needs (tool search/load while deferring,
        ``read_tool_result`` once something was offloaded).  A user tool with
        the same name always wins over a helper.
        """
        system = ToolRegistry()
        helpers = []
        if self._tool_loader.is_deferring(self._tools):
            helpers.extend(self._tool_loader.meta_tools(lambda: self._tools))
        if self._has_offloaded:
            helpers.append(make_read_tool_result(self._artifacts, self._conversation_id))
        for td in helpers:
            if td.name not in self._tools:
                system.add(td)
        self._system_tools = system
        visible = self._tool_loader.visible(self._tools)
        if not system:
            return visible
        active = ToolRegistry()
        for td in [*visible.definitions, *system.definitions]:
            active.add(td)
        return active

    def _dispatch_registry(self, name: str) -> ToolRegistry:
        """Registry that executes *name*: a helper or the user's tools."""
        if name in self._system_tools and name not in self._tools:
            return self._system_tools
        self._tool_loader.mark_used(name, self._tools)
        return self._tools

    def _all_callable_tools(self) -> ToolRegistry:
        """User tools plus active helpers, for parsing simulated tool calls."""
        if not self._system_tools:
            return self._tools
        merged = ToolRegistry()
        for td in [*self._tools.definitions, *self._system_tools.definitions]:
            merged.add(td)
        return merged

    def _shape_tool_results(self, batch: list[tuple[str, str, bool]]) -> list[str]:
        """History content for one round of results (see :func:`shape_tool_results`)."""
        if self._tool_result_policy is None:
            return [text for _name, text, _err in batch]
        contents, offloaded = shape_tool_results(
            batch,
            policy=self._tool_result_policy,
            store=self._artifacts,
            scope=self._conversation_id,
        )
        if offloaded:
            self._has_offloaded = True
        return contents

    def _simulated_system_prompt(self) -> str:
        """System prompt plus the prompt-format description of this round's tools."""
        from .simulated_tools import build_tool_prompt

        tool_prompt = build_tool_prompt(self._active_tools())
        return f"{self._system_prompt}\n\n{tool_prompt}" if self._system_prompt else tool_prompt


# ---------------------------------------------------------------------------
# Tool call / result pairing
# ---------------------------------------------------------------------------


def _call_ids(message: dict[str, Any]) -> list[str]:
    ids = []
    for call in message.get("tool_calls") or ():
        identifier = call.get("id") if isinstance(call, dict) else getattr(call, "id", None)
        if identifier:
            ids.append(str(identifier))
    return ids


def repair_tool_pairs(messages: list[dict[str, Any]]) -> int:
    """Make every tool call have exactly one result, in place.

    A call with no result (the run stopped between the call and its execution)
    gets a synthetic error result right after the results that do exist.  A
    ``tool`` message that does not answer a call in the assistant message
    before it is dropped.  Providers reject both shapes outright.

    Returns:
        How many messages were inserted or dropped.
    """
    changes = 0
    out: list[dict[str, Any]] = []
    i = 0
    n = len(messages)
    while i < n:
        message = messages[i]
        i += 1
        role = message.get("role")
        if role == "tool":
            changes += 1
            continue
        out.append(message)
        if role != "assistant" or not message.get("tool_calls"):
            continue
        ids = _call_ids(message)
        answered: set[str] = set()
        while i < n and messages[i].get("role") == "tool":
            result = messages[i]
            i += 1
            call_id = str(result.get("tool_call_id") or "")
            if call_id in ids and call_id not in answered:
                out.append(result)
                answered.add(call_id)
            else:
                changes += 1
        for call_id in ids:
            if call_id not in answered:
                out.append({"role": "tool", "tool_call_id": call_id, "content": INTERRUPTED_TOOL_RESULT})
                changes += 1
    if changes:
        logger.info("Repaired %d unpaired tool call/result message(s) in history", changes)
        messages[:] = out
    return changes
