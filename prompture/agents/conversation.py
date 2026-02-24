"""Stateful multi-turn conversation support for Prompture."""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Iterator
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Literal, Union, cast

from pydantic import BaseModel

from ..drivers import get_driver_for_model
from ..drivers.base import Driver
from ..extraction.fields import get_registry_snapshot
from ..extraction.tools import (
    clean_json_text,
    convert_value,
    get_field_default,
)
from ..infra.budget import BudgetPolicy, BudgetState, enforce_budget
from ..infra.callbacks import DriverCallbacks
from ..infra.session import UsageSession
from ..media.image import ImageInput, make_image
from ..persistence.serialization import export_conversation, import_conversation
from ..persistence.store import load_from_file, save_to_file
from .persona import Persona, get_persona
from .tools_schema import ToolRegistry

logger = logging.getLogger("prompture.conversation")


class Conversation:
    """Stateful multi-turn conversation with an LLM.

    Maintains a message history across calls so the model can reference
    previous turns.  Works with any Prompture driver.

    Example::

        conv = Conversation("openai/gpt-4", system_prompt="You are a data extractor")
        r1 = conv.ask_for_json("Extract names from: John, age 30", name_schema)
        r2 = conv.ask_for_json("Now extract ages", age_schema)  # sees turn 1
    """

    def __init__(
        self,
        model_name: str | None = None,
        *,
        driver: Driver | None = None,
        system_prompt: str | None = None,
        persona: str | Persona | None = None,
        options: dict[str, Any] | None = None,
        callbacks: DriverCallbacks | None = None,
        tools: ToolRegistry | None = None,
        max_tool_rounds: int = 10,
        max_tool_result_length: int | None = None,
        simulated_tools: bool | Literal["auto"] = "auto",
        conversation_id: str | None = None,
        auto_save: str | Path | None = None,
        tags: list[str] | None = None,
        max_history_messages: int | None = None,
        max_cost: float | None = None,
        max_tokens: int | None = None,
        budget_policy: BudgetPolicy | None = None,
        fallback_models: list[str] | None = None,
        on_model_fallback: Callable[[str, str, Any], None] | None = None,
    ) -> None:
        if system_prompt is not None and persona is not None:
            raise ValueError("Cannot provide both 'system_prompt' and 'persona'. Use one or the other.")

        # Resolve persona
        resolved_persona: Persona | None = None
        if persona is not None:
            if isinstance(persona, str):
                resolved_persona = get_persona(persona)
                if resolved_persona is None:
                    raise ValueError(f"Persona '{persona}' not found in registry.")
            else:
                resolved_persona = persona

        if model_name is None and driver is None:
            # Check persona for model_hint
            if resolved_persona is not None and resolved_persona.model_hint:
                model_name = resolved_persona.model_hint
            else:
                raise ValueError("Either model_name or driver must be provided")

        if driver is not None:
            self._driver = driver
        else:
            self._driver = get_driver_for_model(model_name)  # type: ignore[arg-type, assignment]

        if callbacks is not None:
            self._driver.callbacks = callbacks

        self._model_name: str = model_name if model_name is not None else ""

        # Apply persona: render system_prompt and merge settings
        self._system_prompt: str | None
        if resolved_persona is not None:
            self._system_prompt = resolved_persona.render()
            # Persona settings as defaults, explicit options override
            self._options = {**resolved_persona.settings, **(dict(options) if options else {})}
        else:
            self._system_prompt = system_prompt
            self._options = dict(options) if options else {}

        self._messages: list[dict[str, Any]] = []
        self._usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost": 0.0,
            "turns": 0,
        }
        self._tools = tools or ToolRegistry()
        self._max_tool_rounds = max_tool_rounds
        self._max_tool_result_length = max_tool_result_length
        self._simulated_tools = simulated_tools
        self._max_history_messages = max_history_messages

        # Budget enforcement
        self._max_cost = max_cost
        self._max_tokens = max_tokens
        self._budget_policy = budget_policy
        self._fallback_models = list(fallback_models) if fallback_models else None
        self._on_model_fallback = on_model_fallback

        # Full (pre-truncation) tool results keyed by tool_call_id.
        self._full_tool_results: dict[str, str] = {}

        # Reasoning content from last response
        self._last_reasoning: str | None = None

        # Persistence
        self._conversation_id = conversation_id or str(uuid.uuid4())
        self._auto_save = Path(auto_save) if auto_save else None
        self._metadata: dict[str, Any] = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "tags": list(tags) if tags else [],
        }

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def last_reasoning(self) -> str | None:
        """The reasoning/thinking content from the last LLM response, if any."""
        return self._last_reasoning

    @property
    def messages(self) -> list[dict[str, Any]]:
        """Read-only view of the conversation history."""
        return list(self._messages)

    @property
    def usage(self) -> dict[str, Any]:
        """Accumulated token/cost totals across all turns."""
        return dict(self._usage)

    def clear(self) -> None:
        """Reset message history (keeps system_prompt and driver)."""
        self._messages.clear()

    def add_context(self, role: str, content: str, images: list[ImageInput] | None = None) -> None:
        """Seed the history with a user or assistant message."""
        if role not in ("user", "assistant"):
            raise ValueError("role must be 'user' or 'assistant'")
        msg_content = self._build_content_with_images(content, images)
        self._messages.append({"role": role, "content": msg_content})

    def register_tool(
        self,
        fn: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        """Register a Python function as a tool the LLM can call."""
        self._tools.register(fn, name=name, description=description)

    def usage_summary(self) -> str:
        """Human-readable summary of accumulated usage."""
        u = self._usage
        return f"Conversation: {u['total_tokens']:,} tokens across {u['turns']} turn(s) costing ${u['cost']:.4f}"

    # ------------------------------------------------------------------
    # Persistence properties
    # ------------------------------------------------------------------

    @property
    def conversation_id(self) -> str:
        """Unique identifier for this conversation."""
        return self._conversation_id

    @property
    def tags(self) -> list[str]:
        """Tags attached to this conversation."""
        return list(self._metadata.get("tags", []))

    @tags.setter
    def tags(self, value: list[str]) -> None:
        self._metadata["tags"] = list(value)

    # ------------------------------------------------------------------
    # Export / Import
    # ------------------------------------------------------------------

    def export(self, *, usage_session: UsageSession | None = None, strip_images: bool = False) -> dict[str, Any]:
        """Export conversation state to a JSON-serializable dict."""
        tools_metadata = (
            [
                {"name": td.name, "description": td.description, "parameters": td.parameters}
                for td in self._tools.definitions
            ]
            if self._tools and self._tools.definitions
            else None
        )
        return export_conversation(
            model_name=self._model_name,
            system_prompt=self._system_prompt,
            options=self._options,
            messages=self._messages,
            usage=self._usage,
            max_tool_rounds=self._max_tool_rounds,
            tools_metadata=tools_metadata,
            usage_session=usage_session,
            metadata=self._metadata,
            conversation_id=self._conversation_id,
            strip_images=strip_images,
        )

    @classmethod
    def from_export(
        cls,
        data: dict[str, Any],
        *,
        callbacks: DriverCallbacks | None = None,
        tools: ToolRegistry | None = None,
    ) -> Conversation:
        """Reconstruct a :class:`Conversation` from an export dict.

        The driver is reconstructed from the stored ``model_name``.
        Callbacks and tool *functions* must be re-attached by the caller
        (tool metadata — name/description/parameters — is preserved in
        the export but executable functions cannot be serialized).
        """
        imported = import_conversation(data)

        model_name = imported.get("model_name") or ""
        if not model_name:
            raise ValueError("Cannot restore conversation: export has no model_name")
        conv = cls(
            model_name=model_name,
            system_prompt=imported.get("system_prompt"),
            options=imported.get("options", {}),
            callbacks=callbacks,
            tools=tools,
            max_tool_rounds=imported.get("max_tool_rounds", 10),
            conversation_id=imported.get("conversation_id"),
            tags=imported.get("metadata", {}).get("tags", []),
        )
        conv._messages = imported.get("messages", [])
        conv._usage = imported.get(
            "usage",
            {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost": 0.0,
                "turns": 0,
            },
        )
        meta = imported.get("metadata", {})
        if "created_at" in meta:
            conv._metadata["created_at"] = meta["created_at"]
        return conv

    def save(self, path: str | Path, **kwargs: Any) -> None:
        """Export and write to a JSON file.

        Keyword arguments are forwarded to :meth:`export`.
        """
        save_to_file(self.export(**kwargs), path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        callbacks: DriverCallbacks | None = None,
        tools: ToolRegistry | None = None,
    ) -> Conversation:
        """Load a conversation from a JSON file."""
        data = load_from_file(path)
        return cls.from_export(data, callbacks=callbacks, tools=tools)

    def _maybe_auto_save(self) -> None:
        """Auto-save after each turn if configured.  Errors are silently logged."""
        if self._auto_save is None:
            return
        try:
            self.save(self._auto_save)
        except Exception:
            logger.debug("Auto-save failed for conversation %s", self._conversation_id, exc_info=True)

    # ------------------------------------------------------------------
    # Budget enforcement
    # ------------------------------------------------------------------

    @property
    def budget_remaining(self) -> dict[str, Any] | None:
        """Current budget status, or ``None`` if no budget is configured."""
        if self._budget_policy is None:
            return None
        state = self._build_budget_state()
        return {
            "cost_remaining": state.cost_remaining,
            "tokens_remaining": state.tokens_remaining,
            "cost_fraction": state.cost_fraction,
            "exceeded": state.exceeded,
            "active_model": self._model_name,
        }

    def _build_budget_state(self) -> BudgetState:
        return BudgetState(
            cost_used=self._usage["cost"],
            tokens_used=int(self._usage["total_tokens"]),
            max_cost=self._max_cost,
            max_tokens=self._max_tokens,
        )

    def _check_budget(self) -> None:
        """Enforce budget policy before a driver call. No-op when unconfigured."""
        if self._budget_policy is None:
            return
        state = self._build_budget_state()
        new_model = enforce_budget(
            state,
            self._budget_policy,
            fallback_models=self._fallback_models,
            current_model=self._model_name,
            on_model_fallback=self._on_model_fallback,
        )
        if new_model is not None:
            self._switch_model(new_model)

    def _switch_model(self, new_model: str) -> None:
        """Replace the active driver with one for *new_model*."""
        old_callbacks = self._driver.callbacks
        self._driver = cast(Driver, get_driver_for_model(new_model))
        if old_callbacks is not None:
            self._driver.callbacks = old_callbacks
        self._model_name = new_model

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    @staticmethod
    def _build_content_with_images(text: str, images: list[ImageInput] | None = None) -> str | list[dict[str, Any]]:
        """Return plain string when no images, or a list of content blocks."""
        if not images:
            return text
        blocks: list[dict[str, Any]] = [{"type": "text", "text": text}]
        for img in images:
            ic = make_image(img)
            blocks.append({"type": "image", "source": ic})
        return blocks

    def _truncate_tool_result(self, result_str: str) -> str:
        """Truncate a tool result string if it exceeds the configured limit.

        Returns the original string when no limit is set or the string is
        within bounds.  Otherwise returns a truncated version with a note
        indicating the original length.
        """
        if self._max_tool_result_length is None:
            return result_str
        if len(result_str) <= self._max_tool_result_length:
            return result_str
        logger.debug(
            "Truncating tool result from %d to %d chars",
            len(result_str),
            self._max_tool_result_length,
        )
        return (
            result_str[: self._max_tool_result_length] + f"\n\n[... result truncated ({len(result_str):,} chars total)]"
        )

    def _build_messages(self, user_content: str, images: list[ImageInput] | None = None) -> list[dict[str, Any]]:
        """Build the full messages array for an API call."""
        msgs: list[dict[str, Any]] = []
        if self._system_prompt:
            msgs.append({"role": "system", "content": self._system_prompt})
        msgs.extend(self._messages)
        content = self._build_content_with_images(user_content, images)
        msgs.append({"role": "user", "content": content})
        return msgs

    def _accumulate_usage(self, meta: dict[str, Any]) -> None:
        delta_tokens = meta.get("total_tokens", 0)
        delta_cost = meta.get("cost", 0.0)
        self._usage["prompt_tokens"] += meta.get("prompt_tokens", 0)
        self._usage["completion_tokens"] += meta.get("completion_tokens", 0)
        self._usage["total_tokens"] += delta_tokens
        self._usage["cost"] += delta_cost
        self._usage["turns"] += 1
        logger.debug(
            "[conversation] usage delta tokens=%d cost=%.6f | running total tokens=%d cost=%.6f turns=%d",
            delta_tokens,
            delta_cost,
            self._usage["total_tokens"],
            self._usage["cost"],
            self._usage["turns"],
        )
        # Trim history to sliding window (system prompt lives outside _messages)
        if self._max_history_messages is not None and len(self._messages) > self._max_history_messages:
            self._messages = self._messages[-self._max_history_messages :]
        self._maybe_auto_save()

    def ask(
        self,
        content: str,
        options: dict[str, Any] | None = None,
        images: list[ImageInput] | None = None,
    ) -> str:
        """Send a message and get a raw text response.

        Appends the user message and assistant response to history.
        If tools are registered and the driver supports tool use,
        dispatches to the tool execution loop.

        Args:
            content: The text message to send.
            options: Additional options for the driver.
            images: Optional list of images to include (bytes, path, URL,
                base64 string, or :class:`ImageContent`).
        """
        self._last_reasoning = None

        # Route to appropriate tool handling
        if self._tools:
            use_native = getattr(self._driver, "supports_tool_use", False)
            if self._simulated_tools is True or (self._simulated_tools == "auto" and not use_native):
                return self._ask_with_simulated_tools(content, options, images=images)
            elif use_native and self._simulated_tools is not True:  # type: ignore[comparison-overlap]
                return self._ask_with_tools(content, options, images=images)

        self._check_budget()
        merged = {**self._options, **(options or {})}
        messages = self._build_messages(content, images=images)
        resp = self._driver.generate_messages_with_hooks(messages, merged)

        text: str = resp.get("text", "")
        meta = resp.get("meta", {})
        self._last_reasoning = resp.get("reasoning_content")

        # Record in history — store content with images for context
        user_content = self._build_content_with_images(content, images)
        self._messages.append({"role": "user", "content": user_content})
        self._messages.append({"role": "assistant", "content": text})
        self._accumulate_usage(meta)

        return text

    def _ask_with_tools(
        self,
        content: str,
        options: dict[str, Any] | None = None,
        images: list[ImageInput] | None = None,
    ) -> str:
        """Execute the tool-use loop: send -> check tool_calls -> execute -> re-send."""
        merged = {**self._options, **(options or {})}
        tool_defs = self._tools.to_openai_format()

        # Build messages including user content
        user_content = self._build_content_with_images(content, images)
        self._messages.append({"role": "user", "content": user_content})
        msgs = self._build_messages_raw()

        for _round in range(self._max_tool_rounds):
            self._check_budget()
            resp = self._driver.generate_messages_with_tools_with_hooks(msgs, tool_defs, merged)

            meta = resp.get("meta", {})
            self._accumulate_usage(meta)

            tool_calls = resp.get("tool_calls", [])
            text: str = resp.get("text", "")

            if not tool_calls:
                # No tool calls -> final response
                self._last_reasoning = resp.get("reasoning_content")
                self._messages.append({"role": "assistant", "content": text})
                return text

            # Record assistant message with tool_calls
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": text}
            assistant_msg["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])},
                }
                for tc in tool_calls
            ]
            # Preserve reasoning_content for providers that require it
            # on subsequent requests (e.g. Moonshot reasoning models).
            if resp.get("reasoning_content") is not None:
                assistant_msg["reasoning_content"] = resp["reasoning_content"]

            self._messages.append(assistant_msg)
            msgs.append(assistant_msg)

            # Execute each tool call and append results
            for tc in tool_calls:
                try:
                    result = self._tools.execute(tc["name"], tc["arguments"])
                    result_str = json.dumps(result) if not isinstance(result, str) else result
                except Exception as exc:
                    result_str = f"Error: {exc}"

                # Preserve full result for step extraction before truncating
                self._full_tool_results[tc["id"]] = result_str

                tool_result_msg: dict[str, Any] = {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": self._truncate_tool_result(result_str),
                }
                self._messages.append(tool_result_msg)
                msgs.append(tool_result_msg)

        raise RuntimeError(f"Tool execution loop exceeded {self._max_tool_rounds} rounds")

    def ask_with_tool_events(
        self,
        content: str,
        options: dict[str, Any] | None = None,
        images: list[ImageInput] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Run the tool loop yielding events for each tool call and result.

        Yields dicts with ``type`` in ``("tool_call", "tool_result",
        "text_delta")``.  Tool calls and results are emitted during the
        loop; the final model response is streamed as ``text_delta`` events
        via :meth:`ask_stream` when streaming is supported.

        This method is intended for use by ``Agent.run_stream()`` to
        provide real-time feedback when tools are registered.
        """
        self._last_reasoning = None

        if not self._tools:
            # No tools — just stream directly
            for chunk in self.ask_stream(content, options, images=images):
                yield {"type": "text_delta", "text": chunk}
            return

        use_native = getattr(self._driver, "supports_tool_use", False)
        if self._simulated_tools is True or (self._simulated_tools == "auto" and not use_native):
            # Simulated tools — fall back to non-streaming with events
            yield from self._ask_with_simulated_tool_events(content, options, images=images)
            return

        # Native tool calling with event emission
        merged = {**self._options, **(options or {})}
        tool_defs = self._tools.to_openai_format()

        user_content = self._build_content_with_images(content, images)
        self._messages.append({"role": "user", "content": user_content})
        msgs = self._build_messages_raw()

        for _round in range(self._max_tool_rounds):
            resp = self._driver.generate_messages_with_tools_with_hooks(msgs, tool_defs, merged)

            meta = resp.get("meta", {})
            self._accumulate_usage(meta)

            tool_calls = resp.get("tool_calls", [])
            text = resp.get("text", "")

            if not tool_calls:
                # Final response — stream it if possible
                self._last_reasoning = resp.get("reasoning_content")
                self._messages.append({"role": "assistant", "content": text})
                # Yield final text as deltas (single chunk since we already have the full text)
                yield {"type": "text_delta", "text": text}
                return

            # Record assistant message with tool_calls
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": text}
            assistant_msg["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])},
                }
                for tc in tool_calls
            ]
            if resp.get("reasoning_content") is not None:
                assistant_msg["reasoning_content"] = resp["reasoning_content"]

            self._messages.append(assistant_msg)
            msgs.append(assistant_msg)

            # Execute each tool and yield events
            for tc in tool_calls:
                yield {
                    "type": "tool_call",
                    "name": tc["name"],
                    "arguments": tc["arguments"],
                    "id": tc["id"],
                }
                try:
                    result = self._tools.execute(tc["name"], tc["arguments"])
                    result_str = json.dumps(result) if not isinstance(result, str) else result
                except Exception as exc:
                    result_str = f"Error: {exc}"

                # Yield the FULL result for UI consumers
                yield {
                    "type": "tool_result",
                    "name": tc["name"],
                    "result": result_str,
                    "id": tc["id"],
                }

                # Truncate before adding to conversation to avoid token overflow
                tool_result_msg: dict[str, Any] = {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": self._truncate_tool_result(result_str),
                }
                self._messages.append(tool_result_msg)
                msgs.append(tool_result_msg)

        raise RuntimeError(f"Tool execution loop exceeded {self._max_tool_rounds} rounds")

    def _ask_with_simulated_tool_events(
        self,
        content: str,
        options: dict[str, Any] | None = None,
        images: list[ImageInput] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Simulated tool calling with event emission."""
        from .simulated_tools import build_tool_prompt, format_tool_result, parse_simulated_response

        merged = {**self._options, **(options or {})}
        tool_prompt = build_tool_prompt(self._tools)

        augmented_system = tool_prompt
        if self._system_prompt:
            augmented_system = f"{self._system_prompt}\n\n{tool_prompt}"

        user_content = self._build_content_with_images(content, images)
        self._messages.append({"role": "user", "content": user_content})

        for _round in range(self._max_tool_rounds):
            msgs: list[dict[str, Any]] = []
            msgs.append({"role": "system", "content": augmented_system})
            msgs.extend(self._messages)

            resp = self._driver.generate_messages_with_hooks(msgs, merged)
            text = resp.get("text", "")
            meta = resp.get("meta", {})
            self._accumulate_usage(meta)

            parsed = parse_simulated_response(text, self._tools)

            if parsed["type"] == "final_answer":
                answer = parsed["content"]
                self._messages.append({"role": "assistant", "content": answer})
                yield {"type": "text_delta", "text": answer}
                return

            tool_name = parsed["name"]
            tool_args = parsed["arguments"]

            self._messages.append({"role": "assistant", "content": text})

            yield {"type": "tool_call", "name": tool_name, "arguments": tool_args, "id": ""}

            try:
                result = self._tools.execute(tool_name, tool_args)
                result_msg = format_tool_result(tool_name, result)
            except Exception as exc:
                result_msg = format_tool_result(tool_name, f"Error: {exc}")

            yield {"type": "tool_result", "name": tool_name, "result": result_msg, "id": ""}

            self._messages.append({"role": "user", "content": self._truncate_tool_result(result_msg)})

        raise RuntimeError(f"Simulated tool execution loop exceeded {self._max_tool_rounds} rounds")

    def _ask_with_simulated_tools(
        self,
        content: str,
        options: dict[str, Any] | None = None,
        images: list[ImageInput] | None = None,
    ) -> str:
        """Prompt-based tool calling for drivers without native tool use."""
        from .simulated_tools import build_tool_prompt, format_tool_result, parse_simulated_response

        merged = {**self._options, **(options or {})}
        tool_prompt = build_tool_prompt(self._tools)

        # Augment system prompt with tool descriptions
        augmented_system = tool_prompt
        if self._system_prompt:
            augmented_system = f"{self._system_prompt}\n\n{tool_prompt}"

        # Record user message in history
        user_content = self._build_content_with_images(content, images)
        self._messages.append({"role": "user", "content": user_content})

        for _round in range(self._max_tool_rounds):
            self._check_budget()
            # Build messages with the augmented system prompt
            msgs: list[dict[str, Any]] = []
            msgs.append({"role": "system", "content": augmented_system})
            msgs.extend(self._messages)

            resp = self._driver.generate_messages_with_hooks(msgs, merged)
            text = resp.get("text", "")
            meta = resp.get("meta", {})
            self._accumulate_usage(meta)

            parsed = parse_simulated_response(text, self._tools)

            if parsed["type"] == "final_answer":
                answer: str = parsed["content"]
                self._messages.append({"role": "assistant", "content": answer})
                return answer

            # Tool call
            tool_name = parsed["name"]
            tool_args = parsed["arguments"]

            # Record assistant's tool call as an assistant message
            self._messages.append({"role": "assistant", "content": text})

            try:
                result = self._tools.execute(tool_name, tool_args)
                result_msg = format_tool_result(tool_name, result)
            except Exception as exc:
                result_msg = format_tool_result(tool_name, f"Error: {exc}")

            # Record tool result as a user message (truncated for the LLM)
            self._messages.append({"role": "user", "content": self._truncate_tool_result(result_msg)})

        raise RuntimeError(f"Simulated tool execution loop exceeded {self._max_tool_rounds} rounds")

    def _build_messages_raw(self) -> list[dict[str, Any]]:
        """Build messages array from system prompt + full history (including tool messages)."""
        msgs: list[dict[str, Any]] = []
        if self._system_prompt:
            msgs.append({"role": "system", "content": self._system_prompt})
        msgs.extend(self._messages)
        return msgs

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def ask_stream(
        self,
        content: str,
        options: dict[str, Any] | None = None,
        images: list[ImageInput] | None = None,
    ) -> Iterator[str]:
        """Send a message and yield text chunks as they arrive.

        Falls back to non-streaming :meth:`ask` if the driver doesn't
        support streaming. After iteration completes, the full response
        is recorded in history.
        """
        self._check_budget()
        if not getattr(self._driver, "supports_streaming", False):
            yield self.ask(content, options, images=images)
            return

        merged = {**self._options, **(options or {})}
        messages = self._build_messages(content, images=images)

        user_content = self._build_content_with_images(content, images)
        self._messages.append({"role": "user", "content": user_content})

        driver_name = getattr(self._driver, "model", self._driver.__class__.__name__)

        # Fire on_request callback before streaming begins
        self._driver._fire_callback(
            "on_request",
            {"prompt": None, "messages": messages, "options": merged, "driver": driver_name},
        )

        t0 = time.perf_counter()
        full_text = ""
        try:
            for chunk in self._driver.generate_messages_stream(messages, merged):
                if chunk["type"] == "delta":
                    full_text += chunk["text"]
                    # Fire stream delta callback
                    self._driver._fire_callback(
                        "on_stream_delta",
                        {"text": chunk["text"], "driver": driver_name},
                    )
                    yield chunk["text"]
                elif chunk["type"] == "done":
                    meta = chunk.get("meta", {})
                    elapsed_ms = (time.perf_counter() - t0) * 1000
                    # Fire on_response callback with usage metadata
                    self._driver._fire_callback(
                        "on_response",
                        {
                            "text": chunk.get("text", full_text),
                            "meta": meta,
                            "driver": driver_name,
                            "elapsed_ms": elapsed_ms,
                        },
                    )
                    self._accumulate_usage(meta)
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            self._driver._fire_callback(
                "on_error",
                {"error": exc, "prompt": None, "messages": messages, "options": merged, "driver": driver_name},
            )
            raise

        self._messages.append({"role": "assistant", "content": full_text})

    def ask_for_json(
        self,
        content: str,
        json_schema: dict[str, Any],
        *,
        ai_cleanup: bool = True,
        options: dict[str, Any] | None = None,
        output_format: Literal["json", "toon"] = "json",
        json_mode: Literal["auto", "on", "off"] = "auto",
        images: list[ImageInput] | None = None,
    ) -> dict[str, Any]:
        """Send a message with schema enforcement and get structured JSON back.

        The schema instructions are appended to the prompt but only the
        original *content* is stored in conversation history to keep
        context clean for subsequent turns.
        """

        self._check_budget()
        self._last_reasoning = None

        merged = {**self._options, **(options or {})}

        # Build the full prompt with schema instructions inline (handled by ask_for_json)
        # We use a special approach: call ask_for_json with the driver but pass messages context
        schema_string = json.dumps(json_schema, indent=2)

        # Determine JSON mode
        use_json_mode = False
        if json_mode == "on":
            use_json_mode = True
        elif json_mode == "auto":
            use_json_mode = getattr(self._driver, "supports_json_mode", False)

        if use_json_mode:
            merged = {**merged, "json_mode": True}
            if getattr(self._driver, "supports_json_schema", False):
                merged["json_schema"] = json_schema

        # Build instruction based on JSON mode
        if use_json_mode and getattr(self._driver, "supports_json_schema", False):
            instruct = "Extract data matching the requested schema.\nIf a value is unknown use null."
        elif use_json_mode:
            instruct = (
                "Return a JSON object that validates against this schema:\n"
                f"{schema_string}\n\n"
                "If a value is unknown use null."
            )
        else:
            instruct = (
                "Return only a single JSON object (no markdown, no extra text) that validates against this JSON schema:\n"
                f"{schema_string}\n\n"
                "If a value is unknown use null. Use double quotes for keys and strings."
            )

        full_user_content = f"{content}\n\n{instruct}"

        messages = self._build_messages(full_user_content, images=images)
        resp = self._driver.generate_messages_with_hooks(messages, merged)

        text = resp.get("text", "")
        meta = resp.get("meta", {})
        self._last_reasoning = resp.get("reasoning_content")

        # Store original content (without schema boilerplate) for cleaner context
        # Include images in history so subsequent turns can reference them
        user_content = self._build_content_with_images(content, images)
        self._messages.append({"role": "user", "content": user_content})

        # Parse JSON
        cleaned = clean_json_text(text)
        try:
            json_obj = json.loads(cleaned)
        except json.JSONDecodeError:
            if ai_cleanup:
                from ..extraction.core import clean_json_text_with_ai

                cleaned, cleanup_meta = clean_json_text_with_ai(self._driver, cleaned, self._model_name, merged)
                json_obj = json.loads(cleaned)
                # Track cleanup call tokens in conversation usage
                if cleanup_meta:
                    self._accumulate_usage(cleanup_meta)
            else:
                raise

        # Store assistant response in history
        self._messages.append({"role": "assistant", "content": cleaned})
        self._accumulate_usage(meta)

        model_name = self._model_name
        if "/" in model_name:
            model_name = model_name.split("/", 1)[1]

        usage = {
            **meta,
            "raw_response": resp,
            "model_name": model_name or getattr(self._driver, "model", ""),
        }

        result: dict[str, Any] = {
            "json_string": cleaned,
            "json_object": json_obj,
            "usage": usage,
            "output_format": output_format,
            "reasoning": self._last_reasoning,
        }

        if output_format == "toon":
            try:
                import toon

                result["toon_string"] = toon.encode(json_obj)
            except ImportError:
                raise RuntimeError("TOON requested but 'python-toon' is not installed.") from None

        return result

    def extract_with_model(
        self,
        model_cls: type[BaseModel],
        text: str,
        *,
        instruction_template: str = "Extract information from the following text:",
        ai_cleanup: bool = True,
        output_format: Literal["json", "toon"] = "json",
        options: dict[str, Any] | None = None,
        json_mode: Literal["auto", "on", "off"] = "auto",
        images: list[ImageInput] | None = None,
        reasoning_strategy: str | None = None,
    ) -> dict[str, Any]:
        """Extract structured information into a Pydantic model with conversation context."""
        from ..extraction.core import normalize_field_value
        from ..extraction.reasoning import (
            _strategy_name,
            apply_reasoning_strategy,
            auto_select_reasoning_strategy,
        )

        schema = model_cls.model_json_schema()
        content_prompt = f"{instruction_template} {text}"
        if reasoning_strategy == "auto":
            reasoning_strategy = auto_select_reasoning_strategy(text, schema)
        content_prompt = apply_reasoning_strategy(content_prompt, reasoning_strategy)

        result = self.ask_for_json(
            content=content_prompt,
            json_schema=schema,
            ai_cleanup=ai_cleanup,
            options=options,
            output_format=output_format,
            json_mode=json_mode,
            images=images,
        )
        result["usage"]["reasoning_strategy"] = _strategy_name(reasoning_strategy)

        # Normalize field values
        json_object = result["json_object"]
        schema_properties = schema.get("properties", {})

        for field_name, field_info in model_cls.model_fields.items():
            if field_name in json_object and field_name in schema_properties:
                field_def = {
                    "nullable": not schema_properties[field_name].get("type")
                    or "null"
                    in (
                        schema_properties[field_name].get("anyOf", [])
                        if isinstance(schema_properties[field_name].get("anyOf"), list)
                        else []
                    ),
                    "default": field_info.default
                    if hasattr(field_info, "default") and field_info.default is not ...
                    else None,
                }
                json_object[field_name] = normalize_field_value(
                    json_object[field_name],
                    field_info.annotation,  # type: ignore[arg-type]
                    field_def,
                )

        model_instance = model_cls(**json_object)

        result_dict = {
            "json_string": result["json_string"],
            "json_object": result["json_object"],
            "usage": result["usage"],
        }
        result_dict["model"] = model_instance

        return type(  # type: ignore[no-any-return]
            "ExtractResult",
            (dict,),
            {
                "__getattr__": lambda self, key: self.get(key),
                "__call__": lambda self: self["model"],
            },
        )(result_dict)

    # ------------------------------------------------------------------
    # Internal: stepwise with shared context
    # ------------------------------------------------------------------

    def _stepwise_extract(
        self,
        model_cls: type[BaseModel],
        text: str,
        instruction_template: str,
        ai_cleanup: bool,
        fields: list[str] | None,
        field_definitions: dict[str, Any] | None,
        json_mode: Literal["auto", "on", "off"],
    ) -> dict[str, Union[str, dict[str, Any]]]:
        """Stepwise extraction using conversation context between fields."""
        if field_definitions is None:
            field_definitions = get_registry_snapshot()

        data: dict[str, Any] = {}
        validation_errors: list[str] = []
        field_results: dict[str, Any] = {}

        accumulated_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost": 0.0,
            "model_name": self._model_name,
            "field_usages": {},
        }

        valid_fields = set(model_cls.model_fields.keys())
        if fields is not None:
            invalid_fields = set(fields) - valid_fields
            if invalid_fields:
                raise KeyError(f"Fields not found in model: {', '.join(invalid_fields)}")
            field_items = [(name, model_cls.model_fields[name]) for name in fields]
        else:
            field_items = list(model_cls.model_fields.items())

        # Seed conversation with the source text
        self.add_context("user", f"I need to extract information from this text:\n\n{text}")
        self.add_context(
            "assistant", "I'll help you extract the information from that text. What would you like to extract?"
        )

        for field_name, field_info in field_items:
            logger.debug("[stepwise-conv] Extracting field: %s", field_name)

            field_schema = {
                "value": {
                    "type": "integer" if field_info.annotation is int else "string",
                    "description": field_info.description or f"Value for {field_name}",
                }
            }

            try:
                prompt = instruction_template.format(field_name=field_name)
                result = self.ask_for_json(
                    content=f"{prompt} {text}",
                    json_schema=field_schema,
                    ai_cleanup=ai_cleanup,
                    json_mode=json_mode,
                )

                field_usage = result.get("usage", {})
                accumulated_usage["prompt_tokens"] += field_usage.get("prompt_tokens", 0)
                accumulated_usage["completion_tokens"] += field_usage.get("completion_tokens", 0)
                accumulated_usage["total_tokens"] += field_usage.get("total_tokens", 0)
                accumulated_usage["cost"] += field_usage.get("cost", 0.0)
                accumulated_usage["field_usages"][field_name] = field_usage  # type: ignore[index]

                extracted_value = result["json_object"]["value"]
                if isinstance(extracted_value, dict) and "value" in extracted_value:
                    raw_value = extracted_value["value"]
                else:
                    raw_value = extracted_value

                # Normalize
                from ..extraction.core import normalize_field_value

                field_def: dict[str, Any] = {}
                if field_definitions and field_name in field_definitions:
                    field_def = field_definitions[field_name] if isinstance(field_definitions[field_name], dict) else {}

                nullable = field_def.get("nullable", True)
                default_value = field_def.get("default")
                if (
                    default_value is None
                    and hasattr(field_info, "default")
                    and field_info.default is not ...
                    and str(field_info.default) != "PydanticUndefined"
                ):
                    default_value = field_info.default

                normalize_def = {"nullable": nullable, "default": default_value}
                raw_value = normalize_field_value(raw_value, field_info.annotation, normalize_def)  # type: ignore[arg-type]

                try:
                    converted_value = convert_value(raw_value, field_info.annotation, allow_shorthand=True)  # type: ignore[arg-type]
                    data[field_name] = converted_value
                    field_results[field_name] = {"status": "success", "used_default": False}
                except ValueError as e:
                    error_msg = f"Type conversion failed for {field_name}: {e!s}"
                    has_default = _has_default(field_name, field_info, field_definitions)
                    if not has_default:
                        validation_errors.append(error_msg)
                    default_value = get_field_default(field_name, field_info, field_definitions)
                    data[field_name] = default_value
                    field_results[field_name] = {
                        "status": "conversion_failed",
                        "error": error_msg,
                        "used_default": True,
                    }

            except Exception as e:
                error_msg = f"Extraction failed for {field_name}: {e!s}"
                has_default = _has_default(field_name, field_info, field_definitions)
                if not has_default:
                    validation_errors.append(error_msg)
                default_value = get_field_default(field_name, field_info, field_definitions)
                data[field_name] = default_value
                field_results[field_name] = {"status": "extraction_failed", "error": error_msg, "used_default": True}
                accumulated_usage["field_usages"][field_name] = {  # type: ignore[index]
                    "error": str(e),
                    "status": "failed",
                    "used_default": True,
                    "default_value": default_value,
                }

        if validation_errors:
            accumulated_usage["validation_errors"] = validation_errors

        try:
            model_instance = model_cls(**data)
            model_dict = model_instance.model_dump()

            class ExtendedJSONEncoder(json.JSONEncoder):
                def default(self, obj: Any) -> Any:
                    if isinstance(obj, (datetime, date)):
                        return obj.isoformat()
                    if isinstance(obj, Decimal):
                        return str(obj)
                    return super().default(obj)

            json_string = json.dumps(model_dict, cls=ExtendedJSONEncoder)

            result = {
                "json_string": json_string,
                "json_object": json.loads(json_string),
                "usage": accumulated_usage,
                "field_results": field_results,
            }
            result["model"] = model_instance
            return type(  # type: ignore[no-any-return]
                "ExtractResult",
                (dict,),
                {"__getattr__": lambda self, key: self.get(key), "__call__": lambda self: self["model"]},
            )(result)
        except Exception as e:
            error_msg = f"Model validation error: {e!s}"
            if "validation_errors" not in accumulated_usage:
                accumulated_usage["validation_errors"] = []
            accumulated_usage["validation_errors"].append(error_msg)  # type: ignore[attr-defined]

            error_result = {
                "json_string": "{}",
                "json_object": {},
                "usage": accumulated_usage,
                "field_results": field_results,
                "error": error_msg,
            }
            return type(  # type: ignore[no-any-return]
                "ExtractResult",
                (dict,),
                {"__getattr__": lambda self, key: self.get(key), "__call__": lambda self: None},
            )(error_result)


def _has_default(field_name: str, field_info: Any, field_definitions: dict[str, Any] | None) -> bool:
    """Check whether a Pydantic field has a usable default value."""
    if field_definitions and field_name in field_definitions:
        fd = field_definitions[field_name]
        if isinstance(fd, dict) and "default" in fd:
            return True
    if hasattr(field_info, "default"):
        val = field_info.default
        if val is not ... and str(val) != "PydanticUndefined":
            return True
    return False
