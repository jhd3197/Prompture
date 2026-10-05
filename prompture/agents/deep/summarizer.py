"""Automatic context summarization for :class:`DeepAgent`.

Implements a middleware-style object that hooks into
``Conversation.before_turn``. When the most recent driver call exceeded
``threshold_tokens`` of prompt context, the middleware:

1. Splits ``conversation._messages`` into [old..cutoff] + [cutoff..new]
   where the tail contains the most recent ``keep_last_n`` messages.
2. Asks a summariser driver to compress the old portion into prose.
3. Replaces the old portion with a single synthetic user message
   carrying the summary, preserving the system prompt and the tail.
4. Records a :class:`SummaryEvent` on the :class:`DeepAgentState`.

The summariser uses a separate driver call (same model by default) and
is wrapped in best-effort error handling — if summarisation fails the
agent loop continues unchanged.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Literal

from ..._internal.json_encoder import PromptureJSONEncoder
from ...drivers import get_driver_for_model
from ...drivers.base import Driver
from ..deep_prompts import SUMMARIZER_SYSTEM_PROMPT, SUMMARY_PRELUDE
from ..deep_state import DeepAgentState, SummaryEvent

if TYPE_CHECKING:
    from ..conversation import Conversation

logger = logging.getLogger("prompture.agents.deep.summarizer")

#: Upper bound for the ``"auto"`` threshold.  Large windows still summarise
#: here, so a long run does not pay for a nearly full window on every call.
DEFAULT_SUMMARIZE_CEILING = 80_000
_SUMMARIZE_FLOOR = 2_000
_WINDOW_RATIO = 0.75


def resolve_summarize_threshold(
    model: str,
    *,
    max_output_tokens: int | None = None,
    ceiling: int = DEFAULT_SUMMARIZE_CEILING,
) -> int:
    """Prompt size at which to summarise, from the model's context window.

    The smallest of ``ceiling``, three quarters of the window, and the window minus the
    output reservation (``max_output_tokens`` when given, else the model's own
    output limit).  An unknown window, or an output reservation that leaves
    no room, falls back to ``ceiling`` rather than summarising every turn.
    """
    if "/" not in model:
        return ceiling
    provider, model_id = model.split("/", 1)
    try:
        from ...infra.model_rates import get_model_capabilities

        caps = get_model_capabilities(provider, model_id)
    except Exception:  # pragma: no cover - capability lookup is best-effort
        logger.debug("summarizer: capability lookup failed for %s", model, exc_info=True)
        caps = None
    window = getattr(caps, "context_window", None)
    if not window:
        return ceiling
    candidates = [ceiling, int(window * _WINDOW_RATIO)]
    reserve = max_output_tokens if max_output_tokens is not None else getattr(caps, "max_output_tokens", None)
    if reserve is not None and 0 <= reserve < window:
        candidates.append(window - reserve)
    return max(_SUMMARIZE_FLOOR, min(candidates))


def max_output_option(options: dict[str, Any] | None) -> int | None:
    """The output-token reservation set in driver options, if any."""
    for key in ("max_completion_tokens", "max_tokens", "max_output_tokens"):
        value = (options or {}).get(key)
        if isinstance(value, int) and value >= 0:
            return value
    return None


class _Threshold:
    """Lazily resolved summarisation threshold shared by both middlewares."""

    def _init_threshold(
        self, threshold_tokens: int | Literal["auto"], model: str, max_output_tokens: int | None
    ) -> None:
        self._threshold: int | None = None if threshold_tokens == "auto" else int(threshold_tokens)
        self._threshold_model = model
        self._threshold_max_output = max_output_tokens

    @property
    def threshold_tokens(self) -> int:
        if self._threshold is None:
            self._threshold = resolve_summarize_threshold(
                self._threshold_model, max_output_tokens=self._threshold_max_output
            )
            logger.debug("summarizer: auto threshold for %r is %d tokens", self._threshold_model, self._threshold)
        return self._threshold

    @threshold_tokens.setter
    def threshold_tokens(self, value: int) -> None:
        self._threshold = int(value)


def _stringify_message(msg: dict[str, Any]) -> str:
    """Render a single conversation message as a labeled block of text."""
    role = msg.get("role", "?")
    content = msg.get("content", "")
    if isinstance(content, list):
        # Content blocks (e.g., multimodal). Flatten text parts.
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    parts.append(block["text"])
                elif block.get("type") == "image":
                    parts.append("[image]")
        content = "\n".join(parts)
    elif not isinstance(content, str):
        content = json.dumps(content, cls=PromptureJSONEncoder)

    tool_calls = msg.get("tool_calls")
    if tool_calls:
        tc_lines: list[str] = []
        for tc in tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name", tc.get("name", "?"))
            args = fn.get("arguments", tc.get("arguments", ""))
            tc_lines.append(f"[tool_call:{name}({args})]")
        content = (content + "\n" if content else "") + "\n".join(tc_lines)

    if role == "tool":
        tc_id = msg.get("tool_call_id", "?")
        return f"[tool_result id={tc_id}]\n{content}"

    return f"[{role}]\n{content}"


class SummarizationMiddleware(_Threshold):
    """Hooked into ``Conversation._before_turn``.

    Attributes:
        threshold_tokens: Fire when the last prompt's token count exceeds
            this value. Counted using the last driver response's
            ``prompt_tokens``. ``"auto"`` derives it from the model's
            context window on first use (see
            :func:`resolve_summarize_threshold`).
        keep_last_n: Number of most-recent messages preserved verbatim.
        state: Shared :class:`DeepAgentState` for recording events.
        summariser: Driver used to perform the summarisation call.
    """

    def __init__(
        self,
        threshold_tokens: int | Literal["auto"],
        keep_last_n: int,
        state: DeepAgentState,
        summariser: Driver | str,
        *,
        model: str = "",
        max_output_tokens: int | None = None,
    ) -> None:
        self._init_threshold(threshold_tokens, model, max_output_tokens)
        self.keep_last_n = max(2, int(keep_last_n))
        self.state = state
        if isinstance(summariser, str):
            self._summariser: Driver = get_driver_for_model(summariser)  # type: ignore[assignment]
        else:
            self._summariser = summariser
        # Re-entrancy guard. Without this the summariser's own driver
        # call would recursively trigger the hook.
        self._summarising = False

    # ------------------------------------------------------------------
    # Hook
    # ------------------------------------------------------------------

    def __call__(self, conversation: Conversation, last_prompt_tokens: int) -> None:
        """Conversation-side before_turn callback signature."""
        if self._summarising:
            return
        if last_prompt_tokens < self.threshold_tokens:
            return
        # Need at least keep_last_n + 1 messages to make summarisation
        # worthwhile (we must have something to discard).
        if len(conversation._messages) <= self.keep_last_n + 1:
            logger.debug(
                "summarizer: skipping — only %d messages, keep_last_n=%d",
                len(conversation._messages),
                self.keep_last_n,
            )
            return

        self._summarising = True
        try:
            self._run_summarization(conversation)
        except Exception:
            logger.exception("summarizer: failed; leaving conversation untouched")
        finally:
            self._summarising = False

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _run_summarization(self, conversation: Conversation) -> None:
        msgs = conversation._messages
        # Partition: oldest .. cutoff (to summarise) | tail (verbatim)
        # We never summarise tool-role messages without their preceding
        # assistant tool_call message — to keep things simple, we cut on
        # message-index boundaries but ensure the tail starts on a
        # role-safe boundary (not a "tool" message).
        cutoff = len(msgs) - self.keep_last_n
        while cutoff < len(msgs) and msgs[cutoff].get("role") == "tool":
            # Sliding the cutoff forward avoids orphaning a tool result
            # from its assistant call.
            cutoff += 1
        if cutoff <= 0 or cutoff >= len(msgs):
            return

        head = msgs[:cutoff]
        tail = msgs[cutoff:]

        rendered = "\n\n".join(_stringify_message(m) for m in head)
        summary_text, usage = self._call_summariser(rendered)
        if not summary_text:
            logger.debug("summarizer: empty summary, leaving messages unchanged")
            return

        synthetic = {
            "role": "user",
            "content": SUMMARY_PRELUDE + summary_text,
        }

        # Mutate in place so the conversation's identity is preserved.
        conversation._messages.clear()
        conversation._messages.append(synthetic)
        conversation._messages.extend(tail)

        # Reset last_prompt_tokens so we don't immediately trigger again.
        conversation._last_prompt_tokens = 0

        self.state.summary_events.append(
            SummaryEvent(
                triggered_at_message_index=len(msgs),
                summary_text=summary_text,
                summarized_message_count=len(head),
                summarizer_usage=usage,
            )
        )
        logger.info(
            "summarizer: collapsed %d messages into %d chars of summary",
            len(head),
            len(summary_text),
        )

    def _call_summariser(self, rendered_history: str) -> tuple[str, dict[str, Any]]:
        """Invoke the summariser driver and return (summary_text, usage)."""
        prompt = f"Conversation fragment to summarise:\n\n=====\n{rendered_history}\n=====\n\nProduce the summary now."
        messages = [
            {"role": "system", "content": SUMMARIZER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        try:
            resp = self._summariser.generate_messages(messages, {})
        except AttributeError:
            # Some drivers expose only generate(prompt, options).
            resp = self._summariser.generate(
                f"{SUMMARIZER_SYSTEM_PROMPT}\n\n{prompt}",
                {},
            )
        text = (resp.get("text") if isinstance(resp, dict) else "") or ""
        meta = resp.get("meta", {}) if isinstance(resp, dict) else {}
        return text.strip(), dict(meta)


class AsyncSummarizationMiddleware(_Threshold):
    """Async counterpart of :class:`SummarizationMiddleware`."""

    def __init__(
        self,
        threshold_tokens: int | Literal["auto"],
        keep_last_n: int,
        state: DeepAgentState,
        summariser: Any,  # AsyncDriver or str
        *,
        model: str = "",
        max_output_tokens: int | None = None,
    ) -> None:
        self._init_threshold(threshold_tokens, model, max_output_tokens)
        self.keep_last_n = max(2, int(keep_last_n))
        self.state = state
        if isinstance(summariser, str):
            from ...drivers.async_registry import get_async_driver_for_model

            self._summariser = get_async_driver_for_model(summariser)
        else:
            self._summariser = summariser
        self._summarising = False

    async def __call__(self, conversation: Any, last_prompt_tokens: int) -> None:
        if self._summarising:
            return
        if last_prompt_tokens < self.threshold_tokens:
            return
        if len(conversation._messages) <= self.keep_last_n + 1:
            return
        self._summarising = True
        try:
            await self._run_summarization(conversation)
        except Exception:
            logger.exception("summarizer: failed; leaving conversation untouched")
        finally:
            self._summarising = False

    async def _run_summarization(self, conversation: Any) -> None:
        msgs = conversation._messages
        cutoff = len(msgs) - self.keep_last_n
        while cutoff < len(msgs) and msgs[cutoff].get("role") == "tool":
            cutoff += 1
        if cutoff <= 0 or cutoff >= len(msgs):
            return

        head = msgs[:cutoff]
        tail = msgs[cutoff:]
        rendered = "\n\n".join(_stringify_message(m) for m in head)
        summary_text, usage = await self._call_summariser(rendered)
        if not summary_text:
            return
        synthetic = {"role": "user", "content": SUMMARY_PRELUDE + summary_text}
        conversation._messages.clear()
        conversation._messages.append(synthetic)
        conversation._messages.extend(tail)
        conversation._last_prompt_tokens = 0
        self.state.summary_events.append(
            SummaryEvent(
                triggered_at_message_index=len(msgs),
                summary_text=summary_text,
                summarized_message_count=len(head),
                summarizer_usage=usage,
            )
        )

    async def _call_summariser(self, rendered_history: str) -> tuple[str, dict[str, Any]]:
        prompt = f"Conversation fragment to summarise:\n\n=====\n{rendered_history}\n=====\n\nProduce the summary now."
        messages = [
            {"role": "system", "content": SUMMARIZER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        try:
            resp = await self._summariser.generate_messages(messages, {})
        except AttributeError:
            resp = await self._summariser.generate(
                f"{SUMMARIZER_SYSTEM_PROMPT}\n\n{prompt}",
                {},
            )
        text = (resp.get("text") if isinstance(resp, dict) else "") or ""
        meta = resp.get("meta", {}) if isinstance(resp, dict) else {}
        return text.strip(), dict(meta)
