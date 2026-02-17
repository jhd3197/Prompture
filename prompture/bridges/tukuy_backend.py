"""Bridge a Prompture AsyncDriver to Tukuy's LLMBackend protocol.

Usage::

    from prompture.bridges import TukuyLLMBackend, create_tukuy_backend

    # Option 1: from an existing driver
    driver = get_async_driver_for_model("openai/gpt-4o")
    backend = TukuyLLMBackend(driver)

    # Option 2: convenience factory
    backend = create_tukuy_backend("openai/gpt-4o")

    # Inject into Tukuy SkillContext
    context.config["llm_backend"] = backend
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import AsyncIterator, Callable
from typing import Any

from ..drivers.async_base import AsyncDriver
from ..exceptions import ConfigurationError, DriverError

logger = logging.getLogger("prompture.bridges.tukuy")


class TukuyLLMBackend:
    """Bridges a Prompture AsyncDriver to Tukuy's LLMBackend protocol.

    Implements the ``complete()`` method that Tukuy's ``@instruction``
    decorator expects on the ``llm_backend`` config key.
    """

    def __init__(
        self,
        driver: AsyncDriver,
        *,
        default_model: str | None = None,
        default_temperature: float | None = None,
        default_max_tokens: int | None = None,
        on_complete: Callable[..., Any] | None = None,
    ) -> None:
        if driver is None:
            raise ConfigurationError(
                "TukuyLLMBackend requires a non-None AsyncDriver. "
                "Use create_tukuy_backend() or pass a configured driver."
            )
        self._driver = driver
        self._default_model = default_model
        self._default_temperature = default_temperature
        self._default_max_tokens = default_max_tokens
        self._on_complete = on_complete

    # ------------------------------------------------------------------
    # Tukuy LLMBackend protocol
    # ------------------------------------------------------------------

    async def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Implement Tukuy's LLMBackend.complete protocol.

        Returns::

            {
                "text": str,
                "meta": {
                    "prompt_tokens": int,
                    "completion_tokens": int,
                    "cost": float,
                    "model": str,
                }
            }
        """
        # Build messages
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # Build options
        options: dict[str, Any] = {}

        temp = temperature if temperature is not None else self._default_temperature
        if temp is not None:
            options["temperature"] = temp

        tokens = max_tokens if max_tokens is not None else self._default_max_tokens
        if tokens is not None:
            options["max_tokens"] = tokens

        # Handle structured output
        if json_schema is not None:
            if self._driver.supports_json_schema:
                options["json_mode"] = True
                options["json_schema"] = json_schema
            elif self._driver.supports_json_mode:
                options["json_mode"] = True
                # Inject schema into messages so the model knows the target structure
                messages = AsyncDriver._inject_schema_into_messages(messages, json_schema)

        # Call the driver
        try:
            if self._driver.supports_messages:
                result = await self._driver.generate_messages(messages, options)
            else:
                # Fallback: combine system + prompt into single string
                full_prompt = f"{system}\n\n{prompt}" if system else prompt
                result = await self._driver.generate(full_prompt, options)
        except Exception as exc:
            raise DriverError(
                f"LLM completion failed: {exc}"
            ) from exc

        # Normalize response to Tukuy protocol
        meta = result.get("meta", {})
        response: dict[str, Any] = {
            "text": result.get("text", ""),
            "meta": {
                "prompt_tokens": meta.get("prompt_tokens", 0),
                "completion_tokens": meta.get("completion_tokens", 0),
                "cost": meta.get("cost", 0.0),
                "model": meta.get("model_name", self._default_model or "unknown"),
            },
        }

        # Fire callback if registered (for logging / credit tracking)
        if self._on_complete is not None:
            try:
                cb_result = self._on_complete(response)
                if inspect.isawaitable(cb_result):
                    await cb_result
            except Exception:
                logger.debug("on_complete callback raised; ignoring", exc_info=True)

        return response

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    async def stream(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream LLM response chunks.

        Yields dicts with the shape:

        * ``{"type": "delta", "text": "chunk..."}`` for each chunk
        * ``{"type": "done", "text": "full_text", "meta": {...}}`` as the
          final item

        Falls back to a single ``"done"`` chunk via :meth:`complete` when
        the underlying driver does not support streaming.
        """
        # Fallback: driver does not support streaming
        if not self._driver.supports_streaming:
            result = await self.complete(
                prompt,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
                json_schema=json_schema,
            )
            yield {"type": "done", "text": result["text"], "meta": result["meta"]}
            return

        # Build messages (same logic as complete())
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # Build options
        options: dict[str, Any] = {}

        temp = temperature if temperature is not None else self._default_temperature
        if temp is not None:
            options["temperature"] = temp

        tokens = max_tokens if max_tokens is not None else self._default_max_tokens
        if tokens is not None:
            options["max_tokens"] = tokens

        if json_schema is not None:
            if self._driver.supports_json_schema:
                options["json_mode"] = True
                options["json_schema"] = json_schema
            elif self._driver.supports_json_mode:
                options["json_mode"] = True
                messages = AsyncDriver._inject_schema_into_messages(messages, json_schema)

        # Stream from driver
        full_text = ""
        meta: dict[str, Any] = {}
        try:
            async for chunk in self._driver.generate_messages_stream(messages, options):
                if chunk.get("type") == "delta":
                    full_text += chunk.get("text", "")
                    yield {"type": "delta", "text": chunk.get("text", "")}
                elif chunk.get("type") == "done":
                    full_text = chunk.get("text", full_text)
                    raw_meta = chunk.get("meta", {})
                    meta = {
                        "prompt_tokens": raw_meta.get("prompt_tokens", 0),
                        "completion_tokens": raw_meta.get("completion_tokens", 0),
                        "cost": raw_meta.get("cost", 0.0),
                        "model": raw_meta.get("model_name", self._default_model or "unknown"),
                    }
        except Exception as exc:
            raise DriverError(f"LLM streaming failed: {exc}") from exc

        # Fire on_complete callback
        response: dict[str, Any] = {"text": full_text, "meta": meta}
        if self._on_complete is not None:
            try:
                cb_result = self._on_complete(response)
                if inspect.isawaitable(cb_result):
                    await cb_result
            except Exception:
                logger.debug("on_complete callback raised; ignoring", exc_info=True)

        yield {"type": "done", "text": full_text, "meta": meta}

    # ------------------------------------------------------------------
    # Model switching
    # ------------------------------------------------------------------

    def with_model(self, model_hint: str) -> TukuyLLMBackend:
        """Create a new backend instance targeting a different model.

        Used when an instruction's ``model_hint`` differs from the bot's
        default.  Returns a new :class:`TukuyLLMBackend` with a driver
        for the hinted model.
        """
        from ..drivers import get_async_driver_for_model

        new_driver = get_async_driver_for_model(model_hint)
        return TukuyLLMBackend(
            new_driver,
            default_model=model_hint,
            default_temperature=self._default_temperature,
            default_max_tokens=self._default_max_tokens,
            on_complete=self._on_complete,
        )

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        model = self._default_model or "unknown"
        driver_cls = type(self._driver).__name__
        return f"TukuyLLMBackend(driver={driver_cls}, model={model!r})"


def create_tukuy_backend(
    model: str,
    *,
    env: Any | None = None,
    on_complete: Callable[..., Any] | None = None,
    default_temperature: float | None = None,
    default_max_tokens: int | None = None,
) -> TukuyLLMBackend:
    """Create a :class:`TukuyLLMBackend` from a model string.

    Args:
        model: Model string like ``"openai/gpt-4o"`` or
            ``"claude/claude-sonnet-4-5-20250929"``.
        env: Optional :class:`~prompture.infra.provider_env.ProviderEnvironment`
            for isolated API keys (per-bot).
        on_complete: Callback fired after each LLM completion.
        default_temperature: Default temperature for completions.
        default_max_tokens: Default max tokens for completions.

    Returns:
        A configured :class:`TukuyLLMBackend`.
    """
    from ..drivers import get_async_driver_for_model

    driver = get_async_driver_for_model(model, env=env)
    return TukuyLLMBackend(
        driver,
        default_model=model,
        default_temperature=default_temperature,
        default_max_tokens=default_max_tokens,
        on_complete=on_complete,
    )
