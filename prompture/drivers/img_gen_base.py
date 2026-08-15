"""Base class for image generation drivers."""

from __future__ import annotations

import logging
import time
from typing import Any

from ..infra.callbacks import DriverCallbacks

logger = logging.getLogger("prompture.img_gen_driver")


class ImageGenDriver:
    """Adapter base for image generation. Implement ``generate_image(prompt, options)``.

    Response contract::

        {
            "images": list[ImageContent],
            "meta": {
                "image_count": int,
                "size": str,
                "revised_prompt": str | None,
                "cost": float,
                "model_name": str,
                "raw_response": dict,
            },
        }
    """

    supports_multiple: bool = True
    supports_size_variants: bool = True
    supported_sizes: list[str] = []
    max_images: int = 10

    # Editing capabilities (default: not supported). Drivers that implement
    # editing flip these so callers can feature-detect before offering UI.
    supports_edit: bool = False
    supports_variation: bool = False

    # Whether the driver honors a native ``negative_prompt`` option. Drivers
    # that don't can still receive negative guidance folded into the prompt
    # (see ``prompture.imaging.compose_negative_prompt``).
    supports_negative_prompt: bool = False

    callbacks: DriverCallbacks | None = None

    def generate_image(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        """Generate image(s) from a text prompt.

        Args:
            prompt: Text description of the desired image.
            options: Provider-specific options (size, quality, style, n, etc.).

        Returns:
            Dict with ``images`` and ``meta`` keys.
        """
        raise NotImplementedError

    def edit_image(
        self,
        image: bytes,
        prompt: str,
        options: dict[str, Any],
        *,
        mask: bytes | None = None,
    ) -> dict[str, Any]:
        """Edit an existing image with a text instruction (image-to-image).

        Args:
            image: Source image bytes (PNG/JPEG).
            prompt: Instruction describing the desired change.
            options: Provider-specific options (size, n, model, etc.).
            mask: Optional PNG mask; transparent pixels mark the region to edit
                (inpainting). When omitted the whole image is editable.

        Returns:
            Dict with ``images`` and ``meta`` keys (same contract as
            :meth:`generate_image`).
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support image editing"
        )

    def create_variation(self, image: bytes, options: dict[str, Any]) -> dict[str, Any]:
        """Produce variation(s) of an existing image (no prompt).

        Args:
            image: Source image bytes (PNG).
            options: Provider-specific options (size, n, model, etc.).

        Returns:
            Dict with ``images`` and ``meta`` keys.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support variations"
        )

    # ------------------------------------------------------------------
    # Hook-aware wrapper
    # ------------------------------------------------------------------

    def generate_image_with_hooks(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        """Wrap :meth:`generate_image` with on_request / on_response / on_error callbacks."""
        driver_name = getattr(self, "model", self.__class__.__name__)
        self._fire_callback(
            "on_request",
            {"prompt_length": len(prompt), "options": options, "driver": driver_name},
        )
        t0 = time.perf_counter()
        try:
            resp = self.generate_image(prompt, options)
        except Exception as exc:
            self._fire_callback(
                "on_error",
                {"error": exc, "prompt_length": len(prompt), "options": options, "driver": driver_name},
            )
            raise
        elapsed_ms = (time.perf_counter() - t0) * 1000
        meta = resp.get("meta", {})
        logger.debug(
            "[img_gen] generate driver=%s images=%d cost=%.6f elapsed=%.0fms",
            driver_name,
            meta.get("image_count", 0),
            meta.get("cost", 0.0),
            elapsed_ms,
        )
        self._fire_callback(
            "on_response",
            {
                "image_count": meta.get("image_count", 0),
                "meta": meta,
                "driver": driver_name,
                "elapsed_ms": elapsed_ms,
            },
        )
        return resp

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fire_callback(self, event: str, payload: dict[str, Any]) -> None:
        """Invoke a single callback, swallowing and logging any exception."""
        if self.callbacks is None:
            return
        cb = getattr(self.callbacks, event, None)
        if cb is None:
            return
        try:
            cb(payload)
        except Exception:
            logger.exception("Callback %s raised an exception", event)
