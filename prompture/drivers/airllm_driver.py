import logging
from typing import Any, Optional

from .base import Driver

logger = logging.getLogger(__name__)


class AirLLMDriver(Driver):
    """Driver for AirLLM — run large models (70B+) on consumer GPUs via
    layer-by-layer memory management.

    The ``airllm`` package is a lazy dependency: it is imported on first
    ``generate()`` call so the rest of Prompture works without it installed.
    """

    MODEL_PRICING = {"default": {"prompt": 0.0, "completion": 0.0}}

    def __init__(self, model: str = "meta-llama/Llama-2-7b-hf", compression: Optional[str] = None):
        """
        Args:
            model: HuggingFace repo ID (e.g. ``"meta-llama/Llama-2-70b-hf"``).
            compression: Optional quantization mode — ``"4bit"`` or ``"8bit"``.
        """
        self.model = model
        self.compression = compression
        self.options: dict[str, Any] = {}
        self._llm = None
        self._tokenizer = None

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------
    def _ensure_loaded(self):
        """Load the AirLLM model and tokenizer on first use."""
        if self._llm is not None:
            return

        try:
            from airllm import AutoModel
        except ImportError:
            raise ImportError(
                "The 'airllm' package is required for the AirLLM driver. Install it with: pip install prompture[airllm]"
            ) from None

        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError(
                "The 'transformers' package is required for the AirLLM driver. "
                "Install it with: pip install transformers"
            ) from None

        logger.info(f"Loading AirLLM model: {self.model} (compression={self.compression})")

        load_kwargs: dict[str, Any] = {}
        if self.compression:
            load_kwargs["compression"] = self.compression

        self._llm = AutoModel.from_pretrained(self.model, **load_kwargs)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model)
        logger.info("AirLLM model loaded successfully")

    # ------------------------------------------------------------------
    # Driver interface
    # ------------------------------------------------------------------
    def generate(self, prompt: str, options: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        self._ensure_loaded()

        merged_options = self.options.copy()
        if options:
            merged_options.update(options)

        max_new_tokens = merged_options.get("max_new_tokens", 256)

        # Tokenize
        input_ids = self._tokenizer(prompt, return_tensors="pt").input_ids

        prompt_tokens = input_ids.shape[1]

        logger.debug(f"AirLLM generating with max_new_tokens={max_new_tokens}, prompt_tokens={prompt_tokens}")

        # Generate
        output_ids = self._llm.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
        )

        # Decode only the newly generated tokens (strip the prompt prefix)
        new_tokens = output_ids[0, prompt_tokens:]
        completion_tokens = len(new_tokens)
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)

        total_tokens = prompt_tokens + completion_tokens

        meta = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost": 0.0,
            "raw_response": {
                "model": self.model,
                "compression": self.compression,
                "max_new_tokens": max_new_tokens,
            },
            "model_name": self.model,
        }

        return {"text": text, "meta": meta}
