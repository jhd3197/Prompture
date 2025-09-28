import os
import logging
import google.generativeai as genai
from typing import Any, Dict
from ..driver import Driver

logger = logging.getLogger(__name__)


class GoogleDriver(Driver):
    """Driver for Google's Generative AI API (Gemini)."""

    # Based on current Gemini pricing (as of 2025)
    # Source: https://cloud.google.com/vertex-ai/pricing#gemini_models
    MODEL_PRICING = {
        "gemini-1.5-pro": {
            "prompt": 0.00025,  # $0.25/1M chars input
            "completion": 0.0005  # $0.50/1M chars output
        },
        "gemini-1.5-pro-vision": {
            "prompt": 0.00025,  # $0.25/1M chars input
            "completion": 0.0005  # $0.50/1M chars output
        },
        "gemini-2.5-pro": {
            "prompt": 0.0004,  # $0.40/1M chars input
            "completion": 0.0008  # $0.80/1M chars output
        },
        "gemini-2.5-flash": {
            "prompt": 0.0004,  # $0.40/1M chars input
            "completion": 0.0008  # $0.80/1M chars output
        },
        "gemini-2.5-flash-lite": {
            "prompt": 0.0002,  # $0.20/1M chars input
            "completion": 0.0004  # $0.40/1M chars output
        },
         "gemini-2.0-flash": {
            "prompt": 0.0004,  # $0.40/1M chars input
            "completion": 0.0008  # $0.80/1M chars output
        },
        "gemini-2.0-flash-lite": {
            "prompt": 0.0002,  # $0.20/1M chars input
            "completion": 0.0004  # $0.40/1M chars output
        },
    }

    def __init__(self, api_key: str | None = None, model: str = "gemini-1.5-pro"):
        """Initialize the Google Driver.

        Args:
            api_key: Google API key. If not provided, will look for GOOGLE_API_KEY env var
            model: Model to use. Defaults to "gemini-1.5-pro"
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key not found. Set GOOGLE_API_KEY env var or pass api_key to constructor")

        self.model = model
        if model not in self.MODEL_PRICING:
            raise ValueError(f"Unsupported model: {model}. Must be one of: {list(self.MODEL_PRICING.keys())}")

        # Configure google.generativeai
        genai.configure(api_key=self.api_key)
        self.options: Dict[str, Any] = {}
        
        # Validate connection and model availability
        self._validate_connection()

    def _validate_connection(self):
        """Validate connection to Google's API and model availability."""
        try:
            # List models to validate API key and connectivity
            genai.list_models()
            logger.debug("Connection to Google API validated successfully")
        except Exception as e:
            logger.warning(f"Could not validate connection to Google API: {e}")
            raise

    def generate(self, prompt: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate text using Google's Generative AI.

        Args:
            prompt: The input prompt
            options: Additional options to pass to the model

        Returns:
            Dict containing generated text and metadata
        """
        merged_options = self.options.copy()
        if options:
            merged_options.update(options)

        try:
            logger.debug(f"Initializing {self.model} for generation")
            model = genai.GenerativeModel(self.model)

            # Generate response
            logger.debug(f"Generating with prompt: {prompt}")
            response = model.generate_content(prompt)
            
            if not response.text:
                raise ValueError("Empty response from model")

            # Calculate token usage and cost
            # Note: Using character count as proxy since Google charges per character
            prompt_chars = len(prompt)
            completion_chars = len(response.text)
            
            # Calculate costs
            model_pricing = self.MODEL_PRICING[self.model]
            prompt_cost = (prompt_chars / 1_000_000) * model_pricing["prompt"]
            completion_cost = (completion_chars / 1_000_000) * model_pricing["completion"]
            total_cost = prompt_cost + completion_cost

            meta = {
                "prompt_chars": prompt_chars,
                "completion_chars": completion_chars,
                "total_chars": prompt_chars + completion_chars,
                "cost": total_cost,
                "raw_response": response.prompt_feedback,
                "model_name": self.model,
            }

            return {"text": response.text, "meta": meta}

        except Exception as e:
            logger.error(f"Google API request failed: {e}")
            raise RuntimeError(f"Google API request failed: {e}")