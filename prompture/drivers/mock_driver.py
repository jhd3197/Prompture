from ..driver import Driver
from typing import Dict, Any
import time

class MockDriver(Driver):
    def generate(self, prompt: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Enhanced MockDriver with token usage and cost tracking.

        Uses prompt length and response complexity to generate realistic token counts.
        """
        options = options or {}

        # Simple response logic
        if "Juan is 28" in prompt:
            text = '{"name":"Juan","age":28,"location":"Miami","interests":["basketball","coding"]}'
        else:
            text = '{"name":"Unknown","age":null,"location":null,"interests":[]}'

        # Generate realistic token usage based on prompt and response length
        prompt_tokens = len(prompt.split()) * 2  # Rough token estimation
        completion_tokens = len(text.split()) * 3  # JSON is more efficiently tokenized

        meta = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "cost": 0.0,  # Mock driver is free
            "raw_response": {
                "mock_response": True,
                "processing_time": time.time(),
                "prompt_length": len(prompt),
                "response_length": len(text)
            }
        }

        return {"text": text, "meta": meta}