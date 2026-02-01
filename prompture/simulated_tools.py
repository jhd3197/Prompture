"""Prompt-based tool calling for drivers without native tool use support.

When a driver lacks ``supports_tool_use`` the conversation classes can
fall back to *simulated* tool calling: the available tools are described
in the system prompt, the model is asked to respond with a structured
JSON object (either a tool call or a final answer), and Prompture
parses + dispatches accordingly.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .tools import clean_json_text
from .tools_schema import ToolRegistry

logger = logging.getLogger("prompture.simulated_tools")


def build_tool_prompt(tools: ToolRegistry) -> str:
    """Build a plain-text prompt section describing all registered tools.

    The returned string should be appended to the system prompt so the
    model knows which tools are available and how to call them.
    """
    lines = [
        "You have access to the following tools:",
        "",
        tools.to_prompt_format(),
        "",
        "To use a tool, respond with ONLY a JSON object in this exact format:",
        '{"type": "tool_call", "name": "<tool_name>", "arguments": {<args>}}',
        "",
        "When you have the final answer (after using tools or if no tool is needed), "
        "respond with ONLY a JSON object in this format:",
        '{"type": "final_answer", "content": "<your answer>"}',
        "",
        "IMPORTANT: Your entire response must be a single JSON object. "
        "Do not include any other text, markdown, or explanation outside the JSON.",
    ]
    return "\n".join(lines)


def parse_simulated_response(text: str, tools: ToolRegistry) -> dict[str, Any]:
    """Parse the model's response into a tool call or final answer dict.

    Returns one of:
    - ``{"type": "tool_call", "name": str, "arguments": dict}``
    - ``{"type": "final_answer", "content": str}``
    """
    cleaned = clean_json_text(text).strip()

    # Try JSON parse
    try:
        obj = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        # Non-JSON text → treat as final answer
        logger.debug("Response is not valid JSON, treating as final answer")
        return {"type": "final_answer", "content": text.strip()}

    if not isinstance(obj, dict):
        return {"type": "final_answer", "content": text.strip()}

    # Explicit type discriminator
    resp_type = obj.get("type")

    if resp_type == "tool_call":
        return {
            "type": "tool_call",
            "name": obj.get("name", ""),
            "arguments": obj.get("arguments", {}),
        }

    if resp_type == "final_answer":
        return {
            "type": "final_answer",
            "content": obj.get("content", ""),
        }

    # Infer type from keys when "type" is missing
    if "name" in obj and "arguments" in obj:
        logger.debug("Inferred tool_call from keys (no 'type' field)")
        return {
            "type": "tool_call",
            "name": obj["name"],
            "arguments": obj.get("arguments", {}),
        }

    if "content" in obj:
        logger.debug("Inferred final_answer from keys (no 'type' field)")
        return {
            "type": "final_answer",
            "content": obj["content"],
        }

    # Unrecognised JSON structure → final answer with the raw text
    return {"type": "final_answer", "content": text.strip()}


def format_tool_result(tool_name: str, result: Any) -> str:
    """Format a tool execution result as a user message for the next round."""
    if isinstance(result, str):
        result_str = result
    else:
        try:
            result_str = json.dumps(result)
        except (TypeError, ValueError):
            result_str = str(result)

    return (
        f"Tool '{tool_name}' returned:\n{result_str}\n\n"
        "Continue using the JSON format. Either call another tool or provide your final answer."
    )
