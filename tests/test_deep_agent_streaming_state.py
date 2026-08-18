"""DeepAgent's streaming entry points must carry deep state.

``DeepAgent`` overrode ``run()`` to return a :class:`DeepAgentResult`, but
``iter()``, ``run_stream()`` and ``run_live()`` were inherited from
:class:`Agent` verbatim.  All three therefore returned a plain
:class:`AgentResult`, silently dropping todos, files, sub-agent calls and
summary events — the entire reason for using a DeepAgent.

The original report named only ``run_live``; all three are affected.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from prompture.agents.deep_agent import DeepAgent
from prompture.agents.deep_state import DeepAgentResult
from prompture.drivers.base import Driver

STREAMING_METHODS = ["iter", "run_stream", "run_live"]


class _Scripted(Driver):
    supports_tool_use = False

    def __init__(self) -> None:
        self.model = "scripted/deep"

    def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        return {
            "text": json.dumps({"type": "final_answer", "content": "all done"}),
            "meta": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
                "cost": 0.0,
                "raw_response": {},
            },
        }


def _agent() -> DeepAgent:
    return DeepAgent(driver=_Scripted(), initial_files={"notes.txt": "hello"})


def _drain(stream: Any) -> Any:
    for _ in stream:
        pass
    return stream.result


@pytest.mark.parametrize("method", STREAMING_METHODS)
def test_streaming_returns_a_deep_result(method):
    agent = _agent()

    result = _drain(getattr(agent, method)("do the thing"))

    assert isinstance(result, DeepAgentResult), (
        f"{method}() returned {type(result).__name__}, losing deep state"
    )


@pytest.mark.parametrize("method", STREAMING_METHODS)
def test_streaming_preserves_files(method):
    agent = _agent()

    result = _drain(getattr(agent, method)("do the thing"))

    assert result.files == {"notes.txt": "hello"}


@pytest.mark.parametrize("method", STREAMING_METHODS)
def test_streaming_exposes_the_deep_state_snapshot(method):
    agent = _agent()

    result = _drain(getattr(agent, method)("do the thing"))

    assert result.deep_state is not None
    assert result.todos == []
    assert result.sub_agent_calls == []


@pytest.mark.parametrize("method", STREAMING_METHODS)
def test_streaming_still_yields_events(method):
    """Upgrading the return value must not swallow the stream itself."""
    agent = _agent()
    stream = getattr(agent, method)("do the thing")

    yielded = list(stream)

    assert stream.result is not None
    # iter() yields steps; the others yield events. Either way something came out.
    assert isinstance(yielded, list)


@pytest.mark.parametrize("method", STREAMING_METHODS)
def test_streaming_agrees_with_run(method):
    """The streaming result must match what run() reports."""
    streamed = _drain(getattr(_agent(), method)("do the thing"))
    ran = _agent().run("do the thing")

    assert type(streamed) is type(ran)
    assert streamed.files == ran.files
    assert streamed.output_text == ran.output_text


def test_result_is_none_before_the_stream_is_consumed():
    """The wrapper contract is unchanged: no result until iteration ends."""
    stream = _agent().run_live("do the thing")

    assert stream.result is None
