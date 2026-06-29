"""Recorded-real-response replay for the SmolAgents framework (LAY-3614).

Drives a REAL ``smolagents.CodeAgent`` whose ``OpenAIServerModel`` is backed by a
real ``openai.OpenAI`` client over ``httpx.MockTransport`` serving the captured
OpenAI response, with the real ``SmolAgentsAdapter`` attached. This exercises the
full path — real provider chat.completion response -> real smolagents
``generate()`` / ``ChatMessage.token_usage`` / ``ActionStep`` -> real adapter ->
emitted events — which the unit doubles (hand-built step objects) never combine.
Reuses the openai corpus fixture (smolagents' OpenAI model consumes the
provider's chat.completion response).

The strong tell that the real provider shape flowed through: ``model.invoke``
reports ``tokens_prompt/completion/total = 12/1/13``, which smolagents reads off
``response.usage`` of the recorded body into a real ``TokenUsage`` and the adapter
normalizes off the real ``ActionStep.token_usage`` — not off any hand-built
double. (Unlike pydantic_ai, smolagents surfaces the *requested* ``model_id`` on
the step, so the model assertion is the configured ``gpt-4o-mini``; the tokens are
the response-derived strong values.)

The recorded ``content`` is plain prose ("pong"), not a smolagents code blob, so
the ``CodeAgent`` cannot parse a ``final_answer(...)`` call from it. With
``max_steps=1`` the agent deterministically runs one action step (firing the
adapter's ``ActionStep`` callback with the real token usage) then returns the raw
content as the final answer — a fixed, network-free, deterministic run.
"""

from __future__ import annotations

import httpx
import pytest

import openai

pytest.importorskip("smolagents")  # skips in the base venv (not installed there)

from smolagents import CodeAgent, OpenAIServerModel  # noqa: E402

from layerlens.instrument._capture_config import CaptureConfig
from layerlens.instrument.adapters.frameworks.smolagents import SmolAgentsAdapter

from .conftest import find_event, capture_framework_trace
from ..._recorded import load_recorded, mock_transport


def _agent(fixture):
    transport, _ = mock_transport(fixture)
    # smolagents' ApiModel accepts a pre-built ``client=`` (its public seam,
    # bypassing OpenAIServerModel.create_client); inject the real openai client
    # bound to the MockTransport so the framework does its real deserialization.
    client = openai.OpenAI(api_key="test-key", http_client=httpx.Client(transport=transport))
    model = OpenAIServerModel(model_id="gpt-4o-mini", client=client)
    return CodeAgent(tools=[], model=model, max_steps=1, verbosity_level=0, name="replay_agent")


class TestSmolAgentsRecorded:
    def test_agent_over_recorded_openai(self, mock_client):
        fixture = load_recorded("openai", "default")
        uploaded = capture_framework_trace(mock_client)

        agent = _agent(fixture)
        adapter = SmolAgentsAdapter(mock_client, capture_config=CaptureConfig.full())
        agent = adapter.connect(target=agent)
        result = agent.run("Reply with exactly: pong")
        adapter.disconnect()

        assert result == "pong"

        events = uploaded["events"]

        # The real ActionStep carries the real OpenAI usage parsed off the
        # recorded chat.completion body into smolagents' TokenUsage; the adapter
        # normalizes input_tokens/output_tokens -> tokens_prompt/completion.
        mi = find_event(events, "model.invoke")
        assert mi["payload"]["model"] == "gpt-4o-mini"
        assert mi["payload"]["tokens_prompt"] == 12
        assert mi["payload"]["tokens_completion"] == 1
        assert mi["payload"]["tokens_total"] == 13

        # smolagents emits framework-flat token accounting (no provider pricing).
        cost = find_event(events, "cost.record")
        assert cost["payload"]["framework"] == "smolagents"
        assert cost["payload"]["tokens_prompt"] == 12
        assert cost["payload"]["tokens_completion"] == 1
        assert cost["payload"]["tokens_total"] == 13

        # The run flushed the outer lifecycle boundary with the raw final answer.
        out = find_event(events, "agent.output")
        assert out["payload"]["framework"] == "smolagents"
        assert out["payload"]["output"] == "pong"
