"""Recorded-real-response replay for the Microsoft Agent Framework adapter (LAY-3614).

Drives a REAL ``semantic_kernel.agents.ChatCompletionAgent`` (the single-agent
chat the ``MSAgentFrameworkAdapter`` wraps) backed by an ``openai.AsyncOpenAI``
client over ``httpx.MockTransport`` serving the captured OpenAI chat-completions
response, with the real ``MSAgentFrameworkAdapter`` attached. This exercises the
full path — real provider response shape -> real semantic-kernel
``ChatMessageContent`` / ``CompletionUsage`` objects -> real adapter -> emitted
events — which neither the unit suite (synthetic ``SimpleNamespace`` messages
with hand-built ``metadata={"model":..., "usage": {dict}}``) nor the e2e suite
(real SK content objects but *still* hand-injected dict metadata) ever reach.

Two facts about the REAL SK shape this proves (and the doubles never modelled):

* ``message.metadata["usage"]`` arrives as a real ``openai.types.CompletionUsage``
  *object*, not a ``dict`` — so the adapter's ``_normalize_tokens`` must take its
  ``getattr`` branch to recover ``12 / 1 / 13``. The unit/e2e doubles always pass
  a plain dict, so the object branch was never exercised against a real provider
  shape. The token counts asserted below are read off that real object.
* Real SK ``metadata`` carries **no** ``model`` / ``model_id`` key (the model id
  lives on ``message.ai_model_id`` and ``inner_content.model``), so the adapter
  must recover it FROM THE MESSAGE — otherwise ``model.invoke`` / ``cost.record``
  ship ``model=None`` and the shared price-on-emit hook cannot compute
  ``cost_usd`` (a tokens-only ``cost.record``). The synthetic
  ``metadata={"model": "gpt-4o"}`` doubles masked this stranded-id shape
  entirely. We assert the recovered real-shape outcome — ``model ==
  "gpt-4o-mini"`` off ``ai_model_id``, with a priced ``cost_usd`` — proving the
  recovery fires on the real provider shape the doubles never reach.

Content capture is disabled via the public ``capture_config=`` constructor seam:
on a real run the yielded ``AgentResponseItem`` carries a live
``ChatHistoryAgentThread`` that ``safe_serialize`` cannot reduce to JSON, so
default-on content capture would raise inside ``agent.output`` (an adapter
robustness gap, out of scope here). Disabling content does not touch the token /
usage path this test asserts — that flows through ``message.metadata`` regardless.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from openai import AsyncOpenAI

pytest.importorskip("semantic_kernel.agents")  # skips in the base venv (not installed there)

from semantic_kernel.agents import ChatCompletionAgent  # noqa: E402
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

from layerlens.instrument._capture_config import CaptureConfig
from layerlens.instrument.adapters.frameworks.ms_agent_framework import (
    MSAgentFrameworkAdapter,
)

from .conftest import find_event, find_events, capture_framework_trace
from ..._recorded import load_recorded, mock_transport


def _agent(fixture):
    transport, _ = mock_transport(fixture)
    # OpenAIChatCompletion accepts a caller-supplied AsyncOpenAI via the public
    # ``async_client=`` seam — inject the MockTransport through it so the real
    # SK service does its real chat-completions routing + deserialization.
    async_client = AsyncOpenAI(api_key="test-key", http_client=httpx.AsyncClient(transport=transport))
    service = OpenAIChatCompletion(ai_model_id="gpt-4o-mini", async_client=async_client)
    return ChatCompletionAgent(service=service, name="replay_agent", instructions="reply")


def _drain(agent):
    async def run():
        return [item async for item in agent.invoke(messages="Reply with exactly: pong")]

    return asyncio.run(run())


class TestMSAgentFrameworkRecorded:
    def test_agent_over_recorded_openai(self, mock_client):
        fixture = load_recorded("openai", "default")
        uploaded = capture_framework_trace(mock_client)

        agent = _agent(fixture)
        # Content off: the real AgentResponseItem holds a non-serializable live
        # thread; the token/usage path under test is independent of content.
        adapter = MSAgentFrameworkAdapter(mock_client, capture_config=CaptureConfig(capture_content=False))
        adapter.instrument_chat(agent)
        result = _drain(agent)
        adapter.disconnect()

        assert len(result) == 1
        # AgentResponseItem.message is the ChatMessageContent; its text is "pong".
        assert str(result[0].message.content) == "pong"

        events = uploaded["events"]

        # The run frames the invocation with agent.input / agent.output.
        assert find_event(events, "agent.input")["payload"]["agent_name"] == "replay_agent"
        assert find_event(events, "agent.output")["payload"]["agent_name"] == "replay_agent"

        # The STRONG tell that the real provider response flowed through SK: the
        # token counts come off the real openai CompletionUsage object carried on
        # message.metadata["usage"] (an object, not a dict) — recovered via the
        # adapter's getattr branch, summed to total.
        cost = find_event(events, "cost.record")
        assert cost["payload"]["framework"] == "ms_agent_framework"
        assert cost["payload"]["tokens_prompt"] == 12
        assert cost["payload"]["tokens_completion"] == 1
        assert cost["payload"]["tokens_total"] == 13

        # Real SK strands the model id on message.ai_model_id (metadata carries
        # no model key); the adapter recovers it, so cost.record carries the real
        # model id and the shared price-on-emit hook computes a real cost_usd.
        assert cost["payload"]["model"] == "gpt-4o-mini"
        assert cost["payload"].get("cost_usd") is not None
        assert cost["payload"]["cost_usd"] > 0
        # S22/G5: token telemetry is preserved and a single model.invoke fires,
        # now carrying the recovered model id alongside the real token counts.
        invokes = find_events(events, "model.invoke")
        assert len(invokes) == 1
        assert invokes[0]["payload"].get("model") == "gpt-4o-mini"
        assert invokes[0]["payload"]["tokens_prompt"] == 12
        assert invokes[0]["payload"]["tokens_completion"] == 1
        assert invokes[0]["payload"]["tokens_total"] == 13
