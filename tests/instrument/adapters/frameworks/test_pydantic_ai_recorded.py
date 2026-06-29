"""Recorded-real-response replay for the PydanticAI framework (LAY-3614).

Drives a REAL ``pydantic_ai.Agent`` whose ``OpenAIModel`` is backed by an
``httpx.AsyncClient`` over ``httpx.MockTransport`` serving the captured OpenAI
response, with the real ``PydanticAIAdapter`` attached. This exercises the full
path — real provider response shape -> real pydantic_ai ``ModelResponse`` /
``RunUsage`` objects -> real adapter -> emitted events — which the unit suite
(``TestModel``, a fake model) never reaches. Reuses the openai corpus fixture
(pydantic_ai's OpenAI model consumes the provider's chat.completion response).

The strong tell that the real provider shape flowed through: ``model.invoke``
reports ``gpt-4o-mini-2024-07-18`` (the model echoed in the recorded *response*
body), not the ``gpt-4o-mini`` we *requested* — the adapter read it off the real
parsed ``ModelResponse``, not off our config.
"""

from __future__ import annotations

import httpx
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from layerlens.instrument._capture_config import CaptureConfig
from layerlens.instrument.adapters.frameworks.pydantic_ai import PydanticAIAdapter

from .conftest import find_event, capture_framework_trace
from ..._recorded import load_recorded, mock_transport


def _agent(fixture):
    transport, _ = mock_transport(fixture)
    # pydantic_ai's OpenAI model client is async-only; inject the MockTransport
    # through the provider's documented ``http_client=`` seam.
    provider = OpenAIProvider(api_key="test-key", http_client=httpx.AsyncClient(transport=transport))
    model = OpenAIModel("gpt-4o-mini", provider=provider)
    return Agent(model, name="replay_agent")


class TestPydanticAIRecorded:
    def test_agent_over_recorded_openai(self, mock_client):
        fixture = load_recorded("openai", "default")
        uploaded = capture_framework_trace(mock_client)

        agent = _agent(fixture)
        adapter = PydanticAIAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect(target=agent)
        result = agent.run_sync("Reply with exactly: pong")
        adapter.disconnect()

        assert result.output == "pong"

        events = uploaded["events"]

        # The real per-request ModelResponse carries the real OpenAI model id +
        # usage parsed off the recorded chat.completion body.
        mi = find_event(events, "model.invoke")
        assert mi["payload"]["model"] == "gpt-4o-mini-2024-07-18"
        assert mi["payload"]["tokens_prompt"] == 12
        assert mi["payload"]["tokens_completion"] == 1
        assert mi["payload"]["tokens_total"] == 13

        # The run aggregates the same real usage onto agent.output.
        out = find_event(events, "agent.output")
        assert out["payload"]["status"] == "ok"
        assert out["payload"]["output"] == "pong"
        assert out["payload"]["tokens_prompt"] == 12
        assert out["payload"]["tokens_completion"] == 1
        assert out["payload"]["tokens_total"] == 13
        assert out["payload"]["model_requests"] == 1

        # cost.record echoes the real run-level token accounting.
        cost = find_event(events, "cost.record")
        assert cost["payload"]["framework"] == "pydantic-ai"
        assert cost["payload"]["tokens_total"] == 13
