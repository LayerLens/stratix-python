"""Recorded-real-response replay for the Agno framework (LAY-3614).

Drives a REAL ``agno`` ``Agent`` whose ``OpenAIChat`` model is backed by an
``httpx.Client`` over ``httpx.MockTransport`` serving the captured OpenAI
response, with the real ``AgnoAdapter`` attached. This exercises the full path
— real provider response shape -> real agno ``RunOutput`` / ``RunMetrics``
objects -> real adapter -> emitted events — which the unit doubles (hand-built
metrics objects) never combine with a real provider body. Reuses the openai
corpus fixture (agno's OpenAIChat model consumes the provider's chat.completion
response and rolls usage onto ``RunMetrics``).

The strong tell that the real provider shape flowed through: ``RunMetrics``
reports ``input_tokens=12 / output_tokens=1 / total_tokens=13`` parsed off the
recorded ``usage{}`` block, surfaced by the adapter as flat
``tokens_prompt/completion/total``. (Agno records the *requested* model id
``gpt-4o-mini`` off the agent config, not the response-echoed
``gpt-4o-mini-2024-07-18`` — the adapter reads ``agent.model.id``.)
"""

from __future__ import annotations

from typing import Any, Dict

import httpx
import pytest

pytest.importorskip("agno")  # skips in the base venv (not installed there)

from agno.agent import Agent  # noqa: E402
from agno.models.openai import OpenAIChat

from layerlens.instrument.adapters.frameworks.agno import AgnoAdapter

from .conftest import find_event, capture_framework_trace
from ..._recorded import load_recorded, mock_transport


def _agent(fixture: Dict[str, Any]) -> Agent:
    transport, _ = mock_transport(fixture)
    # agno's OpenAIChat accepts a custom sync httpx.Client through its documented
    # ``http_client=`` seam; the real OpenAI SDK client still deserializes the
    # recorded chat.completion body. ``telemetry=False`` disables agno's own
    # outbound usage ping so the only network is the MockTransport.
    model = OpenAIChat(
        id="gpt-4o-mini",
        api_key="test-key",
        http_client=httpx.Client(transport=transport),
    )
    return Agent(model=model, name="replay_agent", telemetry=False)


class TestAgnoRecorded:
    def test_agent_over_recorded_openai(self, mock_client):
        fixture = load_recorded("openai", "default")
        uploaded = capture_framework_trace(mock_client)

        agent = _agent(fixture)
        adapter = AgnoAdapter(mock_client)
        agent = adapter.connect(target=agent)
        result = agent.run("Reply with exactly: pong")
        adapter.disconnect()

        assert result.content == "pong"

        events = uploaded["events"]

        # The real RunMetrics carries the usage parsed off the recorded
        # chat.completion body; the adapter surfaces it as framework-flat tokens.
        mi = find_event(events, "model.invoke")
        assert mi["payload"]["model"] == "gpt-4o-mini"
        assert mi["payload"]["tokens_prompt"] == 12
        assert mi["payload"]["tokens_completion"] == 1
        assert mi["payload"]["tokens_total"] == 13

        # The run output echoes the real assistant content off RunOutput.content.
        out = find_event(events, "agent.output")
        assert out["payload"]["output"] == "pong"
        assert out["payload"]["model"] == "gpt-4o-mini"

        # cost.record mirrors the same real token accounting.
        cost = find_event(events, "cost.record")
        assert cost["payload"]["framework"] == "agno"
        assert cost["payload"]["tokens_prompt"] == 12
        assert cost["payload"]["tokens_completion"] == 1
        assert cost["payload"]["tokens_total"] == 13
