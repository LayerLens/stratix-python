"""Recorded-real-response replay for the AWS Strands adapter (LAY-3614, G5).

Drives a REAL ``strands.Agent`` whose ``OpenAIModel`` is backed by an
``httpx.AsyncClient`` over ``httpx.MockTransport`` serving a captured
``text/event-stream`` (SSE) ``chat.completions`` response, with the real
``StrandsAdapter`` attached as a hook provider. This exercises the full path —
real streamed provider chunks (incl. the final usage-only chunk) -> the real
Strands event loop -> per-cycle ``event_loop_metrics`` usage -> real adapter ->
emitted ``cost.record`` / ``agent.output`` — which the unit doubles (hand-built
``Mock`` cycles with fixed ``usage`` dicts) never combine with a real streamed
body.

The strong tell that the real streamed shape flowed through: ``cost.record``
carries ``tokens_prompt=12 / tokens_completion=1 / tokens_total=13`` — the exact
counts Strands lifted from the recorded stream's terminal usage chunk (which is
only present because ``OpenAIModel`` forces ``stream_options={"include_usage":
True}``). The agent's content ("pong") is the real streamed assistant delta.
"""

from __future__ import annotations

from typing import Any, Dict

import httpx
import pytest

pytest.importorskip("strands")
pytest.importorskip("openai")

from strands import Agent  # noqa: E402
from strands.models.openai import OpenAIModel  # noqa: E402

from layerlens.instrument._capture_config import CaptureConfig  # noqa: E402
from layerlens.instrument.adapters.frameworks.strands import StrandsAdapter  # noqa: E402

from .conftest import find_event, capture_framework_trace  # noqa: E402
from ..._recorded import load_recorded, mock_transport  # noqa: E402


def _model(fixture: Dict[str, Any]) -> OpenAIModel:
    transport, _ = mock_transport(fixture)
    # Strands builds its own AsyncOpenAI from client_args; the real SDK still
    # parses the recorded SSE stream over the injected MockTransport.
    return OpenAIModel(
        client_args={"api_key": "test-key", "http_client": httpx.AsyncClient(transport=transport)},
        model_id="gpt-4o-mini",
    )


class TestStrandsRecorded:
    def test_agent_over_recorded_openai_stream(self, mock_client):
        fixture = load_recorded("openai", "stream")
        uploaded = capture_framework_trace(mock_client)

        adapter = StrandsAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect()
        try:
            agent = Agent(model=_model(fixture), hooks=[adapter], name="pong_agent")
            result = agent("Reply with exactly: pong")
        finally:
            adapter.disconnect()

        # The real Strands event loop consumed the recorded SSE deltas.
        assert str(result).strip() == "pong"

        events = uploaded["events"]

        mi = find_event(events, "model.invoke")
        assert mi["payload"]["model"] == "gpt-4o-mini"
        assert mi["payload"]["agent_name"] == "pong_agent"

        # cost.record carries the per-cycle usage Strands parsed off the
        # recorded stream's terminal usage chunk (include_usage=True).
        cost = find_event(events, "cost.record")
        assert cost["payload"]["framework"] == "strands"
        assert cost["payload"]["model"] == "gpt-4o-mini"
        assert cost["payload"]["tokens_prompt"] == 12
        assert cost["payload"]["tokens_completion"] == 1
        assert cost["payload"]["tokens_total"] == 13

        # The run output echoes the real streamed assistant content.
        out = find_event(events, "agent.output")
        assert out["payload"]["agent_name"] == "pong_agent"
        assert out["payload"]["output"]["content"][0]["text"] == "pong"
