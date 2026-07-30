"""Recorded-real-response replay for the LlamaIndex framework (LAY-3614).

Drives a REAL ``llama_index.llms.openai.OpenAI`` LLM over ``httpx.MockTransport``
serving the captured OpenAI response, with the real ``LlamaIndexAdapter``
attached (span + event handlers on the root dispatcher). This exercises the full
path — real provider response shape -> real LlamaIndex ``LLMChatEndEvent`` /
``ChatResponse`` objects -> real adapter -> emitted events — which the unit
doubles (hand-built events) and the matrix layer (fake models) never combine.
Reuses the openai corpus fixture (LlamaIndex's OpenAI LLM consumes the
provider's chat.completion response).

The strong tell that the real provider shape flowed through: ``model.invoke``
reports ``gpt-4o-mini-2024-07-18`` (the model echoed in the recorded *response*
body), not the ``gpt-4o-mini`` we *requested* — the adapter read it off the real
parsed ``ChatResponse.raw["model"]``, and the usage off ``raw["usage"]``.
"""

from __future__ import annotations

import httpx
import pytest

pytest.importorskip("llama_index")  # skips in the base venv (not installed there)

from llama_index.llms.openai import OpenAI as LIOpenAI
from llama_index.core.base.llms.types import ChatMessage  # noqa: E402

from layerlens.instrument.adapters.frameworks.llamaindex import LlamaIndexAdapter

from .conftest import find_event, capture_framework_trace
from ..._recorded import load_recorded, mock_transport


def _llm(fixture):
    transport, _ = mock_transport(fixture)
    # LlamaIndex's OpenAI LLM accepts a custom ``http_client=`` (its documented
    # seam); inject the MockTransport so the real OpenAI client deserializes the
    # recorded chat.completion body. ``max_retries=0`` keeps the single replay
    # interaction deterministic.
    return LIOpenAI(
        model="gpt-4o-mini",
        api_key="test-key",
        http_client=httpx.Client(transport=transport),
        max_retries=0,
    )


class TestLlamaIndexRecorded:
    def test_chat_over_recorded_openai(self, mock_client):
        fixture = load_recorded("openai", "default")
        uploaded = capture_framework_trace(mock_client)

        adapter = LlamaIndexAdapter(mock_client)
        adapter.connect()
        llm = _llm(fixture)
        # ``.chat`` is a real LlamaIndex LLM call: the instrumentation API opens a
        # root span and fires LLMChatStart/End around it, so the per-root-span
        # collector flushes a trace on span exit.
        response = llm.chat([ChatMessage(role="user", content="Reply with exactly: pong")])
        adapter.disconnect()

        assert response.message.content == "pong"

        events = uploaded["events"]

        # The real ChatResponse carries the real OpenAI model id + usage parsed
        # off the recorded chat.completion body (raw["model"] / raw["usage"]).
        mi = find_event(events, "model.invoke")
        assert mi["payload"]["framework"] == "llamaindex"
        assert mi["payload"]["model"] == "gpt-4o-mini-2024-07-18"
        assert mi["payload"]["tokens_prompt"] == 12
        assert mi["payload"]["tokens_completion"] == 1
        assert mi["payload"]["tokens_total"] == 13

        # cost.record echoes the same real per-call token accounting.
        cost = find_event(events, "cost.record")
        assert cost["payload"]["framework"] == "llamaindex"
        assert cost["payload"]["model"] == "gpt-4o-mini-2024-07-18"
        assert cost["payload"]["tokens_prompt"] == 12
        assert cost["payload"]["tokens_completion"] == 1
        assert cost["payload"]["tokens_total"] == 13
