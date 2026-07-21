"""Recorded-real-response replay for the LangChain framework (LAY-3614).

Drives a REAL ``langchain_openai.ChatOpenAI`` over ``httpx.MockTransport``
serving the captured OpenAI response, through a real ``prompt | llm`` chain with
the real ``LangChainCallbackHandler`` attached. This exercises the full path —
real provider response shape -> real LangChain callback objects -> real adapter
-> emitted events — which the matrix layer (fake models) and the hand-built unit
doubles never combine. Reuses the openai corpus fixture (the framework consumes
the provider's response).
"""

from __future__ import annotations

import httpx
import pytest

pytest.importorskip("langchain_openai")  # skips in envs without langchain-openai

from langchain_openai import ChatOpenAI  # noqa: E402
from langchain_core.prompts import ChatPromptTemplate  # noqa: E402

from layerlens.instrument._capture_config import CaptureConfig
from layerlens.instrument.adapters.frameworks.langchain import LangChainCallbackHandler

from .conftest import find_event, capture_framework_trace
from ..._recorded import load_recorded, mock_transport


def _chat(fixture):
    transport, _ = mock_transport(fixture)
    return ChatOpenAI(model="gpt-4o-mini", api_key="test-key", http_client=httpx.Client(transport=transport))


class TestLangChainRecorded:
    def test_chain_over_recorded_openai(self, mock_client):
        fixture = load_recorded("openai", "default")
        uploaded = capture_framework_trace(mock_client)
        handler = LangChainCallbackHandler(mock_client)

        chain = ChatPromptTemplate.from_messages([("user", "{q}")]) | _chat(fixture)
        result = chain.invoke({"q": "Reply with exactly: pong"}, config={"callbacks": [handler]})
        assert result.content == "pong"

        events = uploaded["events"]
        # The real LangChain callback objects carry the real OpenAI usage/model.
        mi = find_event(events, "model.invoke")
        assert mi["payload"]["model"] == "gpt-4o-mini-2024-07-18"
        assert mi["payload"]["tokens_prompt"] == 12
        assert mi["payload"]["tokens_completion"] == 1
        assert mi["payload"]["tokens_total"] == 13
        assert mi["payload"]["finish_reason"] == "stop"

        # LangChain emits framework-flat token accounting (no provider pricing).
        cost = find_event(events, "cost.record")
        assert cost["payload"]["framework"] == "langchain"
        assert cost["payload"]["tokens_total"] == 13

    def test_chain_over_recorded_tool_call(self, mock_client):
        # The default-fixture test above never exercises on_llm_end's tool_calls
        # extraction (langchain.py:362-380 -> 434-444): its response has no
        # tool_calls, and the hand-built unit doubles set no gen.message at all.
        # This drives a REAL tool-calling OpenAI response shape through the real
        # ChatOpenAI parser + the real handler, biting the tool.call path that has
        # ZERO coverage in any dimension.
        fixture = load_recorded("openai", "tool_call")
        uploaded = capture_framework_trace(mock_client)
        # capture_content=True so the tool-call arguments (content) are retained;
        # id/tool_name/model are structural and survive either way.
        handler = LangChainCallbackHandler(mock_client, capture_config=CaptureConfig(capture_content=True))

        chain = ChatPromptTemplate.from_messages([("user", "{q}")]) | _chat(fixture)
        chain.invoke({"q": "What's the weather in Paris?"}, config={"callbacks": [handler]})

        events = uploaded["events"]
        mi = find_event(events, "model.invoke")
        assert mi["payload"]["finish_reason"] == "tool_calls"
        assert mi["payload"]["tokens_total"] == 68
        assert mi["payload"]["model"] == "gpt-4o-mini-2024-07-18"

        # The bite: a real tool.call reconstructed from the response's tool_calls.
        # VACUITY CONTROL — the default-fixture test emits NO tool.call, so this
        # find_event raises unless the recorded tool_calls shape flowed through
        # on_llm_end's extraction (the only coverage of that path).
        tc = find_event(events, "tool.call")
        assert tc["payload"]["tool_name"] == "get_weather"
        assert tc["payload"]["id"] == "call_N0cYZsPSiRm7o3khCvuy6J3M"
        assert "Paris" in tc["payload"]["arguments"], "raw tool-call arguments (JSON string) not carried through"
        assert tc["payload"]["model"] == "gpt-4o-mini-2024-07-18"
