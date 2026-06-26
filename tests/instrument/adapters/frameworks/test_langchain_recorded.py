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
