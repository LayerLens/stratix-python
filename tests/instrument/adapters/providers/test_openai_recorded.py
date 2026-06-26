"""Recorded-real-response replay for the OpenAI provider (LAY-3614).

Replays a REAL captured ``chat.completions`` wire body through a real
``openai.OpenAI`` client over ``httpx.MockTransport`` and asserts the adapter's
emitted events. The fixture is the provider's raw response (recorded upstream of
the parser); the assertions are the events (downstream). Unlike the hand-built
``conftest.make_openai_response`` doubles — which fabricate token counts and omit
real fields — these run against the actual OpenAI shape, including
``prompt_tokens_details`` / ``completion_tokens_details`` / ``service_tier``.

See ``tests/instrument/_recorded.py`` for the corpus design + its snapshot limit.
"""

from __future__ import annotations

import httpx

import openai
from layerlens.instrument import trace
from layerlens.instrument.adapters.providers.openai import OpenAIProvider

from ...conftest import find_event
from ..._recorded import load_recorded, mock_transport


def _client(fixture):
    transport, requests = mock_transport(fixture)
    client = openai.OpenAI(api_key="test-key", http_client=httpx.Client(transport=transport))
    return client, requests


class TestOpenAIRecorded:
    def test_default_real_shape(self, mock_client, capture_trace):
        fixture = load_recorded("openai", "default")
        client, _ = _client(fixture)
        provider = OpenAIProvider()
        provider.connect(client)

        @trace(mock_client)
        def agent():
            r = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Reply with exactly: pong"}],
            )
            return r.choices[0].message.content

        assert agent() == "pong"

        events = capture_trace["events"]
        mi = find_event(events, "model.invoke")
        assert mi["payload"]["name"] == "openai.chat.completions.create"
        # The response (dated) model wins over the requested alias.
        assert mi["payload"]["model"] == "gpt-4o-mini-2024-07-18"
        assert mi["payload"]["response_model"] == "gpt-4o-mini-2024-07-18"
        assert mi["payload"]["finish_reason"] == "stop"
        assert mi["payload"]["system_fingerprint"] == "fp_2ff2473d75"
        # Real-shape fields the fabricated doubles never carried:
        assert mi["payload"]["service_tier"] == "default"
        assert mi["payload"]["output_message"] == {"role": "assistant", "content": "pong"}
        usage = mi["payload"]["usage"]
        assert usage["prompt_tokens"] == 12
        assert usage["completion_tokens"] == 1
        assert usage["total_tokens"] == 13
        # prompt_tokens_details / completion_tokens_details are real OpenAI shape:
        assert usage["cached_tokens"] == 0
        assert usage["reasoning_tokens"] == 0

        cost = find_event(events, "cost.record")
        assert cost["payload"]["provider"] == "openai"
        assert cost["payload"]["model"] == "gpt-4o-mini-2024-07-18"
        assert cost["payload"]["total_tokens"] == 13
        assert cost["payload"]["cost_usd"] > 0

        provider.disconnect()

    def test_tool_call_real_shape(self, mock_client, capture_trace):
        fixture = load_recorded("openai", "tool_call")
        client, _ = _client(fixture)
        provider = OpenAIProvider()
        provider.connect(client)

        @trace(mock_client)
        def agent():
            return client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "What's the weather in Paris? Use the tool."}],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                        },
                    }
                ],
            )

        agent()
        events = capture_trace["events"]
        tool_call = find_event(events, "tool.call")
        assert tool_call["payload"]["tool_name"] == "get_weather"
        # arguments arrive as a JSON string on the wire and are parsed downstream.
        assert tool_call["payload"]["arguments"] == {"city": "Paris"}
        assert tool_call["payload"]["provider"] == "openai"

        mi = find_event(events, "model.invoke")
        assert mi["payload"]["finish_reason"] == "tool_calls"
        assert mi["payload"]["usage"]["prompt_tokens"] == 54
        assert mi["payload"]["usage"]["completion_tokens"] == 14
        provider.disconnect()

    def test_provenance_is_stamped(self):
        fixture = load_recorded("openai", "default")
        prov = fixture["provenance"]
        assert prov["provider"] == "openai"
        assert prov["scenario"] == "default"
        # captured_at makes staleness visible (snapshot, not freshness).
        assert prov["captured_at"] and prov["captured_at"] != "pending-creds"
