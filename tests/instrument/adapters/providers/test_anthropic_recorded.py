"""Recorded-real-response replay for the Anthropic provider (LAY-3614).

Replays a REAL captured ``/v1/messages`` wire body through a real
``anthropic.Anthropic`` client over ``httpx.MockTransport`` and asserts the
adapter's emitted events. The fixture is the provider's raw response (recorded
upstream of the parser); the assertions are the events (downstream). Unlike the
hand-built ``conftest`` doubles — which fabricate token counts and omit real
fields — these run against the actual Anthropic Messages shape, including
``cache_creation``/``cache_read_input_tokens``, ``service_tier``, the
``content`` TextBlock / tool_use block layout, and ``stop_reason``.

See ``tests/instrument/_recorded.py`` for the corpus design + its snapshot limit.
"""

from __future__ import annotations

import httpx
import anthropic

from layerlens.instrument import trace
from layerlens.instrument.adapters.providers.anthropic import AnthropicProvider

from ...conftest import find_event
from ..._recorded import load_recorded, mock_transport


def _client(fixture):
    transport, requests = mock_transport(fixture)
    client = anthropic.Anthropic(api_key="test-key", http_client=httpx.Client(transport=transport))
    return client, requests


class TestAnthropicRecorded:
    def test_default_real_shape(self, mock_client, capture_trace):
        fixture = load_recorded("anthropic", "default")
        client, _ = _client(fixture)
        provider = AnthropicProvider()
        provider.connect(client)

        @trace(mock_client)
        def agent():
            r = client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=64,
                messages=[{"role": "user", "content": "Reply with exactly: pong"}],
            )
            return r.content[0].text

        assert agent() == "pong"

        events = capture_trace["events"]
        mi = find_event(events, "model.invoke")
        assert mi["payload"]["name"] == "anthropic.messages.create"
        # The response (dated) model wins over the requested alias.
        assert mi["payload"]["model"] == "claude-haiku-4-5-20251001"
        assert mi["payload"]["response_model"] == "claude-haiku-4-5-20251001"
        assert mi["payload"]["response_id"] == "msg_014aGrUbXt6iC6ZBoMY24njK"
        assert mi["payload"]["stop_reason"] == "end_turn"
        assert mi["payload"]["role"] == "assistant"
        # A single text block collapses to a flat {"type": "text", ...} output.
        assert mi["payload"]["output_message"] == {"type": "text", "text": "pong"}
        # Content-block accounting derived from the real content array.
        assert mi["payload"]["content_block_counts"] == {"text": 1}
        assert mi["payload"]["has_thinking"] is False

        usage = mi["payload"]["usage"]
        # input_tokens(13) + cache_read(0) folds into prompt_tokens.
        assert usage["prompt_tokens"] == 13
        assert usage["completion_tokens"] == 5
        assert usage["input_tokens"] == 13
        assert usage["output_tokens"] == 5
        # cache fields are the real Anthropic shape the fabricated doubles omit:
        assert usage["cached_tokens"] == 0
        assert usage["cache_read_input_tokens"] == 0
        assert usage["cache_creation_input_tokens"] == 0

        cost = find_event(events, "cost.record")
        assert cost["payload"]["provider"] == "anthropic"
        assert cost["payload"]["model"] == "claude-haiku-4-5-20251001"
        assert cost["payload"]["cost_usd"] > 0

        provider.disconnect()

    def test_tool_call_real_shape(self, mock_client, capture_trace):
        fixture = load_recorded("anthropic", "tool_call")
        client, _ = _client(fixture)
        provider = AnthropicProvider()
        provider.connect(client)

        @trace(mock_client)
        def agent():
            return client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=256,
                messages=[{"role": "user", "content": "What's the weather in Paris? Use the tool."}],
                tools=[
                    {
                        "name": "get_weather",
                        "description": "Get the weather for a city",
                        "input_schema": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    }
                ],
            )

        agent()
        events = capture_trace["events"]

        tool_call = find_event(events, "tool.call")
        assert tool_call["payload"]["provider"] == "anthropic"
        assert tool_call["payload"]["tool_name"] == "get_weather"
        assert tool_call["payload"]["type"] == "tool_use"
        assert tool_call["payload"]["id"] == "toolu_01BHG5dvi5Wtv2ZZQFNzh3W7"
        # tool_use input is parsed by the real SDK into a dict downstream.
        assert tool_call["payload"]["arguments"] == {"city": "Paris"}
        assert tool_call["payload"]["model"] == "claude-haiku-4-5-20251001"

        mi = find_event(events, "model.invoke")
        assert mi["payload"]["model"] == "claude-haiku-4-5-20251001"
        assert mi["payload"]["stop_reason"] == "tool_use"
        assert mi["payload"]["content_block_counts"] == {"tool_use": 1}
        assert mi["payload"]["tool_use_names"] == ["get_weather"]
        # The single non-text block keeps the structured {"type": "message"} form.
        assert mi["payload"]["output_message"] == {
            "type": "message",
            "blocks": [
                {
                    "type": "tool_use",
                    "id": "toolu_01BHG5dvi5Wtv2ZZQFNzh3W7",
                    "tool_name": "get_weather",
                    "input": {"city": "Paris"},
                }
            ],
        }

        usage = mi["payload"]["usage"]
        assert usage["prompt_tokens"] == 660
        assert usage["completion_tokens"] == 38
        assert usage["input_tokens"] == 660
        assert usage["output_tokens"] == 38

        provider.disconnect()

    def test_provenance_is_stamped(self):
        fixture = load_recorded("anthropic", "default")
        prov = fixture["provenance"]
        assert prov["provider"] == "anthropic"
        assert prov["scenario"] == "default"
        # captured_at makes staleness visible (snapshot, not freshness).
        assert prov["captured_at"] and prov["captured_at"] != "pending-creds"
