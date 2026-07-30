"""Recorded-replay for the Azure OpenAI provider (LAY-3614) — SEEDED corpus.

azure_openai has no credentials on any machine, so (unlike openai/anthropic)
this fixture is *seeded* from the real openai chat.completions shape plus
Azure's ``prompt_filter_results`` / ``content_filter_results`` members, flagged
``captured_at: pending-creds``. It still exercises the real path: a real
``openai.AzureOpenAI`` client deserializes the seeded body over
``httpx.MockTransport`` and the adapter parses it. Replace with a live capture
when creds exist (the freshness check is the future live-smoke session).
"""

from __future__ import annotations

import httpx

from openai import AzureOpenAI
from layerlens.instrument import trace
from layerlens.instrument._capture_config import CaptureConfig
from layerlens.instrument.adapters.providers.azure_openai import AzureOpenAIProvider

from ...conftest import find_event
from ..._recorded import load_recorded, mock_transport


def _client(fixture):
    transport, requests = mock_transport(fixture)
    client = AzureOpenAI(
        azure_endpoint="https://unit-test.openai.azure.com",
        api_key="test-key",
        api_version="2024-06-01",
        http_client=httpx.Client(transport=transport),
    )
    return client, requests


class TestAzureOpenAIRecorded:
    def test_default_real_shape(self, mock_client, capture_trace):
        fixture = load_recorded("azure_openai", "default")
        client, requests = _client(fixture)
        provider = AzureOpenAIProvider()
        provider.connect(client)

        @trace(mock_client, capture_config=CaptureConfig.full())
        def agent():
            r = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Reply with exactly: pong"}],
            )
            return r.choices[0].message.content

        assert agent() == "pong"
        # Real SDK routed through the Azure deployment URL.
        assert "/deployments/gpt-4o/chat/completions" in requests[0].url.path

        mi = find_event(capture_trace["events"], "model.invoke")
        assert mi["payload"]["model"] == "gpt-4o-2024-05-13"
        assert mi["payload"]["finish_reason"] == "stop"
        assert mi["payload"]["output_message"] == {"role": "assistant", "content": "pong"}
        assert mi["payload"]["usage"]["total_tokens"] == 20
        # Azure resource identity reaches the trace, query string scrubbed.
        assert mi["payload"]["azure_endpoint"].startswith("https://unit-test.openai.azure.com")
        assert "?" not in mi["payload"]["azure_endpoint"]
        provider.disconnect()

    def test_seed_provenance_is_flagged_pending(self):
        prov = load_recorded("azure_openai", "default")["provenance"]
        assert prov["provider"] == "azure_openai"
        # Honest: this is a seed, not a live capture — the gap stays visible.
        assert prov["captured_at"] == "pending-creds"
