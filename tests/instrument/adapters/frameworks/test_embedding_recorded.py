"""Recorded-real-response replay for the embedding adapter (LAY-3614, G5).

Drives the REAL ``openai`` client's ``embeddings.create`` over
``httpx.MockTransport`` serving a captured ``POST /v1/embeddings`` response, with
the real ``EmbeddingAdapter`` wrapping the call. This exercises the full path —
real provider response shape -> the real OpenAI SDK's own ``CreateEmbedding``
deserialization (incl. its base64-embedding decode) -> real adapter parser ->
emitted ``embedding.create`` event — which the unit doubles (hand-built
``SimpleNamespace`` results) never combine with a real provider body.

The strong tells that the real provider shape flowed through: ``dimensions``
equals the real vector length the SDK decoded off the recorded response
(``text-embedding-3-small`` -> 1536), and ``total_tokens`` is the real
``usage.total_tokens`` the SDK lifted from the recorded ``usage`` block.
"""

from __future__ import annotations

from typing import Any, Dict

import httpx
import pytest

pytest.importorskip("openai")

import openai  # noqa: E402
from layerlens.instrument import trace_context  # noqa: E402
from layerlens.instrument.adapters.frameworks.embedding import EmbeddingAdapter  # noqa: E402

from .conftest import find_event, capture_framework_trace  # noqa: E402
from ..._recorded import load_recorded, mock_transport  # noqa: E402


def _client(fixture: Dict[str, Any]) -> Any:
    transport, _ = mock_transport(fixture)
    # The real OpenAI SDK client still deserializes the recorded embeddings body
    # (including its default base64->float decode) over the MockTransport.
    return openai.OpenAI(api_key="test-key", http_client=httpx.Client(transport=transport))


class TestEmbeddingRecorded:
    def test_openai_embeddings_over_recorded(self, mock_client):
        fixture = load_recorded("openai", "embeddings")
        uploaded = capture_framework_trace(mock_client)

        client = _client(fixture)
        adapter = EmbeddingAdapter(mock_client)
        adapter.connect(target=client)  # auto-wraps client.embeddings.create
        try:
            with trace_context(mock_client):
                result = client.embeddings.create(model="text-embedding-3-small", input="hello world")
        finally:
            adapter.disconnect()

        # The real SDK decoded the recorded response into a real embedding vector.
        assert len(result.data[0].embedding) == 1536
        # 2 is the real ``usage.total_tokens`` captured for input "hello world".
        assert result.usage.total_tokens == 2

        evt = find_event(uploaded["events"], "embedding.create")
        assert evt["payload"]["provider"] == "openai"
        assert evt["payload"]["model"] == "text-embedding-3-small"
        assert evt["payload"]["batch_size"] == 1
        # dimensions is the REAL decoded vector length off the recorded body.
        assert evt["payload"]["dimensions"] == 1536
        # total_tokens is the REAL usage the SDK parsed from the recorded response.
        assert evt["payload"]["total_tokens"] == 2
        assert evt["payload"]["latency_ms"] >= 0
