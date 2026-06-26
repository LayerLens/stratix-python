"""Recorded-replay for the Google Vertex provider (LAY-3614) — SEEDED corpus.

google_vertex has no GCP credentials on any machine (the weakest double in the
tree: fully hand-built classes). This seed rebuilds a REAL proto-backed
``vertexai.generative_models.GenerationResponse`` via ``from_dict`` (needs the
SDK, not creds) and drives the adapter against it — including the real
``publishers/google/models/<id>`` resource-name form that the LAY-3615
``_strip_models_prefix`` fix must reduce to the bare model id. Flagged
``captured_at: pending-creds``; replace with a live capture when creds exist.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from layerlens.instrument import trace
from layerlens.instrument.adapters.providers.google_vertex import GoogleVertexProvider

from ...conftest import find_event
from ..._recorded import load_recorded


def _model(response, model_name):
    # Mirror the real vertexai GenerativeModel shape the adapter duck-types on:
    # a ``_model_name`` attribute + a ``generate_content`` method.
    return SimpleNamespace(_model_name=model_name, generate_content=lambda *a, **k: response)


class TestGoogleVertexRecorded:
    def test_real_proto_response(self, mock_client, capture_trace):
        gm = pytest.importorskip("vertexai.generative_models")
        fixture = load_recorded("google_vertex", "default")
        response = gm.GenerationResponse.from_dict(fixture["response"])
        model = _model(response, fixture["model_name"])

        provider = GoogleVertexProvider()
        provider.connect(model)

        @trace(mock_client)
        def agent():
            r = model.generate_content("Reply with exactly: pong", temperature=0)
            return r.candidates[0].content.parts[0].text

        assert agent() == "pong"

        mi = find_event(capture_trace["events"], "model.invoke")
        assert mi["payload"]["name"] == "google_vertex.generate_content"
        # The real 'publishers/google/models/gemini-1.5-flash-002' resource name
        # must strip to the bare id (LAY-3615), not leak the full path.
        assert mi["payload"]["model"] == "gemini-1.5-flash-002"
        assert mi["payload"]["output_message"] == {"role": "model", "content": "pong"}
        assert mi["payload"]["usage"]["prompt_tokens"] == 14
        assert mi["payload"]["usage"]["completion_tokens"] == 1
        assert mi["payload"]["usage"]["total_tokens"] == 15
        assert mi["payload"]["finish_reason"] == "STOP"
        provider.disconnect()

    def test_seed_provenance_is_flagged_pending(self):
        prov = load_recorded("google_vertex", "default")["provenance"]
        assert prov["provider"] == "google_vertex"
        assert prov["captured_at"] == "pending-creds"
