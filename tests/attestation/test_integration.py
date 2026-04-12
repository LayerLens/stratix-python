from __future__ import annotations

import json
from unittest.mock import Mock

from layerlens.instrument import emit, span, trace
from layerlens.attestation import verify_chain, detect_tampering
from layerlens.attestation._envelope import HashScope, AttestationEnvelope


def _make_client():
    """Create a mock client that captures the uploaded trace JSON."""
    client = Mock()
    client.traces = Mock()
    uploaded = {}

    def capture(path):
        with open(path) as f:
            uploaded["data"] = json.load(f)

    client.traces.upload = Mock(side_effect=capture)
    return client, uploaded


class TestTraceAttestation:
    def test_trace_includes_attestation(self):
        """@trace should include attestation data in the upload."""
        client, uploaded = _make_client()

        @trace(client)
        def my_agent(query: str):
            return f"answer to {query}"

        my_agent("hello")

        payload = uploaded["data"][0]
        assert "attestation" in payload
        att = payload["attestation"]
        assert "chain" in att
        assert "root_hash" in att
        assert att["root_hash"].startswith("sha256:")
        assert att["schema_version"] == "1.0"

    def test_trace_with_child_spans(self):
        """Attestation chain should include events from all spans."""
        client, uploaded = _make_client()

        @trace(client)
        def my_agent(query: str):
            with span("step-1"):
                emit("tool.call", {"name": "search", "input": "q"})
            with span("step-2"):
                emit("model.invoke", {"name": "gpt-4"})
            return "done"

        my_agent("test")

        att = uploaded["data"][0]["attestation"]
        chain_events = att["chain"]["events"]
        # agent.input + tool.call + model.invoke + agent.output = 4 events
        assert len(chain_events) == 4

    def test_chain_events_are_linked(self):
        """Verify the chain in the uploaded payload is valid."""
        client, uploaded = _make_client()

        @trace(client)
        def my_agent(query: str):
            with span("step-1"):
                emit("tool.call", {"name": "search", "input": "q"})
            with span("step-2"):
                emit("tool.result", {"output": "result"})
            return "done"

        my_agent("test")

        chain_events = uploaded["data"][0]["attestation"]["chain"]["events"]
        envelopes = [
            AttestationEnvelope(
                hash=e["hash"],
                scope=HashScope(e["scope"]),
                previous_hash=e["previous_hash"],
            )
            for e in chain_events
        ]
        result = verify_chain(envelopes)
        assert result.valid

    def test_trace_error_still_has_attestation(self):
        """Even when the traced function raises, attestation should be present."""
        client, uploaded = _make_client()

        @trace(client)
        def failing_agent():
            with span("step-1"):
                emit("tool.call", {"name": "search", "input": "q"})
            raise ValueError("boom")

        try:
            failing_agent()
        except ValueError:
            pass

        payload = uploaded["data"][0]
        assert "attestation" in payload
        assert payload["attestation"]["root_hash"].startswith("sha256:")

    def test_modifying_event_breaks_chain(self):
        """Changing an event payload must invalidate the attestation."""
        client, uploaded = _make_client()

        @trace(client)
        def my_agent(query: str):
            with span("llm-call"):
                emit("model.invoke", {"name": "gpt-4", "output_message": "the real answer"})
            return "done"

        my_agent("test")

        payload = uploaded["data"][0]
        att = payload["attestation"]
        envelopes = [
            AttestationEnvelope(
                hash=e["hash"],
                scope=HashScope(e["scope"]),
                previous_hash=e["previous_hash"],
            )
            for e in att["chain"]["events"]
        ]

        # The events that were hashed
        original_events = payload["events"]

        # Verify clean data passes
        clean = detect_tampering(envelopes, original_events)
        assert not clean.tampered

        # Tamper: change the model output in the second event
        tampered_events = [dict(e) for e in original_events]
        tampered_events[1] = {**tampered_events[1], "payload": {"name": "gpt-4", "output_message": "a forged answer"}}

        tampered = detect_tampering(envelopes, tampered_events)
        assert tampered.tampered
        assert 1 in tampered.modified_indices
