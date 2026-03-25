from __future__ import annotations

import json
from unittest.mock import Mock

from layerlens.instrument import span, trace
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
        """Attestation chain should include all spans in the tree."""
        client, uploaded = _make_client()

        @trace(client)
        def my_agent(query: str):
            with span("step-1", kind="tool") as s:
                s.output = "result-1"
            with span("step-2", kind="llm") as s:
                s.output = "result-2"
            return "done"

        my_agent("test")

        att = uploaded["data"][0]["attestation"]
        chain_events = att["chain"]["events"]
        # Root span + 2 child spans = 3 events in the chain
        assert len(chain_events) == 3

    def test_chain_events_are_linked(self):
        """Verify the chain in the uploaded payload is valid."""
        client, uploaded = _make_client()

        @trace(client)
        def my_agent(query: str):
            with span("step-1") as s:
                s.output = "r1"
            with span("step-2") as s:
                s.output = "r2"
            return "done"

        my_agent("test")

        chain_events = uploaded["data"][0]["attestation"]["chain"]["events"]
        # Reconstruct envelopes and verify chain integrity
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
            with span("step-1") as s:
                s.output = "ok"
            raise ValueError("boom")

        try:
            failing_agent()
        except ValueError:
            pass

        payload = uploaded["data"][0]
        assert "attestation" in payload
        assert payload["attestation"]["root_hash"].startswith("sha256:")

    def test_modifying_output_breaks_chain(self):
        """Changing what the agent said must invalidate the attestation."""
        client, uploaded = _make_client()

        @trace(client)
        def my_agent(query: str):
            with span("llm-call", kind="llm") as s:
                s.output = "the real answer"
            return "done"

        my_agent("test")

        att = uploaded["data"][0]["attestation"]
        envelopes = [
            AttestationEnvelope(
                hash=e["hash"],
                scope=HashScope(e["scope"]),
                previous_hash=e["previous_hash"],
            )
            for e in att["chain"]["events"]
        ]

        # Build the original span dicts that were hashed (root + child)
        payload = uploaded["data"][0]
        original_spans = []
        for s in [payload] + payload.get("children", []):
            d = {k: v for k, v in s.items() if k not in ("children", "attestation")}
            original_spans.append(d)

        # Verify clean data passes
        clean = detect_tampering(envelopes, original_spans)
        assert not clean.tampered

        # Tamper: change the LLM output
        tampered_spans = [dict(d) for d in original_spans]
        tampered_spans[1] = {**tampered_spans[1], "output": "a forged answer"}

        tampered = detect_tampering(envelopes, tampered_spans)
        assert tampered.tampered
        assert 1 in tampered.modified_indices
