"""
Tests for MCP elicitation request/response event pair.
"""

import pytest

from layerlens.instrument.adapters.protocols.mcp.elicitation import ElicitationTracker


class TestElicitationTracker:
    def setup_method(self):
        self.tracker = ElicitationTracker()

    def test_start_request(self):
        eid = self.tracker.start_request(
            server_name="mcp-server",
            schema={"type": "object"},
            title="Confirm deletion",
        )
        assert eid is not None
        assert self.tracker.is_active(eid)
        assert self.tracker.active_count == 1

    def test_complete_response(self):
        eid = self.tracker.start_request(server_name="mcp-server")
        latency = self.tracker.complete_response(eid, "submit", response={"confirmed": True})
        assert latency is not None
        assert latency >= 0
        assert not self.tracker.is_active(eid)

    def test_cancel_response(self):
        eid = self.tracker.start_request(server_name="mcp-server")
        self.tracker.complete_response(eid, "cancel")
        assert self.tracker.active_count == 0

    def test_unknown_elicitation(self):
        latency = self.tracker.complete_response("unknown-id", "submit")
        assert latency is None

    def test_custom_elicitation_id(self):
        eid = self.tracker.start_request(
            server_name="mcp-server",
            elicitation_id="custom-id-123",
        )
        assert eid == "custom-id-123"
        assert self.tracker.is_active("custom-id-123")

    def test_hash_response(self):
        h = self.tracker.hash_response({"name": "John"})
        assert h.startswith("sha256:")

    def test_hash_schema(self):
        h = self.tracker.hash_schema({"type": "object", "required": ["name"]})
        assert h.startswith("sha256:")


class TestElicitationEvents:
    def test_elicitation_request_event(self, mcp_adapter, mock_stratix):
        mcp_adapter.on_elicitation_request(
            elicitation_id="elic-001",
            server_name="mcp-server",
            schema={"type": "object", "$id": "confirm-schema"},
            title="Confirm deletion",
        )
        assert len(mock_stratix.events) == 1
        event = mock_stratix.events[0][0]
        assert event.event_type == "protocol.elicitation.request"
        assert event.elicitation_id == "elic-001"
        assert event.request_title == "Confirm deletion"
        assert event.schema_ref == "confirm-schema"

    def test_elicitation_response_event(self, mcp_adapter, mock_stratix):
        mcp_adapter.on_elicitation_response(
            elicitation_id="elic-001",
            action="submit",
            response={"confirmed": True},
            latency_ms=1500.0,
        )
        assert len(mock_stratix.events) == 1
        event = mock_stratix.events[0][0]
        assert event.event_type == "protocol.elicitation.response"
        assert event.action == "submit"
        assert event.latency_ms == 1500.0
        assert event.response_hash.startswith("sha256:")

    def test_elicitation_pair(self, mcp_adapter, mock_stratix):
        mcp_adapter.on_elicitation_request(
            elicitation_id="elic-002",
            server_name="mcp-server",
        )
        mcp_adapter.on_elicitation_response(
            elicitation_id="elic-002",
            action="cancel",
        )
        assert len(mock_stratix.events) == 2
        assert mock_stratix.events[0][0].event_type == "protocol.elicitation.request"
        assert mock_stratix.events[1][0].event_type == "protocol.elicitation.response"
