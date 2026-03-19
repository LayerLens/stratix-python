"""
Tests for MCP App invocation events.
"""

import pytest

from layerlens.instrument.adapters.protocols.mcp.mcp_app_handler import (
    normalize_component_type,
    normalize_interaction_result,
    hash_parameters,
    hash_result,
)


class TestMCPAppHelpers:
    def test_normalize_component_type(self):
        assert normalize_component_type("form") == "form"
        assert normalize_component_type("FORM") == "form"
        assert normalize_component_type("confirmation") == "confirmation"
        assert normalize_component_type("picker") == "picker"
        assert normalize_component_type("unknown_type") == "custom"

    def test_normalize_interaction_result(self):
        assert normalize_interaction_result("submitted") == "submitted"
        assert normalize_interaction_result("cancelled") == "cancelled"
        assert normalize_interaction_result("timeout") == "timeout"
        assert normalize_interaction_result("unknown") == "submitted"

    def test_hash_parameters(self):
        h = hash_parameters({"key": "value"})
        assert h.startswith("sha256:")

    def test_hash_result(self):
        h = hash_result({"data": True})
        assert h is not None
        assert h.startswith("sha256:")

    def test_hash_result_none(self):
        assert hash_result(None) is None


class TestMCPAppEvents:
    def test_on_mcp_app_invocation(self, mcp_adapter, mock_stratix):
        mcp_adapter.on_mcp_app_invocation(
            app_id="app-form-1",
            component_type="form",
            interaction_result="submitted",
            parameters={"fields": ["name", "email"]},
            result={"name": "John", "email": "john@example.com"},
        )
        assert len(mock_stratix.events) == 1
        event = mock_stratix.events[0][0]
        assert event.event_type == "protocol.mcp_app.invocation"
        assert event.app_id == "app-form-1"
        assert event.component_type == "form"
        assert event.interaction_result == "submitted"
        assert event.parameters_hash.startswith("sha256:")
        assert event.result_hash.startswith("sha256:")

    def test_on_mcp_app_cancelled(self, mcp_adapter, mock_stratix):
        mcp_adapter.on_mcp_app_invocation(
            app_id="app-confirm-1",
            component_type="confirmation",
            interaction_result="cancelled",
        )
        event = mock_stratix.events[0][0]
        assert event.interaction_result == "cancelled"
        assert event.result_hash is None

    def test_auth_event(self, mcp_adapter, mock_stratix):
        mcp_adapter.on_auth_event(
            auth_type="oauth2",
            success=True,
            details={"provider": "github"},
        )
        assert len(mock_stratix.events) == 1
        event = mock_stratix.events[0][0]
        assert event.event_type == "environment.config"
