"""
Tests for AG-UI event type mapping and stream event processing.
"""

import pytest

from layerlens.instrument.adapters.protocols.agui.event_mapper import (
    map_agui_to_stratix,
    get_all_agui_event_types,
    AGUIEventType,
)


class TestAGUIEventMapper:
    def test_lifecycle_events_map_to_state_change(self):
        for event_type in ("RUN_STARTED", "RUN_FINISHED", "RUN_ERROR"):
            mapping = map_agui_to_stratix(event_type)
            assert mapping["stratix_event"] == "agent.state.change"
            assert mapping["category"] == "lifecycle"

    def test_text_events_map_to_stream(self):
        for event_type in ("TEXT_MESSAGE_START", "TEXT_MESSAGE_CONTENT", "TEXT_MESSAGE_END"):
            mapping = map_agui_to_stratix(event_type)
            assert mapping["stratix_event"] == "protocol.stream.event"
            assert mapping["category"] == "text"

    def test_tool_start_maps_to_tool_call(self):
        mapping = map_agui_to_stratix("TOOL_CALL_START")
        assert mapping["stratix_event"] == "tool.call"

    def test_tool_result_maps_to_tool_call(self):
        mapping = map_agui_to_stratix("TOOL_CALL_RESULT")
        assert mapping["stratix_event"] == "tool.call"

    def test_state_events_map_to_state_change(self):
        for event_type in ("STATE_SNAPSHOT", "STATE_DELTA", "MESSAGES_SNAPSHOT"):
            mapping = map_agui_to_stratix(event_type)
            assert mapping["stratix_event"] == "agent.state.change"

    def test_special_events_map_to_stream(self):
        for event_type in ("STEP_STARTED", "STEP_FINISHED", "RAW"):
            mapping = map_agui_to_stratix(event_type)
            assert mapping["stratix_event"] == "protocol.stream.event"

    def test_unknown_event_type(self):
        mapping = map_agui_to_stratix("UNKNOWN_EVENT")
        assert mapping["stratix_event"] == "protocol.stream.event"
        assert mapping["category"] == "unknown"

    def test_all_event_types_count(self):
        types = get_all_agui_event_types()
        assert len(types) == 16

    def test_enum_values(self):
        assert AGUIEventType.RUN_STARTED.value == "RUN_STARTED"
        assert AGUIEventType.TEXT_MESSAGE_CONTENT.value == "TEXT_MESSAGE_CONTENT"


class TestAGUIStreamProcessing:
    def test_process_lifecycle_events(self, agui_adapter, mock_stratix):
        agui_adapter.on_agui_event("RUN_STARTED", {"runId": "run-1"})
        assert len(mock_stratix.events) >= 1

    def test_process_text_message_sequence(self, agui_adapter, mock_stratix):
        agui_adapter.on_agui_event("TEXT_MESSAGE_START", {"messageId": "m1"})
        agui_adapter.on_agui_event("TEXT_MESSAGE_CONTENT", {"content": "Hello"})
        agui_adapter.on_agui_event("TEXT_MESSAGE_END", {"messageId": "m1"})
        assert len(mock_stratix.events) >= 3

    def test_l6b_gating_suppresses_content_events(self, agui_adapter_no_streams, mock_stratix):
        agui_adapter_no_streams.on_agui_event("TEXT_MESSAGE_START", {"messageId": "m1"})
        initial_count = len(mock_stratix.events)
        agui_adapter_no_streams.on_agui_event("TEXT_MESSAGE_CONTENT", {"content": "Hello"})
        # Content events should be suppressed when l6b_protocol_streams = False
        assert len(mock_stratix.events) == initial_count

    def test_tool_call_events(self, agui_adapter, mock_stratix):
        agui_adapter.on_agui_event("TOOL_CALL_START", {
            "tool_name": "search",
            "args": {"query": "test"},
        })
        # Should emit both protocol.stream.event and tool.call
        assert len(mock_stratix.events) >= 1
