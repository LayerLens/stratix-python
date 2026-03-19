"""
Tests for AG-UI CaptureConfig l6b_protocol_streams gating.
"""

import pytest

from layerlens.instrument.adapters._capture import CaptureConfig, ALWAYS_ENABLED_EVENT_TYPES


class TestL6CaptureConfig:
    def test_l6_fields_exist(self):
        config = CaptureConfig()
        assert hasattr(config, "l6a_protocol_discovery")
        assert hasattr(config, "l6b_protocol_streams")
        assert hasattr(config, "l6c_protocol_lifecycle")

    def test_l6_defaults_true(self):
        config = CaptureConfig()
        assert config.l6a_protocol_discovery is True
        assert config.l6b_protocol_streams is True
        assert config.l6c_protocol_lifecycle is True

    def test_minimal_preset_streams_off(self):
        config = CaptureConfig.minimal()
        assert config.l6b_protocol_streams is False
        assert config.l6a_protocol_discovery is True
        assert config.l6c_protocol_lifecycle is True

    def test_standard_preset_all_on(self):
        config = CaptureConfig.standard()
        assert config.l6a_protocol_discovery is True
        assert config.l6b_protocol_streams is True
        assert config.l6c_protocol_lifecycle is True

    def test_full_preset_all_on(self):
        config = CaptureConfig.full()
        assert config.l6a_protocol_discovery is True
        assert config.l6b_protocol_streams is True
        assert config.l6c_protocol_lifecycle is True

    def test_is_layer_enabled_protocol_events(self):
        config = CaptureConfig()
        assert config.is_layer_enabled("protocol.agent_card") is True
        assert config.is_layer_enabled("protocol.stream.event") is True

    def test_is_layer_enabled_with_l6b_off(self):
        config = CaptureConfig(l6b_protocol_streams=False)
        assert config.is_layer_enabled("protocol.stream.event") is False
        assert config.is_layer_enabled("protocol.agent_card") is True  # L6a still on

    def test_short_label_mapping(self):
        config = CaptureConfig()
        assert config.is_layer_enabled("L6a") is True
        assert config.is_layer_enabled("L6b") is True
        assert config.is_layer_enabled("L6c") is True

    def test_protocol_task_events_always_enabled(self):
        assert "protocol.task.submitted" in ALWAYS_ENABLED_EVENT_TYPES
        assert "protocol.task.completed" in ALWAYS_ENABLED_EVENT_TYPES
        assert "protocol.async_task" in ALWAYS_ENABLED_EVENT_TYPES

    def test_always_enabled_events_bypass_config(self):
        config = CaptureConfig(l6c_protocol_lifecycle=False)
        # Task events are always enabled regardless of config
        assert config.is_layer_enabled("protocol.task.submitted") is True
        assert config.is_layer_enabled("protocol.task.completed") is True
        assert config.is_layer_enabled("protocol.async_task") is True

    def test_mcp_events_gated_by_l5a(self):
        config = CaptureConfig(l5a_tool_calls=False)
        assert config.is_layer_enabled("protocol.elicitation.request") is False
        assert config.is_layer_enabled("protocol.tool.structured_output") is False
        assert config.is_layer_enabled("protocol.mcp_app.invocation") is False

    def test_mcp_events_enabled_with_l5a(self):
        config = CaptureConfig(l5a_tool_calls=True)
        assert config.is_layer_enabled("protocol.elicitation.request") is True
        assert config.is_layer_enabled("protocol.tool.structured_output") is True
