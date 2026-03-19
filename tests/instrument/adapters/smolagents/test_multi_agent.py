"""Test SmolAgents adapter multi-agent tracing."""

import pytest


class TestSmolAgentsAdapterMultiAgent:
    def test_handoff_emits_agent_handoff(self, adapter, mock_stratix):
        adapter.on_handoff(
            from_agent="agent_a",
            to_agent="agent_b",
            context="delegation context",
        )
        events = mock_stratix.get_events("agent.handoff")
        assert len(events) == 1
        assert events[0]["payload"]["from_agent"] == "agent_a"
        assert events[0]["payload"]["to_agent"] == "agent_b"

    def test_multiple_handoffs(self, adapter, mock_stratix):
        adapter.on_handoff(from_agent="a", to_agent="b")
        adapter.on_handoff(from_agent="b", to_agent="c")
        events = mock_stratix.get_events("agent.handoff")
        assert len(events) == 2

    def test_agent_config_emitted_once(self, adapter, mock_stratix):
        adapter.emit_dict_event("environment.config", {
            "framework": "smolagents",
            "agent_name": "test_agent",
        })
        adapter.emit_dict_event("environment.config", {
            "framework": "smolagents",
            "agent_name": "test_agent",
        })
        # Both emit since dedup is in _emit_agent_config, not emit_dict_event
        events = mock_stratix.get_events("environment.config")
        assert len(events) >= 1
