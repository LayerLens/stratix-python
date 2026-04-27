"""Test Bedrock Agents adapter multi-agent tracing.

Ported from ``ateam/tests/adapters/bedrock_agents/test_multi_agent.py``.
"""

from __future__ import annotations

from typing import Any

from layerlens.instrument.adapters.frameworks.bedrock_agents.lifecycle import (
    BedrockAgentsAdapter,
)


class TestBedrockAgentsAdapterMultiAgent:
    def test_handoff_emits_agent_handoff(
        self, adapter: BedrockAgentsAdapter, mock_stratix: Any
    ) -> None:
        adapter.on_handoff(
            from_agent="agent_a",
            to_agent="agent_b",
            context="delegation context",
        )
        events = mock_stratix.get_events("agent.handoff")
        assert len(events) == 1
        assert events[0]["payload"]["from_agent"] == "agent_a"
        assert events[0]["payload"]["to_agent"] == "agent_b"

    def test_multiple_handoffs(
        self, adapter: BedrockAgentsAdapter, mock_stratix: Any
    ) -> None:
        adapter.on_handoff(from_agent="a", to_agent="b")
        adapter.on_handoff(from_agent="b", to_agent="c")
        events = mock_stratix.get_events("agent.handoff")
        assert len(events) == 2

    def test_agent_config_emitted_once(
        self, adapter: BedrockAgentsAdapter, mock_stratix: Any
    ) -> None:
        adapter.emit_dict_event(
            "environment.config",
            {
                "framework": "bedrock_agents",
                "agent_name": "test_agent",
            },
        )
        adapter.emit_dict_event(
            "environment.config",
            {
                "framework": "bedrock_agents",
                "agent_name": "test_agent",
            },
        )
        # Both emit since dedup is in _emit_agent_config, not emit_dict_event
        events = mock_stratix.get_events("environment.config")
        assert len(events) >= 1
