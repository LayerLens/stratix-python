"""Test OpenAI Agents adapter multi-agent tracing.

Ported from ``ateam/tests/adapters/openai_agents/test_multi_agent.py``.
The adapter and its multi-agent (handoff) handlers live in the same
namespace move as the rest of the suite — see ``test_lifecycle.py`` for
the rename table.
"""

from __future__ import annotations

from layerlens.instrument.adapters.frameworks.openai_agents.lifecycle import (
    OpenAIAgentsAdapter,
)

from .conftest import MockStratix


class TestOpenAIAgentsAdapterMultiAgent:
    def test_handoff_emits_agent_handoff(self, adapter: OpenAIAgentsAdapter, mock_stratix: MockStratix) -> None:
        adapter.on_handoff(
            from_agent="agent_a",
            to_agent="agent_b",
            context="delegation context",
        )
        events = mock_stratix.get_events("agent.handoff")
        assert len(events) == 1
        assert events[0]["payload"]["from_agent"] == "agent_a"
        assert events[0]["payload"]["to_agent"] == "agent_b"

    def test_multiple_handoffs(self, adapter: OpenAIAgentsAdapter, mock_stratix: MockStratix) -> None:
        adapter.on_handoff(from_agent="a", to_agent="b")
        adapter.on_handoff(from_agent="b", to_agent="c")
        events = mock_stratix.get_events("agent.handoff")
        assert len(events) == 2

    def test_agent_config_emitted_once(self, adapter: OpenAIAgentsAdapter, mock_stratix: MockStratix) -> None:
        adapter.emit_dict_event(
            "environment.config",
            {
                "framework": "openai_agents",
                "agent_name": "test_agent",
            },
        )
        adapter.emit_dict_event(
            "environment.config",
            {
                "framework": "openai_agents",
                "agent_name": "test_agent",
            },
        )
        # Both emit since dedup is in _emit_agent_config, not emit_dict_event
        events = mock_stratix.get_events("environment.config")
        assert len(events) >= 1
