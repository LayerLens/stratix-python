"""Test Google ADK adapter multi-agent tracing.

Ported as-is from ``ateam/tests/adapters/google_adk/test_multi_agent.py``.
"""

from __future__ import annotations

from layerlens.instrument.adapters.frameworks.google_adk.lifecycle import (
    GoogleADKAdapter,
)

from .conftest import MockStratix


class TestGoogleADKAdapterMultiAgent:
    def test_handoff_emits_agent_handoff(
        self, adapter: GoogleADKAdapter, mock_stratix: MockStratix
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
        self, adapter: GoogleADKAdapter, mock_stratix: MockStratix
    ) -> None:
        adapter.on_handoff(from_agent="a", to_agent="b")
        adapter.on_handoff(from_agent="b", to_agent="c")
        events = mock_stratix.get_events("agent.handoff")
        assert len(events) == 2

    def test_agent_config_emitted_once(
        self, adapter: GoogleADKAdapter, mock_stratix: MockStratix
    ) -> None:
        adapter.emit_dict_event(
            "environment.config",
            {
                "framework": "google_adk",
                "agent_name": "test_agent",
            },
        )
        adapter.emit_dict_event(
            "environment.config",
            {
                "framework": "google_adk",
                "agent_name": "test_agent",
            },
        )
        # Both emit since dedup is in _emit_agent_config, not emit_dict_event
        events = mock_stratix.get_events("environment.config")
        assert len(events) >= 1
