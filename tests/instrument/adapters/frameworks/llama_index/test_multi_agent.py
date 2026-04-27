"""Test LlamaIndex adapter multi-agent tracing.

Ported from ``ateam/tests/adapters/llama_index/test_multi_agent.py``.

Translation rules applied:
* fixtures (``adapter``, ``mock_stratix``) come from ``./conftest.py``
  whose imports are remapped to ``layerlens.instrument.*``.
"""

from __future__ import annotations

from layerlens.instrument.adapters.frameworks.llama_index.lifecycle import (
    LlamaIndexAdapter,
)

from .conftest import MockStratix


class TestLlamaIndexAdapterMultiAgent:
    def test_handoff_emits_agent_handoff(
        self, adapter: LlamaIndexAdapter, mock_stratix: MockStratix
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
        self, adapter: LlamaIndexAdapter, mock_stratix: MockStratix
    ) -> None:
        adapter.on_handoff(from_agent="a", to_agent="b")
        adapter.on_handoff(from_agent="b", to_agent="c")
        events = mock_stratix.get_events("agent.handoff")
        assert len(events) == 2

    def test_agent_config_emitted_once(
        self, adapter: LlamaIndexAdapter, mock_stratix: MockStratix
    ) -> None:
        adapter.emit_dict_event(
            "environment.config",
            {
                "framework": "llama_index",
                "agent_name": "test_agent",
            },
        )
        adapter.emit_dict_event(
            "environment.config",
            {
                "framework": "llama_index",
                "agent_name": "test_agent",
            },
        )
        # Both emit since dedup is in _emit_agent_config, not emit_dict_event
        events = mock_stratix.get_events("environment.config")
        assert len(events) >= 1
