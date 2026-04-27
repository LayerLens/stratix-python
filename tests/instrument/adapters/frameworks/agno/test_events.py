"""Tests for the Agno adapter's CaptureConfig-gated event emission.

Ported from ``ateam/tests/adapters/agno/test_events.py``.

Renames:
- ``stratix.sdk.python.adapters.capture`` →
  ``layerlens.instrument.adapters._base`` (re-exports ``CaptureConfig``).
- ``stratix.sdk.python.adapters.agno.lifecycle`` →
  ``layerlens.instrument.adapters.frameworks.agno.lifecycle``.
"""

from __future__ import annotations

from layerlens.instrument.adapters._base import CaptureConfig
from layerlens.instrument.adapters.frameworks.agno.lifecycle import AgnoAdapter

from .conftest import MockLayerLens


class TestAgnoAdapterEvents:
    def test_capture_config_minimal_gates_l3_l5(self, mock_stratix: MockLayerLens) -> None:
        adapter = AgnoAdapter(stratix=mock_stratix, capture_config=CaptureConfig.minimal())
        adapter.connect()
        adapter.on_llm_call(model="gpt-4o")
        adapter.on_tool_use(tool_name="test")
        assert len(mock_stratix.get_events("model.invoke")) == 0
        assert len(mock_stratix.get_events("tool.call")) == 0

    def test_cross_cutting_always_emitted(self, mock_stratix: MockLayerLens) -> None:
        adapter = AgnoAdapter(stratix=mock_stratix, capture_config=CaptureConfig.minimal())
        adapter.connect()
        adapter.emit_dict_event(
            "agent.state.change", {"framework": "agno", "event_subtype": "test"}
        )
        assert len(mock_stratix.get_events("agent.state.change")) == 1

    def test_tool_use_with_error(
        self, adapter: AgnoAdapter, mock_stratix: MockLayerLens
    ) -> None:
        adapter.on_tool_use(
            tool_name="failing_tool",
            tool_input={"query": "test"},
            error=Exception("tool failed"),
        )
        events = mock_stratix.get_events("tool.call")
        assert len(events) == 1
        assert events[0]["payload"]["error"] == "tool failed"

    def test_tool_use_with_latency(
        self, adapter: AgnoAdapter, mock_stratix: MockLayerLens
    ) -> None:
        adapter.on_tool_use(
            tool_name="slow_tool",
            tool_input={"query": "test"},
            tool_output={"result": "ok"},
            latency_ms=1500.0,
        )
        events = mock_stratix.get_events("tool.call")
        assert len(events) == 1
        assert events[0]["payload"]["latency_ms"] == 1500.0

    def test_handoff_with_context(
        self, adapter: AgnoAdapter, mock_stratix: MockLayerLens
    ) -> None:
        adapter.on_handoff(from_agent="leader", to_agent="writer", context="Write a summary")
        events = mock_stratix.get_events("agent.handoff")
        assert len(events) == 1
        assert events[0]["payload"]["context_hash"] is not None
