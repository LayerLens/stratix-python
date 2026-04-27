"""Tests for the Agno adapter lifecycle and event emission.

Ported from ``ateam/tests/adapters/agno/test_lifecycle.py``.

Renames:
- ``stratix.sdk.python.adapters.agno.lifecycle`` →
  ``layerlens.instrument.adapters.frameworks.agno.lifecycle``
- ``stratix.sdk.python.adapters.base`` →
  ``layerlens.instrument.adapters._base``
- ``stratix.sdk.python.adapters.replay_models.ReplayableTrace`` →
  ``layerlens.instrument.adapters._base.ReplayableTrace``
- ``stratix.sdk.python.adapters.registry._ADAPTER_MODULES`` →
  ``layerlens.instrument.adapters._base.registry._ADAPTER_MODULES``
- The wrapper marker attribute renamed by the source from
  ``_stratix_original`` to ``_layerlens_original``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from layerlens.instrument.adapters._base import (
    AdapterStatus,
    ReplayableTrace,
    AdapterCapability,
)
from layerlens.instrument.adapters.frameworks.agno.lifecycle import AgnoAdapter

from .conftest import MockLayerLens


class TestAgnoAdapterLifecycle:
    def test_adapter_initialization(self) -> None:
        adapter = AgnoAdapter()
        assert adapter.FRAMEWORK == "agno"
        assert adapter.VERSION == "0.1.0"

    def test_adapter_initialization_with_stratix(self, mock_stratix: MockLayerLens) -> None:
        adapter = AgnoAdapter(stratix=mock_stratix)
        assert adapter.has_stratix

    def test_adapter_initialization_legacy_param(self, mock_stratix: MockLayerLens) -> None:
        adapter = AgnoAdapter(stratix_instance=mock_stratix)
        assert adapter.has_stratix

    def test_connect_sets_healthy(self) -> None:
        adapter = AgnoAdapter()
        adapter.connect()
        assert adapter.is_connected
        assert adapter.status == AdapterStatus.HEALTHY

    def test_connect_without_framework(self) -> None:
        """Adapter connects gracefully even when agno is not installed."""
        adapter = AgnoAdapter()
        adapter.connect()
        assert adapter.is_connected
        assert adapter.status == AdapterStatus.HEALTHY

    def test_disconnect_sets_disconnected(self) -> None:
        adapter = AgnoAdapter()
        adapter.connect()
        adapter.disconnect()
        assert not adapter.is_connected
        assert adapter.status == AdapterStatus.DISCONNECTED

    def test_health_check_healthy(self, adapter: AgnoAdapter) -> None:
        health = adapter.health_check()
        assert health.status == AdapterStatus.HEALTHY
        assert health.framework_name == "agno"
        assert health.adapter_version == "0.1.0"
        assert health.error_count == 0
        assert not health.circuit_open

    def test_health_check_disconnected(self) -> None:
        adapter = AgnoAdapter()
        health = adapter.health_check()
        assert health.status == AdapterStatus.DISCONNECTED

    def test_get_adapter_info(self, adapter: AgnoAdapter) -> None:
        info = adapter.get_adapter_info()
        assert info.name == "AgnoAdapter"
        assert info.framework == "agno"
        assert info.version == "0.1.0"
        assert AdapterCapability.TRACE_TOOLS in info.capabilities
        assert AdapterCapability.TRACE_MODELS in info.capabilities
        assert AdapterCapability.TRACE_STATE in info.capabilities
        assert AdapterCapability.TRACE_HANDOFFS in info.capabilities

    def test_serialize_for_replay(self, adapter: AgnoAdapter) -> None:
        trace = adapter.serialize_for_replay()
        assert isinstance(trace, ReplayableTrace)
        assert trace.adapter_name == "AgnoAdapter"
        assert trace.framework == "agno"
        assert trace.trace_id is not None
        assert isinstance(trace.events, list)
        assert isinstance(trace.config, dict)

    def test_null_stratix_pattern(self) -> None:
        adapter = AgnoAdapter()
        adapter.connect()
        # Should not raise even without a STRATIX/LayerLens client
        adapter.emit_dict_event("agent.input", {"framework": "agno"})

    def test_instrument_agent(self, adapter: AgnoAdapter) -> None:
        mock_agent = MagicMock()
        mock_agent.name = "test_agent"
        mock_agent.run = MagicMock()
        mock_agent.arun = AsyncMock()
        mock_agent.tools = []

        adapter.instrument_agent(mock_agent)
        assert hasattr(mock_agent.run, "_layerlens_original")
        assert hasattr(mock_agent.arun, "_layerlens_original")

    def test_instrument_agent_idempotent(self, adapter: AgnoAdapter) -> None:
        mock_agent = MagicMock()
        mock_agent.name = "test_agent"
        mock_agent.run = MagicMock()
        adapter.instrument_agent(mock_agent)
        first_run = mock_agent.run
        adapter.instrument_agent(mock_agent)
        assert mock_agent.run is first_run

    def test_disconnect_unwraps(self, adapter: AgnoAdapter) -> None:
        mock_agent = MagicMock()
        mock_agent.name = "test_agent"
        original_run = MagicMock()
        mock_agent.run = original_run
        adapter.instrument_agent(mock_agent)
        assert hasattr(mock_agent.run, "_layerlens_original")
        adapter.disconnect()
        assert mock_agent.run is original_run


class TestAgnoAdapterEvents:
    def test_on_run_start_emits_agent_input(
        self, adapter: AgnoAdapter, mock_stratix: MockLayerLens
    ) -> None:
        adapter.on_run_start(agent_name="test_agent", input_data="hello")
        events = mock_stratix.get_events("agent.input")
        assert len(events) == 1
        assert events[0]["payload"]["framework"] == "agno"
        assert events[0]["payload"]["agent_name"] == "test_agent"

    def test_on_run_end_emits_agent_output(
        self, adapter: AgnoAdapter, mock_stratix: MockLayerLens
    ) -> None:
        adapter.on_run_start(agent_name="test_agent", input_data="hello")
        adapter.on_run_end(agent_name="test_agent", output="response")
        events = mock_stratix.get_events("agent.output")
        assert len(events) == 1
        assert events[0]["payload"]["duration_ns"] >= 0  # may be 0 in fast test execution

    def test_on_tool_use_emits_tool_call(
        self, adapter: AgnoAdapter, mock_stratix: MockLayerLens
    ) -> None:
        adapter.on_tool_use(
            tool_name="search_web",
            tool_input={"query": "test"},
            tool_output={"result": "ok"},
        )
        events = mock_stratix.get_events("tool.call")
        assert len(events) == 1
        assert events[0]["payload"]["tool_name"] == "search_web"

    def test_on_llm_call_emits_model_invoke(
        self, adapter: AgnoAdapter, mock_stratix: MockLayerLens
    ) -> None:
        adapter.on_llm_call(
            provider="openai",
            model="gpt-4o",
            tokens_prompt=100,
            tokens_completion=50,
            latency_ms=500.0,
        )
        events = mock_stratix.get_events("model.invoke")
        assert len(events) == 1
        assert events[0]["payload"]["model"] == "gpt-4o"

    def test_on_handoff_emits_agent_handoff(
        self, adapter: AgnoAdapter, mock_stratix: MockLayerLens
    ) -> None:
        adapter.on_handoff(from_agent="leader", to_agent="researcher")
        events = mock_stratix.get_events("agent.handoff")
        assert len(events) == 1
        assert events[0]["payload"]["from_agent"] == "leader"
        assert events[0]["payload"]["to_agent"] == "researcher"

    def test_error_in_output(
        self, adapter: AgnoAdapter, mock_stratix: MockLayerLens
    ) -> None:
        adapter.on_run_end(agent_name="test_agent", output=None, error=Exception("test error"))
        events = mock_stratix.get_events("agent.output")
        assert len(events) == 1
        assert "error" in events[0]["payload"]

    def test_state_change_on_run_end(
        self, adapter: AgnoAdapter, mock_stratix: MockLayerLens
    ) -> None:
        adapter.on_run_end(agent_name="test_agent", output="done")
        events = mock_stratix.get_events("agent.state.change")
        assert len(events) == 1
        assert events[0]["payload"]["event_subtype"] == "run_complete"

    def test_state_change_on_error(
        self, adapter: AgnoAdapter, mock_stratix: MockLayerLens
    ) -> None:
        adapter.on_run_end(agent_name="test_agent", output=None, error=Exception("fail"))
        events = mock_stratix.get_events("agent.state.change")
        assert len(events) == 1
        assert events[0]["payload"]["event_subtype"] == "run_failed"


class TestAgnoAdapterRegistry:
    def test_adapter_registered(self) -> None:
        from layerlens.instrument.adapters._base.registry import _ADAPTER_MODULES

        assert "agno" in _ADAPTER_MODULES
        assert _ADAPTER_MODULES["agno"] == "layerlens.instrument.adapters.frameworks.agno"
