"""Test Microsoft Agent Framework adapter lifecycle methods.

Ported as-is from ``ateam/tests/adapters/ms_agent_framework/test_lifecycle.py``.

Translation rules applied:
* ``stratix.sdk.python.adapters.ms_agent_framework.lifecycle`` →
  ``layerlens.instrument.adapters.frameworks.ms_agent_framework.lifecycle``
* ``stratix.sdk.python.adapters.base`` →
  ``layerlens.instrument.adapters._base``
* ``stratix.sdk.python.adapters.capture.CaptureConfig`` →
  ``layerlens.instrument.adapters._base.CaptureConfig``
* ``stratix.sdk.python.adapters.replay_models.ReplayableTrace`` →
  ``layerlens.instrument.adapters._base.ReplayableTrace``
* ``stratix.sdk.python.adapters.registry._ADAPTER_MODULES`` →
  ``layerlens.instrument.adapters._base.registry._ADAPTER_MODULES``
* The wrapper marker attribute renamed by the source from
  ``_stratix_original`` to ``_layerlens_original``.

Multi-tenancy: per the transitional "stratix attribute" pattern (see
migration doc §2.3 step 2 — keystone PR #118 still DRAFT), the
``MockStratix`` / ``EventCollector`` test stub gets an ``org_id``
attribute. The post-merge sweep PR will rebase to canonical kwarg once
#118 lands.
"""

from unittest.mock import AsyncMock, MagicMock

from layerlens.instrument.adapters._base import AdapterCapability, AdapterStatus
from layerlens.instrument.adapters.frameworks.ms_agent_framework.lifecycle import MSAgentAdapter
from layerlens.instrument.adapters._base import ReplayableTrace


class TestMSAgentAdapterLifecycle:
    def test_adapter_initialization(self):
        adapter = MSAgentAdapter()
        assert adapter.FRAMEWORK == "ms_agent_framework"
        assert adapter.VERSION == "0.1.0"

    def test_adapter_initialization_with_stratix(self, mock_stratix):
        adapter = MSAgentAdapter(stratix=mock_stratix)
        assert adapter.has_stratix

    def test_adapter_initialization_legacy_param(self, mock_stratix):
        adapter = MSAgentAdapter(stratix_instance=mock_stratix)
        assert adapter.has_stratix

    def test_connect_sets_healthy(self):
        adapter = MSAgentAdapter()
        adapter.connect()
        assert adapter.is_connected
        assert adapter.status == AdapterStatus.HEALTHY

    def test_connect_without_framework(self):
        """Adapter connects gracefully even when semantic-kernel is not installed."""
        adapter = MSAgentAdapter()
        adapter.connect()
        assert adapter.is_connected
        assert adapter.status == AdapterStatus.HEALTHY

    def test_disconnect_sets_disconnected(self):
        adapter = MSAgentAdapter()
        adapter.connect()
        adapter.disconnect()
        assert not adapter.is_connected
        assert adapter.status == AdapterStatus.DISCONNECTED

    def test_health_check_healthy(self, adapter):
        health = adapter.health_check()
        assert health.status == AdapterStatus.HEALTHY
        assert health.framework_name == "ms_agent_framework"
        assert health.adapter_version == "0.1.0"
        assert health.error_count == 0
        assert not health.circuit_open

    def test_health_check_disconnected(self):
        adapter = MSAgentAdapter()
        health = adapter.health_check()
        assert health.status == AdapterStatus.DISCONNECTED

    def test_get_adapter_info(self, adapter):
        info = adapter.get_adapter_info()
        assert info.name == "MSAgentAdapter"
        assert info.framework == "ms_agent_framework"
        assert info.version == "0.1.0"
        assert AdapterCapability.TRACE_TOOLS in info.capabilities
        assert AdapterCapability.TRACE_MODELS in info.capabilities
        assert AdapterCapability.TRACE_STATE in info.capabilities
        assert AdapterCapability.TRACE_HANDOFFS in info.capabilities

    def test_serialize_for_replay(self, adapter):
        trace = adapter.serialize_for_replay()
        assert isinstance(trace, ReplayableTrace)
        assert trace.adapter_name == "MSAgentAdapter"
        assert trace.framework == "ms_agent_framework"
        assert trace.trace_id is not None
        assert isinstance(trace.events, list)
        assert isinstance(trace.config, dict)

    def test_null_stratix_pattern(self):
        adapter = MSAgentAdapter()
        adapter.connect()
        # Should not raise even without STRATIX
        adapter.emit_dict_event("agent.input", {"framework": "ms_agent_framework"})

    def test_instrument_chat(self, adapter):
        mock_chat = MagicMock()
        mock_chat.name = "test_chat"
        mock_chat.invoke = AsyncMock()
        mock_chat.agents = []

        adapter.instrument_chat(mock_chat)
        assert hasattr(mock_chat.invoke, "_layerlens_original")

    def test_instrument_agent_aliases_instrument_chat(self, adapter):
        mock_chat = MagicMock()
        mock_chat.name = "test_chat"
        mock_chat.invoke = AsyncMock()
        mock_chat.agents = []

        adapter.instrument_agent(mock_chat)
        assert hasattr(mock_chat.invoke, "_layerlens_original")

    def test_instrument_chat_idempotent(self, adapter):
        mock_chat = MagicMock()
        mock_chat.name = "test_chat"
        mock_chat.invoke = AsyncMock()
        adapter.instrument_chat(mock_chat)
        first_invoke = mock_chat.invoke
        adapter.instrument_chat(mock_chat)
        assert mock_chat.invoke is first_invoke

    def test_disconnect_unwraps(self, adapter):
        mock_chat = MagicMock()
        mock_chat.name = "test_chat"
        original_invoke = AsyncMock()
        mock_chat.invoke = original_invoke
        adapter.instrument_chat(mock_chat)
        assert hasattr(mock_chat.invoke, "_layerlens_original")
        adapter.disconnect()
        assert mock_chat.invoke is original_invoke


class TestMSAgentAdapterEvents:
    def test_on_run_start_emits_agent_input(self, adapter, mock_stratix):
        adapter.on_run_start(agent_name="test_agent", input_data="hello")
        events = mock_stratix.get_events("agent.input")
        assert len(events) == 1
        assert events[0]["payload"]["framework"] == "ms_agent_framework"
        assert events[0]["payload"]["agent_name"] == "test_agent"

    def test_on_run_end_emits_agent_output(self, adapter, mock_stratix):
        adapter.on_run_start(agent_name="test_agent", input_data="hello")
        adapter.on_run_end(agent_name="test_agent", output="response")
        events = mock_stratix.get_events("agent.output")
        assert len(events) == 1
        assert events[0]["payload"]["duration_ns"] >= 0  # may be 0 in fast test execution

    def test_on_tool_use_emits_tool_call(self, adapter, mock_stratix):
        adapter.on_tool_use(
            tool_name="get_weather",
            tool_input={"city": "Seattle"},
            tool_output={"temp": "65F"},
        )
        events = mock_stratix.get_events("tool.call")
        assert len(events) == 1
        assert events[0]["payload"]["tool_name"] == "get_weather"

    def test_on_llm_call_emits_model_invoke(self, adapter, mock_stratix):
        adapter.on_llm_call(
            provider="azure_openai",
            model="gpt-4o",
            tokens_prompt=150,
            tokens_completion=75,
            latency_ms=600.0,
        )
        events = mock_stratix.get_events("model.invoke")
        assert len(events) == 1
        assert events[0]["payload"]["model"] == "gpt-4o"

    def test_on_handoff_emits_agent_handoff(self, adapter, mock_stratix):
        adapter.on_handoff(from_agent="agent_a", to_agent="agent_b")
        events = mock_stratix.get_events("agent.handoff")
        assert len(events) == 1
        assert events[0]["payload"]["from_agent"] == "agent_a"
        assert events[0]["payload"]["to_agent"] == "agent_b"
        assert events[0]["payload"]["reason"] == "group_chat_turn"

    def test_error_in_output(self, adapter, mock_stratix):
        adapter.on_run_end(agent_name="test_agent", output=None, error=Exception("test error"))
        events = mock_stratix.get_events("agent.output")
        assert len(events) == 1
        assert "error" in events[0]["payload"]

    def test_state_change_on_run_end(self, adapter, mock_stratix):
        adapter.on_run_end(agent_name="test_agent", output="done")
        events = mock_stratix.get_events("agent.state.change")
        assert len(events) == 1
        assert events[0]["payload"]["event_subtype"] == "run_complete"

    def test_state_change_on_error(self, adapter, mock_stratix):
        adapter.on_run_end(agent_name="test_agent", output=None, error=Exception("fail"))
        events = mock_stratix.get_events("agent.state.change")
        assert len(events) == 1
        assert events[0]["payload"]["event_subtype"] == "run_failed"

    def test_detect_provider_azure_default(self, adapter):
        assert adapter._detect_provider("some-custom-model") == "azure_openai"

    def test_detect_provider_known(self, adapter):
        assert adapter._detect_provider("gpt-4o") == "openai"
        assert adapter._detect_provider("claude-3-opus") == "anthropic"
        assert adapter._detect_provider("phi-3") == "microsoft"


class TestMSAgentAdapterRegistry:
    def test_adapter_registered(self):
        from layerlens.instrument.adapters._base.registry import _ADAPTER_MODULES

        assert "ms_agent_framework" in _ADAPTER_MODULES
        assert (
            _ADAPTER_MODULES["ms_agent_framework"] == "layerlens.instrument.adapters.frameworks.ms_agent_framework"
        )
