"""Test AWS Strands adapter lifecycle methods.

Ported as-is from ``ateam/tests/adapters/strands/test_lifecycle.py``.

Translation rules applied:
* ``stratix.sdk.python.adapters.strands.lifecycle`` →
  ``layerlens.instrument.adapters.frameworks.strands.lifecycle``
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

from unittest.mock import MagicMock

from layerlens.instrument.adapters._base import AdapterCapability, AdapterStatus
from layerlens.instrument.adapters._base import ReplayableTrace
from layerlens.instrument.adapters.frameworks.strands.lifecycle import StrandsAdapter


class TestStrandsAdapterLifecycle:
    def test_adapter_initialization(self):
        adapter = StrandsAdapter()
        assert adapter.FRAMEWORK == "strands"
        assert adapter.VERSION == "0.1.0"

    def test_adapter_initialization_with_stratix(self, mock_stratix):
        adapter = StrandsAdapter(stratix=mock_stratix)
        assert adapter.has_stratix

    def test_adapter_initialization_legacy_param(self, mock_stratix):
        adapter = StrandsAdapter(stratix_instance=mock_stratix)
        assert adapter.has_stratix

    def test_connect_sets_healthy(self):
        adapter = StrandsAdapter()
        adapter.connect()
        assert adapter.is_connected
        assert adapter.status == AdapterStatus.HEALTHY

    def test_connect_without_framework(self):
        """Adapter connects gracefully even when strands is not installed."""
        adapter = StrandsAdapter()
        adapter.connect()
        assert adapter.is_connected
        assert adapter.status == AdapterStatus.HEALTHY

    def test_disconnect_sets_disconnected(self):
        adapter = StrandsAdapter()
        adapter.connect()
        adapter.disconnect()
        assert not adapter.is_connected
        assert adapter.status == AdapterStatus.DISCONNECTED

    def test_health_check_healthy(self, adapter):
        health = adapter.health_check()
        assert health.status == AdapterStatus.HEALTHY
        assert health.framework_name == "strands"
        assert health.adapter_version == "0.1.0"
        assert health.error_count == 0
        assert not health.circuit_open

    def test_health_check_disconnected(self):
        adapter = StrandsAdapter()
        health = adapter.health_check()
        assert health.status == AdapterStatus.DISCONNECTED

    def test_get_adapter_info(self, adapter):
        info = adapter.get_adapter_info()
        assert info.name == "StrandsAdapter"
        assert info.framework == "strands"
        assert info.version == "0.1.0"
        assert AdapterCapability.TRACE_TOOLS in info.capabilities
        assert AdapterCapability.TRACE_MODELS in info.capabilities
        assert AdapterCapability.TRACE_STATE in info.capabilities

    def test_serialize_for_replay(self, adapter):
        trace = adapter.serialize_for_replay()
        assert isinstance(trace, ReplayableTrace)
        assert trace.adapter_name == "StrandsAdapter"
        assert trace.framework == "strands"
        assert trace.trace_id is not None
        assert isinstance(trace.events, list)
        assert isinstance(trace.config, dict)

    def test_null_stratix_pattern(self):
        adapter = StrandsAdapter()
        adapter.connect()
        # Should not raise even without STRATIX
        adapter.emit_dict_event("agent.input", {"framework": "strands"})

    def test_instrument_agent(self, adapter):
        mock_agent = MagicMock()
        mock_agent.name = "test_agent"
        mock_agent.__call__ = MagicMock()
        mock_agent.tools = []

        adapter.instrument_agent(mock_agent)
        assert hasattr(mock_agent.__call__, "_layerlens_original")

    def test_instrument_agent_idempotent(self, adapter):
        mock_agent = MagicMock()
        mock_agent.name = "test_agent"
        mock_agent.__call__ = MagicMock()
        adapter.instrument_agent(mock_agent)
        first_call = mock_agent.__call__
        adapter.instrument_agent(mock_agent)
        assert mock_agent.__call__ is first_call

    def test_disconnect_unwraps(self, adapter):
        mock_agent = MagicMock()
        mock_agent.name = "test_agent"
        original_call = MagicMock()
        mock_agent.__call__ = original_call
        adapter.instrument_agent(mock_agent)
        assert hasattr(mock_agent.__call__, "_layerlens_original")
        adapter.disconnect()
        assert mock_agent.__call__ is original_call


class TestStrandsAdapterEvents:
    def test_on_run_start_emits_agent_input(self, adapter, mock_stratix):
        adapter.on_run_start(agent_name="test_agent", input_data="hello")
        events = mock_stratix.get_events("agent.input")
        assert len(events) == 1
        assert events[0]["payload"]["framework"] == "strands"
        assert events[0]["payload"]["agent_name"] == "test_agent"

    def test_on_run_end_emits_agent_output(self, adapter, mock_stratix):
        adapter.on_run_start(agent_name="test_agent", input_data="hello")
        adapter.on_run_end(agent_name="test_agent", output="response")
        events = mock_stratix.get_events("agent.output")
        assert len(events) == 1
        assert events[0]["payload"]["duration_ns"] >= 0  # may be 0 in fast test execution

    def test_on_tool_use_emits_tool_call(self, adapter, mock_stratix):
        adapter.on_tool_use(
            tool_name="calculator",
            tool_input={"expression": "2+2"},
            tool_output={"result": 4},
        )
        events = mock_stratix.get_events("tool.call")
        assert len(events) == 1
        assert events[0]["payload"]["tool_name"] == "calculator"

    def test_on_llm_call_emits_model_invoke(self, adapter, mock_stratix):
        adapter.on_llm_call(
            provider="bedrock",
            model="anthropic.claude-3-sonnet",
            tokens_prompt=200,
            tokens_completion=100,
            latency_ms=800.0,
        )
        events = mock_stratix.get_events("model.invoke")
        assert len(events) == 1
        assert events[0]["payload"]["model"] == "anthropic.claude-3-sonnet"
        assert events[0]["payload"]["provider"] == "bedrock"

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

    def test_detect_provider_bedrock_default(self, adapter):
        assert adapter._detect_provider("anthropic.claude-3-sonnet") == "bedrock"
        assert adapter._detect_provider("amazon.titan-text") == "bedrock"
        assert adapter._detect_provider("unknown-model") == "bedrock"

    def test_detect_provider_non_bedrock(self, adapter):
        assert adapter._detect_provider("gpt-4o") == "openai"
        assert adapter._detect_provider("gemini-pro") == "google"


class TestStrandsAdapterRegistry:
    def test_adapter_registered(self):
        from layerlens.instrument.adapters._base.registry import _ADAPTER_MODULES

        assert "strands" in _ADAPTER_MODULES
        assert _ADAPTER_MODULES["strands"] == "layerlens.instrument.adapters.frameworks.strands"
