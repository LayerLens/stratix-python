"""Test Microsoft Agent Framework adapter event emission.

Ported as-is from ``ateam/tests/adapters/ms_agent_framework/test_events.py``.

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

from layerlens.instrument.adapters._base import CaptureConfig


class TestMSAgentAdapterEvents:
    def test_capture_config_minimal_gates_l3_l5(self, mock_stratix):
        from layerlens.instrument.adapters.frameworks.ms_agent_framework.lifecycle import MSAgentAdapter

        adapter = MSAgentAdapter(stratix=mock_stratix, capture_config=CaptureConfig.minimal())
        adapter.connect()
        adapter.on_llm_call(model="gpt-4o")
        adapter.on_tool_use(tool_name="test")
        assert len(mock_stratix.get_events("model.invoke")) == 0
        assert len(mock_stratix.get_events("tool.call")) == 0

    def test_cross_cutting_always_emitted(self, mock_stratix):
        from layerlens.instrument.adapters.frameworks.ms_agent_framework.lifecycle import MSAgentAdapter

        adapter = MSAgentAdapter(stratix=mock_stratix, capture_config=CaptureConfig.minimal())
        adapter.connect()
        adapter.emit_dict_event(
            "agent.state.change", {"framework": "ms_agent_framework", "event_subtype": "test"}
        )
        assert len(mock_stratix.get_events("agent.state.change")) == 1

    def test_tool_use_with_error(self, adapter, mock_stratix):
        adapter.on_tool_use(
            tool_name="failing_tool",
            tool_input={"query": "test"},
            error=Exception("tool failed"),
        )
        events = mock_stratix.get_events("tool.call")
        assert len(events) == 1
        assert events[0]["payload"]["error"] == "tool failed"

    def test_tool_use_with_latency(self, adapter, mock_stratix):
        adapter.on_tool_use(
            tool_name="slow_tool",
            tool_input={"query": "test"},
            tool_output={"result": "ok"},
            latency_ms=1200.0,
        )
        events = mock_stratix.get_events("tool.call")
        assert len(events) == 1
        assert events[0]["payload"]["latency_ms"] == 1200.0

    def test_handoff_with_context(self, adapter, mock_stratix):
        adapter.on_handoff(from_agent="planner", to_agent="executor", context="Execute plan step 3")
        events = mock_stratix.get_events("agent.handoff")
        assert len(events) == 1
        assert events[0]["payload"]["context_hash"] is not None
        assert events[0]["payload"]["reason"] == "group_chat_turn"

    def test_llm_call_with_messages_content_enabled(self, mock_stratix):
        from layerlens.instrument.adapters.frameworks.ms_agent_framework.lifecycle import MSAgentAdapter

        config = CaptureConfig(capture_content=True)
        adapter = MSAgentAdapter(stratix=mock_stratix, capture_config=config)
        adapter.connect()
        adapter.on_llm_call(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hello"}],
        )
        events = mock_stratix.get_events("model.invoke")
        assert len(events) == 1
        assert "messages" in events[0]["payload"]
