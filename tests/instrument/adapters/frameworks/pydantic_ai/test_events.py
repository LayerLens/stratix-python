"""Test Pydantic AI adapter event emission.

Ported as-is from ``ateam/tests/adapters/pydantic_ai/test_events.py``.

Translation rules applied:
* ``stratix.sdk.python.adapters.pydantic_ai.lifecycle`` →
  ``layerlens.instrument.adapters.frameworks.pydantic_ai.lifecycle``
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


class TestPydanticAIAdapterEvents:
    def test_on_on_run_start_emits_agent_input(self, adapter, mock_stratix):
        adapter.on_run_start(agent_name="test_agent", input_data="hello")
        events = mock_stratix.get_events("agent.input")
        assert len(events) == 1
        assert events[0]["payload"]["framework"] == "pydantic_ai"

    def test_on_on_run_end_emits_agent_output(self, adapter, mock_stratix):
        adapter.on_run_start(agent_name="test_agent", input_data="hello")
        adapter.on_run_end(agent_name="test_agent", output="response")
        events = mock_stratix.get_events("agent.output")
        assert len(events) == 1

    def test_on_tool_use_emits_tool_call(self, adapter, mock_stratix):
        adapter.on_tool_use(
            tool_name="test_tool",
            tool_input={"query": "test"},
            tool_output={"result": "ok"},
        )
        events = mock_stratix.get_events("tool.call")
        assert len(events) == 1
        assert events[0]["payload"]["tool_name"] == "test_tool"

    def test_on_llm_call_emits_model_invoke(self, adapter, mock_stratix):
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

    def test_capture_config_minimal_gates_l3_l5(self, mock_stratix):
        from layerlens.instrument.adapters.frameworks.pydantic_ai.lifecycle import PydanticAIAdapter

        adapter = PydanticAIAdapter(stratix=mock_stratix, capture_config=CaptureConfig.minimal())
        adapter.connect()
        adapter.on_llm_call(model="gpt-4o")
        adapter.on_tool_use(tool_name="test")
        assert len(mock_stratix.get_events("model.invoke")) == 0
        assert len(mock_stratix.get_events("tool.call")) == 0

    def test_cross_cutting_always_emitted(self, mock_stratix):
        from layerlens.instrument.adapters.frameworks.pydantic_ai.lifecycle import PydanticAIAdapter

        adapter = PydanticAIAdapter(stratix=mock_stratix, capture_config=CaptureConfig.minimal())
        adapter.connect()
        adapter.emit_dict_event(
            "agent.state.change", {"framework": "pydantic_ai", "event_subtype": "test"}
        )
        assert len(mock_stratix.get_events("agent.state.change")) == 1

    def test_error_in_output(self, adapter, mock_stratix):
        adapter.on_run_end(agent_name="test_agent", output=None, error=Exception("test error"))
        events = mock_stratix.get_events("agent.output")
        assert len(events) == 1
        assert "error" in events[0]["payload"]
