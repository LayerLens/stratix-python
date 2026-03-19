"""Test Google ADK adapter event emission."""

import pytest
from layerlens.instrument.adapters._capture import CaptureConfig


class TestGoogleADKAdapterEvents:
    def test_on_on_agent_start_emits_agent_input(self, adapter, mock_stratix):
        adapter.on_agent_start(agent_name="test_agent", input_data="hello")
        events = mock_stratix.get_events("agent.input")
        assert len(events) == 1
        assert events[0]["payload"]["framework"] == "google_adk"

    def test_on_on_agent_end_emits_agent_output(self, adapter, mock_stratix):
        adapter.on_agent_start(agent_name="test_agent", input_data="hello")
        adapter.on_agent_end(agent_name="test_agent", output="response")
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
        from layerlens.instrument.adapters.google_adk.lifecycle import GoogleADKAdapter
        adapter = GoogleADKAdapter(stratix=mock_stratix, capture_config=CaptureConfig.minimal())
        adapter.connect()
        adapter.on_llm_call(model="gpt-4o")
        adapter.on_tool_use(tool_name="test")
        assert len(mock_stratix.get_events("model.invoke")) == 0
        assert len(mock_stratix.get_events("tool.call")) == 0

    def test_cross_cutting_always_emitted(self, mock_stratix):
        from layerlens.instrument.adapters.google_adk.lifecycle import GoogleADKAdapter
        adapter = GoogleADKAdapter(stratix=mock_stratix, capture_config=CaptureConfig.minimal())
        adapter.connect()
        adapter.emit_dict_event("agent.state.change", {"framework": "google_adk", "event_subtype": "test"})
        assert len(mock_stratix.get_events("agent.state.change")) == 1

    def test_error_in_output(self, adapter, mock_stratix):
        adapter.on_agent_end(agent_name="test_agent", output=None, error=Exception("test error"))
        events = mock_stratix.get_events("agent.output")
        assert len(events) == 1
        assert "error" in events[0]["payload"]
