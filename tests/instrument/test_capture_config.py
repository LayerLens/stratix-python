from __future__ import annotations

import dataclasses
from unittest.mock import Mock

import pytest

from layerlens.instrument import trace, CaptureConfig
from .conftest import find_events, find_event


# ---------------------------------------------------------------------------
# CaptureConfig unit tests
# ---------------------------------------------------------------------------


class TestCaptureConfig:
    def test_default_matches_standard(self):
        """Bare CaptureConfig() gives sensible production defaults (matches ateam)."""
        config = CaptureConfig()
        assert config.l1_agent_io is True
        assert config.l2_agent_code is False
        assert config.l3_model_metadata is True
        assert config.l4a_environment_config is True
        assert config.l4b_environment_metrics is False
        assert config.l5a_tool_calls is True
        assert config.l5b_tool_logic is False
        assert config.l5c_tool_environment is False
        assert config.l6a_protocol_discovery is True
        assert config.l6b_protocol_streams is True
        assert config.l6c_protocol_lifecycle is True
        assert config.capture_content is True

    def test_full_preset(self):
        config = CaptureConfig.full()
        for f in dataclasses.fields(config):
            assert getattr(config, f.name) is True

    def test_minimal_preset(self):
        config = CaptureConfig.minimal()
        assert config.l1_agent_io is True
        assert config.l2_agent_code is False
        assert config.l3_model_metadata is False
        assert config.l4a_environment_config is False
        assert config.l4b_environment_metrics is False
        assert config.l5a_tool_calls is False
        assert config.l5b_tool_logic is False
        assert config.l5c_tool_environment is False
        assert config.l6a_protocol_discovery is True
        assert config.l6b_protocol_streams is False
        assert config.l6c_protocol_lifecycle is True
        assert config.capture_content is True

    def test_standard_preset(self):
        """standard() is the same as bare CaptureConfig()."""
        config = CaptureConfig.standard()
        default = CaptureConfig()
        for f in dataclasses.fields(config):
            assert getattr(config, f.name) == getattr(default, f.name)

    def test_frozen(self):
        config = CaptureConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.l1_agent_io = False  # type: ignore[misc]

    def test_to_dict(self):
        config = CaptureConfig.minimal()
        d = config.to_dict()
        assert len(d) == 12  # 11 layers + capture_content
        assert d["l1_agent_io"] is True
        assert d["l3_model_metadata"] is False
        assert d["l5a_tool_calls"] is False
        assert d["capture_content"] is True

    def test_custom_config(self):
        config = CaptureConfig(l1_agent_io=True, l5a_tool_calls=False)
        assert config.l1_agent_io is True
        assert config.l5a_tool_calls is False
        assert config.l3_model_metadata is True  # default

    def test_is_layer_enabled_always_enabled(self):
        config = CaptureConfig.minimal()
        assert config.is_layer_enabled("agent.error") is True
        assert config.is_layer_enabled("cost.record") is True
        assert config.is_layer_enabled("agent.state.change") is True
        assert config.is_layer_enabled("policy.violation") is True
        assert config.is_layer_enabled("protocol.task.submitted") is True
        assert config.is_layer_enabled("protocol.task.completed") is True
        assert config.is_layer_enabled("protocol.async_task") is True

    def test_is_layer_enabled_mapped(self):
        config = CaptureConfig.minimal()
        assert config.is_layer_enabled("agent.input") is True  # L1 on
        assert config.is_layer_enabled("model.invoke") is False  # L3 off
        assert config.is_layer_enabled("tool.call") is False  # L5a off

    def test_is_layer_enabled_unknown_fail_open(self):
        config = CaptureConfig.minimal()
        assert config.is_layer_enabled("unknown.event") is True

    def test_is_layer_enabled_full(self):
        config = CaptureConfig.full()
        assert config.is_layer_enabled("agent.input") is True
        assert config.is_layer_enabled("model.invoke") is True
        assert config.is_layer_enabled("tool.call") is True


# ---------------------------------------------------------------------------
# @trace integration with CaptureConfig
# ---------------------------------------------------------------------------


def _openai_response():
    r = Mock()
    r.choices = [Mock()]
    r.choices[0].message = Mock()
    r.choices[0].message.role = "assistant"
    r.choices[0].message.content = "Hello!"
    r.usage = Mock()
    r.usage.prompt_tokens = 10
    r.usage.completion_tokens = 5
    r.usage.total_tokens = 15
    r.model = "gpt-4"
    return r


class TestCaptureConfigWithTrace:
    def test_full_config_preserves_all(self, mock_client, capture_trace):
        @trace(mock_client, capture_config=CaptureConfig.full())
        def my_agent(query):
            return {"answer": 42}

        my_agent("hello")
        events = capture_trace["events"]
        agent_input = find_event(events, "agent.input")
        agent_output = find_event(events, "agent.output")
        assert agent_input["payload"]["input"] == "hello"
        assert agent_output["payload"]["output"] == {"answer": 42}

    def test_default_is_full(self, mock_client, capture_trace):
        @trace(mock_client)
        def my_agent(query):
            return {"answer": 42}

        my_agent("hello")
        events = capture_trace["events"]
        agent_input = find_event(events, "agent.input")
        assert agent_input["payload"]["input"] == "hello"

    def test_l1_off_strips_agent_io(self, mock_client, capture_trace):
        """When L1 is off, agent.input/agent.output events are suppressed."""
        config = CaptureConfig(l1_agent_io=False)

        @trace(mock_client, capture_config=config)
        def my_agent(query):
            return {"answer": 42}

        result = my_agent("hello")
        assert result == {"answer": 42}  # return value still works
        # With only L1 events and L1 off, no events emitted → no upload
        mock_client.traces.upload.assert_not_called()

    def test_l1_off_preserves_error(self, mock_client, capture_trace):
        config = CaptureConfig(l1_agent_io=False)

        @trace(mock_client, capture_config=config)
        def my_agent():
            raise ValueError("boom")

        with pytest.raises(ValueError):
            my_agent()
        events = capture_trace["events"]
        # agent.error is always enabled
        errors = find_events(events, "agent.error")
        assert len(errors) == 1

    def test_config_stored_in_upload(self, mock_client, capture_trace):
        config = CaptureConfig.standard()

        @trace(mock_client, capture_config=config)
        def my_agent():
            return "ok"

        my_agent()
        stored = capture_trace["capture_config"]
        assert stored == config.to_dict()

    def test_context_cleanup(self, mock_client):
        from layerlens.instrument._context import _current_collector

        @trace(mock_client, capture_config=CaptureConfig.minimal())
        def my_agent():
            return "ok"

        my_agent()
        assert _current_collector.get() is None


# ---------------------------------------------------------------------------
# Provider adapter filtering (L3)
# ---------------------------------------------------------------------------


class TestCaptureConfigWithProviders:
    def test_l3_on_captures_all_metadata(self, mock_client, capture_trace):
        from layerlens.instrument.adapters.providers.openai import OpenAIProvider

        openai_client = Mock()
        openai_client.chat.completions.create = Mock(return_value=_openai_response())

        provider = OpenAIProvider()
        provider.connect(openai_client)

        @trace(mock_client, capture_config=CaptureConfig.full())
        def my_agent():
            return openai_client.chat.completions.create(
                model="gpt-4", messages=[{"role": "user", "content": "Hi"}]
            ).choices[0].message.content

        my_agent()
        events = capture_trace["events"]
        model_invoke = find_event(events, "model.invoke")
        assert model_invoke["payload"]["parameters"]["model"] == "gpt-4"
        assert model_invoke["payload"]["usage"]["total_tokens"] == 15
        assert model_invoke["payload"]["output_message"]["content"] == "Hello!"

    def test_l3_off_suppresses_model_invoke_keeps_cost(self, mock_client, capture_trace):
        """When L3 is off, model.invoke events are suppressed but cost.record (always-enabled) still fires."""
        from layerlens.instrument.adapters.providers.openai import OpenAIProvider

        openai_client = Mock()
        openai_client.chat.completions.create = Mock(return_value=_openai_response())

        provider = OpenAIProvider()
        provider.connect(openai_client)

        config = CaptureConfig(l3_model_metadata=False)

        @trace(mock_client, capture_config=config)
        def my_agent():
            return openai_client.chat.completions.create(
                model="gpt-4",
                temperature=0.7,
                messages=[{"role": "user", "content": "Hi"}],
            ).choices[0].message.content

        my_agent()
        events = capture_trace["events"]

        # model.invoke is gated by L3 — suppressed
        assert len(find_events(events, "model.invoke")) == 0

        # cost.record is always-enabled — still fires
        cost = find_event(events, "cost.record")
        assert cost["payload"]["prompt_tokens"] == 10

    def test_l3_off_anthropic(self, mock_client, capture_trace):
        """When L3 is off with Anthropic, model.invoke suppressed, cost.record still fires."""
        from layerlens.instrument.adapters.providers.anthropic import AnthropicProvider

        anthropic_client = Mock()

        def _anthropic_response():
            r = Mock()
            block = Mock()
            block.type = "text"
            block.text = "I'm Claude!"
            r.content = [block]
            r.usage = Mock()
            r.usage.input_tokens = 20
            r.usage.output_tokens = 10
            r.model = "claude-3-opus"
            r.stop_reason = "end_turn"
            return r

        anthropic_client.messages.create = Mock(return_value=_anthropic_response())

        provider = AnthropicProvider()
        provider.connect(anthropic_client)

        config = CaptureConfig(l3_model_metadata=False)

        @trace(mock_client, capture_config=config)
        def my_agent():
            return anthropic_client.messages.create(
                model="claude-3-opus",
                max_tokens=1024,
                messages=[{"role": "user", "content": "Hi"}],
            ).content[0].text

        my_agent()
        events = capture_trace["events"]

        # model.invoke suppressed by L3
        assert len(find_events(events, "model.invoke")) == 0

        # cost.record always fires
        cost = find_event(events, "cost.record")
        assert cost["payload"]["input_tokens"] == 20

    def test_capture_content_off(self, mock_client, capture_trace):
        from layerlens.instrument.adapters.providers.openai import OpenAIProvider

        openai_client = Mock()
        openai_client.chat.completions.create = Mock(return_value=_openai_response())

        provider = OpenAIProvider()
        provider.connect(openai_client)

        config = CaptureConfig(capture_content=False)

        @trace(mock_client, capture_config=config)
        def my_agent():
            return openai_client.chat.completions.create(
                model="gpt-4", messages=[{"role": "user", "content": "Hi"}]
            ).choices[0].message.content

        my_agent()
        events = capture_trace["events"]
        model_invoke = find_event(events, "model.invoke")

        # Messages and output_message should be stripped
        assert "messages" not in model_invoke["payload"]
        assert "output_message" not in model_invoke["payload"]

        # But usage and params should still be there
        assert model_invoke["payload"]["usage"]["total_tokens"] == 15
        assert model_invoke["payload"]["parameters"]["model"] == "gpt-4"

    def test_minimal_suppresses_model_invoke(self, mock_client, capture_trace):
        from layerlens.instrument.adapters.providers.openai import OpenAIProvider

        openai_client = Mock()
        openai_client.chat.completions.create = Mock(return_value=_openai_response())

        provider = OpenAIProvider()
        provider.connect(openai_client)

        config = CaptureConfig.minimal()

        @trace(mock_client, capture_config=config)
        def my_agent():
            return openai_client.chat.completions.create(
                model="gpt-4", messages=[{"role": "user", "content": "Hi"}]
            ).choices[0].message.content

        my_agent()
        events = capture_trace["events"]

        # model.invoke gated by L3 — should be suppressed in minimal
        model_invokes = find_events(events, "model.invoke")
        assert len(model_invokes) == 0

        # cost.record is always enabled
        cost_records = find_events(events, "cost.record")
        assert len(cost_records) == 1
