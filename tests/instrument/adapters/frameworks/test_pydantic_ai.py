"""Tests for the PydanticAI adapter using the native Hooks capability API.

Tests use PydanticAI's TestModel to exercise the real agent loop with
hooks firing at each lifecycle point — no monkey-patching or mocking of
PydanticAI internals.
"""

from __future__ import annotations

import asyncio
from typing import Optional

import pytest

pydantic_ai = pytest.importorskip("pydantic_ai")
# The adapter targets the ``Hooks`` capability API, which is not yet released
# in the public ``pydantic-ai`` package. Skip the entire module until the API
# lands — otherwise every test errors on ``_root_capability`` access.
pytest.importorskip("pydantic_ai.capabilities.hooks")

from pydantic_ai import Agent  # noqa: E402
from pydantic_ai.models.test import TestModel  # noqa: E402

from layerlens.instrument._capture_config import CaptureConfig  # noqa: E402
from layerlens.instrument.adapters.frameworks.pydantic_ai import PydanticAIAdapter  # noqa: E402

from .conftest import find_event, find_events, capture_framework_trace  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(
    name: Optional[str] = None,
    output_text: str = "Hello!",
    model_name: str = "test",  # noqa: ARG001 — accepted for API stability; TestModel no longer exposes this kwarg
    tools: Optional[list] = None,
) -> Agent:
    """Create a PydanticAI Agent with TestModel for deterministic testing."""
    agent = Agent(
        model=TestModel(custom_output_text=output_text),
        name=name,
    )
    if tools:
        for tool_fn in tools:
            agent.tool_plain(tool_fn)
    return agent


def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"72F in {city}"


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestPydanticAIAdapterLifecycle:
    def test_connect_injects_hooks(self, mock_client):
        adapter = PydanticAIAdapter(mock_client)
        agent = _make_agent()

        caps_before = len(agent._root_capability.capabilities)
        adapter.connect(target=agent)

        assert adapter.is_connected
        assert len(agent._root_capability.capabilities) == caps_before + 1
        info = adapter.adapter_info()
        assert info.name == "pydantic-ai"
        assert info.adapter_type == "framework"
        assert info.connected is True

        adapter.disconnect()

    def test_disconnect_removes_hooks(self, mock_client):
        adapter = PydanticAIAdapter(mock_client)
        agent = _make_agent()
        caps_before = len(agent._root_capability.capabilities)

        adapter.connect(target=agent)
        adapter.disconnect()

        assert not adapter.is_connected
        assert len(agent._root_capability.capabilities) == caps_before

    def test_connect_without_target_raises(self, mock_client):
        adapter = PydanticAIAdapter(mock_client)
        with pytest.raises(ValueError, match="requires a target agent"):
            adapter.connect()

    def test_connect_without_pydantic_ai_raises(self, mock_client, monkeypatch):
        import layerlens.instrument.adapters.frameworks.pydantic_ai as mod

        monkeypatch.setattr(mod, "_HAS_PYDANTIC_AI", False)
        adapter = PydanticAIAdapter(mock_client)
        with pytest.raises(ImportError, match="pydantic-ai"):
            adapter.connect(target=_make_agent())


# ---------------------------------------------------------------------------
# run_sync
# ---------------------------------------------------------------------------


class TestRunSync:
    def test_basic_run(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = PydanticAIAdapter(mock_client)
        agent = _make_agent(output_text="The weather is sunny")

        adapter.connect(target=agent)
        result = agent.run_sync("What is the weather?")
        adapter.disconnect()

        assert result.output == "The weather is sunny"
        events = uploaded["events"]

        inp = find_event(events, "agent.input")
        assert inp["payload"]["framework"] == "pydantic-ai"
        assert inp["payload"]["input"] == "What is the weather?"

        out = find_event(events, "agent.output")
        assert out["payload"]["status"] == "ok"
        assert out["payload"]["output"] == "The weather is sunny"
        assert out["payload"]["latency_ms"] >= 0
        assert out["payload"]["tokens_prompt"] > 0
        assert out["payload"]["tokens_completion"] > 0

        cost = find_event(events, "cost.record")
        assert cost["payload"]["tokens_total"] > 0

    def test_named_agent(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = PydanticAIAdapter(mock_client)
        agent = _make_agent(name="my_agent", output_text="ok")

        adapter.connect(target=agent)
        agent.run_sync("test")
        adapter.disconnect()

        inp = find_event(uploaded["events"], "agent.input")
        assert inp["payload"]["agent_name"] == "my_agent"


# ---------------------------------------------------------------------------
# async run
# ---------------------------------------------------------------------------


class TestRunAsync:
    def test_async_run(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = PydanticAIAdapter(mock_client)
        agent = _make_agent(name="async_agent", output_text="Async result")

        adapter.connect(target=agent)
        result = asyncio.get_event_loop().run_until_complete(agent.run("async test"))
        adapter.disconnect()

        assert result.output == "Async result"

        inp = find_event(uploaded["events"], "agent.input")
        assert inp["payload"]["agent_name"] == "async_agent"
        assert inp["payload"]["input"] == "async test"

        out = find_event(uploaded["events"], "agent.output")
        assert out["payload"]["status"] == "ok"


# ---------------------------------------------------------------------------
# Model invocation events
# ---------------------------------------------------------------------------


class TestModelInvocation:
    def test_model_invoke_emitted(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = PydanticAIAdapter(mock_client)
        agent = _make_agent(output_text="hello")

        adapter.connect(target=agent)
        agent.run_sync("hi")
        adapter.disconnect()

        model_invokes = find_events(uploaded["events"], "model.invoke")
        assert len(model_invokes) >= 1
        # TestModel reports its own model name ("test"); we just assert the
        # adapter captured whatever it was, non-empty.
        assert isinstance(model_invokes[0]["payload"]["model"], str)
        assert model_invokes[0]["payload"]["model"]
        assert model_invokes[0]["payload"]["tokens_prompt"] > 0

    def test_model_invoke_with_tools_has_two_calls(self, mock_client):
        """When a tool is called, TestModel makes 2 model requests:
        first to call the tool, then to produce the final text."""
        uploaded = capture_framework_trace(mock_client)
        adapter = PydanticAIAdapter(mock_client)
        agent = _make_agent(output_text="Done", tools=[get_weather])

        adapter.connect(target=agent)
        agent.run_sync("weather NYC")
        adapter.disconnect()

        model_invokes = find_events(uploaded["events"], "model.invoke")
        assert len(model_invokes) == 2


# ---------------------------------------------------------------------------
# Tool events
# ---------------------------------------------------------------------------


class TestToolEvents:
    def test_tool_call_and_result(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = PydanticAIAdapter(mock_client)
        agent = _make_agent(output_text="Done", tools=[get_weather])

        adapter.connect(target=agent)
        agent.run_sync("weather NYC")
        adapter.disconnect()

        events = uploaded["events"]

        tool_calls = find_events(events, "tool.call")
        assert len(tool_calls) == 1
        assert tool_calls[0]["payload"]["tool_name"] == "get_weather"

        tool_results = find_events(events, "tool.result")
        assert len(tool_results) == 1
        assert tool_results[0]["payload"]["tool_name"] == "get_weather"

    def test_tool_result_has_output(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = PydanticAIAdapter(mock_client, capture_config=CaptureConfig.full())
        agent = _make_agent(output_text="Done", tools=[get_weather])

        adapter.connect(target=agent)
        agent.run_sync("weather NYC")
        adapter.disconnect()

        tool_results = find_events(uploaded["events"], "tool.result")
        assert len(tool_results) == 1
        # The output should contain the tool's return value
        assert "72F" in str(tool_results[0]["payload"]["output"])

    def test_tool_result_has_latency(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = PydanticAIAdapter(mock_client)
        agent = _make_agent(output_text="Done", tools=[get_weather])

        adapter.connect(target=agent)
        agent.run_sync("weather")
        adapter.disconnect()

        tool_results = find_events(uploaded["events"], "tool.result")
        assert len(tool_results) == 1
        assert tool_results[0]["payload"]["latency_ms"] >= 0


# ---------------------------------------------------------------------------
# Span hierarchy
# ---------------------------------------------------------------------------


class TestSpanHierarchy:
    def test_per_step_events_parented_to_root(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = PydanticAIAdapter(mock_client)
        agent = _make_agent(output_text="Done", tools=[get_weather])

        adapter.connect(target=agent)
        agent.run_sync("weather")
        adapter.disconnect()

        events = uploaded["events"]
        root = find_event(events, "agent.input")
        root_span = root["span_id"]

        for evt in find_events(events, "model.invoke"):
            assert evt["parent_span_id"] == root_span
        for evt in find_events(events, "tool.call"):
            assert evt["parent_span_id"] == root_span
        for evt in find_events(events, "tool.result"):
            assert evt["parent_span_id"] == root_span


# ---------------------------------------------------------------------------
# CaptureConfig gating
# ---------------------------------------------------------------------------


class TestCaptureConfigGating:
    def test_no_content_capture_omits_io(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        config = CaptureConfig(capture_content=False)
        adapter = PydanticAIAdapter(mock_client, capture_config=config)
        agent = _make_agent(output_text="done", tools=[get_weather])

        adapter.connect(target=agent)
        agent.run_sync("secret prompt")
        adapter.disconnect()

        events = uploaded["events"]

        inp = find_event(events, "agent.input")
        assert "input" not in inp["payload"]

        tool_calls = find_events(events, "tool.call")
        assert len(tool_calls) >= 1
        assert "input" not in tool_calls[0]["payload"]

        tool_results = find_events(events, "tool.result")
        assert len(tool_results) >= 1
        assert "output" not in tool_results[0]["payload"]

        # cost.record should still exist
        assert len(find_events(events, "cost.record")) == 1

    def test_full_config_includes_content(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        config = CaptureConfig.full()
        adapter = PydanticAIAdapter(mock_client, capture_config=config)
        agent = _make_agent(output_text="Hi Alice", tools=[get_weather])

        adapter.connect(target=agent)
        agent.run_sync("greet Alice")
        adapter.disconnect()

        events = uploaded["events"]

        inp = find_event(events, "agent.input")
        assert inp["payload"]["input"] == "greet Alice"

        out = find_event(events, "agent.output")
        assert out["payload"]["output"] == "Hi Alice"

        tool_calls = find_events(events, "tool.call")
        assert "input" in tool_calls[0]["payload"]


# ---------------------------------------------------------------------------
# Multiple runs
# ---------------------------------------------------------------------------


class TestMultipleRuns:
    def test_sequential_runs_separate_traces(self, mock_client):
        import json

        all_uploads: list = []

        def _capture(path: str) -> None:
            with open(path) as f:
                data = json.load(f)
            all_uploads.append(data[0])

        mock_client.traces.upload.side_effect = _capture

        adapter = PydanticAIAdapter(mock_client)
        agent = _make_agent(output_text="ok")

        adapter.connect(target=agent)
        agent.run_sync("first")
        agent.run_sync("second")
        adapter.disconnect()

        assert len(all_uploads) == 2
        trace_ids = {u["trace_id"] for u in all_uploads}
        assert len(trace_ids) == 2


# ---------------------------------------------------------------------------
# Event structure
# ---------------------------------------------------------------------------


class TestEventStructure:
    def test_event_fields(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = PydanticAIAdapter(mock_client)
        agent = _make_agent(name="test_agent", output_text="ok")

        adapter.connect(target=agent)
        agent.run_sync("hello")
        adapter.disconnect()

        events = uploaded["events"]
        for event in events:
            assert "event_type" in event
            assert "trace_id" in event
            assert "span_id" in event
            assert "sequence_id" in event
            assert "timestamp_ns" in event
            assert "payload" in event

        seq_ids = [e["sequence_id"] for e in events]
        assert seq_ids == sorted(seq_ids)
        assert len(set(seq_ids)) == len(seq_ids)

        trace_ids = {e["trace_id"] for e in events}
        assert len(trace_ids) == 1

    def test_attestation_present(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = PydanticAIAdapter(mock_client)
        agent = _make_agent(output_text="ok")

        adapter.connect(target=agent)
        agent.run_sync("test")
        adapter.disconnect()

        assert uploaded.get("trace_id") is not None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_prompt(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = PydanticAIAdapter(mock_client)
        agent = _make_agent(output_text="ok")

        adapter.connect(target=agent)
        agent.run_sync("")
        adapter.disconnect()

        inp = find_event(uploaded["events"], "agent.input")
        assert inp["payload"]["framework"] == "pydantic-ai"

    def test_pydantic_model_output(self, mock_client):
        from pydantic import BaseModel

        class CityInfo(BaseModel):
            city: str
            temp: int

        uploaded = capture_framework_trace(mock_client)
        adapter = PydanticAIAdapter(mock_client)
        agent = Agent(
            model=TestModel(custom_output_args={"city": "NYC", "temp": 72}),
            output_type=CityInfo,
        )

        adapter.connect(target=agent)
        result = agent.run_sync("weather")
        adapter.disconnect()

        out = find_event(uploaded["events"], "agent.output")
        assert out["payload"]["output"] == {"city": "NYC", "temp": 72}

    def test_zero_token_usage_still_has_tokens(self, mock_client):
        """TestModel always produces some tokens, so we verify they're present."""
        uploaded = capture_framework_trace(mock_client)
        adapter = PydanticAIAdapter(mock_client)
        agent = _make_agent(output_text="ok")

        adapter.connect(target=agent)
        agent.run_sync("test")
        adapter.disconnect()

        out = find_event(uploaded["events"], "agent.output")
        # TestModel always has some token usage
        assert "tokens_prompt" in out["payload"]
        assert len(find_events(uploaded["events"], "cost.record")) == 1

    def test_disconnect_idempotent(self, mock_client):
        adapter = PydanticAIAdapter(mock_client)
        agent = _make_agent()
        adapter.connect(target=agent)
        adapter.disconnect()
        adapter.disconnect()  # should not raise
