"""Tests for the Agno adapter using the real agno package.

Uses a lightweight ``_TestModel`` that subclasses ``agno.models.base.Model``
so we can exercise ``Agent.run()`` / ``Agent.arun()`` without hitting any
external API.
"""

from __future__ import annotations

import asyncio
from typing import Any, Iterator, Optional

import pytest

agno = pytest.importorskip("agno")

from agno.metrics import RunMetrics, ModelMetrics, ToolCallMetrics  # noqa: E402
from agno.agent.agent import Agent  # noqa: E402
from agno.models.base import Model  # noqa: E402
from agno.models.response import ModelResponse, ToolExecution  # noqa: E402

from layerlens.instrument._capture_config import CaptureConfig  # noqa: E402
from layerlens.instrument.adapters.frameworks.agno import (  # noqa: E402
    AgnoAdapter,
    _model_id,
    _extract_tools,
    _extract_tokens,
)

from .conftest import find_event, find_events, capture_framework_trace  # noqa: E402

# ---------------------------------------------------------------------------
# Test model
# ---------------------------------------------------------------------------


class _TestModel(Model):
    """Deterministic model for testing — no network calls."""

    def __init__(
        self,
        content: str = "Hello!",
        input_tokens: int = 10,
        output_tokens: int = 5,
    ) -> None:
        super().__init__(id="test-model", name="TestModel", provider="test")
        self._content = content
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens

    def _make_response(self) -> ModelResponse:
        return ModelResponse(
            content=self._content,
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            total_tokens=self._input_tokens + self._output_tokens,
        )

    def invoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
        return self._make_response()

    async def ainvoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
        return self._make_response()

    def invoke_stream(self, *args: Any, **kwargs: Any) -> Iterator[ModelResponse]:
        yield self._make_response()

    async def ainvoke_stream(self, *args: Any, **kwargs: Any):  # type: ignore[override]
        yield self._make_response()

    def _parse_provider_response(self, response: Any, **kwargs: Any) -> ModelResponse:
        return self._make_response()

    def _parse_provider_response_delta(self, response: Any) -> ModelResponse:
        return self._make_response()

    def response(self, messages: Any, **kwargs: Any) -> ModelResponse:
        resp = self._make_response()
        run_response = kwargs.get("run_response")
        if run_response and run_response.metrics:
            run_response.metrics.input_tokens += resp.input_tokens or 0
            run_response.metrics.output_tokens += resp.output_tokens or 0
            run_response.metrics.total_tokens += resp.total_tokens or 0
        return resp

    async def aresponse(self, messages: Any, **kwargs: Any) -> ModelResponse:
        return self.response(messages, **kwargs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(
    name: str = "test_agent",
    content: str = "Hello!",
    input_tokens: int = 10,
    output_tokens: int = 5,
    model: Optional[_TestModel] = None,
) -> Agent:
    """Create an agno Agent backed by _TestModel."""
    if model is None:
        model = _TestModel(content=content, input_tokens=input_tokens, output_tokens=output_tokens)
    return Agent(model=model, name=name)


def _connect_and_run(
    mock_client: Any,
    *,
    agent: Optional[Agent] = None,
    config: Optional[CaptureConfig] = None,
    message: str = "hello",
) -> dict:
    """Connect an adapter to an agent, run it, and return uploaded events."""
    if agent is None:
        agent = _make_agent()
    uploaded = capture_framework_trace(mock_client)
    adapter = AgnoAdapter(mock_client, capture_config=config)
    adapter.connect(target=agent)
    agent.run(message)
    return uploaded


def _connect_and_arun(
    mock_client: Any,
    *,
    agent: Optional[Agent] = None,
    config: Optional[CaptureConfig] = None,
    message: str = "hello",
) -> dict:
    """Connect an adapter, arun the agent, and return uploaded events."""
    if agent is None:
        agent = _make_agent()
    uploaded = capture_framework_trace(mock_client)
    adapter = AgnoAdapter(mock_client, capture_config=config)
    adapter.connect(target=agent)
    asyncio.get_event_loop().run_until_complete(agent.arun(message))
    return uploaded


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_connect_disconnect(self, mock_client):
        agent = _make_agent()
        adapter = AgnoAdapter(mock_client)
        returned = adapter.connect(target=agent)
        assert returned is agent
        assert adapter.is_connected
        adapter.disconnect()
        assert not adapter.is_connected

    def test_raises_when_agno_missing(self, mock_client, monkeypatch):
        import layerlens.instrument.adapters.frameworks.agno as _mod

        monkeypatch.setattr(_mod, "_HAS_AGNO", False)
        with pytest.raises(ImportError, match="agno"):
            AgnoAdapter(mock_client).connect(target=_make_agent())

    def test_connect_with_no_target(self, mock_client):
        adapter = AgnoAdapter(mock_client)
        adapter.connect(target=None)
        assert adapter.is_connected

    def test_adapter_info(self, mock_client):
        adapter = AgnoAdapter(mock_client)
        assert adapter.adapter_info().name == "agno"
        assert not adapter.adapter_info().connected

    def test_disconnect_restores_originals(self, mock_client):
        agent = _make_agent()
        adapter = AgnoAdapter(mock_client)
        adapter.connect(target=agent)
        # While connected, run should have the _layerlens_original marker
        assert hasattr(agent.run, "_layerlens_original")
        assert hasattr(agent.arun, "_layerlens_original")
        adapter.disconnect()
        # After disconnect, the marker should be gone (originals restored)
        assert not hasattr(agent.run, "_layerlens_original")
        assert not hasattr(agent.arun, "_layerlens_original")

    def test_double_instrument_is_idempotent(self, mock_client):
        agent = _make_agent()
        adapter = AgnoAdapter(mock_client)
        adapter.connect(target=agent)
        first_run = agent.run
        adapter._instrument_agent(agent)
        assert agent.run is first_run


# ---------------------------------------------------------------------------
# Sync run() -- agent I/O
# ---------------------------------------------------------------------------


class TestSyncAgentIO:
    def test_input_and_output(self, mock_client):
        agent = _make_agent(content="world")
        uploaded = _connect_and_run(mock_client, agent=agent, message="hello")
        events = uploaded["events"]

        inp = find_event(events, "agent.input")
        assert inp["payload"]["agent_name"] == "test_agent"
        assert inp["payload"]["framework"] == "agno"
        assert inp["payload"]["input"] == "hello"
        assert inp["payload"]["model"] == "test-model"

        out = find_event(events, "agent.output")
        assert out["payload"]["output"] == "world"
        assert out["payload"]["latency_ms"] > 0
        assert out["payload"]["model"] == "test-model"

    def test_content_gating(self, mock_client):
        agent = _make_agent(content="secret")
        uploaded = _connect_and_run(
            mock_client,
            agent=agent,
            config=CaptureConfig(capture_content=False),
        )
        events = uploaded["events"]
        assert "input" not in find_event(events, "agent.input")["payload"]
        assert "output" not in find_event(events, "agent.output")["payload"]

    def test_error_propagates(self, mock_client):
        agent = _make_agent()
        uploaded = capture_framework_trace(mock_client)
        adapter = AgnoAdapter(mock_client)
        adapter.connect(target=agent)

        # Sabotage the original run to raise
        original = agent.run._layerlens_original

        def _boom(*a: Any, **kw: Any) -> Any:
            raise RuntimeError("boom")

        agent.run._layerlens_original = _boom
        # Re-wrap with the sabotaged original
        adapter._unwrap_agent(agent)
        adapter._originals.pop(id(agent), None)
        agent.run = _boom
        adapter._instrument_agent(agent)

        with pytest.raises(RuntimeError, match="boom"):
            agent.run("fail")

        out = find_event(uploaded["events"], "agent.output")
        assert out["payload"]["error"] == "boom"
        assert out["payload"]["error_type"] == "RuntimeError"

    def test_none_result(self, mock_client):
        agent = _make_agent()
        uploaded = capture_framework_trace(mock_client)
        adapter = AgnoAdapter(mock_client)
        adapter.connect(target=agent)

        # Replace original run with one that returns None
        adapter._unwrap_agent(agent)
        adapter._originals.pop(id(agent), None)
        agent.run = lambda *a, **kw: None
        adapter._instrument_agent(agent)
        agent.run("hi")

        events = uploaded["events"]
        assert find_event(events, "agent.input") is not None
        assert find_event(events, "agent.output") is not None
        assert len(find_events(events, "model.invoke")) == 0

    def test_fallback_agent_name(self, mock_client):
        agent = _make_agent()
        agent.name = None
        uploaded = _connect_and_run(mock_client, agent=agent)
        out = find_event(uploaded["events"], "agent.output")
        assert out["payload"]["agent_name"] == "agno_agent"

    def test_not_connected_passthrough(self, mock_client):
        agent = _make_agent()
        adapter = AgnoAdapter(mock_client)
        adapter.connect(target=agent)
        adapter.disconnect()
        # After disconnect, run should still work (calls original)
        result = agent.run("hi")
        assert result is not None


# ---------------------------------------------------------------------------
# Async arun()
# ---------------------------------------------------------------------------


class TestAsyncRun:
    def test_arun_emits_agent_io(self, mock_client):
        agent = _make_agent(name="async_agent", content="async world")
        uploaded = _connect_and_arun(mock_client, agent=agent)
        events = uploaded["events"]
        inp = find_event(events, "agent.input")
        assert inp["payload"]["agent_name"] == "async_agent"
        out = find_event(events, "agent.output")
        assert out["payload"]["output"] == "async world"

    def test_arun_error_propagates(self, mock_client):
        agent = _make_agent()
        uploaded = capture_framework_trace(mock_client)
        adapter = AgnoAdapter(mock_client)
        adapter.connect(target=agent)

        # Replace arun with one that raises
        adapter._unwrap_agent(agent)
        adapter._originals.pop(id(agent), None)

        async def _boom(*a: Any, **kw: Any) -> Any:
            raise ValueError("async boom")

        agent.arun = _boom
        adapter._instrument_agent(agent)

        with pytest.raises(ValueError, match="async boom"):
            asyncio.get_event_loop().run_until_complete(agent.arun("fail"))

        out = find_event(uploaded["events"], "agent.output")
        assert out["payload"]["error"] == "async boom"


# ---------------------------------------------------------------------------
# Model invoke + cost record
# ---------------------------------------------------------------------------


class TestModelInvoke:
    def test_model_invoke_emitted(self, mock_client):
        agent = _make_agent(input_tokens=100, output_tokens=50)
        uploaded = _connect_and_run(mock_client, agent=agent)
        events = uploaded["events"]
        invoke = find_event(events, "model.invoke")
        assert invoke["payload"]["model"] == "test-model"
        assert invoke["payload"]["tokens_prompt"] == 100
        assert invoke["payload"]["tokens_completion"] == 50
        assert invoke["payload"]["tokens_total"] == 150

    def test_cost_record_emitted(self, mock_client):
        agent = _make_agent(input_tokens=100, output_tokens=50)
        uploaded = _connect_and_run(mock_client, agent=agent)
        events = uploaded["events"]
        cost = find_event(events, "cost.record")
        assert cost["payload"]["tokens_total"] == 150
        assert cost["payload"]["model"] == "test-model"
        invoke = find_event(events, "model.invoke")
        assert cost["parent_span_id"] == invoke["parent_span_id"]

    def test_no_metrics_skips_cost(self, mock_client):
        """When result has no metrics, no cost.record should be emitted."""
        agent = _make_agent()
        uploaded = capture_framework_trace(mock_client)
        adapter = AgnoAdapter(mock_client)
        adapter.connect(target=agent)

        # Replace run with one that returns a result with no metrics
        adapter._unwrap_agent(agent)
        adapter._originals.pop(id(agent), None)

        class _NoMetricsResult:
            content = "ok"
            metrics = None
            tools = None

        agent.run = lambda *a, **kw: _NoMetricsResult()
        adapter._instrument_agent(agent)
        agent.run("hi")

        assert len(find_events(uploaded["events"], "cost.record")) == 0

    def test_zero_tokens_skips_cost(self, mock_client):
        agent = _make_agent(input_tokens=0, output_tokens=0)
        uploaded = _connect_and_run(mock_client, agent=agent)
        assert len(find_events(uploaded["events"], "cost.record")) == 0

    def test_detail_metrics_fallback(self, mock_client):
        """When top-level tokens are absent, adapter falls back to details."""
        agent = _make_agent()
        uploaded = capture_framework_trace(mock_client)
        adapter = AgnoAdapter(mock_client)
        adapter.connect(target=agent)

        # Replace run with result whose metrics use details
        adapter._unwrap_agent(agent)
        adapter._originals.pop(id(agent), None)

        class _DetailResult:
            content = "ok"
            tools = None

            class metrics:
                input_tokens = None
                output_tokens = None
                details = {
                    "openai": [
                        ModelMetrics(input_tokens=40, output_tokens=20),
                        ModelMetrics(input_tokens=60, output_tokens=30),
                    ]
                }

        agent.run = lambda *a, **kw: _DetailResult()
        adapter._instrument_agent(agent)
        agent.run("hi")

        cost = find_event(uploaded["events"], "cost.record")
        assert cost["payload"]["tokens_prompt"] == 100
        assert cost["payload"]["tokens_completion"] == 50
        assert cost["payload"]["tokens_total"] == 150


# ---------------------------------------------------------------------------
# Tool calls
# ---------------------------------------------------------------------------


class TestToolCalls:
    def test_tool_call_and_result(self, mock_client):
        agent = _make_agent()
        uploaded = capture_framework_trace(mock_client)
        adapter = AgnoAdapter(mock_client)
        adapter.connect(target=agent)

        # Replace run with result that has tool executions
        adapter._unwrap_agent(agent)
        adapter._originals.pop(id(agent), None)

        te = ToolExecution(
            tool_name="web_search",
            tool_args={"query": "AI"},
            result="Found 10 results",
            metrics=ToolCallMetrics(duration=0.5),
        )

        class _ToolResult:
            content = "ok"
            metrics = RunMetrics(input_tokens=10, output_tokens=5, total_tokens=15)
            tools = [te]

        agent.run = lambda *a, **kw: _ToolResult()
        adapter._instrument_agent(agent)
        agent.run("search")

        events = uploaded["events"]

        call = find_event(events, "tool.call")
        assert call["payload"]["tool_name"] == "web_search"
        assert call["payload"]["input"] == {"query": "AI"}

        tr = find_event(events, "tool.result")
        assert tr["payload"]["tool_name"] == "web_search"
        assert tr["payload"]["output"] == "Found 10 results"
        assert tr["payload"]["latency_ms"] == 500.0

    def test_tool_content_gating(self, mock_client):
        agent = _make_agent()
        uploaded = capture_framework_trace(mock_client)
        adapter = AgnoAdapter(mock_client, capture_config=CaptureConfig(capture_content=False))
        adapter.connect(target=agent)

        adapter._unwrap_agent(agent)
        adapter._originals.pop(id(agent), None)

        te = ToolExecution(
            tool_name="search",
            tool_args={"q": "secret"},
            result="classified",
        )

        class _ToolResult:
            content = "ok"
            metrics = None
            tools = [te]

        agent.run = lambda *a, **kw: _ToolResult()
        adapter._instrument_agent(agent)
        agent.run("hi")

        events = uploaded["events"]
        assert "input" not in find_event(events, "tool.call")["payload"]
        assert "output" not in find_event(events, "tool.result")["payload"]

    def test_multiple_tools(self, mock_client):
        agent = _make_agent()
        uploaded = capture_framework_trace(mock_client)
        adapter = AgnoAdapter(mock_client)
        adapter.connect(target=agent)

        adapter._unwrap_agent(agent)
        adapter._originals.pop(id(agent), None)

        class _ToolResult:
            content = "ok"
            metrics = None
            tools = [
                ToolExecution(tool_name="search"),
                ToolExecution(tool_name="calculator"),
            ]

        agent.run = lambda *a, **kw: _ToolResult()
        adapter._instrument_agent(agent)
        agent.run("hi")

        events = uploaded["events"]
        assert len(find_events(events, "tool.call")) == 2
        assert len(find_events(events, "tool.result")) == 2

    def test_no_tools_skips_tool_events(self, mock_client):
        agent = _make_agent()
        uploaded = _connect_and_run(mock_client, agent=agent)
        # Real agent run returns empty tools list, no ToolExecution objects
        assert len(find_events(uploaded["events"], "tool.call")) == 0


# ---------------------------------------------------------------------------
# Trace integrity
# ---------------------------------------------------------------------------


class TestTraceIntegrity:
    def test_shared_trace_id(self, mock_client):
        agent = _make_agent(input_tokens=10, output_tokens=5)
        uploaded = _connect_and_run(mock_client, agent=agent)
        trace_ids = {e["trace_id"] for e in uploaded["events"]}
        assert len(trace_ids) == 1

    def test_span_hierarchy(self, mock_client):
        agent = _make_agent(input_tokens=10, output_tokens=5)
        uploaded = _connect_and_run(mock_client, agent=agent)
        events = uploaded["events"]

        root = find_event(events, "agent.input")["span_id"]
        assert find_event(events, "agent.output")["span_id"] == root
        assert find_event(events, "model.invoke")["parent_span_id"] == root

    def test_monotonic_sequence_ids(self, mock_client):
        agent = _make_agent(input_tokens=10, output_tokens=5)
        uploaded = _connect_and_run(mock_client, agent=agent)
        seq = [e["sequence_id"] for e in uploaded["events"]]
        assert seq == sorted(seq)

    def test_flush_produces_trace(self, mock_client):
        agent = _make_agent()
        uploaded = _connect_and_run(mock_client, agent=agent)
        assert uploaded.get("trace_id") is not None


# ---------------------------------------------------------------------------
# Helpers (module-level pure functions)
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_model_id_with_real_model(self):
        agent = _make_agent()
        assert _model_id(agent) == "test-model"

    def test_model_id_none(self):
        agent = _make_agent()
        agent.model = None
        assert _model_id(agent) is None

    def test_model_id_str_fallback(self):
        class _NoIdModel:
            id = None

            def __str__(self) -> str:
                return "claude-3"

        agent = _make_agent()
        agent.model = _NoIdModel()
        assert _model_id(agent) == "claude-3"

    def test_extract_tokens_with_real_metrics(self):
        """Use real agno RunMetrics."""

        class _Result:
            metrics = RunMetrics(input_tokens=10, output_tokens=5, total_tokens=15)

        tokens = _extract_tokens(_Result())
        assert tokens == {"tokens_prompt": 10, "tokens_completion": 5, "tokens_total": 15}

    def test_extract_tokens_none(self):
        class _Result:
            metrics = None

        assert _extract_tokens(_Result()) == {}

    def test_extract_tokens_details(self):
        """Use real agno ModelMetrics in the details fallback."""

        class _Result:
            class metrics:
                input_tokens = None
                output_tokens = None
                details = {
                    "openai": [
                        ModelMetrics(input_tokens=20, output_tokens=10),
                        ModelMetrics(input_tokens=30, output_tokens=15),
                    ],
                }

        tokens = _extract_tokens(_Result())
        assert tokens["tokens_prompt"] == 50
        assert tokens["tokens_completion"] == 25
        assert tokens["tokens_total"] == 75

    def test_extract_tools_empty(self):
        class _Result:
            tools = None

        assert _extract_tools(_Result()) == []

    def test_extract_tools_with_real_tool_execution(self):
        """Use real agno ToolExecution and ToolCallMetrics."""
        te = ToolExecution(
            tool_name="calc",
            tool_args={"x": 1},
            result="42",
            metrics=ToolCallMetrics(duration=0.1),
        )

        class _Result:
            tools = [te]

        tools = _extract_tools(_Result())
        assert len(tools) == 1
        assert tools[0]["tool_name"] == "calc"
        assert tools[0]["latency_ms"] == pytest.approx(100.0)
