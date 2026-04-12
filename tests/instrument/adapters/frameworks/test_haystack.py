"""Tests for the Haystack adapter.

Mocks ``haystack.tracing`` since haystack-ai is not installed in the test env.
Drives the tracer with exact operation names and tags that Haystack uses.
"""

from __future__ import annotations

import threading
from typing import Any, Optional
from unittest.mock import Mock, MagicMock

import pytest

import layerlens.instrument.adapters.frameworks.haystack as _mod
from layerlens.instrument._capture_config import CaptureConfig
from layerlens.instrument.adapters.frameworks.haystack import (
    HaystackAdapter,
    _NullSpan,
    _extract_model,
    _extract_usage,
    _LayerLensTracer,
)

from .conftest import find_event, find_events, capture_framework_trace


@pytest.fixture(autouse=True)
def mock_haystack_tracing():
    mock_tracing = MagicMock()
    _mod._hs_tracing = mock_tracing
    _mod._HAS_HAYSTACK = True
    yield mock_tracing
    _mod._HAS_HAYSTACK = False


def _make_adapter(client: Any, config: Optional[CaptureConfig] = None) -> HaystackAdapter:
    adapter = HaystackAdapter(client, capture_config=config)
    adapter.connect()
    return adapter


def _simulate_pipeline(
    tracer: _LayerLensTracer,
    *,
    input_data: Any = None,
    output_data: Any = None,
    components: Optional[list] = None,
    error: Optional[str] = None,
    max_runs: Optional[int] = None,
) -> None:
    with tracer.trace("haystack.pipeline.run") as pipe:
        if input_data is not None:
            pipe.set_content_tag("haystack.pipeline.input_data", input_data)
        if max_runs is not None:
            pipe.set_tag("haystack.pipeline.max_runs_per_component", max_runs)

        for comp in components or []:
            with tracer.trace("haystack.component.run") as cs:
                cs.set_tag("haystack.component.name", comp["name"])
                cs.set_tag("haystack.component.type", comp["type"])
                if comp.get("model"):
                    cs.set_tag("haystack.model", comp["model"])
                if comp.get("input") is not None:
                    cs.set_content_tag("haystack.component.input", comp["input"])
                if comp.get("output") is not None:
                    cs.set_content_tag("haystack.component.output", comp["output"])
                if comp.get("error"):
                    cs.set_tag("error", True)
                    cs.set_tag("error.message", comp["error"])

        if output_data is not None:
            pipe.set_content_tag("haystack.pipeline.output_data", output_data)
        if error:
            pipe.set_tag("error", True)
            pipe.set_tag("error.message", error)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_connect_disconnect(self, mock_client):
        adapter = HaystackAdapter(mock_client)
        adapter.connect()
        assert adapter.is_connected
        adapter.disconnect()
        assert not adapter.is_connected

    def test_installs_and_restores_tracer(self, mock_client, mock_haystack_tracing):
        original = Mock(spec=[])
        mock_haystack_tracing.tracer.actual_tracer = original
        adapter = HaystackAdapter(mock_client)
        adapter.connect()
        assert isinstance(mock_haystack_tracing.tracer.actual_tracer, _LayerLensTracer)
        adapter.disconnect()
        assert mock_haystack_tracing.tracer.actual_tracer is original

    def test_raises_when_haystack_missing(self, mock_client):
        _mod._HAS_HAYSTACK = False
        with pytest.raises(ImportError, match="haystack"):
            HaystackAdapter(mock_client).connect()

    def test_adapter_info(self, mock_client):
        adapter = HaystackAdapter(mock_client)
        assert adapter.adapter_info().name == "haystack"
        assert not adapter.adapter_info().connected
        adapter.connect()
        assert adapter.adapter_info().connected
        adapter.disconnect()


# ---------------------------------------------------------------------------
# Pipeline spans
# ---------------------------------------------------------------------------


class TestPipelineSpans:
    def test_input_and_output(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = _make_adapter(mock_client)
        _simulate_pipeline(adapter._tracer, input_data={"q": "hello"}, output_data={"a": "world"})

        inp = find_event(uploaded["events"], "agent.input")
        assert inp["payload"]["framework"] == "haystack"
        assert inp["payload"]["input"] == {"q": "hello"}

        out = find_event(uploaded["events"], "agent.output")
        assert out["payload"]["output"] == {"a": "world"}
        assert out["payload"]["latency_ms"] > 0
        adapter.disconnect()

    def test_content_gating(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = _make_adapter(mock_client, config=CaptureConfig(capture_content=False))
        _simulate_pipeline(adapter._tracer, input_data="secret", output_data="classified")

        assert "input" not in find_event(uploaded["events"], "agent.input")["payload"]
        assert "output" not in find_event(uploaded["events"], "agent.output")["payload"]
        adapter.disconnect()

    def test_error(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = _make_adapter(mock_client)
        _simulate_pipeline(adapter._tracer, error="Pipeline failed")
        assert find_event(uploaded["events"], "agent.output")["payload"]["error"] == "Pipeline failed"
        adapter.disconnect()

    def test_exception(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = _make_adapter(mock_client)
        with pytest.raises(ValueError):
            with adapter._tracer.trace("haystack.pipeline.run"):
                raise ValueError("boom")
        assert find_event(uploaded["events"], "agent.output")["payload"]["error"] == "boom"
        adapter.disconnect()

    def test_max_runs_tag(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = _make_adapter(mock_client)
        _simulate_pipeline(adapter._tracer, max_runs=100)
        assert find_event(uploaded["events"], "agent.input")["payload"]["max_runs_per_component"] == 100
        adapter.disconnect()

    def test_flushes_trace(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = _make_adapter(mock_client)
        _simulate_pipeline(adapter._tracer)
        assert uploaded.get("trace_id") is not None
        adapter.disconnect()


# ---------------------------------------------------------------------------
# Generator components
# ---------------------------------------------------------------------------


class TestGeneratorComponents:
    def _gen_component(self, **overrides: Any) -> dict:
        base = {
            "name": "llm",
            "type": "OpenAIChatGenerator",
            "model": "gpt-4o",
            "output": {
                "replies": ["answer"],
                "meta": [{"model": "gpt-4o", "usage": {"prompt_tokens": 100, "completion_tokens": 50}}],
            },
        }
        base.update(overrides)
        return base

    def test_model_invoke_with_tokens(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = _make_adapter(mock_client)
        _simulate_pipeline(adapter._tracer, components=[self._gen_component()])

        invoke = find_event(uploaded["events"], "model.invoke")
        assert invoke["payload"]["model"] == "gpt-4o"
        assert invoke["payload"]["tokens_prompt"] == 100
        assert invoke["payload"]["tokens_completion"] == 50
        assert invoke["payload"]["tokens_total"] == 150
        assert invoke["span_name"] == "component:llm"
        adapter.disconnect()

    def test_cost_record(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = _make_adapter(mock_client)
        _simulate_pipeline(adapter._tracer, components=[self._gen_component()])

        cost = find_event(uploaded["events"], "cost.record")
        assert cost["payload"]["tokens_total"] == 150
        assert cost["payload"]["model"] == "gpt-4o"
        # Parented to model.invoke span
        assert cost["parent_span_id"] == find_event(uploaded["events"], "model.invoke")["span_id"]
        adapter.disconnect()

    def test_chatgenerator_classified(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = _make_adapter(mock_client)
        _simulate_pipeline(
            adapter._tracer,
            components=[
                {"name": "c", "type": "haystack.components.generators.chat.openai.OpenAIChatGenerator"},
            ],
        )
        assert len(find_events(uploaded["events"], "model.invoke")) == 1
        adapter.disconnect()

    def test_content_gating(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = _make_adapter(mock_client, config=CaptureConfig(capture_content=False))
        _simulate_pipeline(adapter._tracer, components=[self._gen_component()])

        invoke = find_event(uploaded["events"], "model.invoke")
        assert "input" not in invoke["payload"]
        assert "output" not in invoke["payload"]
        adapter.disconnect()

    def test_model_from_output_meta(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = _make_adapter(mock_client)
        _simulate_pipeline(
            adapter._tracer,
            components=[
                {
                    "name": "llm",
                    "type": "ChatGenerator",
                    "output": {
                        "replies": ["ok"],
                        "meta": [{"model": "claude-3", "usage": {"prompt_tokens": 5, "completion_tokens": 3}}],
                    },
                }
            ],
        )
        assert find_event(uploaded["events"], "model.invoke")["payload"]["model"] == "claude-3"
        adapter.disconnect()


# ---------------------------------------------------------------------------
# Non-generator components
# ---------------------------------------------------------------------------


class TestToolComponents:
    def test_tool_call_and_result(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = _make_adapter(mock_client)
        _simulate_pipeline(
            adapter._tracer,
            components=[
                {"name": "my_retriever", "type": "BM25Retriever", "input": {"q": "find"}, "output": {"docs": ["d1"]}},
            ],
        )

        call = find_event(uploaded["events"], "tool.call")
        assert call["payload"]["tool_name"] == "my_retriever"
        assert call["payload"]["component_type"] == "BM25Retriever"
        assert call["payload"]["input"] == {"q": "find"}

        result = find_event(uploaded["events"], "tool.result")
        assert result["payload"]["output"] == {"docs": ["d1"]}
        assert result["payload"]["latency_ms"] > 0
        adapter.disconnect()

    def test_content_gating(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = _make_adapter(mock_client, config=CaptureConfig(capture_content=False))
        _simulate_pipeline(
            adapter._tracer,
            components=[
                {"name": "r", "type": "Retriever", "input": "secret", "output": "classified"},
            ],
        )
        assert "input" not in find_event(uploaded["events"], "tool.call")["payload"]
        assert "output" not in find_event(uploaded["events"], "tool.result")["payload"]
        adapter.disconnect()

    def test_component_error(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = _make_adapter(mock_client)
        _simulate_pipeline(
            adapter._tracer,
            components=[
                {"name": "broken", "type": "Custom", "error": "crashed"},
            ],
        )
        assert find_event(uploaded["events"], "tool.result")["payload"]["error"] == "crashed"
        adapter.disconnect()

    def test_prompt_builder_is_tool(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = _make_adapter(mock_client)
        _simulate_pipeline(
            adapter._tracer,
            components=[
                {"name": "pb", "type": "PromptBuilder", "input": {"tpl": "hi"}, "output": {"prompt": "hi"}},
            ],
        )
        assert len(find_events(uploaded["events"], "tool.call")) == 1
        assert len([e for e in uploaded["events"] if e["event_type"] == "agent.code"]) == 0
        adapter.disconnect()


# ---------------------------------------------------------------------------
# Full pipeline with multiple components
# ---------------------------------------------------------------------------


class TestFullPipeline:
    def test_rag_pipeline(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = _make_adapter(mock_client)
        _simulate_pipeline(
            adapter._tracer,
            input_data={"query": "test"},
            output_data={"answer": "result"},
            components=[
                {"name": "retriever", "type": "BM25Retriever"},
                {"name": "prompt_builder", "type": "PromptBuilder"},
                {
                    "name": "llm",
                    "type": "OpenAIChatGenerator",
                    "model": "gpt-4o",
                    "output": {
                        "replies": ["answer"],
                        "meta": [{"usage": {"prompt_tokens": 20, "completion_tokens": 10}}],
                    },
                },
            ],
        )
        events = uploaded["events"]
        assert len(find_events(events, "agent.input")) == 1
        assert len(find_events(events, "agent.output")) == 1
        assert len(find_events(events, "tool.call")) == 2
        assert len(find_events(events, "model.invoke")) == 1
        assert len(find_events(events, "cost.record")) == 1
        adapter.disconnect()


# ---------------------------------------------------------------------------
# Trace integrity
# ---------------------------------------------------------------------------


class TestTraceIntegrity:
    def test_shared_trace_id(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = _make_adapter(mock_client)
        _simulate_pipeline(
            adapter._tracer,
            components=[
                {"name": "r", "type": "Retriever"},
                {
                    "name": "g",
                    "type": "ChatGenerator",
                    "output": {"replies": ["ok"], "meta": [{"usage": {"prompt_tokens": 1, "completion_tokens": 1}}]},
                },
            ],
        )
        assert len({e["trace_id"] for e in uploaded["events"]}) == 1
        adapter.disconnect()

    def test_monotonic_sequence_ids(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = _make_adapter(mock_client)
        _simulate_pipeline(adapter._tracer, components=[{"name": "c", "type": "T"}])
        seq = [e["sequence_id"] for e in uploaded["events"]]
        assert seq == sorted(seq)
        adapter.disconnect()

    def test_span_hierarchy(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = _make_adapter(mock_client)
        _simulate_pipeline(
            adapter._tracer,
            components=[
                {"name": "ret", "type": "Retriever"},
                {"name": "gen", "type": "ChatGenerator"},
            ],
        )
        events = uploaded["events"]
        root = find_event(events, "agent.input")["span_id"]
        assert find_event(events, "tool.call")["parent_span_id"] == root
        assert find_event(events, "model.invoke")["parent_span_id"] == root
        adapter.disconnect()

    def test_internal_spans_skipped(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = _make_adapter(mock_client)
        with adapter._tracer.trace("haystack.pipeline.run"):
            with adapter._tracer.trace("haystack.internal.something") as s:
                s.set_tag("x", "y")
        assert len(find_events(uploaded["events"], "tool.call")) == 0
        adapter.disconnect()


# ---------------------------------------------------------------------------
# Tracer protocol
# ---------------------------------------------------------------------------


class TestTracerProtocol:
    def test_current_span(self, mock_client):
        adapter = _make_adapter(mock_client)
        assert isinstance(adapter._tracer.current_span(), _NullSpan)
        with adapter._tracer.trace("haystack.pipeline.run") as span:
            assert adapter._tracer.current_span() is span
        adapter.disconnect()

    def test_nested_parent_tracking(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = _make_adapter(mock_client)
        with adapter._tracer.trace("haystack.pipeline.run") as pipe:
            with adapter._tracer.trace("haystack.component.run") as comp:
                comp.set_tag("haystack.component.type", "Retriever")
                comp.set_tag("haystack.component.name", "r")
                assert comp._parent_span_id == pipe.span_id
        assert find_event(uploaded["events"], "tool.call")["parent_span_id"] == pipe.span_id
        adapter.disconnect()

    def test_span_protocol_methods(self, mock_client):
        adapter = _make_adapter(mock_client)
        with adapter._tracer.trace("haystack.pipeline.run") as span:
            assert span.raw_span() is None
            data = span.get_correlation_data_for_logs()
            assert data["span_id"] == span.span_id
        adapter.disconnect()


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_pipelines(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = _make_adapter(mock_client)
        errors = []

        def _run(tid: int) -> None:
            try:
                _simulate_pipeline(
                    adapter._tracer,
                    input_data={"t": tid},
                    output_data={"r": tid},
                    components=[{"name": f"c_{tid}", "type": "T"}],
                )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_run, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        adapter.disconnect()

        assert len(find_events(uploaded["events"], "agent.input")) == 5
        assert len(find_events(uploaded["events"], "agent.output")) == 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_extract_model_from_tag(self):
        assert _extract_model({"haystack.model": "gpt-4o"}) == "gpt-4o"

    def test_extract_model_from_meta(self):
        assert _extract_model({"haystack.component.output": {"meta": [{"model": "claude-3"}]}}) == "claude-3"

    def test_extract_model_none(self):
        assert _extract_model({}) is None

    def test_extract_usage(self):
        assert _extract_usage({"meta": [{"usage": {"prompt_tokens": 10}}]}) == {"prompt_tokens": 10}

    def test_extract_usage_none(self):
        assert _extract_usage({}) is None

    def test_nullspan_noop(self):
        ns = _NullSpan()
        ns.set_tag("k", "v")
        ns.set_content_tag("k", "v")
        assert ns.raw_span() is None
        assert ns.get_correlation_data_for_logs() == {}
