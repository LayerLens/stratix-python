"""Tests for Semantic Kernel adapter lifecycle."""

import pytest

from layerlens.instrument.adapters.semantic_kernel.lifecycle import SemanticKernelAdapter
from layerlens.instrument.adapters._base import AdapterStatus, AdapterCapability


class MockStratix:
    def __init__(self):
        self.events = []

    def emit(self, event_type, payload=None):
        self.events.append({"event_type": event_type, "payload": payload})


class MockKernel:
    """Mock SK Kernel."""

    def __init__(self):
        self.plugins = {"MathPlugin": {}, "WebPlugin": {}}
        self._stratix_adapter = None
        self._stratix_filters = None
        self._filters = {}

    def add_filter(self, filter_type, filter_obj):
        self._filters.setdefault(filter_type, []).append(filter_obj)


class TestSemanticKernelAdapterLifecycle:

    def test_connect_sets_healthy(self):
        adapter = SemanticKernelAdapter()
        adapter.connect()
        assert adapter.is_connected
        assert adapter.status == AdapterStatus.HEALTHY

    def test_disconnect(self):
        adapter = SemanticKernelAdapter()
        adapter.connect()
        adapter.disconnect()
        assert not adapter.is_connected
        assert adapter.status == AdapterStatus.DISCONNECTED

    def test_health_check(self):
        adapter = SemanticKernelAdapter()
        adapter.connect()
        health = adapter.health_check()
        assert health.status == AdapterStatus.HEALTHY
        assert health.framework_name == "semantic_kernel"

    def test_get_adapter_info(self):
        adapter = SemanticKernelAdapter()
        info = adapter.get_adapter_info()
        assert info.name == "SemanticKernelAdapter"
        assert info.framework == "semantic_kernel"
        assert AdapterCapability.TRACE_TOOLS in info.capabilities
        assert AdapterCapability.TRACE_MODELS in info.capabilities
        assert AdapterCapability.TRACE_STATE in info.capabilities

    def test_serialize_for_replay(self):
        adapter = SemanticKernelAdapter()
        trace = adapter.serialize_for_replay()
        assert trace.adapter_name == "SemanticKernelAdapter"
        assert trace.framework == "semantic_kernel"
        assert trace.trace_id is not None


class TestSemanticKernelInstrumentation:

    def test_instrument_kernel_registers_filters(self):
        adapter = SemanticKernelAdapter()
        adapter.connect()
        kernel = MockKernel()
        result = adapter.instrument_kernel(kernel)
        assert result is kernel
        assert kernel._stratix_adapter is adapter
        assert len(kernel._filters.get("function_invocation", [])) == 1
        assert len(kernel._filters.get("prompt_rendering", [])) == 1
        assert len(kernel._filters.get("auto_function_invocation", [])) == 1

    def test_instrument_kernel_discovers_plugins(self):
        stratix = MockStratix()
        adapter = SemanticKernelAdapter(stratix=stratix)
        adapter.connect()
        kernel = MockKernel()
        adapter.instrument_kernel(kernel)
        # Should emit environment.config for each plugin
        config_events = [
            e for e in stratix.events
            if e["event_type"] == "environment.config"
        ]
        assert len(config_events) == 2


class TestSemanticKernelEventEmission:

    def test_on_function_start_returns_context(self):
        adapter = SemanticKernelAdapter()
        adapter.connect()
        ctx = adapter.on_function_start(
            plugin_name="Math",
            function_name="add",
        )
        assert "start_ns" in ctx
        assert ctx["plugin_name"] == "Math"
        assert ctx["function_name"] == "add"

    def test_on_function_end_emits_tool_call(self):
        stratix = MockStratix()
        adapter = SemanticKernelAdapter(stratix=stratix)
        adapter.connect()
        ctx = adapter.on_function_start("Math", "add")
        adapter.on_function_end(ctx, result=42)
        tool_events = [
            e for e in stratix.events if e["event_type"] == "tool.call"
        ]
        assert len(tool_events) == 1
        assert tool_events[0]["payload"]["tool_name"] == "Math.add"

    def test_auto_invoked_function(self):
        stratix = MockStratix()
        adapter = SemanticKernelAdapter(stratix=stratix)
        adapter.connect()
        ctx = adapter.on_function_start("Web", "search", auto_invoked=True)
        adapter.on_function_end(ctx, result="results", auto_invoked=True)
        tool_events = [
            e for e in stratix.events if e["event_type"] == "tool.call"
        ]
        assert len(tool_events) == 1
        assert tool_events[0]["payload"]["auto_invoked"] is True

    def test_on_prompt_render(self):
        from layerlens.instrument.adapters._capture import CaptureConfig
        stratix = MockStratix()
        adapter = SemanticKernelAdapter(stratix=stratix, capture_config=CaptureConfig.full())
        adapter.connect()
        adapter.on_prompt_render(
            template="Summarize {{$text}}",
            rendered_prompt="Summarize the document...",
            function_name="summarize",
        )
        code_events = [
            e for e in stratix.events if e["event_type"] == "agent.code"
        ]
        assert len(code_events) == 1
        assert code_events[0]["payload"]["event_subtype"] == "prompt_render"

    def test_on_model_invoke(self):
        stratix = MockStratix()
        adapter = SemanticKernelAdapter(stratix=stratix)
        adapter.connect()
        adapter.on_model_invoke(
            provider="openai",
            model="gpt-4o",
            prompt_tokens=100,
            completion_tokens=50,
            latency_ms=250.0,
        )
        model_events = [
            e for e in stratix.events if e["event_type"] == "model.invoke"
        ]
        cost_events = [
            e for e in stratix.events if e["event_type"] == "cost.record"
        ]
        assert len(model_events) == 1
        assert len(cost_events) == 1
        assert cost_events[0]["payload"]["total_tokens"] == 150

    def test_on_planner_step(self):
        from layerlens.instrument.adapters._capture import CaptureConfig
        stratix = MockStratix()
        adapter = SemanticKernelAdapter(stratix=stratix, capture_config=CaptureConfig.full())
        adapter.connect()
        adapter.on_planner_step(
            planner_type="sequential",
            step_index=0,
            action="Math.add",
            status="completed",
        )
        code_events = [
            e for e in stratix.events if e["event_type"] == "agent.code"
        ]
        assert len(code_events) == 1
        assert code_events[0]["payload"]["planner_type"] == "sequential"

    def test_on_memory_operation(self):
        stratix = MockStratix()
        adapter = SemanticKernelAdapter(stratix=stratix)
        adapter.connect()
        adapter.on_memory_operation(
            operation="search",
            collection="docs",
            query="What is AI?",
            result_count=5,
        )
        tool_events = [
            e for e in stratix.events if e["event_type"] == "tool.call"
        ]
        assert len(tool_events) == 1
        assert tool_events[0]["payload"]["operation"] == "search"
        assert tool_events[0]["payload"]["tool_name"] == "memory.search"

    def test_on_kernel_invoke_start_end(self):
        stratix = MockStratix()
        adapter = SemanticKernelAdapter(stratix=stratix)
        adapter.connect()
        adapter.on_kernel_invoke_start("What is 2+2?")
        adapter.on_kernel_invoke_end("4")
        input_events = [
            e for e in stratix.events if e["event_type"] == "agent.input"
        ]
        output_events = [
            e for e in stratix.events if e["event_type"] == "agent.output"
        ]
        assert len(input_events) == 1
        assert len(output_events) == 1

    def test_function_end_with_error(self):
        stratix = MockStratix()
        adapter = SemanticKernelAdapter(stratix=stratix)
        adapter.connect()
        ctx = adapter.on_function_start("Calc", "divide")
        adapter.on_function_end(ctx, error=ZeroDivisionError("div by zero"))
        tool_events = [
            e for e in stratix.events if e["event_type"] == "tool.call"
        ]
        assert "error" in tool_events[0]["payload"]

    def test_deduplicates_plugin_config(self):
        stratix = MockStratix()
        adapter = SemanticKernelAdapter(stratix=stratix)
        adapter.connect()
        adapter.on_function_start("Math", "add")
        adapter.on_function_start("Math", "multiply")
        # Should only emit one environment.config for "Math"
        config_events = [
            e for e in stratix.events if e["event_type"] == "environment.config"
        ]
        assert len(config_events) == 1


class TestSemanticKernelFilters:

    def test_function_filter_sync(self):
        from layerlens.instrument.adapters.semantic_kernel.filters import STRATIXFunctionFilter
        stratix = MockStratix()
        adapter = SemanticKernelAdapter(stratix=stratix)
        adapter.connect()
        f = STRATIXFunctionFilter(adapter)
        f.on_function_invocation_sync("Plugin", "func", result="ok")
        tool_events = [
            e for e in stratix.events if e["event_type"] == "tool.call"
        ]
        assert len(tool_events) == 1

    def test_prompt_render_filter_sync(self):
        from layerlens.instrument.adapters.semantic_kernel.filters import STRATIXPromptRenderFilter
        from layerlens.instrument.adapters._capture import CaptureConfig
        stratix = MockStratix()
        adapter = SemanticKernelAdapter(stratix=stratix, capture_config=CaptureConfig.full())
        adapter.connect()
        f = STRATIXPromptRenderFilter(adapter)
        f.on_prompt_render_sync(template="test {{x}}", rendered_prompt="test hello")
        code_events = [
            e for e in stratix.events if e["event_type"] == "agent.code"
        ]
        assert len(code_events) == 1

    def test_auto_function_filter_sync(self):
        from layerlens.instrument.adapters.semantic_kernel.filters import STRATIXAutoFunctionFilter
        stratix = MockStratix()
        adapter = SemanticKernelAdapter(stratix=stratix)
        adapter.connect()
        f = STRATIXAutoFunctionFilter(adapter)
        f.on_auto_function_invocation_sync("Web", "search", result="found")
        tool_events = [
            e for e in stratix.events if e["event_type"] == "tool.call"
        ]
        assert len(tool_events) == 1
        assert tool_events[0]["payload"]["auto_invoked"] is True


class TestSemanticKernelMetadata:

    def test_extract_kernel_metadata(self):
        from layerlens.instrument.adapters.semantic_kernel.metadata import SKMetadataExtractor
        ext = SKMetadataExtractor()
        kernel = MockKernel()
        meta = ext.extract_kernel_metadata(kernel)
        assert meta["plugin_count"] == 2
        assert "MathPlugin" in meta["plugin_names"]
