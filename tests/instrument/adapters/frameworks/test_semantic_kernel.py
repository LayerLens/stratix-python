"""Tests for the Semantic Kernel adapter using the SK filter API.

Tests use real Kernel objects and KernelFunctions. Filters are exercised
either through actual kernel.invoke() calls or by directly invoking the
filter callables with mock contexts.
"""
from __future__ import annotations

import asyncio
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest

sk = pytest.importorskip("semantic_kernel")

from semantic_kernel import Kernel  # noqa: E402
from semantic_kernel.functions import kernel_function  # noqa: E402
from semantic_kernel.filters.filter_types import FilterTypes  # noqa: E402

from layerlens.instrument._capture_config import CaptureConfig  # noqa: E402
from layerlens.instrument.adapters.frameworks.semantic_kernel import (  # noqa: E402
    SemanticKernelAdapter,
    _extract_arguments,
    _extract_function_name,
    _extract_plugin_name,
)

from .conftest import capture_framework_trace, find_event, find_events  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MathPlugin:
    @kernel_function(name="add", description="Add two numbers")
    def add(self, a: int, b: int) -> int:
        return a + b

    @kernel_function(name="divide", description="Divide a by b")
    def divide(self, a: int, b: int) -> float:
        return a / b


class TextPlugin:
    @kernel_function(name="upper", description="Uppercase text")
    def upper(self, text: str) -> str:
        return text.upper()


class MockFunction:
    def __init__(self, name: str = "test_func", plugin_name: str = "TestPlugin"):
        self.name = name
        self.plugin_name = plugin_name


class MockContext:
    def __init__(
        self,
        function: Any = None,
        arguments: Any = None,
        result: Any = None,
        rendered_prompt: Optional[str] = None,
        function_call_content: Any = None,
        function_result: Any = None,
        request_sequence_index: int = 0,
        function_sequence_index: int = 0,
    ):
        self.function = function or MockFunction()
        self.arguments = arguments
        self.result = result
        self.rendered_prompt = rendered_prompt
        self.function_call_content = function_call_content
        self.function_result = function_result
        self.request_sequence_index = request_sequence_index
        self.function_sequence_index = function_sequence_index


class MockFunctionCallContent:
    def __init__(self, arguments: Any = None):
        self.arguments = arguments


class MockFunctionResult:
    def __init__(self, value: Any = None):
        self.value = value


def _run(coro: Any) -> Any:
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_connect_registers_filters(self, mock_client):
        kernel = Kernel()
        adapter = SemanticKernelAdapter(mock_client)
        adapter.connect(target=kernel)

        assert adapter.is_connected
        assert len(kernel.function_invocation_filters) == 1
        assert len(kernel.prompt_rendering_filters) == 1
        assert len(kernel.auto_function_invocation_filters) == 1

        info = adapter.adapter_info()
        assert info.name == "semantic_kernel"
        assert info.adapter_type == "framework"
        assert info.connected is True

        adapter.disconnect()

    def test_disconnect_removes_filters(self, mock_client):
        kernel = Kernel()
        adapter = SemanticKernelAdapter(mock_client)
        adapter.connect(target=kernel)
        adapter.disconnect()

        assert not adapter.is_connected
        assert len(kernel.function_invocation_filters) == 0
        assert len(kernel.prompt_rendering_filters) == 0
        assert len(kernel.auto_function_invocation_filters) == 0

    def test_connect_without_target_raises(self, mock_client):
        adapter = SemanticKernelAdapter(mock_client)
        with pytest.raises(ValueError, match="requires a target kernel"):
            adapter.connect()

    def test_connect_without_sk_raises(self, mock_client, monkeypatch):
        import layerlens.instrument.adapters.frameworks.semantic_kernel as mod

        monkeypatch.setattr(mod, "_HAS_SEMANTIC_KERNEL", False)
        adapter = SemanticKernelAdapter(mock_client)
        with pytest.raises(ImportError, match="semantic_kernel"):
            adapter.connect(target=Kernel())

    def test_disconnect_idempotent(self, mock_client):
        kernel = Kernel()
        adapter = SemanticKernelAdapter(mock_client)
        adapter.connect(target=kernel)
        adapter.disconnect()
        adapter.disconnect()  # should not raise


# ---------------------------------------------------------------------------
# Function invocation via real kernel.invoke()
# ---------------------------------------------------------------------------


class TestFunctionInvocation:
    def test_invoke_emits_tool_call(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        kernel = Kernel()
        kernel.add_plugin(MathPlugin(), "MathPlugin")

        adapter = SemanticKernelAdapter(mock_client)
        adapter.connect(target=kernel)

        result = _run(kernel.invoke(plugin_name="MathPlugin", function_name="add", a=2, b=3))
        assert str(result) == "5"

        adapter.disconnect()

        events = uploaded["events"]
        tool_calls = find_events(events, "tool.call")
        assert len(tool_calls) >= 1
        assert tool_calls[0]["payload"]["tool_name"] == "MathPlugin.add"
        assert tool_calls[0]["payload"]["plugin_name"] == "MathPlugin"
        assert tool_calls[0]["payload"]["function_name"] == "add"

        tool_results = find_events(events, "tool.result")
        assert len(tool_results) >= 1
        assert tool_results[0]["payload"]["status"] == "ok"
        assert tool_results[0]["payload"]["latency_ms"] >= 0

    def test_invoke_captures_output(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        kernel = Kernel()
        kernel.add_plugin(MathPlugin(), "MathPlugin")

        adapter = SemanticKernelAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect(target=kernel)

        _run(kernel.invoke(plugin_name="MathPlugin", function_name="add", a=10, b=20))
        adapter.disconnect()

        events = uploaded["events"]
        tool_result = find_event(events, "tool.result")
        assert tool_result["payload"]["output"] == 30

    def test_invoke_error_emits_agent_error(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        kernel = Kernel()
        kernel.add_plugin(MathPlugin(), "MathPlugin")

        adapter = SemanticKernelAdapter(mock_client)
        adapter.connect(target=kernel)

        with pytest.raises(Exception):
            _run(kernel.invoke(plugin_name="MathPlugin", function_name="divide", a=1, b=0))

        adapter.disconnect()

        events = uploaded["events"]
        err = find_event(events, "agent.error")
        assert "division by zero" in err["payload"]["error"]
        assert err["payload"]["error_type"] == "ZeroDivisionError"
        assert err["payload"]["tool_name"] == "MathPlugin.divide"

    def test_sequential_invocations(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        kernel = Kernel()
        kernel.add_plugin(MathPlugin(), "MathPlugin")

        adapter = SemanticKernelAdapter(mock_client)
        adapter.connect(target=kernel)

        _run(kernel.invoke(plugin_name="MathPlugin", function_name="add", a=1, b=2))
        _run(kernel.invoke(plugin_name="MathPlugin", function_name="add", a=3, b=4))
        adapter.disconnect()

        events = uploaded["events"]
        assert len(find_events(events, "tool.call")) == 2
        assert len(find_events(events, "tool.result")) == 2


# ---------------------------------------------------------------------------
# Function invocation filter via direct call
# ---------------------------------------------------------------------------


class TestFunctionInvocationFilter:
    def test_filter_calls_next_and_emits(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        kernel = Kernel()
        adapter = SemanticKernelAdapter(mock_client)
        adapter.connect(target=kernel)

        ctx = MockContext(
            function=MockFunction("greet", "HelloPlugin"),
        )

        async def mock_next(context):
            context.result = MockFunctionResult("Hi")

        _run(adapter._function_invocation_filter(ctx, mock_next))
        adapter.disconnect()

        events = uploaded["events"]
        tool_call = find_event(events, "tool.call")
        assert tool_call["payload"]["plugin_name"] == "HelloPlugin"
        assert tool_call["payload"]["function_name"] == "greet"

        tool_result = find_event(events, "tool.result")
        assert tool_result["payload"]["status"] == "ok"

    def test_filter_propagates_exception(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        kernel = Kernel()
        adapter = SemanticKernelAdapter(mock_client)
        adapter.connect(target=kernel)

        ctx = MockContext()

        async def failing_next(context):
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            _run(adapter._function_invocation_filter(ctx, failing_next))

        adapter.disconnect()

        events = uploaded["events"]
        err = find_event(events, "agent.error")
        assert err["payload"]["error"] == "boom"
        assert err["payload"]["error_type"] == "RuntimeError"


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------


class TestPromptRendering:
    def test_prompt_render_emits_agent_code(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        kernel = Kernel()
        adapter = SemanticKernelAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect(target=kernel)

        ctx = MockContext(
            function=MockFunction("summarize", "TextPlugin"),
            rendered_prompt="Summarize: Hello world",
        )

        async def mock_next(context):
            pass

        _run(adapter._prompt_rendering_filter(ctx, mock_next))
        adapter.disconnect()

        events = uploaded["events"]
        ev = find_event(events, "agent.code")
        assert ev["payload"]["event_subtype"] == "prompt_render"
        assert ev["payload"]["function_name"] == "summarize"
        assert "Summarize" in ev["payload"]["rendered_prompt"]

    def test_prompt_render_no_content_when_disabled(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        kernel = Kernel()
        config = CaptureConfig(l2_agent_code=True, capture_content=False)
        adapter = SemanticKernelAdapter(mock_client, capture_config=config)
        adapter.connect(target=kernel)

        ctx = MockContext(
            function=MockFunction("summarize", "TextPlugin"),
            rendered_prompt="secret prompt",
        )

        async def mock_next(context):
            pass

        _run(adapter._prompt_rendering_filter(ctx, mock_next))
        adapter.disconnect()

        events = uploaded["events"]
        ev = find_event(events, "agent.code")
        assert "rendered_prompt" not in ev["payload"]


# ---------------------------------------------------------------------------
# Auto function invocation (LLM-initiated tool calls)
# ---------------------------------------------------------------------------


class TestAutoFunctionInvocation:
    def test_auto_function_emits_tool_call_and_result(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        kernel = Kernel()
        adapter = SemanticKernelAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect(target=kernel)

        ctx = MockContext(
            function=MockFunction("web_search", "SearchPlugin"),
            function_call_content=MockFunctionCallContent(arguments={"query": "test"}),
            function_result=MockFunctionResult("found it"),
            request_sequence_index=1,
            function_sequence_index=0,
        )

        async def mock_next(context):
            pass

        _run(adapter._auto_function_invocation_filter(ctx, mock_next))
        adapter.disconnect()

        events = uploaded["events"]

        tool_call = find_event(events, "tool.call")
        assert tool_call["payload"]["auto_invoked"] is True
        assert tool_call["payload"]["tool_name"] == "SearchPlugin.web_search"
        assert tool_call["payload"]["input"] == {"query": "test"}
        assert tool_call["payload"]["request_sequence_index"] == 1

        tool_results = find_events(events, "tool.result")
        assert len(tool_results) == 1
        assert tool_results[0]["payload"]["auto_invoked"] is True
        assert tool_results[0]["payload"]["output"] == "found it"
        assert tool_results[0]["payload"]["latency_ms"] >= 0

    def test_auto_function_error(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        kernel = Kernel()
        adapter = SemanticKernelAdapter(mock_client)
        adapter.connect(target=kernel)

        ctx = MockContext(
            function=MockFunction("fail_tool", "ToolPlugin"),
        )

        async def failing_next(context):
            raise ValueError("tool exploded")

        with pytest.raises(ValueError, match="tool exploded"):
            _run(adapter._auto_function_invocation_filter(ctx, failing_next))

        adapter.disconnect()

        events = uploaded["events"]
        # tool.call should still be emitted (before the error)
        tool_call = find_event(events, "tool.call")
        assert tool_call["payload"]["auto_invoked"] is True

        err = find_event(events, "agent.error")
        assert err["payload"]["error"] == "tool exploded"
        assert err["payload"]["auto_invoked"] is True


# ---------------------------------------------------------------------------
# Plugin discovery
# ---------------------------------------------------------------------------


class TestPluginDiscovery:
    def test_discover_plugins_on_connect(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        kernel = Kernel()
        kernel.add_plugin(MathPlugin(), "MathPlugin")
        kernel.add_plugin(TextPlugin(), "TextPlugin")

        adapter = SemanticKernelAdapter(mock_client)
        adapter.connect(target=kernel)
        adapter.disconnect()

        events = uploaded["events"]
        config_events = find_events(events, "environment.config")
        plugin_names = {e["payload"]["plugin_name"] for e in config_events}
        assert "MathPlugin" in plugin_names
        assert "TextPlugin" in plugin_names

    def test_new_plugin_discovered_on_first_call(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        kernel = Kernel()
        adapter = SemanticKernelAdapter(mock_client)
        adapter.connect(target=kernel)

        # Invoke filter directly with a plugin not yet seen
        ctx = MockContext(function=MockFunction("do_stuff", "NewPlugin"))

        async def mock_next(context):
            context.result = MockFunctionResult("ok")

        _run(adapter._function_invocation_filter(ctx, mock_next))
        adapter.disconnect()

        events = uploaded["events"]
        config_events = find_events(events, "environment.config")
        names = {e["payload"]["plugin_name"] for e in config_events}
        assert "NewPlugin" in names

    def test_duplicate_plugin_not_rediscovered(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        kernel = Kernel()
        adapter = SemanticKernelAdapter(mock_client)
        adapter.connect(target=kernel)

        ctx1 = MockContext(function=MockFunction("f1", "SamePlugin"))
        ctx2 = MockContext(function=MockFunction("f2", "SamePlugin"))

        async def mock_next(context):
            context.result = MockFunctionResult("ok")

        _run(adapter._function_invocation_filter(ctx1, mock_next))
        _run(adapter._function_invocation_filter(ctx2, mock_next))
        adapter.disconnect()

        events = uploaded["events"]
        config_events = find_events(events, "environment.config")
        same_plugin = [e for e in config_events if e["payload"]["plugin_name"] == "SamePlugin"]
        assert len(same_plugin) == 1


# ---------------------------------------------------------------------------
# CaptureConfig gating
# ---------------------------------------------------------------------------


class TestCaptureConfigGating:
    def test_no_content_strips_io(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        kernel = Kernel()
        adapter = SemanticKernelAdapter(mock_client, capture_config=CaptureConfig(capture_content=False))
        adapter.connect(target=kernel)

        ctx = MockContext(
            function=MockFunction("search", "Plugin"),
            arguments={"secret": "key"},
        )

        async def mock_next(context):
            context.result = MockFunctionResult("classified")

        _run(adapter._function_invocation_filter(ctx, mock_next))
        adapter.disconnect()

        events = uploaded["events"]
        tool_call = find_event(events, "tool.call")
        assert "input" not in tool_call["payload"]

        tool_result = find_event(events, "tool.result")
        assert "output" not in tool_result["payload"]

    def test_full_config_includes_io(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        kernel = Kernel()
        adapter = SemanticKernelAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect(target=kernel)

        ctx = MockContext(
            function=MockFunction("search", "Plugin"),
            arguments={"query": "test"},
        )

        async def mock_next(context):
            context.result = MockFunctionResult("results")

        _run(adapter._function_invocation_filter(ctx, mock_next))
        adapter.disconnect()

        events = uploaded["events"]
        tool_call = find_event(events, "tool.call")
        assert tool_call["payload"]["input"] == {"query": "test"}

        tool_result = find_event(events, "tool.result")
        assert tool_result["payload"]["output"] == "results"


# ---------------------------------------------------------------------------
# LLM call wrapping
# ---------------------------------------------------------------------------


class MockUsage:
    def __init__(self, prompt_tokens: int = 0, completion_tokens: int = 0):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class MockChatMessage:
    def __init__(self, text: str = "Hello!", model_id: str = "gpt-4o", usage: Any = None):
        self.content = text
        self.ai_model_id = model_id
        self.metadata = {"usage": usage} if usage else {}


class MockChatService:
    """Minimal mock that looks like a ChatCompletionClientBase to the adapter."""

    def __init__(self, response_text: str = "Hello!", model_id: str = "gpt-4o",
                 prompt_tokens: int = 100, completion_tokens: int = 50):
        self.ai_model_id = model_id
        self._response = MockChatMessage(
            text=response_text,
            model_id=model_id,
            usage=MockUsage(prompt_tokens, completion_tokens),
        )

    async def _inner_get_chat_message_contents(self, chat_history: Any, settings: Any) -> list:
        return [self._response]


class TestLLMCallWrapping:
    def _register_mock_service(self, kernel, service):
        """Register a mock service directly on the kernel."""
        kernel.services["mock"] = service

    def test_model_invoke_emitted(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        kernel = Kernel()
        service = MockChatService(prompt_tokens=100, completion_tokens=50)
        self._register_mock_service(kernel, service)

        adapter = SemanticKernelAdapter(mock_client)
        adapter.connect(target=kernel)

        # Call the wrapped method directly
        _run(service._inner_get_chat_message_contents(None, None))

        adapter.disconnect()

        events = uploaded["events"]
        model_invoke = find_event(events, "model.invoke")
        assert model_invoke["payload"]["model"] == "gpt-4o"
        assert model_invoke["payload"]["tokens_prompt"] == 100
        assert model_invoke["payload"]["tokens_completion"] == 50
        assert model_invoke["payload"]["latency_ms"] >= 0

    def test_cost_record_emitted(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        kernel = Kernel()
        service = MockChatService(prompt_tokens=200, completion_tokens=100)
        self._register_mock_service(kernel, service)

        adapter = SemanticKernelAdapter(mock_client)
        adapter.connect(target=kernel)
        _run(service._inner_get_chat_message_contents(None, None))
        adapter.disconnect()

        events = uploaded["events"]
        cost = find_event(events, "cost.record")
        assert cost["payload"]["tokens_total"] == 300
        assert cost["payload"]["model"] == "gpt-4o"

    def test_no_cost_record_without_tokens(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        kernel = Kernel()
        service = MockChatService(prompt_tokens=0, completion_tokens=0)
        self._register_mock_service(kernel, service)

        adapter = SemanticKernelAdapter(mock_client)
        adapter.connect(target=kernel)
        _run(service._inner_get_chat_message_contents(None, None))
        adapter.disconnect()

        events = uploaded["events"]
        cost_events = find_events(events, "cost.record")
        assert len(cost_events) == 0

    def test_llm_error_emits_agent_error(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        kernel = Kernel()
        service = MockChatService()
        self._register_mock_service(kernel, service)

        # Replace inner method with one that fails
        original = service._inner_get_chat_message_contents

        async def failing_inner(chat_history, settings):
            raise RuntimeError("API timeout")

        adapter = SemanticKernelAdapter(mock_client)
        adapter.connect(target=kernel)

        # The adapter wrapped the original, so replace the original call path
        # We need to set up the service to fail BEFORE connect wraps it
        # Let's test by reconnecting
        adapter.disconnect()

        service._inner_get_chat_message_contents = failing_inner
        adapter.connect(target=kernel)

        with pytest.raises(RuntimeError, match="API timeout"):
            _run(service._inner_get_chat_message_contents(None, None))

        adapter.disconnect()

        events = uploaded["events"]
        err = find_event(events, "agent.error")
        assert err["payload"]["error"] == "API timeout"
        assert err["payload"]["model"] == "gpt-4o"

    def test_disconnect_restores_original(self, mock_client):
        kernel = Kernel()
        service = MockChatService()
        self._register_mock_service(kernel, service)

        adapter = SemanticKernelAdapter(mock_client)
        adapter.connect(target=kernel)
        # After connect, the method is our wrapper (an instance attribute, not the class method)
        assert "_traced_inner" in service._inner_get_chat_message_contents.__name__

        adapter.disconnect()
        # After disconnect, the instance override is removed and the class method is accessible again
        assert "_traced_inner" not in service._inner_get_chat_message_contents.__name__


# ---------------------------------------------------------------------------
# Span hierarchy
# ---------------------------------------------------------------------------


class TestSpanHierarchy:
    def test_events_share_root_span(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        kernel = Kernel()
        kernel.add_plugin(MathPlugin(), "MathPlugin")

        adapter = SemanticKernelAdapter(mock_client)
        adapter.connect(target=kernel)

        _run(kernel.invoke(plugin_name="MathPlugin", function_name="add", a=1, b=2))
        adapter.disconnect()

        events = uploaded["events"]
        # All events should share the same root span (via parent_span_id)
        parent_spans = {e.get("parent_span_id") for e in events if e.get("parent_span_id")}
        # There should be at most one root
        assert len(parent_spans) <= 2  # root_span_id from _ensure_collector + our root


# ---------------------------------------------------------------------------
# Event structure
# ---------------------------------------------------------------------------


class TestEventStructure:
    def test_event_fields(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        kernel = Kernel()
        kernel.add_plugin(MathPlugin(), "MathPlugin")

        adapter = SemanticKernelAdapter(mock_client)
        adapter.connect(target=kernel)

        _run(kernel.invoke(plugin_name="MathPlugin", function_name="add", a=1, b=2))
        adapter.disconnect()

        events = uploaded["events"]
        for event in events:
            assert "event_type" in event
            assert "trace_id" in event
            assert "span_id" in event
            assert "sequence_id" in event
            assert "timestamp_ns" in event
            assert "payload" in event
            assert event["payload"]["framework"] == "semantic_kernel"

        seq_ids = [e["sequence_id"] for e in events]
        assert seq_ids == sorted(seq_ids)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_extract_plugin_name_from_function(self):
        ctx = MockContext(function=MockFunction(plugin_name="MyPlugin"))
        assert _extract_plugin_name(ctx) == "MyPlugin"

    def test_extract_plugin_name_fallback(self):
        class Ctx:
            function = None
            plugin_name = "FallbackPlugin"

        assert _extract_plugin_name(Ctx()) == "FallbackPlugin"

    def test_extract_function_name(self):
        ctx = MockContext(function=MockFunction(name="my_func"))
        assert _extract_function_name(ctx) == "my_func"

    def test_extract_arguments_dict(self):
        ctx = MockContext(arguments={"x": 1, "y": 2})
        assert _extract_arguments(ctx) == {"x": 1, "y": 2}

    def test_extract_arguments_none(self):
        ctx = MockContext(arguments=None)
        assert _extract_arguments(ctx) is None

    def test_extract_arguments_mapping(self):
        """SK KernelArguments has .items() but isn't a dict."""
        class FakeArgs:
            def items(self):
                return [("a", 1)]

        ctx = MockContext(arguments=FakeArgs())
        assert _extract_arguments(ctx) == {"a": 1}
