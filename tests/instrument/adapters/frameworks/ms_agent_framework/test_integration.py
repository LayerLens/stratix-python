"""Integration tests for Microsoft Agent Framework adapter using real SDK types.

These tests verify that MSAgentAdapter correctly captures events from
actual Semantic Kernel Agent types -- not mocks. The SDK must be installed:
    pip install semantic-kernel

Tests are skipped if semantic-kernel is not installed.

Ported as-is from ``ateam/tests/adapters/ms_agent_framework/test_integration.py``.

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

from __future__ import annotations

import hashlib
from typing import Any

import pytest

from layerlens.instrument.adapters._base import AdapterCapability, AdapterStatus
from layerlens.instrument.adapters._base import CaptureConfig
from layerlens.instrument.adapters.frameworks.ms_agent_framework.lifecycle import MSAgentAdapter

# ---------------------------------------------------------------------------
# Try to import semantic-kernel; tests that need it will skipif unavailable
# ---------------------------------------------------------------------------

_has_semantic_kernel = False
_sk_version: str | None = None

try:
    import semantic_kernel  # type: ignore[import-untyped]

    _has_semantic_kernel = True
    _sk_version = getattr(semantic_kernel, "__version__", "unknown")
except ImportError:
    pass

needs_semantic_kernel = pytest.mark.skipif(
    not _has_semantic_kernel,
    reason="semantic-kernel not installed",
)


# ---------------------------------------------------------------------------
# EventCollector -- real collector, not a mock
# ---------------------------------------------------------------------------


class EventCollector:
    """Real event collector -- accumulates events for assertions."""

    def __init__(self) -> None:
        self.org_id: str = "test-org"
        self.events: list[dict[str, Any]] = []
        self.traces_started: int = 0
        self.traces_ended: int = 0

    def emit(self, event_type: str, payload: dict[str, Any]) -> None:
        self.events.append({"type": event_type, "payload": payload})

    def start_trace(self, **kwargs: Any) -> str:
        self.traces_started += 1
        return f"trace-{self.traces_started}"

    def end_trace(self, **kwargs: Any) -> None:
        self.traces_ended += 1

    def get_events(self, event_type: str | None = None) -> list[dict[str, Any]]:
        if event_type:
            return [e for e in self.events if e["type"] == event_type]
        return self.events


# ---------------------------------------------------------------------------
# Fake SK Agent objects for testing (no auth/kernel required)
# ---------------------------------------------------------------------------


class FakeKernelPlugin:
    """Stub matching semantic_kernel.KernelPlugin interface."""

    def __init__(self, name: str) -> None:
        self.name = name


class FakeKernel:
    """Stub matching semantic_kernel.Kernel interface."""

    def __init__(self, plugins: dict[str, Any] | None = None) -> None:
        self.plugins = plugins or {}


class FakeAgent:
    """Stub matching semantic_kernel.agents.ChatCompletionAgent surface."""

    def __init__(
        self,
        name: str = "test_agent",
        instructions: str = "You are a helpful assistant.",
        kernel: FakeKernel | None = None,
    ) -> None:
        self.name = name
        self.instructions = instructions
        self.kernel = kernel or FakeKernel()


class FakeFunctionCallItem:
    """Stub for a function call content item in a ChatMessageContent."""

    def __init__(self, name: str, arguments: str | None = None) -> None:
        self.name = name
        self.arguments = arguments

    @property
    def __class__(self) -> type:
        # Make type(item).__name__ return "FunctionCallContent"
        return type("FunctionCallContent", (), {})


class FakeFunctionResultItem:
    """Stub for a function result content item in a ChatMessageContent."""

    def __init__(self, name: str, result: str | None = None) -> None:
        self.name = name
        self.result = result

    @property
    def __class__(self) -> type:
        return type("FunctionResultContent", (), {})


class FakeChatMessageContent:
    """Stub matching semantic_kernel.contents.ChatMessageContent."""

    def __init__(
        self,
        agent_name: str | None = None,
        content: str = "",
        items: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.agent_name = agent_name
        self.name = agent_name
        self.content = content
        self.items = items or []
        self.metadata = metadata or {}


class FakeUsage:
    """Stub matching SK CompletionUsage."""

    def __init__(self, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class FakeChat:
    """Stub matching AgentChat or AgentGroupChat invoke surface."""

    def __init__(
        self,
        name: str = "test_chat",
        agents: list[FakeAgent] | None = None,
        selection_strategy: Any = None,
        termination_strategy: Any = None,
    ) -> None:
        self.name = name
        self.agents = agents
        self.agent = agents[0] if agents else None
        self.selection_strategy = selection_strategy
        self.termination_strategy = termination_strategy
        self._invoke_called = False

    async def invoke(self, *args: Any, **kwargs: Any) -> Any:
        """Async generator yielding messages."""
        self._invoke_called = True
        yield FakeChatMessageContent(content="test response")

    async def invoke_stream(self, *args: Any, **kwargs: Any) -> Any:
        """Async generator yielding streaming messages."""
        yield FakeChatMessageContent(content="streamed response")


class FakeSelectionStrategy:
    pass


class FakeTerminationStrategy:
    pass


# ---------------------------------------------------------------------------
# Adapter construction
# ---------------------------------------------------------------------------


class TestAdapterConstruction:
    """Verify adapter constructs correctly with various configurations."""

    def test_adapter_framework_metadata(self) -> None:
        """Adapter should expose correct framework name."""
        adapter = MSAgentAdapter()
        assert adapter.FRAMEWORK == "ms_agent_framework"
        assert adapter.VERSION == "0.1.0"

    def test_adapter_capabilities(self) -> None:
        """MS Agent adapter must declare all four capabilities."""
        collector = EventCollector()
        adapter = MSAgentAdapter(stratix=collector)
        adapter.connect()
        info = adapter.get_adapter_info()
        assert AdapterCapability.TRACE_HANDOFFS in info.capabilities
        assert AdapterCapability.TRACE_TOOLS in info.capabilities
        assert AdapterCapability.TRACE_MODELS in info.capabilities
        assert AdapterCapability.TRACE_STATE in info.capabilities
        assert info.name == "MSAgentAdapter"

    def test_capture_config_propagates(self) -> None:
        """CaptureConfig correctly controls which events are captured."""
        collector = EventCollector()
        config = CaptureConfig(
            l3_model_metadata=True,
            l5a_tool_calls=False,
            l1_agent_io=True,
        )
        adapter = MSAgentAdapter(stratix=collector, capture_config=config)
        assert adapter._capture_config.l3_model_metadata is True
        assert adapter._capture_config.l5a_tool_calls is False

    @needs_semantic_kernel
    def test_connect_detects_sdk_version(self) -> None:
        """connect() should discover the installed semantic-kernel version."""
        collector = EventCollector()
        adapter = MSAgentAdapter(stratix=collector)
        adapter.connect()
        health = adapter.health_check()
        assert health.framework_version is not None
        assert health.framework_version == _sk_version

    def test_connect_without_sdk_still_healthy(self) -> None:
        """connect() should succeed even without SDK."""
        collector = EventCollector()
        adapter = MSAgentAdapter(stratix=collector)
        adapter.connect()
        assert adapter._status == AdapterStatus.HEALTHY


# ---------------------------------------------------------------------------
# Chat instrumentation (invoke wrapping)
# ---------------------------------------------------------------------------


class TestChatInstrumentation:
    """Verify instrument_chat wraps invoke methods on chat objects."""

    def test_instrument_chat_wraps_invoke(self) -> None:
        """instrument_chat should replace invoke with a traced version."""
        collector = EventCollector()
        adapter = MSAgentAdapter(stratix=collector)
        adapter.connect()

        chat = FakeChat(name="test_chat")
        original_invoke = chat.invoke

        adapter.instrument_chat(chat)

        # invoke should be replaced
        assert chat.invoke is not original_invoke
        # Original should be accessible via _layerlens_original
        assert hasattr(chat.invoke, "_layerlens_original")

    def test_instrument_chat_wraps_invoke_stream(self) -> None:
        """instrument_chat should also wrap invoke_stream."""
        collector = EventCollector()
        adapter = MSAgentAdapter(stratix=collector)
        adapter.connect()

        chat = FakeChat(name="test_chat")
        original_stream = chat.invoke_stream

        adapter.instrument_chat(chat)

        assert chat.invoke_stream is not original_stream
        assert hasattr(chat.invoke_stream, "_layerlens_original")

    def test_instrument_chat_emits_config(self) -> None:
        """instrument_chat should emit environment.config on first encounter."""
        collector = EventCollector()
        adapter = MSAgentAdapter(stratix=collector)
        adapter.connect()

        agents = [FakeAgent(name="coder"), FakeAgent(name="reviewer")]
        chat = FakeChat(
            name="code_review_chat",
            agents=agents,
            selection_strategy=FakeSelectionStrategy(),
            termination_strategy=FakeTerminationStrategy(),
        )

        adapter.instrument_chat(chat)

        config_events = collector.get_events("environment.config")
        assert len(config_events) == 1
        payload = config_events[0]["payload"]
        assert payload["framework"] == "ms_agent_framework"
        assert payload["chat_name"] == "code_review_chat"
        assert payload["chat_type"] == "FakeChat"
        assert payload["agents"] == ["coder", "reviewer"]
        assert payload["selection_strategy"] == "FakeSelectionStrategy"
        assert payload["termination_strategy"] == "FakeTerminationStrategy"

    def test_instrument_chat_idempotent(self) -> None:
        """Calling instrument_chat twice should not double-wrap."""
        collector = EventCollector()
        adapter = MSAgentAdapter(stratix=collector)
        adapter.connect()

        chat = FakeChat(name="test_chat")
        adapter.instrument_chat(chat)
        first_invoke = chat.invoke

        adapter.instrument_chat(chat)
        assert chat.invoke is first_invoke  # Same wrapper


# ---------------------------------------------------------------------------
# Lifecycle hooks (manual API)
# ---------------------------------------------------------------------------


class TestLifecycleHooks:
    """Verify manual lifecycle hook methods emit correct events."""

    def test_run_start_end_roundtrip(self) -> None:
        """on_run_start + on_run_end should emit input/output with duration."""
        collector = EventCollector()
        adapter = MSAgentAdapter(stratix=collector)
        adapter.connect()

        adapter.on_run_start(agent_name="planner", input_data="Build a plan")
        adapter.on_run_end(agent_name="planner", output="Plan: step 1, step 2")

        input_events = collector.get_events("agent.input")
        assert len(input_events) == 1
        assert input_events[0]["payload"]["input"] == "Build a plan"

        output_events = collector.get_events("agent.output")
        assert len(output_events) == 1
        assert output_events[0]["payload"]["output"] == "Plan: step 1, step 2"
        assert output_events[0]["payload"]["duration_ns"] >= 0

    def test_run_end_emits_state_change(self) -> None:
        """on_run_end should also emit agent.state.change."""
        collector = EventCollector()
        adapter = MSAgentAdapter(stratix=collector)
        adapter.connect()

        adapter.on_run_start(agent_name="worker", input_data="do work")
        adapter.on_run_end(agent_name="worker", output="done")

        state_events = collector.get_events("agent.state.change")
        assert len(state_events) == 1
        assert state_events[0]["payload"]["event_subtype"] == "run_complete"

    def test_run_end_with_error_emits_failed_state(self) -> None:
        """on_run_end with error should emit run_failed state change."""
        collector = EventCollector()
        adapter = MSAgentAdapter(stratix=collector)
        adapter.connect()

        adapter.on_run_start(agent_name="failing", input_data="crash")
        adapter.on_run_end(agent_name="failing", error=RuntimeError("Kernel failed"))

        output_events = collector.get_events("agent.output")
        assert len(output_events) == 1
        assert "Kernel failed" in output_events[0]["payload"]["error"]

        state_events = collector.get_events("agent.state.change")
        assert len(state_events) == 1
        assert state_events[0]["payload"]["event_subtype"] == "run_failed"

    def test_on_tool_use(self) -> None:
        """on_tool_use should emit tool.call."""
        collector = EventCollector()
        adapter = MSAgentAdapter(stratix=collector)
        adapter.connect()

        adapter.on_tool_use(
            tool_name="SearchPlugin_search",
            tool_input={"query": "AI trends"},
            tool_output={"results": ["trend1", "trend2"]},
            latency_ms=550.0,
        )

        events = collector.get_events("tool.call")
        assert len(events) == 1
        payload = events[0]["payload"]
        assert payload["tool_name"] == "SearchPlugin_search"
        assert payload["latency_ms"] == 550.0

    def test_on_llm_call(self) -> None:
        """on_llm_call should emit model.invoke."""
        collector = EventCollector()
        adapter = MSAgentAdapter(stratix=collector)
        adapter.connect()

        adapter.on_llm_call(
            provider="azure_openai",
            model="gpt-4o",
            tokens_prompt=200,
            tokens_completion=100,
        )

        events = collector.get_events("model.invoke")
        assert len(events) == 1
        assert events[0]["payload"]["model"] == "gpt-4o"
        assert events[0]["payload"]["provider"] == "azure_openai"

    def test_disconnected_adapter_emits_nothing(self) -> None:
        """Lifecycle hooks should no-op when adapter is disconnected."""
        collector = EventCollector()
        adapter = MSAgentAdapter(stratix=collector)
        # NOT calling connect()

        adapter.on_run_start(agent_name="test", input_data="hello")
        adapter.on_run_end(agent_name="test", output="world")
        adapter.on_tool_use(tool_name="t", tool_input={})
        adapter.on_llm_call(model="m")
        adapter.on_handoff(from_agent="a", to_agent="b")

        assert len(collector.events) == 0


# ---------------------------------------------------------------------------
# Message processing (tool calls, model info, handoffs from messages)
# ---------------------------------------------------------------------------


class TestMessageProcessing:
    """Verify _process_message extracts events from ChatMessageContent."""

    def test_function_call_in_message(self) -> None:
        """FunctionCall items should emit tool.call events."""
        collector = EventCollector()
        adapter = MSAgentAdapter(stratix=collector)
        adapter.connect()

        msg = FakeChatMessageContent(
            agent_name="coder",
            items=[FakeFunctionCallItem(name="write_code", arguments='{"lang": "python"}')],
        )

        adapter._process_message(None, msg, "coder")

        events = collector.get_events("tool.call")
        assert len(events) == 1
        assert events[0]["payload"]["tool_name"] == "write_code"

    def test_function_result_in_message(self) -> None:
        """FunctionResult items should emit tool.call events with output."""
        collector = EventCollector()
        adapter = MSAgentAdapter(stratix=collector)
        adapter.connect()

        msg = FakeChatMessageContent(
            agent_name="coder",
            items=[FakeFunctionResultItem(name="write_code", result="code written")],
        )

        adapter._process_message(None, msg, "coder")

        events = collector.get_events("tool.call")
        assert len(events) == 1
        assert events[0]["payload"]["tool_output"] == "code written"

    def test_agent_turn_transition_emits_handoff(self) -> None:
        """When message agent differs from current agent, emit agent.handoff."""
        collector = EventCollector()
        adapter = MSAgentAdapter(stratix=collector)
        adapter.connect()

        msg = FakeChatMessageContent(agent_name="reviewer")
        adapter._process_message(None, msg, "coder")

        events = collector.get_events("agent.handoff")
        assert len(events) == 1
        assert events[0]["payload"]["from_agent"] == "coder"
        assert events[0]["payload"]["to_agent"] == "reviewer"
        assert events[0]["payload"]["reason"] == "group_chat_turn"

    def test_same_agent_no_handoff(self) -> None:
        """Same agent name should not trigger a handoff event."""
        collector = EventCollector()
        adapter = MSAgentAdapter(stratix=collector)
        adapter.connect()

        msg = FakeChatMessageContent(agent_name="coder")
        adapter._process_message(None, msg, "coder")

        events = collector.get_events("agent.handoff")
        assert len(events) == 0

    def test_model_metadata_in_message(self) -> None:
        """Model info in metadata should emit model.invoke."""
        collector = EventCollector()
        adapter = MSAgentAdapter(stratix=collector)
        adapter.connect()

        msg = FakeChatMessageContent(
            agent_name="assistant",
            metadata={"model": "gpt-4o"},
        )

        adapter._process_message(None, msg, "assistant")

        events = collector.get_events("model.invoke")
        assert len(events) == 1
        assert events[0]["payload"]["model"] == "gpt-4o"

    def test_usage_metadata_emits_cost_record(self) -> None:
        """Usage metadata should emit cost.record."""
        collector = EventCollector()
        adapter = MSAgentAdapter(stratix=collector)
        adapter.connect()

        msg = FakeChatMessageContent(
            agent_name="assistant",
            metadata={
                "model": "gpt-4o",
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            },
        )

        adapter._process_message(None, msg, "assistant")

        cost_events = collector.get_events("cost.record")
        assert len(cost_events) == 1
        assert cost_events[0]["payload"]["tokens_prompt"] == 100
        assert cost_events[0]["payload"]["tokens_completion"] == 50


# ---------------------------------------------------------------------------
# Handoff events
# ---------------------------------------------------------------------------


class TestHandoffEvents:
    """Verify agent.handoff events for group chat turn transitions."""

    def test_handoff_event(self) -> None:
        """on_handoff should emit agent.handoff with context hash."""
        collector = EventCollector()
        adapter = MSAgentAdapter(stratix=collector)
        adapter.connect()

        adapter.on_handoff(
            from_agent="planner",
            to_agent="executor",
            context="Execute step 3 of the plan",
        )

        events = collector.get_events("agent.handoff")
        assert len(events) == 1
        payload = events[0]["payload"]
        assert payload["from_agent"] == "planner"
        assert payload["to_agent"] == "executor"
        assert payload["reason"] == "group_chat_turn"
        expected_hash = hashlib.sha256(b"Execute step 3 of the plan").hexdigest()
        assert payload["context_hash"] == expected_hash

    def test_handoff_without_context(self) -> None:
        """Handoff without context should have None context_hash."""
        collector = EventCollector()
        adapter = MSAgentAdapter(stratix=collector)
        adapter.connect()

        adapter.on_handoff(from_agent="a", to_agent="b")

        events = collector.get_events("agent.handoff")
        assert len(events) == 1
        assert events[0]["payload"]["context_hash"] is None

    def test_handoff_always_emitted_with_minimal_config(self) -> None:
        """agent.handoff is cross-cutting and should emit even with minimal config."""
        collector = EventCollector()
        config = CaptureConfig.minimal()
        adapter = MSAgentAdapter(stratix=collector, capture_config=config)
        adapter.connect()

        adapter.on_handoff(from_agent="x", to_agent="y")

        events = collector.get_events("agent.handoff")
        assert len(events) == 1


# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------


class TestProviderDetection:
    """Verify _detect_provider correctly identifies LLM providers."""

    def test_openai_models(self) -> None:
        adapter = MSAgentAdapter()
        assert adapter._detect_provider("gpt-4o") == "openai"
        assert adapter._detect_provider("o1-preview") == "openai"
        assert adapter._detect_provider("o3-mini") == "openai"

    def test_anthropic_models(self) -> None:
        adapter = MSAgentAdapter()
        assert adapter._detect_provider("claude-3-opus") == "anthropic"
        assert adapter._detect_provider("claude-opus-4") == "anthropic"

    def test_google_models(self) -> None:
        adapter = MSAgentAdapter()
        assert adapter._detect_provider("gemini-2.0-flash") == "google"

    def test_microsoft_models(self) -> None:
        adapter = MSAgentAdapter()
        assert adapter._detect_provider("phi-4") == "microsoft"

    def test_meta_models(self) -> None:
        adapter = MSAgentAdapter()
        assert adapter._detect_provider("llama-3.3-70b") == "meta"

    def test_mistral_models(self) -> None:
        adapter = MSAgentAdapter()
        assert adapter._detect_provider("mistral-large") == "mistral"
        assert adapter._detect_provider("mixtral-8x7b") == "mistral"

    def test_unknown_defaults_to_azure(self) -> None:
        adapter = MSAgentAdapter()
        assert adapter._detect_provider("some-custom-model") == "azure_openai"

    def test_none_returns_none(self) -> None:
        adapter = MSAgentAdapter()
        assert adapter._detect_provider(None) is None


# ---------------------------------------------------------------------------
# CaptureConfig gating
# ---------------------------------------------------------------------------


class TestCaptureConfigGating:
    """Verify that CaptureConfig correctly gates event emission."""

    def test_minimal_config_blocks_l3_l5(self) -> None:
        """Minimal config should block model.invoke and tool.call."""
        collector = EventCollector()
        config = CaptureConfig.minimal()
        adapter = MSAgentAdapter(stratix=collector, capture_config=config)
        adapter.connect()

        adapter.on_llm_call(model="gpt-4o")
        adapter.on_tool_use(tool_name="search")

        assert len(collector.get_events("model.invoke")) == 0
        assert len(collector.get_events("tool.call")) == 0

    def test_minimal_config_allows_l1(self) -> None:
        """Minimal config should still allow agent.input/output."""
        collector = EventCollector()
        config = CaptureConfig.minimal()
        adapter = MSAgentAdapter(stratix=collector, capture_config=config)
        adapter.connect()

        adapter.on_run_start(agent_name="test", input_data="hello")
        adapter.on_run_end(agent_name="test", output="world")

        assert len(collector.get_events("agent.input")) == 1
        assert len(collector.get_events("agent.output")) == 1

    def test_content_capture_enabled(self) -> None:
        """capture_content=True should include messages in model.invoke."""
        collector = EventCollector()
        config = CaptureConfig(capture_content=True)
        adapter = MSAgentAdapter(stratix=collector, capture_config=config)
        adapter.connect()

        adapter.on_llm_call(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hello"}],
        )

        events = collector.get_events("model.invoke")
        assert len(events) == 1
        assert "messages" in events[0]["payload"]

    def test_content_capture_disabled(self) -> None:
        """capture_content=False should exclude messages."""
        collector = EventCollector()
        config = CaptureConfig(capture_content=False)
        adapter = MSAgentAdapter(stratix=collector, capture_config=config)
        adapter.connect()

        adapter.on_llm_call(
            model="gpt-4o",
            messages=[{"role": "user", "content": "secret"}],
        )

        events = collector.get_events("model.invoke")
        assert len(events) == 1
        assert "messages" not in events[0]["payload"]


# ---------------------------------------------------------------------------
# Adapter lifecycle
# ---------------------------------------------------------------------------


class TestAdapterLifecycle:
    """Verify adapter lifecycle management."""

    def test_connect_disconnect(self) -> None:
        """connect() and disconnect() should transition status correctly."""
        collector = EventCollector()
        adapter = MSAgentAdapter(stratix=collector)
        adapter.connect()
        assert adapter._status == AdapterStatus.HEALTHY
        adapter.disconnect()
        assert adapter._status == AdapterStatus.DISCONNECTED

    def test_disconnect_unwraps_chats(self) -> None:
        """disconnect() should restore original invoke methods."""
        collector = EventCollector()
        adapter = MSAgentAdapter(stratix=collector)
        adapter.connect()

        chat = FakeChat(name="test_chat")
        original_invoke = chat.invoke
        adapter.instrument_chat(chat)
        assert chat.invoke is not original_invoke

        adapter.disconnect()
        # After disconnect, originals dict should be cleared
        assert len(adapter._originals) == 0
        assert len(adapter._wrapped_chats) == 0

    def test_serialization_for_replay(self) -> None:
        """serialize_for_replay should produce a valid ReplayableTrace."""
        collector = EventCollector()
        adapter = MSAgentAdapter(stratix=collector)
        adapter.connect()

        adapter.on_run_start(agent_name="test", input_data="hello")
        adapter.on_run_end(agent_name="test", output="world")

        trace = adapter.serialize_for_replay()
        assert trace.adapter_name == "MSAgentAdapter"
        assert trace.framework == "ms_agent_framework"
        assert len(trace.events) >= 2

    def test_null_stratix_pattern(self) -> None:
        """Adapter should work (no-op) without a stratix instance."""
        adapter = MSAgentAdapter()
        adapter.connect()
        # Should not raise
        adapter.on_run_start(agent_name="test", input_data="hello")
        adapter.on_run_end(agent_name="test", output="world")
        adapter.on_tool_use(tool_name="test", tool_input={})
        adapter.on_llm_call(model="test")
        adapter.on_handoff(from_agent="a", to_agent="b")
