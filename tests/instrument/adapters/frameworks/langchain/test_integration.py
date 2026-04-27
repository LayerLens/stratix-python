"""Integration tests for LangChain adapter using the REAL LangChain SDK.

Ported as-is from ``ateam/tests/adapters/langchain/test_integration.py``.

These tests verify that ``LayerLensCallbackHandler`` correctly captures
events from actual LangChain operations — not mocks. The SDK must be
installed:

    pip install 'layerlens[langchain]'

Tests are skipped if ``langchain_core`` is not installed.

Translation rules applied:
* ``stratix.sdk.python.adapters.base`` →
  ``layerlens.instrument.adapters._base``
* ``stratix.sdk.python.adapters.capture`` →
  ``layerlens.instrument.adapters._base``
* ``stratix.sdk.python.adapters.langchain.callbacks`` →
  ``layerlens.instrument.adapters.frameworks.langchain.callbacks``
* ``STRATIXCallbackHandler`` → ``LayerLensCallbackHandler``
"""

from __future__ import annotations

from typing import Any

import pytest

langchain = pytest.importorskip("langchain_core", reason="langchain-core not installed")

from langchain_core.outputs import (  # noqa: E402  # type: ignore[import-not-found,unused-ignore]
    LLMResult,
    ChatGeneration,
)
from langchain_core.messages import (  # noqa: E402  # type: ignore[import-not-found,unused-ignore]
    AIMessage,
    HumanMessage,
)
from langchain_core.callbacks import CallbackManager  # noqa: E402  # type: ignore[import-not-found,unused-ignore]

from layerlens.instrument.adapters._base import (  # noqa: E402
    AdapterStatus,
    CaptureConfig,
)
from layerlens.instrument.adapters.frameworks.langchain.callbacks import (  # noqa: E402
    LayerLensCallbackHandler,
)

# ---------------------------------------------------------------------------
# Test STRATIX instance that collects events
# ---------------------------------------------------------------------------


class EventCollector:
    """Real event collector — not a mock. Accumulates events for assertions."""

    def __init__(self) -> None:
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
# Adapter construction with real LangChain types
# ---------------------------------------------------------------------------


class TestAdapterWithRealSDK:
    """Verify adapter constructs and connects with real LangChain classes."""

    def test_handler_is_valid_langchain_callback(self) -> None:
        """LayerLensCallbackHandler must satisfy LangChain's callback protocol."""
        collector = EventCollector()
        handler = LayerLensCallbackHandler(stratix=collector)

        # LangChain requires these attributes on callback handlers
        assert hasattr(handler, "raise_error")
        assert hasattr(handler, "ignore_llm")
        assert hasattr(handler, "ignore_chain")
        assert hasattr(handler, "ignore_agent")

    def test_handler_registers_in_callback_manager(self) -> None:
        """Handler can be added to a real LangChain CallbackManager."""
        collector = EventCollector()
        handler = LayerLensCallbackHandler(stratix=collector)

        # This is the real LangChain CallbackManager — not a mock
        manager = CallbackManager(handlers=[handler])
        assert len(manager.handlers) >= 1
        assert any(isinstance(h, LayerLensCallbackHandler) for h in manager.handlers)

    def test_capture_config_propagates(self) -> None:
        """CaptureConfig correctly controls which events are captured."""
        collector = EventCollector()
        config = CaptureConfig(
            l3_model_metadata=True,
            l5a_tool_calls=False,
            l1_agent_io=True,
        )
        handler = LayerLensCallbackHandler(
            stratix=collector,
            capture_config=config,
        )
        assert handler._capture_config.l3_model_metadata is True
        assert handler._capture_config.l5a_tool_calls is False


# ---------------------------------------------------------------------------
# LLM callback events with real LangChain types
# ---------------------------------------------------------------------------


class TestLLMCallbackEvents:
    """Verify LLM events are captured using real LangChain result types."""

    def test_on_llm_roundtrip_captures_model_invoke(self) -> None:
        """Full LLM start→end should emit a model.invoke event."""
        collector = EventCollector()
        handler = LayerLensCallbackHandler(stratix=collector)
        handler.connect()

        import uuid

        run_id = uuid.uuid4()
        handler.on_llm_start(
            serialized={"name": "ChatOpenAI", "kwargs": {"model_name": "gpt-4o"}},
            prompts=["What is the capital of France?"],
            run_id=run_id,
        )

        result = LLMResult(
            generations=[[ChatGeneration(message=AIMessage(content="Paris"))]],
            llm_output={"token_usage": {"total_tokens": 15}},
        )
        handler.on_llm_end(response=result, run_id=run_id)

        # Adapter combines start+end into a single model.invoke event
        model_events = [e for e in collector.events if e["type"] == "model.invoke"]
        assert len(model_events) == 1
        event = model_events[0]
        assert event["payload"]["output"] == "Paris"
        assert event["payload"]["token_usage"]["total_tokens"] == 15

    def test_on_llm_error_captures_error_in_event(self) -> None:
        """on_llm_error should capture the error in the model.invoke event."""
        collector = EventCollector()
        handler = LayerLensCallbackHandler(stratix=collector)
        handler.connect()

        import uuid

        run_id = uuid.uuid4()
        handler.on_llm_start(
            serialized={"name": "ChatOpenAI"},
            prompts=["fail"],
            run_id=run_id,
        )

        handler.on_llm_error(
            error=ValueError("Rate limit exceeded"),
            run_id=run_id,
        )

        # Error should be captured in the model.invoke event
        model_events = [e for e in collector.events if e["type"] == "model.invoke"]
        assert len(model_events) == 1
        assert "Rate limit exceeded" in model_events[0]["payload"].get("error", "")


# ---------------------------------------------------------------------------
# Chain callback events
# ---------------------------------------------------------------------------


class TestChainCallbackEvents:
    """Verify chain events using real LangChain types."""

    def test_chain_events_are_internal(self) -> None:
        """Chain events are internal plumbing — adapter may not emit them.

        This test verifies the adapter does not crash on chain callbacks,
        regardless of whether it emits events for them.
        """
        collector = EventCollector()
        handler = LayerLensCallbackHandler(stratix=collector)
        handler.connect()

        import uuid

        run_id = uuid.uuid4()

        # These should not raise — even if no events are emitted
        handler.on_chain_start(
            serialized={"name": "RetrievalQAChain"},
            inputs={"query": "What is AI?"},
            run_id=run_id,
        )
        handler.on_chain_end(
            outputs={"result": "AI is artificial intelligence."},
            run_id=run_id,
        )

        # No assertion on event count — chains are internal


# ---------------------------------------------------------------------------
# Tool callback events
# ---------------------------------------------------------------------------


class TestToolCallbackEvents:
    """Verify tool events using real LangChain types."""

    def test_on_tool_roundtrip_captures_tool_call(self) -> None:
        """Full tool start→end should emit a single tool.call event."""
        collector = EventCollector()
        handler = LayerLensCallbackHandler(stratix=collector)
        handler.connect()

        import uuid

        run_id = uuid.uuid4()

        handler.on_tool_start(
            serialized={"name": "search_tool"},
            input_str="quantum computing",
            run_id=run_id,
        )

        handler.on_tool_end(
            output="Quantum computing uses qubits...",
            run_id=run_id,
        )

        # Adapter combines start+end into a single tool.call event
        tool_events = [e for e in collector.events if e["type"] == "tool.call"]
        assert len(tool_events) == 1
        assert tool_events[0]["payload"]["tool_name"] == "search_tool"
        assert "quantum computing" in str(tool_events[0]["payload"].get("input", ""))

    def test_tool_events_disabled_by_capture_config(self) -> None:
        """When l5a_tool_calls=False, tool events should not be captured."""
        collector = EventCollector()
        config = CaptureConfig(l5a_tool_calls=False)
        handler = LayerLensCallbackHandler(
            stratix=collector,
            capture_config=config,
        )
        handler.connect()

        import uuid

        run_id = uuid.uuid4()
        handler.on_tool_start(
            serialized={"name": "search_tool"},
            input_str="test",
            run_id=run_id,
        )
        handler.on_tool_end(output="result", run_id=run_id)

        # Tool events should be suppressed
        tool_events = [e for e in collector.events if "tool" in e.get("type", "").lower()]
        assert len(tool_events) == 0


# ---------------------------------------------------------------------------
# Message types
# ---------------------------------------------------------------------------


class TestMessageTypeHandling:
    """Verify the adapter handles real LangChain message types correctly."""

    def test_human_message(self) -> None:
        """HumanMessage should be recognized as user input."""
        msg = HumanMessage(content="Hello")
        assert msg.content == "Hello"
        assert msg.type == "human"

    def test_ai_message(self) -> None:
        """AIMessage should be recognized as model output."""
        msg = AIMessage(content="Hi there!")
        assert msg.content == "Hi there!"
        assert msg.type == "ai"

    def test_chat_generation_wraps_ai_message(self) -> None:
        """ChatGeneration from a real LLMResult contains an AIMessage."""
        gen = ChatGeneration(message=AIMessage(content="Paris"))
        assert gen.message.content == "Paris"
        assert isinstance(gen.message, AIMessage)

    def test_llm_result_structure(self) -> None:
        """LLMResult from real LangChain has expected structure."""
        result = LLMResult(
            generations=[[ChatGeneration(message=AIMessage(content="test"))]],
            llm_output={"token_usage": {"total_tokens": 10}},
        )
        assert len(result.generations) == 1
        assert len(result.generations[0]) == 1
        # ``llm_output`` is typed as ``Optional[dict]`` upstream — narrow it
        # before subscripting to keep both pyright and mypy happy.
        assert result.llm_output is not None
        assert result.llm_output["token_usage"]["total_tokens"] == 10


# ---------------------------------------------------------------------------
# Adapter lifecycle
# ---------------------------------------------------------------------------


class TestAdapterLifecycle:
    """Verify adapter lifecycle with real SDK."""

    def test_connect_disconnect(self) -> None:
        """connect() and disconnect() should not raise."""
        collector = EventCollector()
        handler = LayerLensCallbackHandler(stratix=collector)
        handler.connect()
        assert handler._status == AdapterStatus.HEALTHY
        handler.disconnect()

    def test_adapter_has_framework_metadata(self) -> None:
        """Adapter should expose its framework name and version."""
        collector = EventCollector()
        handler = LayerLensCallbackHandler(stratix=collector)
        assert handler.FRAMEWORK == "langchain"
        assert handler.VERSION is not None
