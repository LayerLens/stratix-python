"""Tests for the AutoGen adapter using real autogen_core event classes.

Events are created from ``autogen_core.logging`` and dispatched through
the adapter's logging handler, exactly as they would be in production.

Requires autogen-core >= 0.4 (Python >= 3.10).
"""

from __future__ import annotations

# Skip entire module when autogen_core is not available.
import sys
import logging
from typing import Any, Optional

import pytest

if sys.version_info < (3, 10):
    pytest.skip("autogen-core requires Python >= 3.10", allow_module_level=True)
try:
    import autogen_core  # noqa: F401
except (ImportError, TypeError):
    pytest.skip("autogen-core not installed or incompatible", allow_module_level=True)

from autogen_core import EVENT_LOGGER_NAME, AgentId  # noqa: E402
from autogen_core.logging import (  # noqa: E402
    MessageKind,
    LLMCallEvent,
    MessageEvent,
    DeliveryStage,
    ToolCallEvent,
    LLMStreamEndEvent,
    MessageDroppedEvent,
    MessageHandlerExceptionEvent,
    AgentConstructionExceptionEvent,
)

from layerlens.instrument._capture_config import CaptureConfig  # noqa: E402
from layerlens.instrument.adapters.frameworks.autogen import (  # noqa: E402
    AutoGenAdapter,
    _enum_name,
    _get_field,
    _extract_model,
)

from .conftest import find_event, find_events, capture_framework_trace  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _setup(mock_client: Any, config: Optional[CaptureConfig] = None) -> tuple:
    uploaded = capture_framework_trace(mock_client)
    adapter = AutoGenAdapter(mock_client, capture_config=config)
    adapter.connect()
    return adapter, uploaded


def _log_and_flush(adapter: AutoGenAdapter, *events: Any) -> None:
    """Log events to the real autogen event logger, then disconnect."""
    logger = logging.getLogger(EVENT_LOGGER_NAME)
    for event in events:
        logger.info(event)
    adapter.disconnect()


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_connect_disconnect(self, mock_client):
        adapter = AutoGenAdapter(mock_client)
        adapter.connect()
        assert adapter.is_connected
        adapter.disconnect()
        assert not adapter.is_connected

    def test_adapter_info(self, mock_client):
        adapter = AutoGenAdapter(mock_client)
        info = adapter.adapter_info()
        assert info.name == "autogen"
        assert not info.connected

    def test_handler_attached_to_logger(self, mock_client):
        adapter = AutoGenAdapter(mock_client)
        adapter.connect()
        logger = logging.getLogger(EVENT_LOGGER_NAME)
        handler_types = [type(h).__name__ for h in logger.handlers]
        assert "_LayerLensHandler" in handler_types
        adapter.disconnect()
        handler_types = [type(h).__name__ for h in logger.handlers]
        assert "_LayerLensHandler" not in handler_types

    def test_disconnect_flushes_trace(self, mock_client):
        adapter, uploaded = _setup(mock_client)
        _log_and_flush(
            adapter,
            LLMCallEvent(
                messages=[],
                response={"model": "gpt-4o"},
                prompt_tokens=10,
                completion_tokens=5,
            ),
        )
        assert uploaded.get("trace_id") is not None


# ---------------------------------------------------------------------------
# LLM call events
# ---------------------------------------------------------------------------


class TestLLMCall:
    def test_model_invoke_emitted(self, mock_client):
        adapter, uploaded = _setup(mock_client)
        _log_and_flush(
            adapter,
            LLMCallEvent(
                messages=[{"role": "user", "content": "What is 2+2?"}],
                response={"model": "gpt-4o", "choices": [{"message": {"content": "4"}}]},
                prompt_tokens=50,
                completion_tokens=10,
            ),
        )
        events = uploaded["events"]
        me = find_event(events, "model.invoke")
        assert me["payload"]["framework"] == "autogen"
        assert me["payload"]["model"] == "gpt-4o"
        assert me["payload"]["tokens_prompt"] == 50
        assert me["payload"]["tokens_completion"] == 10
        assert me["payload"]["tokens_total"] == 60
        assert me["payload"]["messages"] == [{"role": "user", "content": "What is 2+2?"}]

    def test_cost_record_emitted(self, mock_client):
        adapter, uploaded = _setup(mock_client)
        _log_and_flush(
            adapter,
            LLMCallEvent(
                messages=[],
                response={"model": "gpt-4o-mini"},
                prompt_tokens=100,
                completion_tokens=25,
            ),
        )
        cost = find_event(uploaded["events"], "cost.record")
        assert cost["payload"]["tokens_total"] == 125
        assert cost["payload"]["model"] == "gpt-4o-mini"

    def test_zero_tokens_no_cost(self, mock_client):
        adapter, uploaded = _setup(mock_client)
        _log_and_flush(
            adapter,
            LLMCallEvent(
                messages=[],
                response={},
                prompt_tokens=0,
                completion_tokens=0,
            ),
        )
        me = find_event(uploaded["events"], "model.invoke")
        assert "tokens_prompt" not in me["payload"]
        assert len(find_events(uploaded["events"], "cost.record")) == 0

    def test_stream_end_handled_same(self, mock_client):
        adapter, uploaded = _setup(mock_client)
        _log_and_flush(
            adapter,
            LLMStreamEndEvent(
                response={"model": "gpt-4o"},
                prompt_tokens=30,
                completion_tokens=15,
            ),
        )
        me = find_event(uploaded["events"], "model.invoke")
        assert me["payload"]["tokens_total"] == 45

    def test_agent_id_from_context(self, mock_client):
        """agent_id is set from MessageHandlerContext at event creation time.

        Outside a running runtime the context is unavailable, so agent_id
        is None and omitted from the payload.  This test verifies the adapter
        handles that gracefully.
        """
        adapter, uploaded = _setup(mock_client)
        _log_and_flush(
            adapter,
            LLMCallEvent(
                messages=[],
                response={},
                prompt_tokens=10,
                completion_tokens=5,
            ),
        )
        me = find_event(uploaded["events"], "model.invoke")
        # No runtime context => agent_id is None => not in payload
        assert "agent_id" not in me["payload"]

    def test_content_gating(self, mock_client):
        adapter, uploaded = _setup(mock_client, config=CaptureConfig(capture_content=False))
        _log_and_flush(
            adapter,
            LLMCallEvent(
                messages=[{"role": "user", "content": "secret"}],
                response={"model": "gpt-4o", "choices": [{"message": {"content": "classified"}}]},
                prompt_tokens=10,
                completion_tokens=5,
            ),
        )
        me = find_event(uploaded["events"], "model.invoke")
        assert "messages" not in me["payload"]
        assert "output_message" not in me["payload"]


# ---------------------------------------------------------------------------
# Tool call events
# ---------------------------------------------------------------------------


class TestToolCall:
    def test_tool_call_emitted(self, mock_client):
        adapter, uploaded = _setup(mock_client)
        _log_and_flush(
            adapter,
            ToolCallEvent(
                tool_name="get_weather",
                arguments={"city": "NYC"},
                result='{"temp": 72}',
            ),
        )
        tc = find_event(uploaded["events"], "tool.call")
        assert tc["payload"]["tool_name"] == "get_weather"
        assert tc["payload"]["input"] == {"city": "NYC"}
        assert tc["payload"]["output"] == '{"temp": 72}'

    def test_tool_content_gating(self, mock_client):
        adapter, uploaded = _setup(mock_client, config=CaptureConfig(capture_content=False))
        _log_and_flush(
            adapter,
            ToolCallEvent(
                tool_name="search",
                arguments={"q": "secret"},
                result="classified",
            ),
        )
        tc = find_event(uploaded["events"], "tool.call")
        assert tc["payload"]["tool_name"] == "search"
        assert "input" not in tc["payload"]
        assert "output" not in tc["payload"]

    def test_multiple_tool_calls(self, mock_client):
        adapter, uploaded = _setup(mock_client)
        _log_and_flush(
            adapter,
            ToolCallEvent(tool_name="search", arguments={}, result="found"),
            ToolCallEvent(tool_name="summarize", arguments={}, result="short"),
        )
        assert len(find_events(uploaded["events"], "tool.call")) == 2


# ---------------------------------------------------------------------------
# Message events
# ---------------------------------------------------------------------------


class TestMessage:
    def test_direct_message_emits_agent_input(self, mock_client):
        adapter, uploaded = _setup(mock_client)
        _log_and_flush(
            adapter,
            MessageEvent(
                payload="Hello, can you help?",
                sender=AgentId("user_proxy", "default"),
                receiver=AgentId("assistant", "default"),
                kind=MessageKind.DIRECT,
                delivery_stage=DeliveryStage.SEND,
            ),
        )
        msg = find_event(uploaded["events"], "agent.input")
        assert msg["payload"]["sender"] == "user_proxy/default"
        assert msg["payload"]["receiver"] == "assistant/default"
        assert msg["payload"]["message_kind"] == "DIRECT"
        assert msg["payload"]["delivery_stage"] == "SEND"

    def test_respond_message_emits_agent_output(self, mock_client):
        adapter, uploaded = _setup(mock_client)
        _log_and_flush(
            adapter,
            MessageEvent(
                payload="The answer is 42",
                sender=AgentId("assistant", "default"),
                receiver=AgentId("user_proxy", "default"),
                kind=MessageKind.RESPOND,
                delivery_stage=DeliveryStage.SEND,
            ),
        )
        out = find_event(uploaded["events"], "agent.output")
        assert "The answer is 42" in out["payload"]["content"]

    def test_publish_message(self, mock_client):
        adapter, uploaded = _setup(mock_client)
        _log_and_flush(
            adapter,
            MessageEvent(
                payload="Broadcast",
                sender=AgentId("orchestrator", "default"),
                receiver=AgentId("chat", "default"),
                kind=MessageKind.PUBLISH,
                delivery_stage=DeliveryStage.SEND,
            ),
        )
        msg = find_event(uploaded["events"], "agent.input")
        assert msg["payload"]["message_kind"] == "PUBLISH"

    def test_none_sender_receiver(self, mock_client):
        adapter, uploaded = _setup(mock_client)
        _log_and_flush(
            adapter,
            MessageEvent(
                payload="orphan",
                sender=None,
                receiver=None,
                kind=MessageKind.DIRECT,
                delivery_stage=DeliveryStage.SEND,
            ),
        )
        msg = find_event(uploaded["events"], "agent.input")
        assert "sender" not in msg["payload"]
        assert "receiver" not in msg["payload"]

    def test_large_message_truncated(self, mock_client):
        adapter, uploaded = _setup(mock_client)
        _log_and_flush(
            adapter,
            MessageEvent(
                payload="x" * 5000,
                sender=None,
                receiver=None,
                kind=MessageKind.DIRECT,
                delivery_stage=DeliveryStage.SEND,
            ),
        )
        msg = find_event(uploaded["events"], "agent.input")
        assert len(msg["payload"]["content"]) <= 2010  # truncate adds "..."

    def test_content_gating(self, mock_client):
        adapter, uploaded = _setup(mock_client, config=CaptureConfig(capture_content=False))
        _log_and_flush(
            adapter,
            MessageEvent(
                payload="secret message",
                sender=None,
                receiver=None,
                kind=MessageKind.DIRECT,
                delivery_stage=DeliveryStage.SEND,
            ),
        )
        msg = find_event(uploaded["events"], "agent.input")
        assert "content" not in msg["payload"]


# ---------------------------------------------------------------------------
# Error events
# ---------------------------------------------------------------------------


class TestErrors:
    def test_message_dropped(self, mock_client):
        adapter, uploaded = _setup(mock_client)
        _log_and_flush(
            adapter,
            MessageDroppedEvent(
                payload="blocked",
                sender=AgentId("user", "default"),
                receiver=AgentId("assistant", "default"),
                kind=MessageKind.DIRECT,
            ),
        )
        err = find_event(uploaded["events"], "agent.error")
        assert err["payload"]["dropped"] is True
        assert err["payload"]["sender"] == "user/default"

    def test_handler_exception(self, mock_client):
        adapter, uploaded = _setup(mock_client)
        _log_and_flush(
            adapter,
            MessageHandlerExceptionEvent(
                payload="bad message",
                handling_agent=AgentId("assistant", "default"),
                exception=RuntimeError("Handler crashed"),
            ),
        )
        err = find_event(uploaded["events"], "agent.error")
        assert "Handler crashed" in err["payload"]["error"]
        # Real autogen events stringify exceptions in kwargs, so the
        # adapter sees a plain string and falls back to "Exception".
        assert err["payload"]["error_type"] == "Exception"
        assert err["payload"]["agent_id"] == "assistant/default"

    def test_construction_exception(self, mock_client):
        adapter, uploaded = _setup(mock_client)
        _log_and_flush(
            adapter,
            AgentConstructionExceptionEvent(
                agent_id=AgentId("broken_agent", "default"),
                exception=TypeError("Missing required param"),
            ),
        )
        err = find_event(uploaded["events"], "agent.error")
        assert "Missing required param" in err["payload"]["error"]
        # Same as above: exception is stringified in kwargs.
        assert err["payload"]["error_type"] == "Exception"
        assert err["payload"]["agent_id"] == "broken_agent/default"

    def test_string_exception_fallback(self, mock_client):
        adapter, uploaded = _setup(mock_client)
        _log_and_flush(
            adapter,
            MessageHandlerExceptionEvent(
                payload="bad",
                handling_agent=AgentId("a", "d"),
                exception="serialized error",
            ),
        )
        err = find_event(uploaded["events"], "agent.error")
        assert err["payload"]["error"] == "serialized error"
        assert err["payload"]["error_type"] == "Exception"


# ---------------------------------------------------------------------------
# Full conversation flow
# ---------------------------------------------------------------------------


class TestFullConversation:
    def test_complete_flow(self, mock_client):
        adapter, uploaded = _setup(mock_client)
        logger = logging.getLogger(EVENT_LOGGER_NAME)

        # User sends message
        logger.info(
            MessageEvent(
                payload="What's the weather?",
                sender=AgentId("user_proxy", "default"),
                receiver=AgentId("assistant", "default"),
                kind=MessageKind.DIRECT,
                delivery_stage=DeliveryStage.SEND,
            )
        )
        # LLM call
        logger.info(
            LLMCallEvent(
                messages=[{"role": "user", "content": "What's the weather?"}],
                response={"model": "gpt-4o"},
                prompt_tokens=50,
                completion_tokens=15,
            )
        )
        # Tool call
        logger.info(
            ToolCallEvent(
                tool_name="get_weather",
                arguments={"city": "NYC"},
                result='{"temp": 72}',
            )
        )
        # Second LLM call
        logger.info(
            LLMCallEvent(
                messages=[],
                response={"model": "gpt-4o"},
                prompt_tokens=80,
                completion_tokens=20,
            )
        )
        # Agent responds
        logger.info(
            MessageEvent(
                payload="It's 72F in NYC",
                sender=AgentId("assistant", "default"),
                receiver=AgentId("user_proxy", "default"),
                kind=MessageKind.RESPOND,
                delivery_stage=DeliveryStage.SEND,
            )
        )

        adapter.disconnect()
        events = uploaded["events"]
        types = [e["event_type"] for e in events]

        assert "agent.input" in types
        assert "model.invoke" in types
        assert "tool.call" in types
        assert "cost.record" in types
        assert "agent.output" in types
        assert len(find_events(events, "model.invoke")) == 2


# ---------------------------------------------------------------------------
# Trace integrity
# ---------------------------------------------------------------------------


class TestTraceIntegrity:
    def test_shared_trace_id(self, mock_client):
        adapter, uploaded = _setup(mock_client)
        _log_and_flush(
            adapter,
            LLMCallEvent(messages=[], response={}, prompt_tokens=10, completion_tokens=5),
            ToolCallEvent(tool_name="t", arguments={}, result="r"),
        )
        trace_ids = {e["trace_id"] for e in uploaded["events"]}
        assert len(trace_ids) == 1

    def test_monotonic_sequence_ids(self, mock_client):
        adapter, uploaded = _setup(mock_client)
        logger = logging.getLogger(EVENT_LOGGER_NAME)
        for i in range(5):
            logger.info(
                LLMCallEvent(
                    messages=[],
                    response={},
                    prompt_tokens=10 * (i + 1),
                    completion_tokens=5,
                )
            )
        adapter.disconnect()
        seq = [e["sequence_id"] for e in uploaded["events"]]
        assert seq == sorted(seq)

    def test_all_events_parented_to_root(self, mock_client):
        adapter, uploaded = _setup(mock_client)
        _log_and_flush(
            adapter,
            LLMCallEvent(messages=[], response={}, prompt_tokens=10, completion_tokens=5),
            ToolCallEvent(tool_name="t", arguments={}, result="r"),
        )
        events = uploaded["events"]
        parent_ids = {e.get("parent_span_id") for e in events}
        assert len(parent_ids) == 1

    def test_unknown_event_type_ignored(self, mock_client):
        adapter, uploaded = _setup(mock_client)
        logger = logging.getLogger(EVENT_LOGGER_NAME)

        class UnknownEvent:
            pass

        logger.info(UnknownEvent())
        adapter.disconnect()
        assert len(uploaded["events"]) == 0

    def test_none_event_does_not_crash(self, mock_client):
        adapter, _ = _setup(mock_client)
        logger = logging.getLogger(EVENT_LOGGER_NAME)
        logger.info(None)
        adapter.disconnect()


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


class TestConcurrency:
    def test_multiple_llm_calls_accumulated(self, mock_client):
        adapter, uploaded = _setup(mock_client)
        logger = logging.getLogger(EVENT_LOGGER_NAME)
        for i in range(5):
            logger.info(
                LLMCallEvent(
                    messages=[],
                    response={"model": "gpt-4o"},
                    prompt_tokens=10 * (i + 1),
                    completion_tokens=5 * (i + 1),
                )
            )
        adapter.disconnect()
        model_events = find_events(uploaded["events"], "model.invoke")
        assert len(model_events) == 5
        token_totals = sorted(e["payload"]["tokens_total"] for e in model_events)
        assert token_totals == [15, 30, 45, 60, 75]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_get_field_from_kwargs(self):
        e = LLMCallEvent(
            messages=[{"role": "user", "content": "hi"}],
            response={"model": "gpt-4o"},
            prompt_tokens=100,
            completion_tokens=50,
        )
        assert _get_field(e, "messages") == [{"role": "user", "content": "hi"}]
        assert _get_field(e, "prompt_tokens") == 100
        assert _get_field(e, "missing") is None
        assert _get_field(e, "missing", 42) == 42

    def test_get_field_from_attr(self):
        class E:
            model = "claude-3"

        assert _get_field(E(), "model") == "claude-3"

    def test_extract_model_from_response(self):
        e = LLMCallEvent(
            messages=[],
            response={"model": "gpt-4o"},
            prompt_tokens=0,
            completion_tokens=0,
        )
        assert _extract_model(e) == "gpt-4o"

    def test_extract_model_from_kwargs(self):
        # Real events don't have a top-level "model" kwarg, but _extract_model
        # falls back to checking kwargs["model"] if response has none.
        e = LLMCallEvent(
            messages=[],
            response={},
            model="claude-3",
            prompt_tokens=0,
            completion_tokens=0,
        )
        assert _extract_model(e) == "claude-3"

    def test_extract_model_none(self):
        e = LLMCallEvent(
            messages=[],
            response={},
            prompt_tokens=0,
            completion_tokens=0,
        )
        assert _extract_model(e) is None

    def test_enum_name_with_real_enums(self):
        assert _enum_name(MessageKind.DIRECT) == "DIRECT"
        assert _enum_name(MessageKind.RESPOND) == "RESPOND"
        assert _enum_name(MessageKind.PUBLISH) == "PUBLISH"
        assert _enum_name(DeliveryStage.SEND) == "SEND"
        assert _enum_name(DeliveryStage.DELIVER) == "DELIVER"

    def test_enum_name_with_stringified_enums(self):
        # Real events stringify enums in kwargs (e.g. "MessageKind.DIRECT").
        assert _enum_name("MessageKind.DIRECT") == "DIRECT"
        assert _enum_name("DeliveryStage.SEND") == "SEND"

    def test_enum_name_plain(self):
        assert _enum_name("PUBLISH") == "PUBLISH"
