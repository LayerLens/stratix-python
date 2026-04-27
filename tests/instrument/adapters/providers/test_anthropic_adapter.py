"""Unit tests for the Anthropic provider adapter.

Mocked at the SDK-response shape level. Verifies that the adapter wraps
``client.messages.create`` and ``client.messages.stream`` correctly,
emits the expected events, and restores the original methods on
disconnect.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List
from unittest import mock

import pytest

from layerlens.instrument.adapters._base import (
    AdapterStatus,
    CaptureConfig,
    AdapterCapability,
)
from layerlens.instrument.adapters.providers.anthropic_adapter import (
    ADAPTER_CLASS,
    AnthropicAdapter,
)


class _RecordingStratix:
    # Multi-tenant test stand-in: every recording client carries an
    # org_id so adapters constructed with this stratix pass the
    # BaseAdapter fail-fast check. Tests asserting cross-tenant
    # isolation override this default.
    org_id: str = "test-org"

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def emit(self, *args: Any, **kwargs: Any) -> None:
        if len(args) == 2 and isinstance(args[0], str):
            self.events.append({"event_type": args[0], "payload": args[1]})
        elif len(args) == 1:
            self.events.append({"event_type": None, "payload": args[0]})


def _make_response(
    *,
    text: str = "hello",
    input_tokens: int = 10,
    output_tokens: int = 5,
    stop_reason: str = "end_turn",
    response_id: str = "msg-abc",
    response_model: str = "claude-sonnet-4-5-20250929",
    tool_uses: List[Any] = None,
    cache_creation: int = None,
    cache_read: int = None,
) -> Any:
    """Build an object that quacks like an Anthropic Message."""
    content_blocks = [SimpleNamespace(type="text", text=text)]
    if tool_uses:
        content_blocks.extend(tool_uses)

    usage_kwargs: Dict[str, Any] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }
    if cache_creation is not None:
        usage_kwargs["cache_creation_input_tokens"] = cache_creation
    if cache_read is not None:
        usage_kwargs["cache_read_input_tokens"] = cache_read

    return SimpleNamespace(
        id=response_id,
        model=response_model,
        content=content_blocks,
        stop_reason=stop_reason,
        usage=SimpleNamespace(**usage_kwargs),
    )


def _make_client(*, returns: Any = None, raises: Exception = None) -> Any:
    def _create(**kwargs: Any) -> Any:
        if raises is not None:
            raise raises
        return returns

    def _stream(**kwargs: Any) -> Any:
        if raises is not None:
            raise raises
        return mock.MagicMock()

    messages = mock.MagicMock()
    messages.create = _create
    messages.stream = _stream

    return SimpleNamespace(messages=messages)


# ---------------------------------------------------------------------------
# Lifecycle + metadata
# ---------------------------------------------------------------------------


class TestAnthropicAdapterLifecycle:
    def test_adapter_class_export(self) -> None:
        assert ADAPTER_CLASS is AnthropicAdapter

    def test_framework_and_version(self) -> None:
        adapter = AnthropicAdapter(org_id="test-org")
        assert adapter.FRAMEWORK == "anthropic"
        assert adapter.VERSION == "0.1.0"

    def test_connect_disconnect(self) -> None:
        adapter = AnthropicAdapter(org_id="test-org")
        adapter.connect()
        assert adapter.is_connected is True
        assert adapter.status == AdapterStatus.HEALTHY
        adapter.disconnect()
        assert adapter.is_connected is False
        assert adapter.status == AdapterStatus.DISCONNECTED

    def test_get_adapter_info(self) -> None:
        adapter = AnthropicAdapter(org_id="test-org")
        info = adapter.get_adapter_info()
        assert info.framework == "anthropic"
        assert info.name == "AnthropicAdapter"
        assert AdapterCapability.TRACE_MODELS in info.capabilities


# ---------------------------------------------------------------------------
# Wrapping messages.create
# ---------------------------------------------------------------------------


class TestAnthropicCreateWrap:
    def test_connect_replaces_create_and_stream(self) -> None:
        adapter = AnthropicAdapter(org_id="test-org")
        client = _make_client(returns=_make_response())
        original_create = client.messages.create
        original_stream = client.messages.stream

        adapter.connect_client(client)

        assert client.messages.create is not original_create
        assert client.messages.stream is not original_stream
        assert "messages.create" in adapter._originals
        assert "messages.stream" in adapter._originals

    def test_successful_call_emits_event_set(self) -> None:
        stratix = _RecordingStratix()
        adapter = AnthropicAdapter(stratix=stratix, capture_config=CaptureConfig.full())
        adapter.connect()

        client = _make_client(returns=_make_response())
        adapter.connect_client(client)

        client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=100,
            messages=[{"role": "user", "content": "hi"}],
        )

        types = [e["event_type"] for e in stratix.events]
        assert "model.invoke" in types
        assert "cost.record" in types

        invoke = next(e for e in stratix.events if e["event_type"] == "model.invoke")
        assert invoke["payload"]["provider"] == "anthropic"
        assert invoke["payload"]["model"] == "claude-sonnet-4-5-20250929"
        assert invoke["payload"]["prompt_tokens"] == 10
        assert invoke["payload"]["completion_tokens"] == 5
        assert invoke["payload"]["total_tokens"] == 15
        assert invoke["payload"]["finish_reason"] == "end_turn"
        assert invoke["payload"]["response_id"] == "msg-abc"

    def test_system_prompt_recorded_as_has_system(self) -> None:
        stratix = _RecordingStratix()
        adapter = AnthropicAdapter(stratix=stratix, capture_config=CaptureConfig.full())
        adapter.connect()
        client = _make_client(returns=_make_response())
        adapter.connect_client(client)

        client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=100,
            system="You are concise.",
            messages=[{"role": "user", "content": "hi"}],
        )

        invoke = next(e for e in stratix.events if e["event_type"] == "model.invoke")
        assert invoke["payload"]["parameters"].get("has_system") is True
        # The normalized messages include the system prompt as the first entry.
        msgs = invoke["payload"].get("messages")
        assert msgs is not None
        assert msgs[0]["role"] == "system"

    def test_tools_count_captured(self) -> None:
        stratix = _RecordingStratix()
        adapter = AnthropicAdapter(stratix=stratix, capture_config=CaptureConfig.full())
        adapter.connect()
        client = _make_client(returns=_make_response())
        adapter.connect_client(client)

        client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=100,
            messages=[{"role": "user", "content": "x"}],
            tools=[{"name": "calc"}, {"name": "search"}],
        )

        invoke = next(e for e in stratix.events if e["event_type"] == "model.invoke")
        assert invoke["payload"]["parameters"]["tools_count"] == 2

    def test_tool_use_blocks_emit_tool_calls(self) -> None:
        stratix = _RecordingStratix()
        adapter = AnthropicAdapter(stratix=stratix, capture_config=CaptureConfig.full())
        adapter.connect()

        tool_use = SimpleNamespace(
            type="tool_use",
            name="get_weather",
            id="tool-1",
            input={"city": "SF"},
        )
        client = _make_client(returns=_make_response(tool_uses=[tool_use]))
        adapter.connect_client(client)
        client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=100,
            messages=[{"role": "user", "content": "weather"}],
        )

        tool_events = [e for e in stratix.events if e["event_type"] == "tool.call"]
        assert len(tool_events) == 1
        assert tool_events[0]["payload"]["tool_name"] == "get_weather"
        assert tool_events[0]["payload"]["tool_input"] == {"city": "SF"}

    def test_cache_metadata_captured(self) -> None:
        stratix = _RecordingStratix()
        adapter = AnthropicAdapter(stratix=stratix, capture_config=CaptureConfig.full())
        adapter.connect()
        client = _make_client(returns=_make_response(cache_creation=100, cache_read=200))
        adapter.connect_client(client)
        client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=100,
            messages=[{"role": "user", "content": "x"}],
        )
        invoke = next(e for e in stratix.events if e["event_type"] == "model.invoke")
        assert invoke["payload"]["cache_creation_input_tokens"] == 100
        assert invoke["payload"]["cache_read_input_tokens"] == 200

    def test_provider_error_emits_policy_violation(self) -> None:
        stratix = _RecordingStratix()
        adapter = AnthropicAdapter(stratix=stratix, capture_config=CaptureConfig.full())
        adapter.connect()
        client = _make_client(raises=RuntimeError("rate limited"))
        adapter.connect_client(client)

        with pytest.raises(RuntimeError):
            client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=100,
                messages=[{"role": "user", "content": "x"}],
            )

        types = [e["event_type"] for e in stratix.events]
        assert "model.invoke" in types
        assert "policy.violation" in types

    def test_disconnect_restores_originals(self) -> None:
        adapter = AnthropicAdapter(org_id="test-org")
        adapter.connect()
        client = _make_client(returns=_make_response())
        original_create = client.messages.create
        adapter.connect_client(client)
        assert client.messages.create is not original_create

        adapter.disconnect()
        assert client.messages.create is original_create


# ---------------------------------------------------------------------------
# Token extraction
# ---------------------------------------------------------------------------


class TestUsageExtraction:
    def test_basic(self) -> None:
        usage = SimpleNamespace(
            input_tokens=100,
            output_tokens=50,
        )
        result = AnthropicAdapter._extract_usage_from_obj(usage)
        assert result.prompt_tokens == 100
        assert result.completion_tokens == 50
        assert result.total_tokens == 150

    def test_with_cache_read(self) -> None:
        usage = SimpleNamespace(
            input_tokens=100,
            output_tokens=50,
            cache_read_input_tokens=20,
        )
        result = AnthropicAdapter._extract_usage_from_obj(usage)
        assert result.cached_tokens == 20

    def test_with_thinking_tokens(self) -> None:
        usage = SimpleNamespace(
            input_tokens=100,
            output_tokens=50,
            thinking_tokens=30,
        )
        result = AnthropicAdapter._extract_usage_from_obj(usage)
        assert result.reasoning_tokens == 30


# ---------------------------------------------------------------------------
# Cost calculation
# ---------------------------------------------------------------------------


class TestCostCalculation:
    def test_known_model_priced(self) -> None:
        stratix = _RecordingStratix()
        adapter = AnthropicAdapter(stratix=stratix, capture_config=CaptureConfig.full())
        adapter.connect()
        client = _make_client(returns=_make_response(input_tokens=1000, output_tokens=500))
        adapter.connect_client(client)
        client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=100,
            messages=[{"role": "user", "content": "x"}],
        )
        cost = next(e for e in stratix.events if e["event_type"] == "cost.record")
        # claude-sonnet-4-5-20250929: 0.003 input + 0.015 output per 1k
        # => 1000 * 0.003 / 1000 + 500 * 0.015 / 1000 = 0.003 + 0.0075 = 0.0105
        assert cost["payload"]["api_cost_usd"] == pytest.approx(0.0105, rel=1e-4)


# ---------------------------------------------------------------------------
# Stream event processing
# ---------------------------------------------------------------------------


class TestAnthropicStreaming:
    def test_stream_emits_one_consolidated_invoke(self) -> None:
        """Iterating a streamed response must emit exactly one model.invoke."""
        stratix = _RecordingStratix()
        adapter = AnthropicAdapter(stratix=stratix, capture_config=CaptureConfig.full())
        adapter.connect()

        # Build a synthetic event stream that mirrors Anthropic's wire shape.
        message_start = SimpleNamespace(
            type="message_start",
            message=SimpleNamespace(
                id="msg-1",
                model="claude-sonnet-4-5-20250929",
                usage=SimpleNamespace(input_tokens=10, output_tokens=0),
            ),
        )
        block_delta = SimpleNamespace(
            type="content_block_delta",
            delta=SimpleNamespace(type="text_delta", text="hello"),
        )
        message_delta = SimpleNamespace(
            type="message_delta",
            delta=SimpleNamespace(stop_reason="end_turn"),
            usage=SimpleNamespace(output_tokens=5),
        )

        events = iter([message_start, block_delta, message_delta])

        # We bypass connect_client and exercise _wrap_stream_response directly
        # because the streaming public API uses a context manager.
        wrapped = adapter._wrap_stream_response(events, model="claude-sonnet-4-5-20250929", params={}, start_ns=0)
        for _ in wrapped:
            pass

        invokes = [e for e in stratix.events if e["event_type"] == "model.invoke"]
        assert len(invokes) == 1
        payload = invokes[0]["payload"]
        assert payload.get("streaming") is True
        assert payload["finish_reason"] == "end_turn"
        # We accumulate the text content into the output_message.
        assert payload.get("output_message") is not None
