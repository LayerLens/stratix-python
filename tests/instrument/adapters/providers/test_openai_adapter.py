"""Unit tests for the OpenAI provider adapter.

Tests are mocked — no real OpenAI API is contacted. Verifies that the
adapter wraps ``client.chat.completions.create`` and
``client.embeddings.create`` correctly, emits the expected events, and
restores the original methods on disconnect.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List
from unittest import mock

import pytest

from layerlens.instrument.adapters._base import (
    EventSink,
    AdapterStatus,
    CaptureConfig,
    AdapterCapability,
)
from layerlens.instrument.adapters.providers._base.tokens import NormalizedTokenUsage
from layerlens.instrument.adapters.providers.openai_adapter import (
    ADAPTER_CLASS,
    OpenAIAdapter,
)


class _RecordingStratix:
    """Captures every event the adapter emits for assertion."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def emit(self, *args: Any, **kwargs: Any) -> None:
        # The adapter calls emit_dict_event which calls
        # _stratix.emit(event_type, payload). Capture both forms.
        if len(args) == 2 and isinstance(args[0], str):
            event_type, payload = args
            self.events.append({"event_type": event_type, "payload": payload})
        elif len(args) == 1:
            self.events.append({"event_type": None, "payload": args[0]})


def _make_response(
    *,
    content: str = "hello",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    total_tokens: int = 15,
    finish_reason: str = "stop",
    response_id: str = "chatcmpl-abc",
    response_model: str = "gpt-4o",
    tool_calls: List[Any] = None,
) -> Any:
    """Build an object that quacks like an OpenAI ChatCompletion."""
    message = SimpleNamespace(
        role="assistant",
        content=content,
        tool_calls=tool_calls or None,
    )
    choice = SimpleNamespace(
        message=message,
        finish_reason=finish_reason,
        index=0,
    )
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        prompt_tokens_details=None,
        completion_tokens_details=None,
    )
    return SimpleNamespace(
        id=response_id,
        model=response_model,
        choices=[choice],
        usage=usage,
        system_fingerprint="fp-xyz",
        service_tier="default",
    )


def _make_client(*, returns: Any = None, raises: Exception = None) -> Any:
    """Build an object that quacks like an OpenAI Client."""

    def _create(**kwargs: Any) -> Any:
        if raises is not None:
            raise raises
        return returns

    def _embed(**kwargs: Any) -> Any:
        if raises is not None:
            raise raises
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3])],
            model=kwargs.get("model"),
            usage=SimpleNamespace(
                prompt_tokens=8,
                completion_tokens=0,
                total_tokens=8,
                prompt_tokens_details=None,
                completion_tokens_details=None,
            ),
        )

    completions = mock.MagicMock()
    completions.create = _create
    chat = SimpleNamespace(completions=completions)

    embeddings = mock.MagicMock()
    embeddings.create = _embed

    return SimpleNamespace(chat=chat, embeddings=embeddings)


# ---------------------------------------------------------------------------
# Lifecycle + metadata
# ---------------------------------------------------------------------------


class TestOpenAIAdapterLifecycle:
    def test_adapter_class_export(self) -> None:
        """Registry uses the ``ADAPTER_CLASS`` convention."""
        assert ADAPTER_CLASS is OpenAIAdapter

    def test_framework_and_version(self) -> None:
        adapter = OpenAIAdapter(org_id="test-org")
        assert adapter.FRAMEWORK == "openai"
        assert adapter.VERSION == "0.1.0"

    def test_connect_and_disconnect(self) -> None:
        adapter = OpenAIAdapter(org_id="test-org")
        adapter.connect()
        assert adapter.is_connected is True
        assert adapter.status == AdapterStatus.HEALTHY

        adapter.disconnect()
        assert adapter.is_connected is False
        assert adapter.status == AdapterStatus.DISCONNECTED

    def test_get_adapter_info(self) -> None:
        adapter = OpenAIAdapter(org_id="test-org")
        info = adapter.get_adapter_info()
        assert info.framework == "openai"
        assert info.name == "OpenAIAdapter"
        assert AdapterCapability.TRACE_MODELS in info.capabilities
        assert AdapterCapability.TRACE_TOOLS in info.capabilities

    def test_health_check(self) -> None:
        adapter = OpenAIAdapter(org_id="test-org")
        adapter.connect()
        h = adapter.health_check()
        assert h.framework_name == "openai"
        assert h.status == AdapterStatus.HEALTHY
        assert h.error_count == 0
        assert h.circuit_open is False

    def test_serialize_for_replay(self) -> None:
        adapter = OpenAIAdapter(
            stratix=_RecordingStratix(),
            capture_config=CaptureConfig.full(),
        )
        adapter.connect()
        rt = adapter.serialize_for_replay()
        assert rt.framework == "openai"
        assert "capture_config" in rt.config


# ---------------------------------------------------------------------------
# Wrapping chat.completions.create
# ---------------------------------------------------------------------------


class TestOpenAIChatWrap:
    def test_connect_client_replaces_create(self) -> None:
        adapter = OpenAIAdapter(org_id="test-org")
        client = _make_client(returns=_make_response())
        original = client.chat.completions.create

        adapter.connect_client(client)

        assert client.chat.completions.create is not original
        assert "chat.completions.create" in adapter._originals

    def test_successful_call_emits_model_invoke_and_cost(self) -> None:
        stratix = _RecordingStratix()
        adapter = OpenAIAdapter(stratix=stratix, capture_config=CaptureConfig.full())
        adapter.connect()

        client = _make_client(returns=_make_response())
        adapter.connect_client(client)

        client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hello"}],
            temperature=0.7,
        )

        types = [e["event_type"] for e in stratix.events]
        assert "model.invoke" in types
        assert "cost.record" in types

        invoke = next(e for e in stratix.events if e["event_type"] == "model.invoke")
        assert invoke["payload"]["model"] == "gpt-4o"
        assert invoke["payload"]["provider"] == "openai"
        assert invoke["payload"]["prompt_tokens"] == 10
        assert invoke["payload"]["completion_tokens"] == 5
        assert invoke["payload"]["total_tokens"] == 15
        assert invoke["payload"]["latency_ms"] >= 0
        assert invoke["payload"]["parameters"]["temperature"] == 0.7
        # Output message captured because capture_content=True (full preset).
        assert "output_message" in invoke["payload"]

    def test_messages_normalized_into_invoke(self) -> None:
        stratix = _RecordingStratix()
        adapter = OpenAIAdapter(stratix=stratix, capture_config=CaptureConfig.full())
        adapter.connect()

        client = _make_client(returns=_make_response())
        adapter.connect_client(client)

        client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi"},
            ],
        )

        invoke = next(e for e in stratix.events if e["event_type"] == "model.invoke")
        msgs = invoke["payload"].get("messages")
        assert msgs is not None
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_capture_content_false_omits_messages(self) -> None:
        stratix = _RecordingStratix()
        adapter = OpenAIAdapter(
            stratix=stratix,
            capture_config=CaptureConfig(capture_content=False),
        )
        adapter.connect()

        client = _make_client(returns=_make_response())
        adapter.connect_client(client)

        client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "secret"}],
        )

        invoke = next(e for e in stratix.events if e["event_type"] == "model.invoke")
        assert "messages" not in invoke["payload"]
        assert "output_message" not in invoke["payload"]

    def test_response_metadata_captured(self) -> None:
        stratix = _RecordingStratix()
        adapter = OpenAIAdapter(stratix=stratix, capture_config=CaptureConfig.full())
        adapter.connect()

        client = _make_client(returns=_make_response(response_id="resp-42"))
        adapter.connect_client(client)
        client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "x"}],
        )

        invoke = next(e for e in stratix.events if e["event_type"] == "model.invoke")
        assert invoke["payload"]["response_id"] == "resp-42"
        assert invoke["payload"]["finish_reason"] == "stop"
        assert invoke["payload"]["system_fingerprint"] == "fp-xyz"

    def test_tool_calls_emitted(self) -> None:
        stratix = _RecordingStratix()
        adapter = OpenAIAdapter(stratix=stratix, capture_config=CaptureConfig.full())
        adapter.connect()

        function = SimpleNamespace(
            name="get_weather",
            arguments='{"city": "SF"}',
        )
        tool_call = SimpleNamespace(id="call-1", function=function)

        client = _make_client(returns=_make_response(tool_calls=[tool_call]))
        adapter.connect_client(client)
        client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "weather"}],
        )

        tool_events = [e for e in stratix.events if e["event_type"] == "tool.call"]
        assert len(tool_events) == 1
        assert tool_events[0]["payload"]["tool_name"] == "get_weather"
        assert tool_events[0]["payload"]["tool_input"] == {"city": "SF"}
        assert tool_events[0]["payload"]["tool_call_id"] == "call-1"

    def test_provider_error_emits_policy_violation(self) -> None:
        stratix = _RecordingStratix()
        adapter = OpenAIAdapter(stratix=stratix, capture_config=CaptureConfig.full())
        adapter.connect()

        client = _make_client(raises=RuntimeError("rate limited"))
        adapter.connect_client(client)

        with pytest.raises(RuntimeError):
            client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "x"}],
            )

        types = [e["event_type"] for e in stratix.events]
        assert "model.invoke" in types
        assert "policy.violation" in types

        invoke = next(e for e in stratix.events if e["event_type"] == "model.invoke")
        assert invoke["payload"]["error"] == "rate limited"

    def test_disconnect_restores_original(self) -> None:
        adapter = OpenAIAdapter(org_id="test-org")
        adapter.connect()

        client = _make_client(returns=_make_response())
        original = client.chat.completions.create
        adapter.connect_client(client)
        assert client.chat.completions.create is not original

        adapter.disconnect()
        assert client.chat.completions.create is original


# ---------------------------------------------------------------------------
# Wrapping embeddings.create
# ---------------------------------------------------------------------------


class TestOpenAIEmbeddingsWrap:
    def test_embeddings_emit_invoke_and_cost(self) -> None:
        stratix = _RecordingStratix()
        adapter = OpenAIAdapter(stratix=stratix, capture_config=CaptureConfig.full())
        adapter.connect()

        client = _make_client()
        adapter.connect_client(client)

        client.embeddings.create(model="text-embedding-3-small", input="hello")

        types = [e["event_type"] for e in stratix.events]
        assert "model.invoke" in types
        assert "cost.record" in types

        invoke = next(e for e in stratix.events if e["event_type"] == "model.invoke")
        assert invoke["payload"].get("request_type") == "embedding"


# ---------------------------------------------------------------------------
# Token usage extraction
# ---------------------------------------------------------------------------


class TestUsageExtraction:
    def test_extract_from_obj_with_basic_fields(self) -> None:
        usage = SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            prompt_tokens_details=None,
            completion_tokens_details=None,
        )
        result = OpenAIAdapter._extract_usage_from_obj(usage)
        assert result.prompt_tokens == 100
        assert result.completion_tokens == 50
        assert result.total_tokens == 150
        assert result.cached_tokens is None

    def test_extract_from_obj_with_cached_tokens(self) -> None:
        details = SimpleNamespace(cached_tokens=20)
        usage = SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            prompt_tokens_details=details,
            completion_tokens_details=None,
        )
        result = OpenAIAdapter._extract_usage_from_obj(usage)
        assert result.cached_tokens == 20

    def test_extract_from_obj_with_reasoning_tokens(self) -> None:
        comp_details = SimpleNamespace(reasoning_tokens=30)
        usage = SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            prompt_tokens_details=None,
            completion_tokens_details=comp_details,
        )
        result = OpenAIAdapter._extract_usage_from_obj(usage)
        assert result.reasoning_tokens == 30


# ---------------------------------------------------------------------------
# Capture config gating on model.invoke
# ---------------------------------------------------------------------------


class TestCaptureGating:
    def test_l3_disabled_drops_model_invoke(self) -> None:
        """When L3 model_metadata is off, model.invoke is dropped."""
        stratix = _RecordingStratix()
        adapter = OpenAIAdapter(
            stratix=stratix,
            capture_config=CaptureConfig(l3_model_metadata=False),
        )
        adapter.connect()

        client = _make_client(returns=_make_response())
        adapter.connect_client(client)
        client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "x"}],
        )

        types = [e["event_type"] for e in stratix.events]
        assert "model.invoke" not in types
        # cost.record is cross-cutting and STILL emits.
        assert "cost.record" in types


# ---------------------------------------------------------------------------
# Sink dispatch integration
# ---------------------------------------------------------------------------


class _MemorySink(EventSink):
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def send(self, event_type: str, payload: Dict[str, Any], timestamp_ns: int) -> None:
        self.events.append({"event_type": event_type, "payload": payload})

    def flush(self) -> None:  # pragma: no cover - no buffering
        pass

    def close(self) -> None:  # pragma: no cover - nothing to finalize
        pass


class TestSinkIntegration:
    def test_sink_receives_emitted_events(self) -> None:
        sink = _MemorySink()
        adapter = OpenAIAdapter(
            stratix=_RecordingStratix(),
            capture_config=CaptureConfig.full(),
        )
        adapter.add_sink(sink)
        adapter.connect()

        client = _make_client(returns=_make_response())
        adapter.connect_client(client)
        client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "x"}],
        )

        types = [e["event_type"] for e in sink.events]
        assert "model.invoke" in types
        assert "cost.record" in types


# ---------------------------------------------------------------------------
# Pricing integration (cost.record must include api_cost_usd for known model)
# ---------------------------------------------------------------------------


class TestCostCalculation:
    def test_known_model_gets_priced_cost(self) -> None:
        stratix = _RecordingStratix()
        adapter = OpenAIAdapter(stratix=stratix, capture_config=CaptureConfig.full())
        adapter.connect()

        client = _make_client(returns=_make_response(prompt_tokens=1000, completion_tokens=500, total_tokens=1500))
        adapter.connect_client(client)
        client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "x"}],
        )

        cost = next(e for e in stratix.events if e["event_type"] == "cost.record")
        assert cost["payload"]["api_cost_usd"] is not None
        # gpt-4o = 0.0025 input + 0.01 output per 1k => 0.0025 + 0.005 = 0.0075
        assert cost["payload"]["api_cost_usd"] == pytest.approx(0.0075, rel=1e-4)

    def test_unknown_model_marked_pricing_unavailable(self) -> None:
        stratix = _RecordingStratix()
        adapter = OpenAIAdapter(stratix=stratix, capture_config=CaptureConfig.full())
        adapter.connect()

        client = _make_client(returns=_make_response(response_model="unobtanium-xyz"))
        adapter.connect_client(client)
        client.chat.completions.create(
            model="unobtanium-xyz",
            messages=[{"role": "user", "content": "x"}],
        )

        cost = next(e for e in stratix.events if e["event_type"] == "cost.record")
        assert cost["payload"]["api_cost_usd"] is None
        assert cost["payload"].get("pricing_unavailable") is True


# ---------------------------------------------------------------------------
# NormalizedTokenUsage compute_total / with_auto_total
# ---------------------------------------------------------------------------


class TestNormalizedTokenUsage:
    def test_with_auto_total_computes_when_zero(self) -> None:
        u = NormalizedTokenUsage.with_auto_total(prompt_tokens=10, completion_tokens=5)
        assert u.total_tokens == 15

    def test_with_auto_total_respects_explicit_total(self) -> None:
        u = NormalizedTokenUsage.with_auto_total(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=99,
        )
        assert u.total_tokens == 99

    def test_compute_total_returns_fresh_instance(self) -> None:
        original = NormalizedTokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=0)
        recomputed = original.compute_total()
        assert original.total_tokens == 0
        assert recomputed.total_tokens == 15
        assert recomputed is not original
