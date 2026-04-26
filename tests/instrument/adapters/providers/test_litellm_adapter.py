"""Unit tests for the LiteLLM provider adapter."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List
from datetime import datetime, timezone

import pytest

from layerlens.instrument.adapters._base import AdapterStatus, CaptureConfig
from layerlens.instrument.adapters.providers.litellm_adapter import (
    ADAPTER_CLASS,
    LiteLLMAdapter,
    LayerLensLiteLLMCallback,
    detect_provider,
)


class _RecordingStratix:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def emit(self, *args: Any, **kwargs: Any) -> None:
        if len(args) == 2 and isinstance(args[0], str):
            self.events.append({"event_type": args[0], "payload": args[1]})


# ---------------------------------------------------------------------------
# detect_provider
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model,expected",
    [
        ("openai/gpt-4o", "openai"),
        ("anthropic/claude-sonnet", "anthropic"),
        ("azure/my-deployment", "azure_openai"),
        ("bedrock/anthropic.claude-3-5-sonnet", "aws_bedrock"),
        ("vertex_ai/gemini-1.5-pro", "google_vertex"),
        ("ollama/llama3", "ollama"),
        ("cohere/command-r", "cohere"),
        ("groq/llama3-70b", "groq"),
        ("gpt-4o", "openai"),
        ("o1-mini", "openai"),
        ("claude-3-5-sonnet", "anthropic"),
        ("gemini-2.0-flash", "google_vertex"),
        ("llama-3.1-70b", "meta"),
        ("mistral-large", "mistral"),
        ("totally-unknown-model", "unknown"),
        ("", "unknown"),
    ],
)
def test_detect_provider_table(model: str, expected: str) -> None:
    assert detect_provider(model) == expected


# ---------------------------------------------------------------------------
# Adapter lifecycle
# ---------------------------------------------------------------------------


def test_adapter_class_export() -> None:
    assert ADAPTER_CLASS is LiteLLMAdapter


def test_backward_compat_alias() -> None:
    """STRATIX* alias preserved for users coming from ateam."""
    from layerlens.instrument.adapters.providers.litellm_adapter import (
        STRATIXLiteLLMCallback,
    )

    assert STRATIXLiteLLMCallback is LayerLensLiteLLMCallback


def test_connect_registers_callback_with_litellm() -> None:
    import litellm  # type: ignore[import-not-found,unused-ignore]

    adapter = LiteLLMAdapter()
    try:
        adapter.connect()
        assert adapter.status in (AdapterStatus.HEALTHY, AdapterStatus.DEGRADED)
        if adapter.status == AdapterStatus.HEALTHY:
            assert adapter._callback in litellm.callbacks
    finally:
        adapter.disconnect()


def test_disconnect_removes_callback() -> None:
    import litellm  # type: ignore[import-not-found,unused-ignore]

    adapter = LiteLLMAdapter()
    adapter.connect()
    cb = adapter._callback
    if cb is not None and adapter.status == AdapterStatus.HEALTHY:
        assert cb in litellm.callbacks
    adapter.disconnect()
    if cb is not None:
        assert cb not in litellm.callbacks
    assert adapter.status == AdapterStatus.DISCONNECTED


# ---------------------------------------------------------------------------
# Callback handlers
# ---------------------------------------------------------------------------


def _make_response_obj() -> Any:
    message = SimpleNamespace(role="assistant", content="hello", tool_calls=None)
    choice = SimpleNamespace(message=message, finish_reason="stop", index=0)
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    return SimpleNamespace(
        id="chatcmpl-x",
        model="gpt-4o",
        choices=[choice],
        usage=usage,
    )


def test_log_success_event_emits_invoke_and_cost() -> None:
    stratix = _RecordingStratix()
    adapter = LiteLLMAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    cb = LayerLensLiteLLMCallback(adapter)

    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    end = datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc)

    cb.log_success_event(
        kwargs={
            "model": "openai/gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.5,
        },
        response_obj=_make_response_obj(),
        start_time=start,
        end_time=end,
    )

    types = [e["event_type"] for e in stratix.events]
    assert "model.invoke" in types
    assert "cost.record" in types

    invoke = next(e for e in stratix.events if e["event_type"] == "model.invoke")
    assert invoke["payload"]["provider"] == "openai"
    # Latency ~ 1s = 1000 ms.
    assert 900 < invoke["payload"]["latency_ms"] < 1100


def test_log_failure_event_emits_policy_violation() -> None:
    stratix = _RecordingStratix()
    adapter = LiteLLMAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    cb = LayerLensLiteLLMCallback(adapter)

    cb.log_failure_event(
        kwargs={
            "model": "anthropic/claude-sonnet",
            "messages": [{"role": "user", "content": "x"}],
            "exception": "rate limited",
        },
        response_obj=None,
        start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
        end_time=datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
    )

    types = [e["event_type"] for e in stratix.events]
    assert "model.invoke" in types
    assert "policy.violation" in types

    invoke = next(e for e in stratix.events if e["event_type"] == "model.invoke")
    assert invoke["payload"]["error"] == "rate limited"
    assert invoke["payload"]["provider"] == "anthropic"


def test_log_stream_event_marks_streaming() -> None:
    stratix = _RecordingStratix()
    adapter = LiteLLMAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    cb = LayerLensLiteLLMCallback(adapter)

    cb.log_stream_event(
        kwargs={"model": "openai/gpt-4o-mini"},
        response_obj=_make_response_obj(),
        start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
        end_time=datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
    )

    invoke = next(e for e in stratix.events if e["event_type"] == "model.invoke")
    assert invoke["payload"]["streaming"] is True
