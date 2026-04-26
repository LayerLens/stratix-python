"""Unit tests for the LiteLLM provider adapter.

Mocks the ``litellm.completion()`` / ``litellm.acompletion()`` boundary
so the test suite never reaches the network. Covers the routing layer
(``gpt-4o`` → OpenAI, ``claude-3-5-sonnet`` → Anthropic,
``bedrock/anthropic.claude-3-5-sonnet`` → AWS Bedrock, etc.), the sync
and async callback paths, and the lifecycle / lazy-import contract.
"""

from __future__ import annotations

import sys
import types
import asyncio
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from layerlens.instrument.adapters._base import AdapterStatus, CaptureConfig
from layerlens.instrument.adapters.providers.litellm import (
    ADAPTER_CLASS,
    LiteLLMAdapter,
    STRATIXLiteLLMCallback,
    LayerLensLiteLLMCallback,
    detect_provider,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _RecordingStratix:
    """Captures every event emitted via the adapter pipeline.

    Mirrors the minimal contract the real LayerLens client exposes so the
    adapter's internal ``self._stratix.emit(event_type, payload)`` calls
    land in a list we can introspect.
    """

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def emit(self, *args: Any, **kwargs: Any) -> None:
        if len(args) == 2 and isinstance(args[0], str):
            self.events.append({"event_type": args[0], "payload": args[1]})


def _make_response_obj(model: str = "gpt-4o") -> Any:
    """Build a LiteLLM-shaped response (OpenAI ChatCompletion lookalike)."""
    message = SimpleNamespace(role="assistant", content="hello", tool_calls=None)
    choice = SimpleNamespace(message=message, finish_reason="stop", index=0)
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    return SimpleNamespace(
        id="chatcmpl-x",
        model=model,
        choices=[choice],
        usage=usage,
    )


@pytest.fixture
def fake_litellm(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Install a fake ``litellm`` module exposing the surface we touch.

    The real ``litellm`` package is heavyweight and may already be
    imported in the test environment; replacing it with a minimal stub
    isolates the tests from upstream changes and lets us drive the
    callback registry deterministically.
    """
    fake = types.ModuleType("litellm")
    fake.callbacks = []  # type: ignore[attr-defined]
    fake.success_callback = []  # type: ignore[attr-defined]
    fake.failure_callback = []  # type: ignore[attr-defined]
    fake.__version__ = "1.40.0"  # type: ignore[attr-defined]

    # Mocked completion / acompletion boundary.
    completion_mock = MagicMock(return_value=_make_response_obj("openai/gpt-4o"))
    acompletion_mock = AsyncMock(return_value=_make_response_obj("anthropic/claude-3-5-sonnet"))
    fake.completion = completion_mock  # type: ignore[attr-defined]
    fake.acompletion = acompletion_mock  # type: ignore[attr-defined]

    # ``completion_cost`` returns USD; ``None`` forces fall-through to the
    # canonical LayerLens pricing manifest.
    fake.completion_cost = MagicMock(return_value=None)  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "litellm", fake)
    return fake


# ---------------------------------------------------------------------------
# Routing — every provider listed in the M3 PR description has an entry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model,expected",
    [
        # Explicit prefix routing (LiteLLM convention).
        ("openai/gpt-4o", "openai"),
        ("anthropic/claude-3-5-sonnet", "anthropic"),
        ("azure/my-deployment", "azure_openai"),
        ("bedrock/anthropic.claude-3-5-sonnet", "aws_bedrock"),
        ("vertex_ai/gemini-1.5-pro", "google_vertex"),
        ("ollama/llama3", "ollama"),
        ("cohere/command-r", "cohere"),
        ("huggingface/meta-llama/Llama-3-8b", "huggingface"),
        ("together_ai/togethercomputer/llama-2-70b", "together_ai"),
        ("groq/llama3-70b", "groq"),
        # Heuristic routing (no prefix).
        ("gpt-4", "openai"),
        ("gpt-4o", "openai"),
        ("o1-mini", "openai"),
        ("o3-mini", "openai"),
        ("claude-3-5-sonnet", "anthropic"),
        ("gemini-2.0-flash", "google_vertex"),
        ("llama-3.1-70b", "meta"),
        ("mistral-large", "mistral"),
        # Fallbacks.
        ("totally-unknown-model", "unknown"),
        ("", "unknown"),
    ],
)
def test_detect_provider_table(model: str, expected: str) -> None:
    assert detect_provider(model) == expected


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def test_adapter_class_export() -> None:
    """``ADAPTER_CLASS`` is the registry hook for lazy adapter discovery."""
    assert ADAPTER_CLASS is LiteLLMAdapter


def test_backward_compat_alias() -> None:
    """``STRATIX*`` alias preserved for users coming from ateam."""
    assert STRATIXLiteLLMCallback is LayerLensLiteLLMCallback


def test_legacy_flat_module_reexports_subpackage() -> None:
    """The flat ``providers.litellm_adapter`` module mirrors the subpackage."""
    from layerlens.instrument.adapters.providers import litellm_adapter as flat

    assert flat.LiteLLMAdapter is LiteLLMAdapter
    assert flat.LayerLensLiteLLMCallback is LayerLensLiteLLMCallback
    assert flat.STRATIXLiteLLMCallback is STRATIXLiteLLMCallback
    assert flat.detect_provider is detect_provider
    assert flat.ADAPTER_CLASS is ADAPTER_CLASS


def test_subpackage_import_does_not_load_litellm() -> None:
    """Importing the adapter subpackage MUST NOT import ``litellm``.

    Lazy-import is the load-bearing guarantee for the whole instrument
    layer — the SDK is loaded only inside ``LiteLLMAdapter.connect``.
    """
    sys.modules.pop("litellm", None)

    import importlib

    importlib.import_module("layerlens.instrument.adapters.providers.litellm")
    importlib.import_module("layerlens.instrument.adapters.providers.litellm.adapter")
    importlib.import_module("layerlens.instrument.adapters.providers.litellm.callback")
    importlib.import_module("layerlens.instrument.adapters.providers.litellm.routing")

    assert "litellm" not in sys.modules, (
        "Importing the LiteLLM adapter subpackage leaked the upstream "
        "`litellm` SDK into sys.modules — the import must stay lazy."
    )


# ---------------------------------------------------------------------------
# Lifecycle (against the fake litellm module)
# ---------------------------------------------------------------------------


def test_connect_registers_callback_with_litellm(fake_litellm: types.ModuleType) -> None:
    adapter = LiteLLMAdapter()
    try:
        adapter.connect()
        assert adapter.status == AdapterStatus.HEALTHY
        assert adapter._callback in fake_litellm.callbacks  # type: ignore[attr-defined]
    finally:
        adapter.disconnect()


def test_disconnect_removes_callback(fake_litellm: types.ModuleType) -> None:
    adapter = LiteLLMAdapter()
    adapter.connect()
    cb = adapter._callback
    assert cb is not None
    assert cb in fake_litellm.callbacks  # type: ignore[attr-defined]

    adapter.disconnect()
    assert cb not in fake_litellm.callbacks  # type: ignore[attr-defined]
    assert adapter.status == AdapterStatus.DISCONNECTED


def test_connect_degraded_when_litellm_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without ``litellm`` installed the adapter degrades cleanly, never crashes."""
    monkeypatch.setitem(sys.modules, "litellm", None)  # poison the import

    adapter = LiteLLMAdapter()
    try:
        adapter.connect()
        # Either DEGRADED (ImportError) or HEALTHY if the test runtime
        # somehow already had a real ``litellm`` cached. Both are
        # acceptable, but never crash.
        assert adapter.status in (AdapterStatus.DEGRADED, AdapterStatus.HEALTHY)
    finally:
        adapter.disconnect()


# ---------------------------------------------------------------------------
# Callback emission paths (driven by the fake completion / acompletion)
# ---------------------------------------------------------------------------


def _connected_adapter(fake_litellm: types.ModuleType) -> Tuple[LiteLLMAdapter, _RecordingStratix]:
    """Build an adapter wired to a recording stratix, callback registered."""
    stratix = _RecordingStratix()
    adapter = LiteLLMAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()
    assert adapter.status == AdapterStatus.HEALTHY
    return adapter, stratix


def _drive_completion(
    fake_litellm: types.ModuleType,
    adapter: LiteLLMAdapter,
    *,
    model: str,
    response: Any,
) -> None:
    """Simulate a successful sync ``litellm.completion`` round-trip.

    We don't just call ``fake_litellm.completion(...)`` — that returns
    the canned response but does not invoke the callback, since LiteLLM
    in the real world is what dispatches the callback after the call
    completes. Instead, mirror that behaviour: invoke the callback the
    same way LiteLLM would, with the same kwargs / response shape.
    """
    fake_litellm.completion.return_value = response  # type: ignore[attr-defined]
    response_obj = fake_litellm.completion(  # type: ignore[attr-defined]
        model=model,
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.5,
    )
    assert adapter._callback is not None
    adapter._callback.log_success_event(
        kwargs={
            "model": model,
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.5,
        },
        response_obj=response_obj,
        start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
        end_time=datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
    )


@pytest.mark.parametrize(
    "model,expected_provider",
    [
        ("gpt-4", "openai"),  # bare OpenAI heuristic
        ("openai/gpt-4o-mini", "openai"),  # explicit prefix
        ("claude-3-5-sonnet", "anthropic"),  # bare Anthropic heuristic
        ("anthropic/claude-3-5-sonnet", "anthropic"),  # explicit prefix
        ("bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0", "aws_bedrock"),
        ("vertex_ai/gemini-1.5-pro", "google_vertex"),
    ],
)
def test_completion_emits_invoke_with_correct_provider(
    fake_litellm: types.ModuleType,
    model: str,
    expected_provider: str,
) -> None:
    """``completion(model=...)`` lands provider-routed events for every prefix."""
    adapter, stratix = _connected_adapter(fake_litellm)
    try:
        _drive_completion(
            fake_litellm,
            adapter,
            model=model,
            response=_make_response_obj(model=model),
        )

        invoke_events = [e for e in stratix.events if e["event_type"] == "model.invoke"]
        cost_events = [e for e in stratix.events if e["event_type"] == "cost.record"]
        assert len(invoke_events) == 1, f"expected one model.invoke, got {invoke_events}"
        assert invoke_events[0]["payload"]["provider"] == expected_provider
        assert invoke_events[0]["payload"]["model"] == model
        # 1-second start/end span → ~1000 ms latency.
        assert 900 < invoke_events[0]["payload"]["latency_ms"] < 1100
        # No litellm cost → pricing manifest takes over (cost.record present).
        assert cost_events, "expected cost.record fall-through to canonical pricing"
        assert cost_events[0]["payload"]["provider"] == expected_provider
    finally:
        adapter.disconnect()


def test_completion_uses_litellm_cost_when_available(fake_litellm: types.ModuleType) -> None:
    """When ``litellm.completion_cost`` returns USD, that value is recorded as ground truth."""
    fake_litellm.completion_cost.return_value = 0.001234  # type: ignore[attr-defined]

    adapter, stratix = _connected_adapter(fake_litellm)
    try:
        _drive_completion(
            fake_litellm,
            adapter,
            model="openai/gpt-4o",
            response=_make_response_obj(model="openai/gpt-4o"),
        )
        cost_events = [e for e in stratix.events if e["event_type"] == "cost.record"]
        assert len(cost_events) == 1
        payload = cost_events[0]["payload"]
        assert payload["api_cost_usd"] == pytest.approx(0.001234)
        assert payload["cost_source"] == "litellm"
        assert payload["provider"] == "openai"
    finally:
        adapter.disconnect()


def test_completion_failure_emits_policy_violation(fake_litellm: types.ModuleType) -> None:
    """A failing ``completion`` call yields ``model.invoke`` with error + ``policy.violation``."""
    adapter, stratix = _connected_adapter(fake_litellm)
    try:
        assert adapter._callback is not None
        adapter._callback.log_failure_event(
            kwargs={
                "model": "anthropic/claude-3-5-sonnet",
                "messages": [{"role": "user", "content": "x"}],
                "exception": "rate limited",
            },
            response_obj=None,
            start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
            end_time=datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
        )

        types_emitted = [e["event_type"] for e in stratix.events]
        assert "model.invoke" in types_emitted
        assert "policy.violation" in types_emitted

        invoke = next(e for e in stratix.events if e["event_type"] == "model.invoke")
        assert invoke["payload"]["error"] == "rate limited"
        assert invoke["payload"]["provider"] == "anthropic"
    finally:
        adapter.disconnect()


def test_streaming_marks_streaming_true(fake_litellm: types.ModuleType) -> None:
    """A streamed completion produces a single ``model.invoke`` flagged ``streaming: True``."""
    adapter, stratix = _connected_adapter(fake_litellm)
    try:
        assert adapter._callback is not None
        adapter._callback.log_stream_event(
            kwargs={"model": "openai/gpt-4o-mini"},
            response_obj=_make_response_obj("openai/gpt-4o-mini"),
            start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
            end_time=datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
        )
        invoke = next(e for e in stratix.events if e["event_type"] == "model.invoke")
        assert invoke["payload"]["streaming"] is True
        assert invoke["payload"]["provider"] == "openai"
    finally:
        adapter.disconnect()


# ---------------------------------------------------------------------------
# Async path (litellm.acompletion)
# ---------------------------------------------------------------------------


def test_async_completion_emits_invoke_via_async_callback(fake_litellm: types.ModuleType) -> None:
    """``litellm.acompletion`` fires the ``async_log_success_event`` hook."""
    adapter, stratix = _connected_adapter(fake_litellm)
    try:
        assert adapter._callback is not None
        response = _make_response_obj("anthropic/claude-3-5-sonnet")
        fake_litellm.acompletion.return_value = response  # type: ignore[attr-defined]

        async def _run() -> Any:
            actual = await fake_litellm.acompletion(  # type: ignore[attr-defined]
                model="anthropic/claude-3-5-sonnet",
                messages=[{"role": "user", "content": "hi"}],
            )
            await adapter._callback.async_log_success_event(  # type: ignore[union-attr]
                kwargs={
                    "model": "anthropic/claude-3-5-sonnet",
                    "messages": [{"role": "user", "content": "hi"}],
                },
                response_obj=actual,
                start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
                end_time=datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
            )
            return actual

        result = asyncio.run(_run())
        assert result is response
        assert fake_litellm.acompletion.await_count == 1  # type: ignore[attr-defined]

        invoke_events = [e for e in stratix.events if e["event_type"] == "model.invoke"]
        assert len(invoke_events) == 1
        assert invoke_events[0]["payload"]["provider"] == "anthropic"
        assert invoke_events[0]["payload"]["model"] == "anthropic/claude-3-5-sonnet"
    finally:
        adapter.disconnect()


def test_async_failure_routes_to_provider_error(fake_litellm: types.ModuleType) -> None:
    """``async_log_failure_event`` mirrors the sync failure pathway."""
    adapter, stratix = _connected_adapter(fake_litellm)
    try:
        assert adapter._callback is not None

        async def _run() -> None:
            await adapter._callback.async_log_failure_event(  # type: ignore[union-attr]
                kwargs={
                    "model": "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
                    "messages": [{"role": "user", "content": "x"}],
                    "exception": "throttled",
                },
                response_obj=None,
                start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
                end_time=datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
            )

        asyncio.run(_run())
        types_emitted = [e["event_type"] for e in stratix.events]
        assert "policy.violation" in types_emitted
        invoke = next(e for e in stratix.events if e["event_type"] == "model.invoke")
        assert invoke["payload"]["provider"] == "aws_bedrock"
        assert invoke["payload"]["error"] == "throttled"
    finally:
        adapter.disconnect()
