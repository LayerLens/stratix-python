"""Unit tests for the Azure OpenAI provider adapter."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List
from unittest import mock

from layerlens.instrument.adapters._base import AdapterStatus, CaptureConfig
from layerlens.instrument.adapters.providers.azure_openai_adapter import (
    ADAPTER_CLASS,
    AzureOpenAIAdapter,
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


def _make_response() -> Any:
    message = SimpleNamespace(role="assistant", content="hello", tool_calls=None)
    choice = SimpleNamespace(message=message, finish_reason="stop", index=0)
    usage = SimpleNamespace(
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        prompt_tokens_details=None,
        completion_tokens_details=None,
    )
    return SimpleNamespace(
        id="chatcmpl-azure",
        model="gpt-4o",
        choices=[choice],
        usage=usage,
        system_fingerprint="fp-az",
    )


def _make_client(*, returns: Any = None) -> Any:
    completions = mock.MagicMock()
    completions.create = lambda **kw: returns

    embeddings = mock.MagicMock()
    embeddings.create = lambda **kw: SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=8,
            completion_tokens=0,
            total_tokens=8,
            prompt_tokens_details=None,
            completion_tokens_details=None,
        ),
    )

    chat = SimpleNamespace(completions=completions)
    return SimpleNamespace(
        chat=chat,
        embeddings=embeddings,
        _base_url="https://my-resource.openai.azure.com/?api-key=secret",
        _api_version="2024-08-01",
    )


def test_adapter_class_export() -> None:
    assert ADAPTER_CLASS is AzureOpenAIAdapter


def test_lifecycle() -> None:
    a = AzureOpenAIAdapter(org_id="test-org")
    a.connect()
    assert a.status == AdapterStatus.HEALTHY
    a.disconnect()
    assert a.status == AdapterStatus.DISCONNECTED


def test_endpoint_sanitization_strips_query_string() -> None:
    """Token leakage prevention: query string is removed from azure_endpoint metadata."""
    raw = "https://my-resource.openai.azure.com/path/?api-key=SECRET"
    sanitized = AzureOpenAIAdapter._sanitize_endpoint(raw)
    assert sanitized is not None
    assert "SECRET" not in sanitized
    assert "my-resource.openai.azure.com" in sanitized


def test_uses_azure_pricing() -> None:
    """Azure adapter must compute cost from AZURE_PRICING (different rates than OpenAI)."""
    stratix = _RecordingStratix()
    adapter = AzureOpenAIAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()
    client = _make_client(returns=_make_response())
    adapter.connect_client(client)

    client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "x"}],
    )

    cost = next(e for e in stratix.events if e["event_type"] == "cost.record")
    # AZURE_PRICING for gpt-4o: 0.00275 input + 0.011 output per 1k.
    # 10 prompt + 5 completion = 10 * 0.00275 / 1000 + 5 * 0.011 / 1000
    # = 0.0000275 + 0.000055 = 0.0000825
    assert cost["payload"]["api_cost_usd"] is not None
    expected = 10 * 0.00275 / 1000 + 5 * 0.011 / 1000
    assert abs(cost["payload"]["api_cost_usd"] - expected) < 1e-6


def test_azure_metadata_in_payload() -> None:
    stratix = _RecordingStratix()
    adapter = AzureOpenAIAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()
    client = _make_client(returns=_make_response())
    adapter.connect_client(client)

    client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "x"}],
    )

    invoke = next(e for e in stratix.events if e["event_type"] == "model.invoke")
    assert invoke["payload"]["api_version"] == "2024-08-01"
    # Endpoint has no query string after sanitization.
    assert "api-key" not in invoke["payload"]["azure_endpoint"]


def test_disconnect_restores_originals() -> None:
    adapter = AzureOpenAIAdapter(org_id="test-org")
    adapter.connect()
    client = _make_client(returns=_make_response())
    original = client.chat.completions.create
    adapter.connect_client(client)
    assert client.chat.completions.create is not original
    adapter.disconnect()
    assert client.chat.completions.create is original
