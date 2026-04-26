"""Unit tests for the Azure OpenAI provider adapter.

Tests are mocked — no real Azure OpenAI API is contacted. Verifies that
the adapter wraps ``client.chat.completions.create`` and
``client.embeddings.create`` correctly, captures Azure-specific
metadata (deployment, endpoint, api-version), uses the
:data:`AZURE_PRICING` table for cost calculations, sanitizes endpoint
URLs to prevent token leakage, and restores originals on disconnect.

Where the adapter is wire-bridged through the ``openai`` SDK, an
``httpx.MockTransport`` (via ``respx``) stands in for the real Azure
endpoint to keep tests fully offline while still exercising the SDK's
HTTP path.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import httpx
import respx
import pytest

from layerlens.instrument.adapters._base import (
    EventSink,
    AdapterStatus,
    CaptureConfig,
    AdapterCapability,
)
from layerlens.instrument.adapters.providers._base.tokens import NormalizedTokenUsage
from layerlens.instrument.adapters.providers._base.pricing import AZURE_PRICING
from layerlens.instrument.adapters.providers.azure_openai_adapter import (
    ADAPTER_CLASS,
    AzureOpenAIAdapter,
)

# ---------------------------------------------------------------------------
# Helpers — quack-alike Azure OpenAI client / response objects.
# ---------------------------------------------------------------------------


class _RecordingStratix:
    """Captures every event the adapter emits for assertion."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def emit(self, *args: Any, **kwargs: Any) -> None:
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
    response_id: str = "chatcmpl-azure",
    response_model: str = "gpt-4o",
    system_fingerprint: str = "fp-az",
    tool_calls: Optional[List[Any]] = None,
) -> Any:
    """Build an object that quacks like an Azure ``ChatCompletion``.

    Azure's response shape is identical to the OpenAI response shape — the
    ``openai`` SDK is the same on both backends.
    """
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
        system_fingerprint=system_fingerprint,
    )


def _make_client(
    *,
    returns: Any = None,
    raises: Optional[Exception] = None,
    base_url: str = "https://my-resource.openai.azure.com/?api-key=secret",
    api_version: Optional[str] = "2024-08-01",
    custom_query: Optional[Dict[str, str]] = None,
) -> Any:
    """Build an object that quacks like an ``AzureOpenAI`` client."""

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

    completions = SimpleNamespace(create=_create)
    chat = SimpleNamespace(completions=completions)
    embeddings = SimpleNamespace(create=_embed)

    return SimpleNamespace(
        chat=chat,
        embeddings=embeddings,
        _base_url=base_url,
        _api_version=api_version,
        _custom_query=custom_query,
    )


# ---------------------------------------------------------------------------
# Lifecycle + adapter metadata.
# ---------------------------------------------------------------------------


class TestAzureOpenAIAdapterLifecycle:
    def test_adapter_class_export(self) -> None:
        """Registry uses the ``ADAPTER_CLASS`` convention."""
        assert ADAPTER_CLASS is AzureOpenAIAdapter

    def test_framework_and_version(self) -> None:
        adapter = AzureOpenAIAdapter()
        assert adapter.FRAMEWORK == "azure_openai"
        assert adapter.VERSION == "0.1.0"

    def test_connect_and_disconnect(self) -> None:
        adapter = AzureOpenAIAdapter()
        adapter.connect()
        assert adapter.is_connected is True
        assert adapter.status == AdapterStatus.HEALTHY

        adapter.disconnect()
        assert adapter.is_connected is False
        assert adapter.status == AdapterStatus.DISCONNECTED

    def test_get_adapter_info(self) -> None:
        adapter = AzureOpenAIAdapter()
        info = adapter.get_adapter_info()
        assert info.framework == "azure_openai"
        assert info.name == "AzureOpenAIAdapter"
        assert AdapterCapability.TRACE_MODELS in info.capabilities
        assert AdapterCapability.TRACE_TOOLS in info.capabilities

    def test_health_check(self) -> None:
        adapter = AzureOpenAIAdapter()
        adapter.connect()
        h = adapter.health_check()
        assert h.framework_name == "azure_openai"
        assert h.status == AdapterStatus.HEALTHY
        assert h.error_count == 0
        assert h.circuit_open is False

    def test_serialize_for_replay(self) -> None:
        adapter = AzureOpenAIAdapter(
            stratix=_RecordingStratix(),
            capture_config=CaptureConfig.full(),
        )
        adapter.connect()
        rt = adapter.serialize_for_replay()
        assert rt.framework == "azure_openai"
        assert "capture_config" in rt.config


# ---------------------------------------------------------------------------
# Endpoint sanitization (Azure-specific safeguard against token leakage).
# ---------------------------------------------------------------------------


class TestEndpointSanitization:
    def test_strips_query_string(self) -> None:
        raw = "https://my-resource.openai.azure.com/path/?api-key=SECRET"
        sanitized = AzureOpenAIAdapter._sanitize_endpoint(raw)
        assert sanitized is not None
        assert "SECRET" not in sanitized
        assert "api-key" not in sanitized
        assert sanitized.startswith("https://my-resource.openai.azure.com/path")

    def test_strips_fragment(self) -> None:
        raw = "https://my-resource.openai.azure.com/?api-key=SECRET#frag"
        sanitized = AzureOpenAIAdapter._sanitize_endpoint(raw)
        assert sanitized is not None
        assert "frag" not in sanitized
        assert "SECRET" not in sanitized

    def test_none_returns_none(self) -> None:
        assert AzureOpenAIAdapter._sanitize_endpoint(None) is None

    def test_handles_url_objects(self) -> None:
        """``httpx.URL`` and similar URL objects are str()'d before parsing."""
        url = httpx.URL("https://my-resource.openai.azure.com/?api-key=x")
        sanitized = AzureOpenAIAdapter._sanitize_endpoint(url)
        assert sanitized is not None
        assert "api-key" not in sanitized

    def test_preserves_path(self) -> None:
        raw = (
            "https://my-resource.openai.azure.com/openai/deployments/"
            "gpt-4o-prod/?api-version=2024-08-01"
        )
        sanitized = AzureOpenAIAdapter._sanitize_endpoint(raw)
        assert sanitized is not None
        assert "/openai/deployments/gpt-4o-prod" in sanitized
        assert "api-version" not in sanitized


# ---------------------------------------------------------------------------
# Wrapping chat.completions.create.
# ---------------------------------------------------------------------------


class TestAzureOpenAIChatWrap:
    def test_connect_client_replaces_create(self) -> None:
        adapter = AzureOpenAIAdapter()
        client = _make_client(returns=_make_response())
        original = client.chat.completions.create

        adapter.connect_client(client)

        assert client.chat.completions.create is not original
        assert "chat.completions.create" in adapter._originals

    def test_connect_client_captures_azure_metadata(self) -> None:
        """``connect_client`` reads endpoint + api-version off the client."""
        adapter = AzureOpenAIAdapter()
        client = _make_client(api_version="2024-10-21")
        adapter.connect_client(client)

        assert adapter._azure_metadata["api_version"] == "2024-10-21"
        # Endpoint is sanitized (api-key query removed).
        assert "api-key" not in adapter._azure_metadata["azure_endpoint"]

    def test_custom_query_overrides_api_version(self) -> None:
        """Some Azure clients put api-version in ``_custom_query`` instead."""
        adapter = AzureOpenAIAdapter()
        client = _make_client(
            api_version=None,
            custom_query={"api-version": "2024-12-01-preview"},
        )
        adapter.connect_client(client)

        assert adapter._azure_metadata["api_version"] == "2024-12-01-preview"

    def test_successful_call_emits_model_invoke_and_cost(self) -> None:
        stratix = _RecordingStratix()
        adapter = AzureOpenAIAdapter(stratix=stratix, capture_config=CaptureConfig.full())
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
        assert invoke["payload"]["provider"] == "azure_openai"
        assert invoke["payload"]["model"] == "gpt-4o"
        assert invoke["payload"]["prompt_tokens"] == 10
        assert invoke["payload"]["completion_tokens"] == 5
        assert invoke["payload"]["total_tokens"] == 15
        assert invoke["payload"]["latency_ms"] >= 0
        assert invoke["payload"]["parameters"]["temperature"] == 0.7

    def test_messages_normalized_into_invoke(self) -> None:
        stratix = _RecordingStratix()
        adapter = AzureOpenAIAdapter(stratix=stratix, capture_config=CaptureConfig.full())
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
        adapter = AzureOpenAIAdapter(
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
        adapter = AzureOpenAIAdapter(stratix=stratix, capture_config=CaptureConfig.full())
        adapter.connect()

        client = _make_client(returns=_make_response(response_id="resp-az-42"))
        adapter.connect_client(client)
        client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "x"}],
        )

        invoke = next(e for e in stratix.events if e["event_type"] == "model.invoke")
        assert invoke["payload"]["response_id"] == "resp-az-42"
        assert invoke["payload"]["finish_reason"] == "stop"
        assert invoke["payload"]["system_fingerprint"] == "fp-az"
        # Azure metadata is merged into model.invoke too.
        assert invoke["payload"]["api_version"] == "2024-08-01"
        assert "azure_endpoint" in invoke["payload"]

    def test_azure_endpoint_in_invoke_is_sanitized(self) -> None:
        """The endpoint surfaced on every event must NOT contain the api-key."""
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
        assert "api-key" not in invoke["payload"]["azure_endpoint"]
        assert "secret" not in invoke["payload"]["azure_endpoint"]

    def test_tool_calls_emitted(self) -> None:
        stratix = _RecordingStratix()
        adapter = AzureOpenAIAdapter(stratix=stratix, capture_config=CaptureConfig.full())
        adapter.connect()

        function = SimpleNamespace(
            name="get_weather",
            arguments='{"city": "Seattle"}',
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
        assert tool_events[0]["payload"]["tool_input"] == {"city": "Seattle"}
        assert tool_events[0]["payload"]["tool_call_id"] == "call-1"

    def test_provider_error_emits_policy_violation(self) -> None:
        stratix = _RecordingStratix()
        adapter = AzureOpenAIAdapter(stratix=stratix, capture_config=CaptureConfig.full())
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
        # Azure metadata is included on error invokes.
        assert invoke["payload"]["api_version"] == "2024-08-01"

    def test_disconnect_restores_original(self) -> None:
        adapter = AzureOpenAIAdapter()
        adapter.connect()

        client = _make_client(returns=_make_response())
        original = client.chat.completions.create
        adapter.connect_client(client)
        assert client.chat.completions.create is not original

        adapter.disconnect()
        assert client.chat.completions.create is original


# ---------------------------------------------------------------------------
# Wrapping embeddings.create.
# ---------------------------------------------------------------------------


class TestAzureOpenAIEmbeddingsWrap:
    def test_embeddings_emit_invoke_and_cost(self) -> None:
        stratix = _RecordingStratix()
        adapter = AzureOpenAIAdapter(stratix=stratix, capture_config=CaptureConfig.full())
        adapter.connect()

        client = _make_client()
        adapter.connect_client(client)

        client.embeddings.create(model="text-embedding-3-small", input="hello")

        types = [e["event_type"] for e in stratix.events]
        assert "model.invoke" in types
        assert "cost.record" in types

        invoke = next(e for e in stratix.events if e["event_type"] == "model.invoke")
        assert invoke["payload"].get("request_type") == "embedding"
        # Azure metadata flows through embeddings too.
        assert invoke["payload"].get("api_version") == "2024-08-01"

    def test_embedding_error_still_emits_invoke(self) -> None:
        stratix = _RecordingStratix()
        adapter = AzureOpenAIAdapter(stratix=stratix, capture_config=CaptureConfig.full())
        adapter.connect()

        client = _make_client(raises=RuntimeError("embed failed"))
        adapter.connect_client(client)

        with pytest.raises(RuntimeError):
            client.embeddings.create(model="text-embedding-3-small", input="x")

        types = [e["event_type"] for e in stratix.events]
        assert "model.invoke" in types

        invoke = next(e for e in stratix.events if e["event_type"] == "model.invoke")
        assert invoke["payload"]["error"] == "embed failed"
        assert invoke["payload"].get("request_type") == "embedding"


# ---------------------------------------------------------------------------
# Capture config gating on model.invoke.
# ---------------------------------------------------------------------------


class TestAzureCaptureGating:
    def test_l3_disabled_drops_model_invoke(self) -> None:
        """When L3 model_metadata is off, model.invoke is dropped."""
        stratix = _RecordingStratix()
        adapter = AzureOpenAIAdapter(
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
# Sink dispatch integration.
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


class TestAzureSinkIntegration:
    def test_sink_receives_emitted_events(self) -> None:
        sink = _MemorySink()
        adapter = AzureOpenAIAdapter(
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
# Pricing — Azure rates differ from OpenAI public rates.
# ---------------------------------------------------------------------------


class TestAzureCostCalculation:
    def test_known_model_uses_azure_pricing(self) -> None:
        """Azure adapter must compute cost from AZURE_PRICING (NOT public PRICING)."""
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
        # 10 prompt + 5 completion => 10 * 0.00275 / 1000 + 5 * 0.011 / 1000
        # = 0.0000275 + 0.000055 = 0.0000825
        assert cost["payload"]["api_cost_usd"] is not None
        expected = 10 * 0.00275 / 1000 + 5 * 0.011 / 1000
        assert cost["payload"]["api_cost_usd"] == pytest.approx(expected, rel=1e-4)

    def test_azure_pricing_differs_from_public_openai(self) -> None:
        """Sanity: AZURE_PRICING['gpt-4o'] != OpenAI public PRICING['gpt-4o']."""
        from layerlens.instrument.adapters.providers._base.pricing import PRICING

        # Azure markup: gpt-4o is 0.00275 vs OpenAI public 0.0025.
        assert AZURE_PRICING["gpt-4o"]["input"] != PRICING["gpt-4o"]["input"]

    def test_unknown_model_marked_pricing_unavailable(self) -> None:
        stratix = _RecordingStratix()
        adapter = AzureOpenAIAdapter(stratix=stratix, capture_config=CaptureConfig.full())
        adapter.connect()

        client = _make_client(
            returns=_make_response(response_model="unobtanium-azure-deployment")
        )
        adapter.connect_client(client)
        client.chat.completions.create(
            model="unobtanium-azure-deployment",
            messages=[{"role": "user", "content": "x"}],
        )

        cost = next(e for e in stratix.events if e["event_type"] == "cost.record")
        assert cost["payload"]["api_cost_usd"] is None
        assert cost["payload"].get("pricing_unavailable") is True


# ---------------------------------------------------------------------------
# NormalizedTokenUsage extraction (delegated to OpenAI shared helpers).
# ---------------------------------------------------------------------------


class TestUsageExtraction:
    def test_extract_from_obj_with_basic_fields(self) -> None:
        from layerlens.instrument.adapters.providers.openai_adapter import OpenAIAdapter

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

    def test_normalized_token_usage_auto_total(self) -> None:
        u = NormalizedTokenUsage.with_auto_total(prompt_tokens=10, completion_tokens=5)
        assert u.total_tokens == 15


# ---------------------------------------------------------------------------
# Wire-level test using a real ``openai.AzureOpenAI`` client + respx.
#
# This ensures the adapter still works when the SDK actually drives an
# HTTP request through ``httpx`` — exactly the path it takes in production
# against ``*.openai.azure.com``. The respx route returns a canned Azure
# response shape so no real Azure resource is contacted.
# ---------------------------------------------------------------------------


_AZURE_DEPLOYMENT = "gpt-4o-prod"
_AZURE_RESOURCE = "my-resource"
_AZURE_API_VERSION = "2024-08-01-preview"
_AZURE_BASE = f"https://{_AZURE_RESOURCE}.openai.azure.com"


def _canned_chat_response() -> Dict[str, Any]:
    return {
        "id": "chatcmpl-azure-respx",
        "object": "chat.completion",
        "created": 1730000000,
        "model": "gpt-4o-2024-08-06",
        "system_fingerprint": "fp-azure-respx",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "respx says hi",
                    "tool_calls": None,
                    "refusal": None,
                },
                "finish_reason": "stop",
                "logprobs": None,
            }
        ],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 4,
            "total_tokens": 16,
            "prompt_tokens_details": {"cached_tokens": 0},
            "completion_tokens_details": {"reasoning_tokens": 0},
        },
    }


class TestAzureOpenAIRespxIntegration:
    """Drive a real openai.AzureOpenAI client through ``respx``."""

    def test_chat_completion_via_respx(self) -> None:
        try:
            from openai import AzureOpenAI
        except ImportError:  # pragma: no cover - optional extra
            pytest.skip("openai SDK not installed (extra: providers-azure-openai)")

        # The openai SDK builds:
        #   POST /openai/deployments/{deployment}/chat/completions
        url = (
            f"{_AZURE_BASE}/openai/deployments/{_AZURE_DEPLOYMENT}/chat/completions"
        )

        with respx.mock(assert_all_called=True) as router:
            route = router.post(url__regex=rf"^{_AZURE_BASE}/openai/deployments/.*").mock(
                return_value=httpx.Response(200, json=_canned_chat_response()),
            )

            stratix = _RecordingStratix()
            adapter = AzureOpenAIAdapter(stratix=stratix, capture_config=CaptureConfig.full())
            adapter.connect()

            client = AzureOpenAI(
                api_key="test-key",
                api_version=_AZURE_API_VERSION,
                azure_endpoint=_AZURE_BASE,
            )
            adapter.connect_client(client)

            response = client.chat.completions.create(
                model=_AZURE_DEPLOYMENT,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=20,
            )

            assert route.called
            # Verify the SDK actually shipped the request and we parsed it.
            assert response.choices[0].message.content == "respx says hi"
            assert response.usage is not None
            assert response.usage.prompt_tokens == 12
            assert url  # silence unused-var lint when assertion above narrows path

            # Verify the adapter saw it.
            types = [e["event_type"] for e in stratix.events]
            assert "model.invoke" in types
            assert "cost.record" in types

            invoke = next(e for e in stratix.events if e["event_type"] == "model.invoke")
            assert invoke["payload"]["provider"] == "azure_openai"
            assert invoke["payload"]["prompt_tokens"] == 12
            assert invoke["payload"]["completion_tokens"] == 4
            # The api-version flows through to the event payload.
            assert invoke["payload"]["api_version"] == _AZURE_API_VERSION
            # azure_endpoint is captured and sanitized.
            ep = invoke["payload"]["azure_endpoint"]
            assert "openai.azure.com" in ep
            assert "api-key" not in ep

            adapter.disconnect()

    def test_chat_completion_request_body_shape(self) -> None:
        """Verify the SDK's outbound request still contains our messages."""
        try:
            from openai import AzureOpenAI
        except ImportError:  # pragma: no cover - optional extra
            pytest.skip("openai SDK not installed (extra: providers-azure-openai)")

        captured_bodies: List[Dict[str, Any]] = []

        def _handler(request: httpx.Request) -> httpx.Response:
            captured_bodies.append(json.loads(request.content.decode()))
            return httpx.Response(200, json=_canned_chat_response())

        with respx.mock(assert_all_called=True) as router:
            router.post(url__regex=rf"^{_AZURE_BASE}/openai/deployments/.*").mock(
                side_effect=_handler,
            )

            adapter = AzureOpenAIAdapter()
            adapter.connect()

            client = AzureOpenAI(
                api_key="test-key",
                api_version=_AZURE_API_VERSION,
                azure_endpoint=_AZURE_BASE,
            )
            adapter.connect_client(client)

            client.chat.completions.create(
                model=_AZURE_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": "be brief"},
                    {"role": "user", "content": "ping"},
                ],
            )

            assert len(captured_bodies) == 1
            body = captured_bodies[0]
            assert body["messages"][0]["role"] == "system"
            assert body["messages"][1]["content"] == "ping"

            adapter.disconnect()
