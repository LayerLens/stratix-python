"""Deterministic doubles for the Azure OpenAI provider adapter (LAY-3582 / T8).

Azure OpenAI is credential-gated (no Azure subscription), so these tests stand
in for live verification. A real ``openai.AzureOpenAI`` client is constructed
against a fake endpoint with an ``httpx.MockTransport`` injected through the
``http_client=`` seam, so deployment URL routing, the ``api-version`` query
param, the ``api-key`` header, and response parsing all run through the real
SDK with no network. The response JSON mirrors a real Azure chat.completions
payload, including ``prompt_filter_results`` / ``content_filter_results``.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import httpx
import pytest

import openai
from openai import AzureOpenAI
from openai.types.chat import ChatCompletion
from layerlens.instrument import trace
from layerlens.instrument.adapters.providers.azure_openai import (
    AzureOpenAIProvider,
    _scrubbed_endpoint,
    instrument_azure_openai,
    uninstrument_azure_openai,
)

from ...conftest import find_event, find_events

_ENDPOINT = "https://unit-test.openai.azure.com"
_API_VERSION = "2024-06-01"
_API_KEY = "fake-azure-key"

_CONTENT_FILTER = {
    "hate": {"filtered": False, "severity": "safe"},
    "self_harm": {"filtered": False, "severity": "safe"},
    "sexual": {"filtered": False, "severity": "safe"},
    "violence": {"filtered": False, "severity": "safe"},
}


def _chat_completion_json(
    content: str = "Hello from Azure!",
    model: str = "gpt-4o-2024-05-13",
    prompt_tokens: int = 14,
    completion_tokens: int = 6,
) -> Dict[str, Any]:
    """Realistic Azure OpenAI chat.completions response body."""
    return {
        "id": "chatcmpl-azure-0001",
        "object": "chat.completion",
        "created": 1717418400,
        "model": model,
        "system_fingerprint": "fp_azure_5f2a1b",
        "prompt_filter_results": [{"prompt_index": 0, "content_filter_results": _CONTENT_FILTER}],
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
                "content_filter_results": _CONTENT_FILTER,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _chat_completion_with_tool_calls_json(
    tool_name: str = "get_weather",
    arguments: str = '{"city": "Seattle"}',
    model: str = "gpt-4o-2024-05-13",
) -> Dict[str, Any]:
    """Realistic Azure OpenAI response whose assistant message dispatches a tool.

    Azure's ``requires_tool_call=True`` live contract (``_registry.py``) has had
    zero unit/double assertion — the inherited OpenAI ``extract_tool_calls`` had
    never been exercised on any Azure response (LAY-3615). This pins that path.
    """
    return {
        "id": "chatcmpl-azure-tool-0001",
        "object": "chat.completion",
        "created": 1717418400,
        "model": model,
        "system_fingerprint": "fp_azure_5f2a1b",
        "prompt_filter_results": [{"prompt_index": 0, "content_filter_results": _CONTENT_FILTER}],
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_azure_abc123",
                            "type": "function",
                            "function": {"name": tool_name, "arguments": arguments},
                        }
                    ],
                },
                "content_filter_results": _CONTENT_FILTER,
            }
        ],
        "usage": {"prompt_tokens": 50, "completion_tokens": 12, "total_tokens": 62},
    }


def _make_client(
    response_json: Optional[Dict[str, Any]] = None,
    status_code: int = 200,
) -> tuple:
    """Real AzureOpenAI client over httpx.MockTransport. Returns (client, requests)."""
    requests: List[httpx.Request] = []
    payload = response_json if response_json is not None else _chat_completion_json()

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(status_code, json=payload)

    client = AzureOpenAI(
        azure_endpoint=_ENDPOINT,
        api_key=_API_KEY,
        api_version=_API_VERSION,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    return client, requests


# ---------------------------------------------------------------------------
# Emit events
# ---------------------------------------------------------------------------


class TestEmitsEvents:
    def test_model_invoke_and_azure_pricing(self, mock_client, capture_trace):
        client, requests = _make_client()
        provider = AzureOpenAIProvider()
        provider.connect(client)

        @trace(mock_client)
        def my_agent():
            r = client.chat.completions.create(
                model="gpt-4o",  # Azure deployment name
                messages=[{"role": "user", "content": "Hello?"}],
                temperature=0.2,
            )
            return r.choices[0].message.content

        assert my_agent() == "Hello from Azure!"

        # The real SDK routed the request through the Azure deployment URL.
        request = requests[0]
        assert request.url.host == "unit-test.openai.azure.com"
        assert "/deployments/gpt-4o/chat/completions" in request.url.path
        assert request.url.params["api-version"] == _API_VERSION
        assert request.headers["api-key"] == _API_KEY
        sent = json.loads(request.content)
        assert sent["messages"] == [{"role": "user", "content": "Hello?"}]
        assert sent["temperature"] == 0.2

        events = capture_trace["events"]
        mi = find_event(events, "model.invoke")
        # The Azure adapter reuses the OpenAI patch surface, so the event name
        # (and the provider derived from it) is the openai one.
        assert mi["payload"]["name"] == "openai.chat.completions.create"
        # Response model (underlying model) wins over the deployment name.
        assert mi["payload"]["model"] == "gpt-4o-2024-05-13"
        assert mi["payload"]["response_model"] == "gpt-4o-2024-05-13"
        assert mi["payload"]["response_id"] == "chatcmpl-azure-0001"
        assert mi["payload"]["system_fingerprint"] == "fp_azure_5f2a1b"
        assert mi["payload"]["finish_reason"] == "stop"
        assert mi["payload"]["latency_ms"] > 0
        assert mi["payload"]["messages"] == [{"role": "user", "content": "Hello?"}]
        assert mi["payload"]["output_message"] == {"role": "assistant", "content": "Hello from Azure!"}
        assert mi["payload"]["usage"]["prompt_tokens"] == 14
        assert mi["payload"]["usage"]["completion_tokens"] == 6
        assert mi["payload"]["usage"]["total_tokens"] == 20
        assert mi["payload"]["parameters"]["model"] == "gpt-4o"
        assert mi["payload"]["parameters"]["temperature"] == 0.2

        cost = find_event(events, "cost.record")
        assert cost["payload"]["provider"] == "openai"  # derived from the event name
        assert cost["payload"]["model"] == "gpt-4o-2024-05-13"
        assert cost["payload"]["total_tokens"] == 20
        # AZURE_PRICING path: dated model falls back to gpt-4o Azure rates
        # (14 * 0.00275/1k + 6 * 0.011/1k = 0.0001045). The base PRICING table
        # would have produced 0.000095, so this pins the Azure override.
        assert cost["payload"]["cost_usd"] == pytest.approx(0.0001045)

        provider.disconnect()

    def test_model_invoke_includes_azure_endpoint(self, mock_client, capture_trace):
        """The Azure resource endpoint captured at connect() must be emitted on
        model.invoke so the azure identity reaches the trace (P4 / LAY-3582).

        Azure reuses the OpenAI patch surface, so the event name (and derived
        provider) is the openai one and the response body carries no endpoint —
        the captured ``_endpoint`` is the only azure-resource signal, and it was
        never emitted.
        """
        client, _ = _make_client()
        provider = AzureOpenAIProvider()
        provider.connect(client)

        endpoint = provider._endpoint
        assert endpoint is not None and endpoint.startswith("https://unit-test.openai.azure.com")

        @trace(mock_client)
        def my_agent():
            client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello?"}],
            )

        my_agent()

        mi = find_event(capture_trace["events"], "model.invoke")
        assert mi["payload"]["azure_endpoint"] == endpoint
        # Never leak the query string (api-key / api-version) into the trace.
        assert "?" not in mi["payload"]["azure_endpoint"]

    def test_tool_call_emitted_from_azure_response(self, mock_client, capture_trace):
        client, _ = _make_client(response_json=_chat_completion_with_tool_calls_json())
        provider = AzureOpenAIProvider()
        provider.connect(client)

        @trace(mock_client)
        def my_agent():
            return client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Weather in Seattle?"}],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                        },
                    }
                ],
            )

        my_agent()
        events = capture_trace["events"]
        tool_call = find_event(events, "tool.call")
        assert tool_call["payload"]["tool_name"] == "get_weather"
        assert tool_call["payload"]["arguments"] == {"city": "Seattle"}
        # Azure reuses the OpenAI patch surface, so provider derives to "openai".
        assert tool_call["payload"]["provider"] == "openai"
        assert tool_call["payload"]["model"] == "gpt-4o-2024-05-13"
        # The model.invoke still lands alongside the tool.call.
        mi = find_event(events, "model.invoke")
        assert mi["payload"]["finish_reason"] == "tool_calls"
        provider.disconnect()

    def test_error_emits_agent_error(self, mock_client, capture_trace):
        client, _ = _make_client(
            response_json={
                "error": {
                    "code": "DeploymentNotFound",
                    "message": "The API deployment for this resource does not exist.",
                }
            },
            status_code=404,
        )
        provider = AzureOpenAIProvider()
        provider.connect(client)

        @trace(mock_client)
        def my_agent():
            try:
                client.chat.completions.create(model="missing-deployment", messages=[])
            except openai.NotFoundError:
                pass
            return "recovered"

        my_agent()
        events = capture_trace["events"]
        error = find_event(events, "agent.error")
        assert error["payload"]["name"] == "openai.chat.completions.create"
        assert error["payload"]["error_type"] == "NotFoundError"
        assert "deployment" in error["payload"]["error"].lower()
        assert "latency_ms" in error["payload"]
        assert not find_events(events, "model.invoke")

        provider.disconnect()


# ---------------------------------------------------------------------------
# Endpoint scrubbing
# ---------------------------------------------------------------------------


class TestEndpointScrubbing:
    def test_connect_records_scrubbed_endpoint(self):
        client, _ = _make_client()
        provider = AzureOpenAIProvider()
        provider.connect(client)

        endpoint = provider._endpoint
        assert endpoint is not None
        assert endpoint.startswith("https://unit-test.openai.azure.com")
        # Never record the query string (api-version / sig material).
        assert "?" not in endpoint
        assert "api-version" not in endpoint
        provider.disconnect()

    def test_scrubbed_endpoint_strips_query(self):
        class FakeClient:
            base_url = f"{_ENDPOINT}/openai?api-key=secret&api-version={_API_VERSION}"

        assert _scrubbed_endpoint(FakeClient()) == f"{_ENDPOINT}/openai"


# ---------------------------------------------------------------------------
# Azure response metadata
# ---------------------------------------------------------------------------


class TestAzureMeta:
    def test_extract_meta_surfaces_azure_attributes(self):
        # The SDK attaches azure attrs to the response object on some paths;
        # OpenAI response models allow extra fields, so build one with them.
        response = ChatCompletion(
            **_chat_completion_json(),
            api_version=_API_VERSION,
            deployment="gpt-4o",
        )
        meta = AzureOpenAIProvider.extract_meta(response)
        assert meta["azure_api_version"] == _API_VERSION
        assert meta["azure_deployment"] == "gpt-4o"
        # Base OpenAI extraction still applies.
        assert meta["response_model"] == "gpt-4o-2024-05-13"
        assert meta["usage"]["total_tokens"] == 20

    def test_extract_meta_without_azure_attributes(self):
        response = ChatCompletion(**_chat_completion_json())
        meta = AzureOpenAIProvider.extract_meta(response)
        assert "azure_api_version" not in meta
        assert "azure_deployment" not in meta


# ---------------------------------------------------------------------------
# Passthrough / lifecycle
# ---------------------------------------------------------------------------


class TestPassthrough:
    def test_no_events_outside_trace(self, mock_client):
        client, requests = _make_client()
        provider = AzureOpenAIProvider()
        provider.connect(client)

        response = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "Hi"}])
        assert response.choices[0].message.content == "Hello from Azure!"
        assert len(requests) == 1
        assert not mock_client.traces.upload.called
        provider.disconnect()


class TestLifecycle:
    def test_adapter_info_and_pricing_table(self):
        provider = AzureOpenAIProvider()
        info = provider.adapter_info()
        assert info.name == "azure_openai"
        assert info.adapter_type == "provider"
        assert info.connected is False
        # Azure-specific pricing override is wired in.
        assert provider.pricing_table is not None
        assert "gpt-4o" in provider.pricing_table

    def test_instrument_and_uninstrument(self):
        client, _ = _make_client()
        original = client.chat.completions.create
        provider = instrument_azure_openai(client)
        assert isinstance(provider, AzureOpenAIProvider)
        assert client.chat.completions.create is not original
        uninstrument_azure_openai()  # must not raise
