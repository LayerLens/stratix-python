"""Sample: instrument the real Azure OpenAI client with the LayerLens adapter.

Runs a single chat completion through ``AzureOpenAIAdapter`` with an
``HttpEventSink`` pointed at atlas-app. Every event the adapter emits
(``model.invoke``, ``cost.record``, optional ``tool.call``) is shipped
to the platform's telemetry ingest endpoint with Azure-specific metadata
(``azure_endpoint``, ``api_version``, the deployment used as ``model``).

Required environment:

* ``AZURE_OPENAI_API_KEY`` — the resource's API key.
* ``AZURE_OPENAI_ENDPOINT`` — e.g. ``https://my-resource.openai.azure.com/``.
* ``AZURE_OPENAI_API_VERSION`` — e.g. ``2024-08-01-preview``.
* ``AZURE_OPENAI_DEPLOYMENT`` — the deployment name you configured in the
  Azure portal (NOT the underlying base model name). The adapter passes
  this as ``model=`` to ``client.chat.completions.create``.
* ``LAYERLENS_STRATIX_API_KEY`` — your LayerLens API key (optional;
  defaults to anonymous if unset).
* ``LAYERLENS_STRATIX_BASE_URL`` — atlas-app base URL (optional;
  defaults to ``https://api.layerlens.ai/api/v1``).

Run::

    pip install 'layerlens[providers-azure-openai]'
    python -m samples.instrument.azure_openai.main

Mock-fixture mode (no real Azure call): set ``LAYERLENS_SAMPLE_MOCK=1``
and the sample stubs the SDK transport with a canned response so the
event flow can be exercised offline. Useful in CI / smoke tests.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict

from layerlens.instrument.transport.sink_http import HttpEventSink

from layerlens.instrument.adapters._base import CaptureConfig
from layerlens.instrument.adapters.providers.azure_openai_adapter import AzureOpenAIAdapter


def _canned_chat_response() -> Dict[str, Any]:
    """Used in mock-fixture mode (LAYERLENS_SAMPLE_MOCK=1)."""
    return {
        "id": "chatcmpl-azure-mock",
        "object": "chat.completion",
        "created": 1730000000,
        "model": "gpt-4o-2024-08-06",
        "system_fingerprint": "fp-azure-mock",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "2 + 2 = 4",
                    "tool_calls": None,
                    "refusal": None,
                },
                "finish_reason": "stop",
                "logprobs": None,
            }
        ],
        "usage": {
            "prompt_tokens": 14,
            "completion_tokens": 6,
            "total_tokens": 20,
            "prompt_tokens_details": {"cached_tokens": 0},
            "completion_tokens_details": {"reasoning_tokens": 0},
        },
    }


def _build_mock_client() -> Any:
    """Build an AzureOpenAI client whose own httpx transport returns canned data.

    A scoped ``httpx.MockTransport`` is installed on the AzureOpenAI client
    only — the global httpx pool (and therefore the LayerLens HttpEventSink)
    is unaffected. No respx side-effects, no test-only globals leaking into
    a sample script.
    """
    import httpx

    from openai import AzureOpenAI

    base = os.environ.get(
        "AZURE_OPENAI_ENDPOINT", "https://layerlens-sample.openai.azure.com/"
    ).rstrip("/")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")

    def _handler(request: httpx.Request) -> httpx.Response:
        # Only the chat/completions deployment URL gets a canned reply.
        if "/openai/deployments/" in request.url.path and request.url.path.endswith(
            "/chat/completions"
        ):
            return httpx.Response(200, json=_canned_chat_response())
        return httpx.Response(404, json={"error": "not mocked"})

    transport = httpx.MockTransport(_handler)
    http_client = httpx.Client(transport=transport)

    return AzureOpenAI(
        api_key="mock-key",
        api_version=api_version,
        azure_endpoint=base,
        http_client=http_client,
    )


def main() -> int:
    mock_mode = os.environ.get("LAYERLENS_SAMPLE_MOCK") == "1"

    if not mock_mode:
        missing = [
            var
            for var in (
                "AZURE_OPENAI_API_KEY",
                "AZURE_OPENAI_ENDPOINT",
                "AZURE_OPENAI_API_VERSION",
                "AZURE_OPENAI_DEPLOYMENT",
            )
            if not os.environ.get(var)
        ]
        if missing:
            print(
                "Missing required Azure env vars: " + ", ".join(missing) + "\n"
                "Either set them or run with LAYERLENS_SAMPLE_MOCK=1 for the "
                "offline mock fixture.",
                file=sys.stderr,
            )
            return 2

    try:
        from openai import AzureOpenAI
    except ImportError:
        print(
            "openai package not installed. Install with:\n"
            "    pip install 'layerlens[providers-azure-openai]'",
            file=sys.stderr,
        )
        return 2

    sink = HttpEventSink(
        adapter_name="azure_openai",
        path="/telemetry/spans",
        max_batch=10,
        flush_interval_s=1.0,
    )

    adapter = AzureOpenAIAdapter(capture_config=CaptureConfig.standard())
    adapter.add_sink(sink)
    adapter.connect()

    if mock_mode:
        client = _build_mock_client()
        deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mock-deployment")
    else:
        client = AzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        )
        deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]

    adapter.connect_client(client)

    try:
        response = client.chat.completions.create(
            model=deployment,  # Azure: this is the DEPLOYMENT name, not the base model.
            messages=[
                {"role": "system", "content": "You are a concise assistant."},
                {"role": "user", "content": "What is 2 + 2?"},
            ],
            max_tokens=20,
        )
        choice = response.choices[0].message.content if response.choices else "(empty)"
        usage = response.usage
        print(f"Response: {choice}")
        if usage is not None:
            print(
                f"Tokens - prompt: {usage.prompt_tokens}, completion: "
                f"{usage.completion_tokens}, total: {usage.total_tokens}"
            )
    finally:
        sink.close()
        adapter.disconnect()

    print("Telemetry shipped. Check the LayerLens dashboard adapter health page.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
