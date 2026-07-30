"""Offline streaming + async + redaction + attestation floor for azure_openai.

Closes the W1 census cells that ``test_azure_openai.py`` never exercised. Every
lane drives a REAL ``openai.AzureOpenAI`` / ``openai.AsyncAzureOpenAI`` client
over an ``httpx.MockTransport`` (the proven ``test_azure_openai._make_client``
seam) so deployment-URL routing, ``api-version`` / ``api-key`` handling, SSE
parsing, async awaiting and response deserialization all run through the real
SDK — no Azure subscription, no network, no spend:

* Streaming    — a real ``stream=True`` call over an SSE ``MockTransport``
                 (``text/event-stream``) is aggregated into ONE ``model.invoke``
                 carrying the concatenated deltas + the terminal usage-only
                 chunk's token counts + streaming timing. Bite: a chunk-drop or
                 aggregation regression changes the reassembled content / usage.
* Async        — a real ``AsyncAzureOpenAI`` ``await create`` emits a
                 ``model.invoke`` + ``cost.record`` with real usage and a
                 measured latency. Bite: a sync-miswrap of the async surface
                 would run the extractors on the *un-awaited coroutine* → no
                 usage, no cost.record, ``output_message is None`` — and the
                 patched method would no longer be a coroutine function.
* Redaction    — ``capture_content=False`` strips ``messages`` / ``output_message``
                 (metadata like ``usage`` survives) and a SENTINEL never reaches
                 the stored trace, each with a ``True`` vacuity control proving
                 the assertion is not vacuous.
* Attestation  — the captured azure trace's attestation chain verifies offline
                 (mirrors the live harness ``_assert_attestation``).
"""

from __future__ import annotations

import json
import asyncio
import inspect
from typing import Any, Dict, List, Tuple, Optional

import httpx

from openai import AzureOpenAI, AsyncAzureOpenAI
from layerlens.instrument import trace
from layerlens.instrument._capture_config import CaptureConfig
from layerlens.instrument.adapters.providers.azure_openai import AzureOpenAIProvider

from ...conftest import find_event, find_events

_ENDPOINT = "https://unit-test.openai.azure.com"
_API_VERSION = "2024-06-01"
_API_KEY = "fake-azure-key"

SENTINEL = "LL-SENTINEL-7f3a9c2e"


# ---------------------------------------------------------------------------
# Real-SDK-over-MockTransport builders (mirror test_azure_openai._make_client)
# ---------------------------------------------------------------------------
def _chat_completion_json(
    content: str = "Hello from Azure!",
    model: str = "gpt-4o-2024-05-13",
    prompt_tokens: int = 14,
    completion_tokens: int = 6,
) -> Dict[str, Any]:
    return {
        "id": "chatcmpl-azure-0001",
        "object": "chat.completion",
        "created": 1717418400,
        "model": model,
        "system_fingerprint": "fp_azure_5f2a1b",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _sse_body(
    content_parts: Tuple[str, ...] = ("Hello", " from", " Azure!"),
    *,
    model: str = "gpt-4o-2024-05-13",
    prompt_tokens: int = 14,
    completion_tokens: int = 6,
) -> str:
    """A real Azure chat.completions SSE stream: role chunk, N content deltas, a
    finish chunk, then a terminal usage-only chunk (choices=[]), then [DONE]."""
    rid = "chatcmpl-azure-stream-0001"
    chunks: List[Dict[str, Any]] = [
        {
            "id": rid,
            "object": "chat.completion.chunk",
            "created": 1717418400,
            "model": model,
            "system_fingerprint": "fp_azure_5f2a1b",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
        }
    ]
    for piece in content_parts:
        chunks.append(
            {
                "id": rid,
                "object": "chat.completion.chunk",
                "created": 1717418400,
                "model": model,
                "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
            }
        )
    chunks.append(
        {
            "id": rid,
            "object": "chat.completion.chunk",
            "created": 1717418400,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
    )
    # Terminal usage-only chunk — only emitted by Azure when the request carried
    # stream_options={"include_usage": True}; choices=[] is the real shape.
    chunks.append(
        {
            "id": rid,
            "object": "chat.completion.chunk",
            "created": 1717418400,
            "model": model,
            "choices": [],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
    )
    return "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"


def _json_client(
    response_json: Optional[Dict[str, Any]] = None,
    *,
    async_: bool = False,
) -> Tuple[Any, List[httpx.Request]]:
    requests: List[httpx.Request] = []
    payload = response_json if response_json is not None else _chat_completion_json()

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)
    kwargs: Dict[str, Any] = dict(azure_endpoint=_ENDPOINT, api_key=_API_KEY, api_version=_API_VERSION)
    if async_:
        client: Any = AsyncAzureOpenAI(http_client=httpx.AsyncClient(transport=transport), **kwargs)
    else:
        client = AzureOpenAI(http_client=httpx.Client(transport=transport), **kwargs)
    return client, requests


def _sse_client(sse_text: str) -> Tuple[AzureOpenAI, List[httpx.Request]]:
    requests: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            text=sse_text,
            headers={"content-type": "text/event-stream; charset=utf-8"},
        )

    client = AzureOpenAI(
        azure_endpoint=_ENDPOINT,
        api_key=_API_KEY,
        api_version=_API_VERSION,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    return client, requests


# ---------------------------------------------------------------------------
# 1. Offline streaming via httpx.MockTransport SSE
# ---------------------------------------------------------------------------
class TestStreamingOffline:
    def test_sse_stream_aggregates_into_one_model_invoke(self, mock_client, capture_trace):
        client, requests = _sse_client(_sse_body(content_parts=("Hello", " from", " Azure!")))
        provider = AzureOpenAIProvider()
        provider.connect(client)

        @trace(mock_client, capture_config=CaptureConfig.full())
        def my_agent() -> str:
            stream = client.chat.completions.create(
                model="gpt-4o",  # Azure deployment name
                messages=[{"role": "user", "content": "Hi?"}],
                stream=True,
                stream_options={"include_usage": True},
            )
            parts: List[str] = []
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    parts.append(chunk.choices[0].delta.content)
            return "".join(parts)

        # The real SDK reassembled the streamed deltas.
        assert my_agent() == "Hello from Azure!"

        # Real Azure deployment routing happened over the MockTransport.
        req = requests[0]
        assert req.url.host == "unit-test.openai.azure.com"
        assert "/deployments/gpt-4o/chat/completions" in req.url.path
        assert req.url.params["api-version"] == _API_VERSION

        events = capture_trace["events"]
        # Exactly ONE model.invoke for the whole stream (not one per chunk).
        invokes = find_events(events, "model.invoke")
        assert len(invokes) == 1
        mi = invokes[0]["payload"]
        assert mi["name"] == "openai.chat.completions.create"
        assert mi["framework"] == "azure_openai"
        # Aggregated output is the CONCATENATION of every content delta — a
        # dropped/mis-ordered chunk (or capturing only the first/last) breaks this.
        assert mi["output_message"] == {"role": "assistant", "content": "Hello from Azure!"}
        assert mi["finish_reason"] == "stop"
        # Response (dated) model wins over the deployment name.
        assert mi["model"] == "gpt-4o-2024-05-13"
        assert mi["parameters"]["model"] == "gpt-4o"
        assert mi["parameters"]["stream"] is True
        # Usage came from the terminal usage-only chunk (choices=[]); dropping it
        # would leave usage absent and suppress cost.record below.
        assert mi["usage"]["prompt_tokens"] == 14
        assert mi["usage"]["completion_tokens"] == 6
        assert mi["usage"]["total_tokens"] == 20
        # Streaming-specific timing only present on the streamed path.
        assert mi["streaming_duration_ms"] >= 0
        assert "ttft_ms" in mi

        cost = find_event(events, "cost.record")["payload"]
        assert cost["provider"] == "openai"  # derived from the openai patch surface
        assert cost["model"] == "gpt-4o-2024-05-13"
        assert cost["total_tokens"] == 20
        assert cost["cost_usd"] > 0
        provider.disconnect()


# ---------------------------------------------------------------------------
# 2. AsyncAzureOpenAI coverage (guards a sync-miswrap of the async surface)
# ---------------------------------------------------------------------------
class TestAsyncOffline:
    def test_async_create_emits_usage_and_latency(self, mock_client, capture_trace):
        client, requests = _json_client(
            _chat_completion_json(content="Async hello", prompt_tokens=11, completion_tokens=4),
            async_=True,
        )
        provider = AzureOpenAIProvider()
        provider.connect(client)

        # Structural guard: the async surface MUST be wrapped by the async
        # wrapper. A sync-miswrap replaces it with a plain function that returns
        # the un-awaited coroutine.
        assert inspect.iscoroutinefunction(client.chat.completions.create)

        @trace(mock_client, capture_config=CaptureConfig.full())
        def my_agent() -> str:
            async def _run() -> str:
                r = await client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": "Hi async?"}],
                )
                return r.choices[0].message.content

            return asyncio.run(_run())

        assert my_agent() == "Async hello"
        assert requests[0].url.host == "unit-test.openai.azure.com"

        events = capture_trace["events"]
        mi = find_event(events, "model.invoke")["payload"]
        assert mi["framework"] == "azure_openai"
        # Real awaited response — a sync-miswrap would extract from the coroutine
        # and produce output_message=None with no usage block.
        assert mi["output_message"] == {"role": "assistant", "content": "Async hello"}
        assert mi["usage"]["prompt_tokens"] == 11
        assert mi["usage"]["completion_tokens"] == 4
        assert mi["usage"]["total_tokens"] == 15
        # Real awaited latency (measured after the await, not coroutine construction).
        assert mi["latency_ms"] > 0

        # cost.record only exists because usage was extracted from the AWAITED
        # response — the sharpest behavioral bite for the sync-miswrap.
        cost = find_event(events, "cost.record")["payload"]
        assert cost["model"] == "gpt-4o-2024-05-13"
        assert cost["total_tokens"] == 15
        assert cost["cost_usd"] > 0
        provider.disconnect()


# ---------------------------------------------------------------------------
# 3. Content-redaction mode (paired vacuity control + SENTINEL sweep)
# ---------------------------------------------------------------------------
class TestRedactionFloor:
    def _run(self, mock_client, config, *, content: str, prompt: str) -> None:
        client, _ = _json_client(_chat_completion_json(content=content))
        provider = AzureOpenAIProvider()
        provider.connect(client)

        @trace(mock_client, capture_config=config)
        def my_agent() -> str:
            r = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
            )
            return r.choices[0].message.content

        my_agent()
        provider.disconnect()

    def test_content_stripped_when_capture_content_false(self, mock_client, capture_trace):
        self._run(
            mock_client,
            CaptureConfig(capture_content=False),
            content=f"secret {SENTINEL}",
            prompt=f"remember {SENTINEL}",
        )
        mi = find_event(capture_trace["events"], "model.invoke")["payload"]
        assert "messages" not in mi
        assert "output_message" not in mi
        # Redaction removes CONTENT, not metadata.
        assert mi["usage"]["total_tokens"] == 20
        assert mi["parameters"]["model"] == "gpt-4o"

    def test_content_present_when_capture_content_true(self, mock_client, capture_trace):
        """Vacuity control: the strip assertion above is only meaningful if the
        SAME path DOES carry content when capture is on."""
        self._run(
            mock_client,
            CaptureConfig.full(),
            content="Hello from Azure!",
            prompt="Say hello",
        )
        mi = find_event(capture_trace["events"], "model.invoke")["payload"]
        assert mi["output_message"] == {"role": "assistant", "content": "Hello from Azure!"}
        assert mi["messages"] == [{"role": "user", "content": "Say hello"}]

    def test_sentinel_never_leaks_when_redacted(self, mock_client, capture_trace):
        self._run(
            mock_client,
            CaptureConfig(capture_content=False),
            content=f"secret {SENTINEL}",
            prompt=f"remember {SENTINEL}",
        )
        blob = json.dumps(capture_trace["events"])
        assert SENTINEL not in blob, "content redaction leaked the SENTINEL into the stored trace"

    def test_sentinel_present_when_capture_on(self, mock_client, capture_trace):
        """Vacuity control for the sweep above."""
        self._run(
            mock_client,
            CaptureConfig.full(),
            content=f"secret {SENTINEL}",
            prompt=f"remember {SENTINEL}",
        )
        assert SENTINEL in json.dumps(capture_trace["events"])


# ---------------------------------------------------------------------------
# 4. Offline attestation-chain verification over a captured azure trace
# ---------------------------------------------------------------------------
class TestAttestationOffline:
    def test_attestation_chain_verifies_offline(self, mock_client, capture_trace):
        from layerlens.attestation._verify import verify_chain
        from layerlens.attestation._envelope import HashScope, AttestationEnvelope

        client, _ = _json_client()
        provider = AzureOpenAIProvider()
        provider.connect(client)

        @trace(mock_client, capture_config=CaptureConfig.full())
        def my_agent() -> str:
            r = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello?"}],
            )
            return r.choices[0].message.content

        assert my_agent() == "Hello from Azure!"
        provider.disconnect()

        events = capture_trace["events"]
        raw = (capture_trace["attestation"].get("chain") or {}).get("events") or []
        envelopes = [
            AttestationEnvelope(
                hash=e["hash"],
                scope=HashScope(e["scope"]),
                previous_hash=e.get("previous_hash"),
            )
            for e in raw
        ]
        assert envelopes, "no attestation envelopes captured"
        assert len(envelopes) == len(events)
        result = verify_chain(envelopes)
        assert result.valid, f"attestation chain invalid: {result.error}"
