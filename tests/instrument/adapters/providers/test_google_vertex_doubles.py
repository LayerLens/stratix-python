"""Deterministic doubles for the Google Vertex provider adapter (LAY-3582 / T8).

Google Vertex is credential-gated (no GCP project), so these tests stand in
for live verification. The adapter is duck-typed (it wraps any object exposing
``generate_content`` / ``generate_content_async``), so the doubles here are
plain classes that mirror the REAL ``vertexai.generative_models``
``GenerationResponse`` shape: ``candidates[].content.parts[].text``,
enum-like ``finish_reason`` (``.name``/``.value``), and ``usage_metadata``
with ``prompt_token_count`` / ``candidates_token_count`` /
``total_token_count`` (+ ``thoughts_token_count`` on newer SDKs).
Where the lock provides ``google-cloud-aiplatform``, the extractors are also
run against a REAL proto-backed ``GenerationResponse`` built via
``from_dict`` (no credentials needed).

RESIDUAL RISK (register input): the real vertexai ``GenerativeModel``
object, its credential flow, and the gRPC/REST transport remain unexercised —
they require GCP credentials. Notably, ``vertexai``'s ``GenerativeModel``
stores a fully-qualified resource name (``publishers/google/models/<id>`` or
``projects/.../models/<id>``) in ``_model_name``, which the adapter's
``_strip_models_prefix`` does not strip; whether pricing lookups resolve for
real vertexai (as opposed to ``google-generativeai``-style ``models/<id>``
names, covered here) is unverifiable without live access.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

import pytest

from layerlens.instrument import trace
from layerlens.instrument.adapters.providers.google_vertex import GoogleVertexProvider

from ...conftest import find_event, find_events

# ---------------------------------------------------------------------------
# Doubles mirroring vertexai.generative_models.GenerationResponse
# ---------------------------------------------------------------------------


class FakePart:
    def __init__(self, text: Optional[str] = None, function_call: Any = None) -> None:
        self.text = text
        self.function_call = function_call


class FakeContent:
    def __init__(self, parts: List[FakePart], role: str = "model") -> None:
        self.role = role
        self.parts = parts


class FakeFinishReason:
    """Enum-like double (proto-plus enums expose .name and .value)."""

    def __init__(self, name: str, value: int) -> None:
        self.name = name
        self.value = value


class FakeCandidate:
    def __init__(self, content: FakeContent, finish_reason: Optional[FakeFinishReason]) -> None:
        self.content = content
        self.finish_reason = finish_reason
        self.index = 0
        self.safety_ratings: List[Any] = []


class FakeUsageMetadata:
    def __init__(
        self,
        prompt_token_count: int,
        candidates_token_count: int,
        total_token_count: int,
        thoughts_token_count: Optional[int] = None,
    ) -> None:
        self.prompt_token_count = prompt_token_count
        self.candidates_token_count = candidates_token_count
        self.total_token_count = total_token_count
        if thoughts_token_count is not None:
            self.thoughts_token_count = thoughts_token_count


class FakeGenerationResponse:
    def __init__(
        self,
        text: str = "Gemini says hello.",
        prompt_tokens: int = 1234,
        completion_tokens: int = 87,
        total_tokens: Optional[int] = 1321,
        finish_reason: Optional[str] = "STOP",
        thoughts_token_count: Optional[int] = None,
        candidates: Optional[List[FakeCandidate]] = None,
    ) -> None:
        if candidates is not None:
            self.candidates = candidates
        else:
            self.candidates = [
                FakeCandidate(
                    content=FakeContent(parts=[FakePart(text=text)]),
                    finish_reason=FakeFinishReason(finish_reason, 1) if finish_reason else None,
                )
            ]
        self.usage_metadata = FakeUsageMetadata(
            prompt_token_count=prompt_tokens,
            candidates_token_count=completion_tokens,
            total_token_count=total_tokens if total_tokens is not None else 0,
            thoughts_token_count=thoughts_token_count,
        )


class FakeGenerativeModel:
    """google-generativeai-style model surface: model_name + generate_content(_async)."""

    def __init__(self, model_name: str = "models/gemini-2.5-pro", response: Optional[Any] = None) -> None:
        self.model_name = model_name
        self._response = response if response is not None else FakeGenerationResponse()
        self.calls: List[Dict[str, Any]] = []

    def generate_content(self, contents: Any, **kwargs: Any) -> Any:
        self.calls.append({"method": "sync", "contents": contents, **kwargs})
        if isinstance(self._response, Exception):
            raise self._response
        return self._response

    async def generate_content_async(self, contents: Any, **kwargs: Any) -> Any:
        self.calls.append({"method": "async", "contents": contents, **kwargs})
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


def _connect(response: Optional[Any] = None, model_name: str = "models/gemini-2.5-pro") -> tuple:
    model = FakeGenerativeModel(model_name=model_name, response=response)
    provider = GoogleVertexProvider()
    provider.connect(model)
    return provider, model


# ---------------------------------------------------------------------------
# Sync generate_content
# ---------------------------------------------------------------------------


class TestSyncGenerateContent:
    def test_model_invoke_usage_and_cost(self, mock_client, capture_trace):
        provider, model = _connect()

        @trace(mock_client)
        def my_agent():
            r = model.generate_content("Summarize the Q2 report", temperature=0.1)
            return r.candidates[0].content.parts[0].text

        assert my_agent() == "Gemini says hello."

        events = capture_trace["events"]
        mi = find_event(events, "model.invoke")
        assert mi["payload"]["name"] == "google_vertex.generate_content"
        assert mi["payload"]["model"] == "gemini-2.5-pro"  # models/ prefix stripped
        assert mi["payload"]["output_message"] == {"role": "model", "content": "Gemini says hello."}
        assert mi["payload"]["usage"]["prompt_tokens"] == 1234
        assert mi["payload"]["usage"]["completion_tokens"] == 87
        assert mi["payload"]["usage"]["total_tokens"] == 1321
        assert mi["payload"]["finish_reason"] == "STOP"
        assert mi["payload"]["parameters"]["temperature"] == 0.1
        assert mi["payload"]["latency_ms"] >= 0

        cost = find_event(events, "cost.record")
        assert cost["payload"]["provider"] == "google_vertex"
        assert cost["payload"]["model"] == "gemini-2.5-pro"
        assert cost["payload"]["total_tokens"] == 1321
        # PRICING gemini-2.5-pro: 1234 * 0.00125/1k + 87 * 0.01/1k.
        assert cost["payload"]["cost_usd"] == pytest.approx(0.0024125)

        provider.disconnect()

    def test_error_emits_agent_error(self, mock_client, capture_trace):
        provider, model = _connect(response=RuntimeError("503 The service is currently unavailable"))

        @trace(mock_client)
        def my_agent():
            try:
                model.generate_content("Hi")
            except RuntimeError:
                pass
            return "recovered"

        my_agent()
        error = find_event(capture_trace["events"], "agent.error")
        assert error["payload"]["name"] == "google_vertex.generate_content"
        assert "503" in error["payload"]["error"]
        provider.disconnect()


# ---------------------------------------------------------------------------
# Async generate_content_async
# ---------------------------------------------------------------------------


class TestAsyncGenerateContent:
    def test_model_invoke_usage_and_cost(self, mock_client, capture_trace):
        provider, model = _connect()
        seen: List[Any] = []

        @trace(mock_client)
        def my_agent():
            r = asyncio.run(model.generate_content_async("Summarize the Q2 report"))
            seen.append(r)
            return r.candidates[0].content.parts[0].text

        assert my_agent() == "Gemini says hello."
        # Passthrough: the caller receives the exact response object.
        assert seen[0] is model._response
        assert model.calls[0]["method"] == "async"

        events = capture_trace["events"]
        mi = find_event(events, "model.invoke")
        assert mi["payload"]["name"] == "google_vertex.generate_content"
        assert mi["payload"]["model"] == "gemini-2.5-pro"
        assert mi["payload"]["usage"]["prompt_tokens"] == 1234
        assert mi["payload"]["usage"]["completion_tokens"] == 87
        assert mi["payload"]["usage"]["total_tokens"] == 1321

        cost = find_event(events, "cost.record")
        assert cost["payload"]["provider"] == "google_vertex"
        assert cost["payload"]["cost_usd"] == pytest.approx(0.0024125)

        provider.disconnect()

    def test_async_error_emits_agent_error(self, mock_client, capture_trace):
        provider, model = _connect(response=RuntimeError("DeadlineExceeded: 504"))

        @trace(mock_client)
        def my_agent():
            try:
                asyncio.run(model.generate_content_async("Hi"))
            except RuntimeError:
                pass
            return "recovered"

        my_agent()
        error = find_event(capture_trace["events"], "agent.error")
        assert error["payload"]["name"] == "google_vertex.generate_content"
        assert "DeadlineExceeded" in error["payload"]["error"]
        provider.disconnect()


# ---------------------------------------------------------------------------
# Usage extraction edge cases
# ---------------------------------------------------------------------------


class TestUsageExtraction:
    def test_total_falls_back_to_prompt_plus_completion(self):
        response = FakeGenerationResponse(prompt_tokens=100, completion_tokens=25, total_tokens=0)
        meta = GoogleVertexProvider.extract_meta(response)
        assert meta["usage"]["total_tokens"] == 125

    def test_thoughts_tokens_surface_as_reasoning_tokens(self):
        response = FakeGenerationResponse(thoughts_token_count=64)
        meta = GoogleVertexProvider.extract_meta(response)
        assert meta["usage"]["reasoning_tokens"] == 64

    def test_no_candidates_yields_no_output(self, mock_client, capture_trace):
        response = FakeGenerationResponse(candidates=[])
        provider, model = _connect(response=response)

        @trace(mock_client)
        def my_agent():
            model.generate_content("Hi")
            return "done"

        my_agent()
        mi = find_event(capture_trace["events"], "model.invoke")
        assert mi["payload"]["output_message"] is None
        # Usage still extracted even without candidates.
        assert mi["payload"]["usage"]["prompt_tokens"] == 1234
        provider.disconnect()


# ---------------------------------------------------------------------------
# Real vertexai SDK response objects (no credentials required)
# ---------------------------------------------------------------------------


class TestRealVertexSdkResponse:
    def test_extractors_against_proto_backed_response(self):
        """Cross-check the doubles above against a REAL proto-backed
        GenerationResponse (built via from_dict — needs the SDK, not creds)."""
        gm = pytest.importorskip("vertexai.generative_models")
        response = gm.GenerationResponse.from_dict(
            {
                "candidates": [
                    {
                        "content": {"role": "model", "parts": [{"text": "Gemini says hello."}]},
                        "finish_reason": "STOP",
                    }
                ],
                "usage_metadata": {
                    "prompt_token_count": 1234,
                    "candidates_token_count": 87,
                    "total_token_count": 1321,
                },
            }
        )

        output = GoogleVertexProvider.extract_output(response)
        assert output == {"role": "model", "content": "Gemini says hello."}

        meta = GoogleVertexProvider.extract_meta(response)
        assert meta["usage"]["prompt_tokens"] == 1234
        assert meta["usage"]["completion_tokens"] == 87
        assert meta["usage"]["total_tokens"] == 1321
        assert meta["finish_reason"] == "STOP"

        assert GoogleVertexProvider.extract_tool_calls(response) == []


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_disconnect_restores_methods(self, mock_client, capture_trace):
        model = FakeGenerativeModel()
        sync_orig = model.generate_content
        async_orig = model.generate_content_async
        provider = GoogleVertexProvider()
        provider.connect(model)
        assert model.generate_content != sync_orig
        provider.disconnect()
        # Bound methods compare equal when __self__/__func__ match.
        assert model.generate_content == sync_orig
        assert model.generate_content_async == async_orig

        @trace(mock_client)
        def my_agent():
            model.generate_content("Hi")
            return "done"

        my_agent()
        assert not find_events(capture_trace["events"], "model.invoke")
