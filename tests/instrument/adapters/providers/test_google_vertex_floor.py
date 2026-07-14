"""Offline privacy + error + attestation + params + streaming floor for the
``google_vertex`` provider (ADP-W2).

Closes the W2 census ◑/gap cells that were previously proven only in the
credential-gated (``blocked=true``) live lane or only with synthetic inputs, so
a regression now fails in plain CI with no GCP credentials and no network. Every
assertion is driven through the REAL ``GoogleVertexProvider`` wrapping a REAL
proto-backed ``vertexai.generative_models.GenerationResponse`` (built via
``from_dict`` — needs the SDK, not creds) or a REAL ``google.api_core`` SDK
exception. The Vertex ``GenerativeModel`` is duck-typed by design (the adapter
wraps any object exposing ``generate_content`` / ``generate_content_async``), so
the model surface is a thin local double whose only job is to hand back the real
response object / raise the real exception — the *response shape* and the
*exception class* are the real things under test.

* Redaction   — ``capture_content=False`` strips ``messages`` AND
                ``output_message`` from ``model.invoke`` (usage + safe params
                remain) with a ``True`` vacuity control, plus a SENTINEL sweep
                over the serialized events (absent when off, present when on),
                including through the streaming-aggregation path.
* Error-paths — REAL ``google.api_core.exceptions`` (``ResourceExhausted`` 429 /
                ``DeadlineExceeded`` 504 / ``PermissionDenied`` 403) fed through
                the instrumented sync + async call surface as ``agent.error``
                with ``error_type`` == the real SDK class name (not the synthetic
                ``RuntimeError`` the existing doubles suite uses).
* Attestation — the captured trace's hash chain verifies offline (one envelope
                per event) with a broken-link tamper control.
* Params      — the 7-item ``_CAPTURE_PARAMS`` allowlist is enforced end-to-end:
                every allowlisted key that was passed appears in
                ``model.invoke.parameters`` and non-allowlisted kwargs (carrying
                a SENTINEL) do not.
* Cost        — ``cost.record.cost_usd`` is present + exact on a real Gemini
                token shape (bare model id resolved from the real
                ``publishers/google/models/<id>`` resource form, LAY-3615).
* Streaming   — ``aggregate_stream`` / ``_AggregatedVertexResponse`` over a
                multi-chunk stream (text deltas + a ``function_call`` part +
                ``usage_metadata`` on the final chunk) so
                ``model.invoke`` / ``tool.call`` / ``cost.record`` survive
                aggregation with the final-chunk usage + finish_reason.
"""

from __future__ import annotations

import json
import asyncio
from typing import Any, List, Tuple, Optional

import pytest
from google.api_core.exceptions import (
    DeadlineExceeded,
    PermissionDenied,
    ResourceExhausted,
)
from vertexai.generative_models import GenerationResponse

from layerlens.instrument import trace
from layerlens.instrument._capture_config import CaptureConfig
from layerlens.instrument.adapters.providers.google_vertex import GoogleVertexProvider

from ...conftest import find_event

SENTINEL = "LL-SENTINEL-7f3a9c2e"

# The real Vertex ``GenerativeModel`` stores the reconciled resource name, not a
# bare id — this exercises ``_strip_models_prefix`` (LAY-3615) so pricing + the
# model field resolve against the real shape.
_MODEL_RESOURCE = "publishers/google/models/gemini-2.5-pro"


# ---------------------------------------------------------------------------
# Real proto-backed response / chunk builders (no credentials required)
# ---------------------------------------------------------------------------
def _response(
    text: str = "Gemini says hello.",
    *,
    prompt: int = 1000,
    completion: int = 200,
    total: Optional[int] = None,
    finish: str = "STOP",
    tool_calls: Optional[List[Tuple[str, dict]]] = None,
) -> GenerationResponse:
    """A REAL ``vertexai`` proto ``GenerationResponse`` with the given content."""
    parts: List[dict] = []
    if text:
        parts.append({"text": text})
    for name, args in tool_calls or []:
        parts.append({"function_call": {"name": name, "args": args}})
    return GenerationResponse.from_dict(
        {
            "candidates": [{"content": {"role": "model", "parts": parts}, "finish_reason": finish}],
            "usage_metadata": {
                "prompt_token_count": prompt,
                "candidates_token_count": completion,
                "total_token_count": total if total is not None else prompt + completion,
            },
        }
    )


def _chunk(
    *,
    text: Optional[str] = None,
    tool_call: Optional[Tuple[str, dict]] = None,
    finish: Optional[str] = None,
    usage: Optional[Tuple[int, int, int]] = None,
) -> GenerationResponse:
    """A REAL proto streaming chunk. Non-final chunks omit usage/finish, mirroring
    Vertex's wire contract (cumulative ``usage_metadata`` + finish on the LAST
    chunk)."""
    parts: List[dict] = []
    if text is not None:
        parts.append({"text": text})
    if tool_call is not None:
        name, args = tool_call
        parts.append({"function_call": {"name": name, "args": args}})
    candidate: dict = {"content": {"role": "model", "parts": parts}}
    if finish is not None:
        candidate["finish_reason"] = finish
    payload: dict = {"candidates": [candidate]}
    if usage is not None:
        p, c, t = usage
        payload["usage_metadata"] = {
            "prompt_token_count": p,
            "candidates_token_count": c,
            "total_token_count": t,
        }
    return GenerationResponse.from_dict(payload)


class _VertexModel:
    """Duck-typed stand-in for a ``vertexai`` ``GenerativeModel``.

    The adapter is explicitly duck-typed (it wraps any object exposing
    ``generate_content`` / ``generate_content_async``). This double only routes
    the call — the payload it returns is a REAL proto ``GenerationResponse`` (or
    a list of them for streaming), and an error scenario raises a REAL
    ``google.api_core`` exception. Network is the sole thing mocked.
    """

    def __init__(self, response: Any, *, model_name: str = _MODEL_RESOURCE) -> None:
        self.model_name = model_name
        self._response = response
        self.calls: List[dict] = []

    def generate_content(self, contents: Any = None, **kwargs: Any) -> Any:
        self.calls.append({"contents": contents, **kwargs})
        if isinstance(self._response, Exception):
            raise self._response
        if kwargs.get("stream") is True:
            return iter(self._response)
        return self._response

    async def generate_content_async(self, contents: Any = None, **kwargs: Any) -> Any:
        self.calls.append({"contents": contents, **kwargs})
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


def _run(model: _VertexModel, mock_client: Any, config: CaptureConfig, *, prompt: str = "Hi", **call_kwargs: Any) -> Any:
    """Drive the REAL provider over the model double under an active trace."""
    provider = GoogleVertexProvider()
    provider.connect(model)
    try:

        @trace(mock_client, capture_config=config)
        def my_agent() -> Any:
            r = model.generate_content(contents=prompt, **call_kwargs)
            if r.candidates and r.candidates[0].content.parts:
                first = r.candidates[0].content.parts[0]
                return getattr(first, "text", None) or "done"
            return "done"

        return my_agent()
    finally:
        provider.disconnect()


# ---------------------------------------------------------------------------
# Redaction floor (offline, credential-free)
# ---------------------------------------------------------------------------
class TestRedactionFloor:
    def test_content_stripped_when_capture_content_false(self, mock_client, capture_trace):
        _run(
            _VertexModel(_response("I am Gemini!")),
            mock_client,
            CaptureConfig(capture_content=False),
            prompt="Hi",
            temperature=0.2,
        )

        mi = find_event(capture_trace["events"], "model.invoke")
        assert "messages" not in mi["payload"]
        assert "output_message" not in mi["payload"]
        # Redaction removes CONTENT, not metadata: usage + safe params remain.
        assert mi["payload"]["usage"]["completion_tokens"] == 200
        assert mi["payload"]["parameters"]["temperature"] == 0.2
        assert mi["payload"]["model"] == "gemini-2.5-pro"

    def test_content_present_when_capture_content_true(self, mock_client, capture_trace):
        """Vacuity control: the SAME path DOES carry content when capture is on."""
        _run(_VertexModel(_response("I am Gemini!")), mock_client, CaptureConfig.full(), prompt="Hi")

        mi = find_event(capture_trace["events"], "model.invoke")
        assert mi["payload"]["output_message"] == {"role": "model", "content": "I am Gemini!"}
        assert mi["payload"]["messages"] == "Hi"

    def test_sentinel_never_leaks_when_redacted(self, mock_client, capture_trace):
        _run(
            _VertexModel(_response(f"Secret is {SENTINEL}")),
            mock_client,
            CaptureConfig(capture_content=False),
            prompt=f"Remember {SENTINEL}",
        )
        blob = json.dumps(capture_trace["events"])
        assert SENTINEL not in blob, "content redaction leaked the SENTINEL into the stored trace"

    def test_sentinel_present_when_capture_on(self, mock_client, capture_trace):
        """Vacuity control for the sweep above."""
        _run(
            _VertexModel(_response(f"Secret is {SENTINEL}")),
            mock_client,
            CaptureConfig.full(),
            prompt=f"Remember {SENTINEL}",
        )
        assert SENTINEL in json.dumps(capture_trace["events"])


# ---------------------------------------------------------------------------
# Real error-shape floor (feeds real google.api_core SDK exceptions)
# ---------------------------------------------------------------------------
class TestRealErrorShape:
    def test_resource_exhausted_emits_agent_error(self, mock_client, capture_trace):
        err = ResourceExhausted("Quota exceeded for quota metric 'generate_content_requests'")
        provider = GoogleVertexProvider()
        provider.connect(_VertexModel(err))
        model = provider._client

        @trace(mock_client, capture_config=CaptureConfig.full())
        def my_agent():
            try:
                model.generate_content(contents="Hi")
            except ResourceExhausted:
                pass
            return "handled"

        my_agent()
        provider.disconnect()

        error = find_event(capture_trace["events"], "agent.error")
        # The REAL SDK exception class name — not the synthetic RuntimeError.
        assert error["payload"]["error_type"] == "ResourceExhausted"
        assert error["payload"]["name"] == "google_vertex.generate_content"
        assert "429" in error["payload"]["error"]
        assert "latency_ms" in error["payload"]

    def test_permission_denied_emits_agent_error(self, mock_client, capture_trace):
        err = PermissionDenied("The caller does not have permission")
        provider = GoogleVertexProvider()
        provider.connect(_VertexModel(err))
        model = provider._client

        @trace(mock_client, capture_config=CaptureConfig.full())
        def my_agent():
            try:
                model.generate_content(contents="Hi")
            except PermissionDenied:
                pass
            return "handled"

        my_agent()
        provider.disconnect()

        error = find_event(capture_trace["events"], "agent.error")
        assert error["payload"]["error_type"] == "PermissionDenied"
        assert "403" in error["payload"]["error"]

    def test_deadline_exceeded_async_emits_agent_error(self, mock_client, capture_trace):
        err = DeadlineExceeded("Deadline of 60.0s exceeded")
        provider = GoogleVertexProvider()
        provider.connect(_VertexModel(err))
        model = provider._client

        @trace(mock_client, capture_config=CaptureConfig.full())
        def my_agent():
            try:
                asyncio.run(model.generate_content_async(contents="Hi"))
            except DeadlineExceeded:
                pass
            return "handled"

        my_agent()
        provider.disconnect()

        error = find_event(capture_trace["events"], "agent.error")
        assert error["payload"]["error_type"] == "DeadlineExceeded"
        assert error["payload"]["name"] == "google_vertex.generate_content"
        assert "504" in error["payload"]["error"]


# ---------------------------------------------------------------------------
# Offline attestation-chain verification (+ tamper control)
# ---------------------------------------------------------------------------
class TestAttestationOffline:
    @staticmethod
    def _envelopes(capture_trace):
        from layerlens.attestation._envelope import HashScope, AttestationEnvelope

        raw = (capture_trace["attestation"].get("chain") or {}).get("events") or []
        return [
            AttestationEnvelope(hash=e["hash"], scope=HashScope(e["scope"]), previous_hash=e.get("previous_hash"))
            for e in raw
        ]

    def test_attestation_chain_verifies_offline(self, mock_client, capture_trace):
        from layerlens.attestation._verify import verify_chain

        _run(_VertexModel(_response()), mock_client, CaptureConfig.full())

        envelopes = self._envelopes(capture_trace)
        assert envelopes, "no attestation envelopes captured"
        assert len(envelopes) == len(capture_trace["events"])
        result = verify_chain(envelopes)
        assert result.valid, f"attestation chain invalid: {result.error}"

    def test_tampered_chain_is_detected(self, mock_client, capture_trace):
        """Control: mutating one link's previous_hash must break verification —
        proves the check above is not vacuously green."""
        from layerlens.attestation._verify import verify_chain
        from layerlens.attestation._envelope import HashScope, AttestationEnvelope

        _run(_VertexModel(_response()), mock_client, CaptureConfig.full())
        envelopes = self._envelopes(capture_trace)
        assert len(envelopes) >= 2

        tampered = list(envelopes)
        broken = envelopes[1]
        tampered[1] = AttestationEnvelope(
            hash=broken.hash,
            scope=HashScope.EVENT,
            previous_hash="deadbeef" * 8,  # no longer matches envelopes[0].hash
        )
        result = verify_chain(tampered)
        assert not result.valid
        assert result.break_index == 1


# ---------------------------------------------------------------------------
# Params allowlist enforced end-to-end (all 7 _CAPTURE_PARAMS + unknown dropped)
# ---------------------------------------------------------------------------
class TestParamsAllowlist:
    def test_all_capture_params_kept_unknown_dropped(self, mock_client, capture_trace):
        _run(
            _VertexModel(_response()),
            mock_client,
            CaptureConfig.full(),
            prompt="Hi",
            # all 7 allowlisted _CAPTURE_PARAMS
            temperature=0.3,
            max_output_tokens=512,
            top_p=0.95,
            top_k=40,
            stream=False,
            generation_config={"candidate_count": 1},
            tools=[{"function_declarations": [{"name": "get_policy_details"}]}],
            # non-allowlisted kwargs — must NOT reach parameters, must NOT leak SENTINEL.
            labels={"trace": SENTINEL},
            system_instruction=f"You are {SENTINEL}",
            safety_settings=[{"category": "HARM", "threshold": "BLOCK_NONE"}],
        )

        params = find_event(capture_trace["events"], "model.invoke")["payload"]["parameters"]
        # Every allowlisted key survives with its passed value.
        assert params["temperature"] == 0.3
        assert params["max_output_tokens"] == 512
        assert params["top_p"] == 0.95
        assert params["top_k"] == 40
        assert params["stream"] is False
        assert params["generation_config"] == {"candidate_count": 1}
        assert params["tools"] == [{"function_declarations": [{"name": "get_policy_details"}]}]
        # The allowlist is a positive filter: unknown kwargs are excluded.
        assert "labels" not in params
        assert "system_instruction" not in params
        assert "safety_settings" not in params
        # And no non-allowlisted value's SENTINEL leaks via the params path.
        assert SENTINEL not in json.dumps(params)


# ---------------------------------------------------------------------------
# Cost floor — real Gemini token shape (Group-B adjudication: RUN it)
# ---------------------------------------------------------------------------
class TestCostFloor:
    def test_cost_usd_present_and_exact(self, mock_client, capture_trace):
        _run(_VertexModel(_response(prompt=1000, completion=200)), mock_client, CaptureConfig.full())

        cost = find_event(capture_trace["events"], "cost.record")
        assert cost["payload"]["provider"] == "google_vertex"
        assert cost["payload"]["model"] == "gemini-2.5-pro"
        assert cost["payload"]["cost_usd"] is not None
        # gemini-2.5-pro PRICING: 1000 * 0.00125/1k (input) + 200 * 0.01/1k (output).
        assert cost["payload"]["cost_usd"] == pytest.approx(0.00325)


# ---------------------------------------------------------------------------
# Streaming floor — aggregate_stream / _AggregatedVertexResponse offline
# ---------------------------------------------------------------------------
class TestStreamingFloor:
    def _drive(self, model, mock_client, config, *, prompt="weather?"):
        provider = GoogleVertexProvider()
        provider.connect(model)
        try:

            @trace(mock_client, capture_config=config)
            def my_agent():
                collected = []
                for chunk in model.generate_content(contents=prompt, stream=True):
                    collected.append(chunk)
                return len(collected)

            return my_agent()
        finally:
            provider.disconnect()

    def test_aggregation_preserves_text_tool_and_cost(self, mock_client, capture_trace):
        chunks = [
            _chunk(text="The weather in "),
            _chunk(text="Paris is sunny."),
            _chunk(tool_call=("get_weather", {"city": "Paris"})),
            _chunk(finish="STOP", usage=(20, 8, 28)),  # cumulative usage on the LAST chunk
        ]
        n = self._drive(_VertexModel(chunks), mock_client, CaptureConfig.full())
        # Iterator contract preserved: every chunk is re-yielded to the caller.
        assert n == 4

        events = capture_trace["events"]
        mi = find_event(events, "model.invoke")
        content = mi["payload"]["output_message"]["content"]
        # Both text deltas survive aggregation (a dropped chunk would fail this).
        assert "The weather in" in content
        assert "Paris is sunny." in content
        # Usage + finish come from the FINAL chunk, not an earlier zeroed one.
        assert mi["payload"]["usage"]["prompt_tokens"] == 20
        assert mi["payload"]["usage"]["completion_tokens"] == 8
        assert mi["payload"]["finish_reason"] == "STOP"
        assert "ttft_ms" in mi["payload"]

        tc = find_event(events, "tool.call")
        assert tc["payload"]["tool_name"] == "get_weather"
        assert tc["payload"]["arguments"] == {"city": "Paris"}

        cost = find_event(events, "cost.record")
        # gemini-2.5-pro: 20 * 0.00125/1k + 8 * 0.01/1k.
        assert cost["payload"]["cost_usd"] == pytest.approx(0.000105)

    def test_streamed_content_redacted_when_off(self, mock_client, capture_trace):
        """Redaction holds THROUGH the streaming-aggregation path too."""
        chunks = [
            _chunk(text=f"streamed secret {SENTINEL}"),
            _chunk(finish="STOP", usage=(5, 3, 8)),
        ]
        self._drive(_VertexModel(chunks), mock_client, CaptureConfig(capture_content=False))
        assert SENTINEL not in json.dumps(capture_trace["events"])
