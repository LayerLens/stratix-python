"""Offline error + attestation + redaction + cost floor for the Semantic Kernel adapter.

Closes the W2 census cells that the existing ``test_semantic_kernel.py`` proves
only via hand-built ``Mock*`` doubles or a synthetic ``RuntimeError`` string, by
driving a *real* ``semantic_kernel.Kernel`` whose ``OpenAIChatCompletion`` service
is backed by an ``AsyncOpenAI`` client over ``httpx.MockTransport`` (the proven
seam from ``test_semantic_kernel_recorded.py``) — the network is the ONLY mock;
the Kernel, the prompt function, the SK filter dispatch, the openai client's real
deserialization, and the adapter's own parser are all real. So a regression fails
in plain CI with no credentials and no network:

* Error-paths  — a REAL ``openai.AuthenticationError`` is *raised by the real
                 AsyncOpenAI client deserializing a real 401 body* served over the
                 mocked transport (NOT the synthetic ``RuntimeError("API timeout")``
                 the existing suite feeds a hand-built ``MockChatService``). Real
                 SK wraps the provider error in its own ``ServiceResponseException``
                 at the chat-service boundary, so it surfaces as ``agent.error``
                 with the honest real-SDK wrapper class name
                 ``error_type == "ServiceResponseException"``, the configured model,
                 and the underlying openai error text (``AuthenticationError`` /
                 ``401`` / ``invalid_api_key``) flowing through verbatim.
* Attestation  — a real prompt-function ``kernel.invoke`` over the recorded openai
                 response flushes a trace whose attestation chain reconstructs and
                 ``verify_chain(...)`` returns valid; one envelope per event; a
                 tamper control breaks link 1 to prove the check is not vacuous.
* Redaction    — the same real prompt-function lifecycle with ``capture_content=False``
                 keeps the structural events (tool.call/model.invoke/tool.result)
                 but strips ``input`` / ``rendered_prompt`` / ``output`` — and a
                 SENTINEL sweep over ``json.dumps(events)`` — from the stored trace,
                 with a ``capture_content=True`` vacuity control proving the same
                 path DOES carry the content otherwise.
* Cost         — the ``cost.record`` emitted off the REAL ``CompletionUsage``
                 parsed from the recorded body (token triple 12/1/13) carries a
                 non-None ``cost_usd`` computed offline from the shared ``PRICING``
                 table (gpt-4o-mini), so a framework trace is never tokens-only.

The single-agent kernel path is exercised end-to-end here; the multi-agent
``AgentGroupChat`` honest-graph path already has strong unit coverage in
``test_semantic_kernel.py::TestHonestGraphContract``.

* Streaming     — a real ``kernel.invoke_stream`` over the recorded *streaming*
                 body attributes the LLM call (``model.invoke`` + a priced
                 ``cost.record``) exactly like the non-streaming path. SK consumes
                 the stream lazily *after* the function-invocation filter's run has
                 already closed, so the adapter shadow-wraps
                 ``_inner_get_streaming_chat_message_contents`` and emits the
                 accumulated usage into a run of its own — closing the former
                 streaming gap where a streaming run emitted no model + cost
                 telemetry at all.
"""

from __future__ import annotations

import json
import asyncio
import dataclasses

import pytest

sk = pytest.importorskip("semantic_kernel")

import httpx  # noqa: E402
from semantic_kernel import Kernel  # noqa: E402
from semantic_kernel.connectors.ai.open_ai import (  # noqa: E402
    OpenAIChatCompletion,
    OpenAIChatPromptExecutionSettings,
)
from semantic_kernel.exceptions.kernel_exceptions import KernelInvokeException  # noqa: E402

from openai import AsyncOpenAI  # noqa: E402
from layerlens.attestation._verify import verify_chain  # noqa: E402
from layerlens.attestation._envelope import HashScope, AttestationEnvelope  # noqa: E402
from layerlens.instrument._capture_config import CaptureConfig  # noqa: E402
from layerlens.instrument.adapters.frameworks.semantic_kernel import SemanticKernelAdapter  # noqa: E402

from .conftest import find_event, find_events, capture_framework_trace  # noqa: E402
from ..._recorded import load_recorded, mock_transport  # noqa: E402

SENTINEL = "LL-SENTINEL-7f3a9c2e"
_MODEL = "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Real-Kernel-over-mocked-transport helpers (the proven recorded seam)
# ---------------------------------------------------------------------------
def _kernel_over_transport(transport: httpx.MockTransport, *, model: str = _MODEL) -> Kernel:
    """A real ``Kernel`` whose OpenAI chat service routes through ``transport``.

    SK's OpenAI chat service is async-only; the MockTransport is injected through
    the documented ``async_client=`` seam so the *real* openai client does its
    real routing + deserialization against the served body. ``ai_model_id`` is the
    model we request (SK reports the configured id, not the response echo)."""
    async_client = AsyncOpenAI(api_key="test-key", http_client=httpx.AsyncClient(transport=transport))
    service = OpenAIChatCompletion(ai_model_id=model, async_client=async_client)
    kernel = Kernel()
    kernel.add_service(service)
    return kernel


def _add_prompt_fn(kernel: Kernel, *, prompt: str = "Question: {{$question}}"):
    """Register a real prompt (template) function — invoking it opens the adapter's
    run boundary via the function-invocation filter, renders the prompt (prompt
    filter), and routes the LLM call through the patched chat service."""
    return kernel.add_function(
        plugin_name="replay",
        function_name="say",
        prompt=prompt,
        prompt_execution_settings=OpenAIChatPromptExecutionSettings(max_tokens=10),
    )


def _drive_prompt(mock_client, transport, config, *, question, prompt="Question: {{$question}}"):
    """Connect the adapter, invoke a real prompt function once, return the captured
    trace. Connect happens with an empty kernel (no plugins) so plugin discovery
    flushes nothing — the single ``kernel.invoke`` is the only trace flushed."""
    uploaded = capture_framework_trace(mock_client)
    kernel = _kernel_over_transport(transport)
    adapter = SemanticKernelAdapter(mock_client, capture_config=config)
    adapter.connect(target=kernel)
    fn = _add_prompt_fn(kernel, prompt=prompt)
    result = asyncio.run(kernel.invoke(fn, question=question))
    adapter.disconnect()
    return uploaded, result


def _recorded_transport() -> httpx.MockTransport:
    fixture = load_recorded("openai", "default")
    transport, _ = mock_transport(fixture)
    return transport


# ---------------------------------------------------------------------------
# Real error-shape floor — a genuine openai SDK exception, raised the real way
# ---------------------------------------------------------------------------
class TestRealErrorShape:
    def test_real_openai_401_surfaces_as_agent_error(self, mock_client):
        # A real 401 body — the real AsyncOpenAI client deserializes it and raises
        # a genuine ``openai.AuthenticationError`` (401 is not retried), which the
        # adapter's patched chat service catches. NOT the synthetic
        # RuntimeError("API timeout") the existing suite feeds a MockChatService.
        body = {
            "error": {
                "message": "Incorrect API key provided.",
                "type": "invalid_request_error",
                "code": "invalid_api_key",
            }
        }

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json=body)

        transport = httpx.MockTransport(handler)

        uploaded = capture_framework_trace(mock_client)
        kernel = _kernel_over_transport(transport)
        adapter = SemanticKernelAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect(target=kernel)
        fn = _add_prompt_fn(kernel)
        # The 401 propagates out of kernel.invoke (SK wraps it in a
        # KernelInvokeException); the LLM-level agent.error is emitted before the
        # re-raise, and the function filter's finally still flushes the trace.
        with pytest.raises(KernelInvokeException):
            asyncio.run(kernel.invoke(fn, question="hi"))
        adapter.disconnect()

        events = uploaded["events"]
        errors = find_events(events, "agent.error")
        # The LLM-level failure is the one carrying the model (the tool-level wrap
        # carries SK's FunctionExecutionException and has no model).
        llm_errors = [e for e in errors if "model" in e["payload"]]
        assert len(llm_errors) == 1, (
            f"expected exactly one model-bearing agent.error, saw {[e['payload'] for e in errors]}"
        )
        payload = llm_errors[0]["payload"]

        # Honest real-SDK classification — bite: lost if the adapter stops emitting
        # on LLM failure, misclassifies, or drops the model. The real SK chat
        # service wraps the provider error in ServiceResponseException.
        assert payload["error_type"] == "ServiceResponseException"
        assert payload["model"] == _MODEL
        assert payload["framework"] == "semantic_kernel"
        assert "latency_ms" in payload
        # The REAL underlying openai exception flows through verbatim (bite:
        # dropped/mangled error text fails here) — proof a genuine SDK error, not
        # a synthetic string, reached the adapter. "401"/"invalid_api_key" survive
        # the secret-scrub chokepoint.
        assert "AuthenticationError" in payload["error"]
        assert "401" in payload["error"]
        assert "invalid_api_key" in payload["error"]


# ---------------------------------------------------------------------------
# Offline attestation-chain verification over a real prompt-function run
# ---------------------------------------------------------------------------
class TestAttestationOffline:
    def test_attestation_chain_verifies_over_real_prompt_fn(self, mock_client):
        uploaded, result = _drive_prompt(
            mock_client, _recorded_transport(), CaptureConfig.full(), question="Reply with exactly: pong"
        )
        assert str(result) == "pong"

        events = uploaded["events"]
        assert events, "real prompt-function invoke must flush a non-empty trace"
        chain = (uploaded["attestation"] or {}).get("chain") or {}
        raw = chain.get("events") or []
        envelopes = [
            AttestationEnvelope(hash=e["hash"], scope=HashScope(e["scope"]), previous_hash=e.get("previous_hash"))
            for e in raw
        ]
        assert envelopes, "no attestation envelopes captured for the real SK trace"
        assert len(envelopes) == len(events), (
            f"attestation chain has {len(envelopes)} envelopes for {len(events)} events"
        )
        assert (uploaded["attestation"] or {}).get("root_hash") is not None

        result_ok = verify_chain(envelopes)
        assert result_ok.valid, f"attestation chain invalid: {result_ok.error}"

        # Vacuity control: verify_chain must REJECT a broken interior link, proving
        # the pass above is not trivially true.
        assert len(envelopes) >= 2
        tampered = list(envelopes)
        tampered[1] = AttestationEnvelope(
            hash=tampered[1].hash,
            scope=tampered[1].scope,
            previous_hash="sha256:deadbeef-not-the-prior-hash",
        )
        broken = verify_chain(tampered)
        assert not broken.valid and broken.break_index == 1, "verify_chain failed to detect a broken link"


# ---------------------------------------------------------------------------
# Redaction content-absence over a real prompt-function lifecycle
# ---------------------------------------------------------------------------
# All layers on so agent.code (l2) fires; content-off differs from the control
# ONLY in capture_content, so the redaction assertion isolates content gating.
_CONTENT_OFF = dataclasses.replace(CaptureConfig.full(), capture_content=False)


class TestRedactionFloor:
    def test_content_present_when_capturing(self, mock_client):
        """Vacuity control: with capture_content=True the SAME real lifecycle DOES
        carry the SENTINEL and the content keys it rides on (tool args, rendered
        prompt) — without this the redaction sweep below could pass trivially."""
        uploaded, _ = _drive_prompt(
            mock_client, _recorded_transport(), CaptureConfig.full(), question=f"Remember {SENTINEL}"
        )
        events = uploaded["events"]
        assert SENTINEL in json.dumps(events), "control run must carry the SENTINEL when capturing content"
        assert "input" in find_event(events, "tool.call")["payload"]
        assert "rendered_prompt" in find_event(events, "agent.code")["payload"]
        assert "output" in find_event(events, "tool.result")["payload"]

    def test_content_absent_when_not_capturing(self, mock_client):
        """capture_content=False keeps the structural events but strips the tool
        arguments, the rendered prompt, the tool output — and the SENTINEL."""
        uploaded, _ = _drive_prompt(mock_client, _recorded_transport(), _CONTENT_OFF, question=f"Remember {SENTINEL}")
        events = uploaded["events"]
        assert events, "the lifecycle must still emit structural events without content"

        # Structure survives (the run still happened) ...
        assert find_events(events, "tool.call"), "tool.call structural event dropped under capture_content=False"
        assert find_events(events, "model.invoke"), "model.invoke dropped under capture_content=False"
        assert find_events(events, "tool.result"), "tool.result structural event dropped under capture_content=False"

        # ... but no content leaks.
        # 1) SENTINEL sweep over the serialized trace.
        assert SENTINEL not in json.dumps(events), "PRIVACY LEAK: SENTINEL survived capture_content=False"
        # 2) The content keys are absent from every payload that would carry them.
        assert "input" not in find_event(events, "tool.call")["payload"], "tool.call leaked 'input'"
        assert "output" not in find_event(events, "tool.result")["payload"], "tool.result leaked 'output'"
        for e in find_events(events, "agent.code"):
            assert "rendered_prompt" not in e["payload"], "agent.code leaked 'rendered_prompt'"


# ---------------------------------------------------------------------------
# Cost floor — cost_usd computed offline on the REAL recorded token shape
# ---------------------------------------------------------------------------
class TestCostFloor:
    def test_cost_usd_present_on_real_token_shape(self, mock_client):
        uploaded, result = _drive_prompt(
            mock_client, _recorded_transport(), CaptureConfig.full(), question="Reply with exactly: pong"
        )
        assert str(result) == "pong"

        cost = find_event(uploaded["events"], "cost.record")
        # Real token triple parsed off the recorded CompletionUsage (the strong
        # tell the real provider shape flowed through the real SK objects).
        assert cost["payload"]["tokens_prompt"] == 12
        assert cost["payload"]["tokens_completion"] == 1
        assert cost["payload"]["tokens_total"] == 13
        assert cost["payload"]["model"] == _MODEL
        assert cost["payload"]["framework"] == "semantic_kernel"

        # cost_usd is filled offline from the shared PRICING table — bite: None if
        # the framework price-on-emit hook regresses (a tokens-only cost.record).
        cost_usd = cost["payload"].get("cost_usd")
        assert cost_usd is not None, "cost.record shipped tokens-only (no cost_usd) — framework pricing regressed"
        assert cost_usd > 0
        # gpt-4o-mini: input 0.00015/1k, output 0.0006/1k over 12/1 tokens.
        expected = 12 / 1000 * 0.00015 + 1 / 1000 * 0.0006
        assert abs(cost_usd - expected) < 1e-9, f"cost_usd {cost_usd} != expected {expected}"


# ---------------------------------------------------------------------------
# Streaming attribution floor — a real ``kernel.invoke_stream`` LLM call must
# carry model.invoke + a priced cost.record, exactly like the non-streaming path
# ---------------------------------------------------------------------------
class TestStreamingCostFloor:
    def test_streaming_llm_call_is_attributed(self, mock_client):
        """A real ``kernel.invoke_stream`` over the recorded *streaming* body must
        attribute the LLM call — ``model.invoke`` + a priced ``cost.record`` — just
        like the non-streaming path.

        SK streams the response lazily: ``get_streaming_chat_message_contents ->
        _inner_get_streaming_chat_message_contents`` is consumed by
        ``KernelFunction.invoke_stream`` *after* the function-invocation filter's
        run boundary has already closed and flushed. An adapter that patches only
        the non-streaming ``_inner_get_chat_message_contents`` therefore drops ALL
        model + cost telemetry for a streaming customer (spend under-reported).
        Bite: RED (0 model.invoke / 0 cost.record) until the streaming inner method
        is shadow-wrapped and the accumulated usage emitted."""
        fixture = load_recorded("openai", "stream")
        transport, _ = mock_transport(fixture)
        uploaded = capture_framework_trace(mock_client)
        kernel = _kernel_over_transport(transport)
        adapter = SemanticKernelAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect(target=kernel)
        fn = _add_prompt_fn(kernel, prompt="Q: {{$question}}")

        async def _consume():
            async for _ in kernel.invoke_stream(fn, question="stream please"):
                pass

        asyncio.run(_consume())
        adapter.disconnect()

        events = uploaded["events"]
        # A streaming LLM call must be attributed exactly like the non-streaming one.
        assert len(find_events(events, "model.invoke")) >= 1, "streaming LLM call emitted no model.invoke"
        assert len(find_events(events, "cost.record")) >= 1, "streaming LLM call emitted no cost.record"

        # Stronger bite: the REAL streamed token triple (12/1/13 off the recorded
        # usage chunk) flows through, attributed to the configured model, and the
        # shared framework price-on-emit hook fills a non-None cost_usd — so a
        # streaming trace is never model-less or tokens-only.
        mi = find_event(events, "model.invoke")
        assert mi["payload"]["model"] == _MODEL
        assert mi["payload"]["framework"] == "semantic_kernel"

        cost = find_event(events, "cost.record")
        assert cost["payload"]["tokens_prompt"] == 12
        assert cost["payload"]["tokens_completion"] == 1
        assert cost["payload"]["tokens_total"] == 13
        assert cost["payload"]["model"] == _MODEL
        cost_usd = cost["payload"].get("cost_usd")
        assert cost_usd is not None, "streaming cost.record shipped tokens-only (no cost_usd)"
        assert cost_usd > 0
