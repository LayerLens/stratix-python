"""Offline error + attestation + redaction floor for the Microsoft Agent Framework adapter.

Closes the W2 census ◑/gap cells that the existing ``test_ms_agent_framework.py``
proves only via synthetic ``SimpleNamespace`` messages or a hand-rolled
``RuntimeError("kaboom")`` string, by driving a *real*
``semantic_kernel.agents.ChatCompletionAgent`` (the single-agent chat the
``MSAgentFrameworkAdapter`` wraps) backed by an ``openai.AsyncOpenAI`` client over
``httpx.MockTransport`` — the proven seam from ``test_ms_agent_framework_recorded.py``.
The network is the ONLY mock; the SK agent, its async chat service, the openai
client's real deserialization, and the adapter's own async-generator wrapper /
message parser are all real. So a regression fails in plain CI with no
credentials and no network:

* Error-paths  — a REAL ``openai.AuthenticationError`` is *raised by the real
                 AsyncOpenAI client deserializing a real 401 body* served over the
                 mocked transport (NOT the synthetic ``RuntimeError("kaboom")`` the
                 existing suite yields). Real SK wraps the provider error in its own
                 ``ServiceResponseException`` at the chat-service boundary, so the
                 adapter's ``except BaseException`` finally-block surfaces it as
                 ``agent.error`` with the honest real-SDK wrapper class name
                 ``error_type == "ServiceResponseException"`` and the underlying
                 openai error text (``AuthenticationError`` / ``401`` /
                 ``invalid_api_key``) flowing through verbatim.
* Attestation  — a real ``agent.invoke`` over the recorded openai response flushes a
                 trace whose attestation chain reconstructs and ``verify_chain(...)``
                 returns valid; one envelope per event; a tamper control breaks link
                 1 to prove the check is not vacuous.
* Redaction    — a real agent run whose assistant reply carries a SENTINEL, with
                 ``capture_content=False``, keeps the structural events
                 (agent.input/model.invoke/cost.record/agent.output) but strips the
                 ``output`` content field — and a SENTINEL sweep over
                 ``json.dumps(events)`` — from the stored trace, with a
                 ``capture_content=True`` vacuity control proving the same path DOES
                 carry the content otherwise.

Cost fidelity (closed by ``TestCostFloor`` below): on a REAL SK run the model id
is stranded on ``message.ai_model_id`` / ``message.inner_content.model`` while the
message-level ``metadata`` dict carries no ``model`` key — the adapter's
``_emit_model_metadata`` used to read only that empty ``metadata["model"]``, so
``model.invoke`` / ``cost.record`` shipped ``model=None`` and the shared framework
price-on-emit hook short-circuited (a tokens-only ``cost.record``). The adapter
now recovers the model id from the message (``ai_model_id`` first, then
``inner_content.model``), so the real model id reaches ``model.invoke`` /
``cost.record`` and the hook computes a real ``cost_usd`` — matching the sibling
``SemanticKernelAdapter``, which reads the configured ``ai_model_id`` and prices
the same way.

Note: ``test_ms_agent_framework_recorded.py`` disables content capture citing a
``ChatHistoryAgentThread`` that ``safe_serialize`` "would raise inside
agent.output"; on the pinned ``semantic-kernel==1.36.0`` that no longer
reproduces (the real ``AgentResponseItem`` model-dumps cleanly), so the
``capture_content=True`` control and the attestation run below serialize the real
response object without error.
"""

from __future__ import annotations

import json
import asyncio
import dataclasses

import pytest

pytest.importorskip("semantic_kernel.agents")

import httpx  # noqa: E402
from semantic_kernel.agents import ChatCompletionAgent  # noqa: E402
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion  # noqa: E402

from openai import AsyncOpenAI  # noqa: E402
from layerlens.attestation._verify import verify_chain  # noqa: E402
from layerlens.attestation._envelope import HashScope, AttestationEnvelope  # noqa: E402
from layerlens.instrument._capture_config import CaptureConfig  # noqa: E402
from layerlens.instrument.adapters.frameworks.ms_agent_framework import (  # noqa: E402
    MSAgentFrameworkAdapter,
)

from .conftest import find_event, find_events, capture_framework_trace  # noqa: E402
from ..._recorded import load_recorded, mock_transport  # noqa: E402

SENTINEL = "LL-SENTINEL-7f3a9c2e"
_MODEL = "gpt-4o-mini"

# Content-off config that differs from the control ONLY in ``capture_content`` so
# the redaction assertion isolates the content gate (mirrors the SK-adapter floor).
_CONTENT_OFF = dataclasses.replace(CaptureConfig.full(), capture_content=False)


# ---------------------------------------------------------------------------
# Real-agent-over-mocked-transport helpers (the proven recorded seam)
# ---------------------------------------------------------------------------
def _agent_over_transport(transport: httpx.MockTransport, *, model: str = _MODEL) -> ChatCompletionAgent:
    """A real ``ChatCompletionAgent`` whose OpenAI chat service routes through
    ``transport``. SK's OpenAI chat service is async-only; the MockTransport is
    injected through the documented ``async_client=`` seam so the *real* openai
    client does its real routing + deserialization against the served body."""
    async_client = AsyncOpenAI(api_key="test-key", http_client=httpx.AsyncClient(transport=transport))
    service = OpenAIChatCompletion(ai_model_id=model, async_client=async_client)
    return ChatCompletionAgent(service=service, name="replay_agent", instructions="reply")


def _drain(agent: ChatCompletionAgent, *, prompt: str = "Reply with exactly: pong"):
    async def run():
        return [item async for item in agent.invoke(messages=prompt)]

    return asyncio.run(run())


def _recorded_transport() -> httpx.MockTransport:
    fixture = load_recorded("openai", "default")
    transport, _ = mock_transport(fixture)
    return transport


def _sentinel_transport(sentinel: str) -> httpx.MockTransport:
    """A real chat-completion body whose assistant reply carries ``sentinel`` and a
    real usage triple (so the structural model.invoke/cost.record still emit)."""
    body = {
        "id": "chatcmpl-floor",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "gpt-4o-mini-2024-07-18",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": f"Answer: {sentinel}"},
            }
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 1, "total_tokens": 13},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    return httpx.MockTransport(handler)


def _streaming_transport(*, model: str = "gpt-4o-mini-2024-07-18") -> httpx.MockTransport:
    """A real OpenAI chat-completions *streaming* body: several content-delta
    chunks (each carrying the model id, no usage) followed by a terminal usage
    chunk — exactly the shape SK's ``invoke_stream`` consumes. SK forces
    ``stream_options={"include_usage": True}``, so the real token usage rides the
    final (empty-choices) chunk while every content fragment carries only the
    ``ai_model_id``."""
    chunks = [
        {"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {"content": "po"}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {"content": "ng"}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 2, "total_tokens": 14}},
    ]
    base = {"id": "chatcmpl-stream", "object": "chat.completion.chunk", "created": 1700000000, "model": model}
    body = "".join(f"data: {json.dumps({**base, **c})}\n\n" for c in chunks) + "data: [DONE]\n\n"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body.encode("utf-8"), headers={"content-type": "text/event-stream"})

    return httpx.MockTransport(handler)


def _drain_stream(agent: ChatCompletionAgent, *, prompt: str = "Reply with exactly: pong"):
    async def run():
        return [item async for item in agent.invoke_stream(messages=prompt)]

    return asyncio.run(run())


def _drive(mock_client, transport, config, *, prompt: str = "Reply with exactly: pong"):
    """Instrument the real agent, run one ``invoke`` to completion, return the
    captured trace + result. ``instrument_chat`` (not ``connect``) is the proven
    recorded-test entry point — it wraps the agent's async-gen ``invoke``."""
    uploaded = capture_framework_trace(mock_client)
    agent = _agent_over_transport(transport)
    adapter = MSAgentFrameworkAdapter(mock_client, capture_config=config)
    adapter.instrument_chat(agent)
    result = _drain(agent, prompt=prompt)
    adapter.disconnect()
    return uploaded, result


# ---------------------------------------------------------------------------
# Real error-shape floor — a genuine openai SDK exception, raised the real way
# ---------------------------------------------------------------------------
class TestRealErrorShape:
    def test_real_openai_401_surfaces_as_agent_error(self, mock_client):
        # A real 401 body — the real AsyncOpenAI client deserializes it and raises a
        # genuine ``openai.AuthenticationError`` (401 is not retried), which the real
        # SK chat service wraps and re-raises through ``agent.invoke``. NOT the
        # synthetic ``RuntimeError("kaboom")`` the existing suite yields.
        body = {
            "error": {
                "message": "Incorrect API key provided.",
                "type": "invalid_request_error",
                "code": "invalid_api_key",
            }
        }

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json=body)

        uploaded = capture_framework_trace(mock_client)
        agent = _agent_over_transport(httpx.MockTransport(handler))
        adapter = MSAgentFrameworkAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.instrument_chat(agent)
        # The 401 propagates out of ``agent.invoke``; the adapter's wrapper catches
        # it in its ``finally`` block, emits ``agent.error``, and re-raises.
        with pytest.raises(Exception, match="401"):
            _drain(agent)
        adapter.disconnect()

        events = uploaded["events"]
        errors = find_events(events, "agent.error")
        assert len(errors) == 1, f"expected exactly one agent.error, saw {[e['payload'] for e in errors]}"
        payload = errors[0]["payload"]

        # Honest real-SDK classification — bite: lost if the adapter stops emitting
        # on failure or misclassifies. The real SK chat service wraps the provider
        # error in ``ServiceResponseException`` (``type(exc).__name__``).
        assert payload["error_type"] == "ServiceResponseException"
        assert payload["status"] == "error"
        assert payload["framework"] == "ms_agent_framework"
        assert "latency_ms" in payload
        # The REAL underlying openai exception flows through verbatim (bite:
        # dropped/mangled error text fails here) — proof a genuine SDK error, not a
        # synthetic string, reached the adapter. These survive the secret-scrub
        # chokepoint (no secret-shaped substring).
        assert "AuthenticationError" in payload["error"]
        assert "401" in payload["error"]
        assert "Incorrect API key provided" in payload["error"]
        assert "invalid_api_key" in payload["error"]


# ---------------------------------------------------------------------------
# Offline attestation-chain verification over a real agent run
# ---------------------------------------------------------------------------
class TestAttestationOffline:
    def test_attestation_chain_verifies_over_real_agent(self, mock_client):
        uploaded, result = _drive(
            mock_client, _recorded_transport(), CaptureConfig.full(), prompt="Reply with exactly: pong"
        )
        # AgentResponseItem.message is the ChatMessageContent; its text is "pong".
        assert str(result[0].message.content) == "pong"

        events = uploaded["events"]
        assert events, "real agent invoke must flush a non-empty trace"
        chain = (uploaded["attestation"] or {}).get("chain") or {}
        raw = chain.get("events") or []
        envelopes = [
            AttestationEnvelope(hash=e["hash"], scope=HashScope(e["scope"]), previous_hash=e.get("previous_hash"))
            for e in raw
        ]
        assert envelopes, "no attestation envelopes captured for the real agent trace"
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
# Redaction content-absence over a real agent lifecycle
# ---------------------------------------------------------------------------
class TestRedactionFloor:
    def test_content_present_when_capturing(self, mock_client):
        """Vacuity control: with capture_content=True the SAME real run DOES carry
        the SENTINEL and the ``output`` content key it rides on — without this the
        redaction sweep below could pass trivially."""
        uploaded, _ = _drive(mock_client, _sentinel_transport(SENTINEL), CaptureConfig.full())
        events = uploaded["events"]
        assert SENTINEL in json.dumps(events), "control run must carry the SENTINEL when capturing content"
        assert "output" in find_event(events, "agent.output")["payload"]

    def test_content_absent_when_not_capturing(self, mock_client):
        """capture_content=False keeps the structural events but strips the assistant
        reply content — and the SENTINEL — from the stored trace."""
        uploaded, _ = _drive(mock_client, _sentinel_transport(SENTINEL), _CONTENT_OFF)
        events = uploaded["events"]
        assert events, "the lifecycle must still emit structural events without content"

        # Structure survives (the run still happened) ...
        assert find_events(events, "agent.input"), "agent.input dropped under capture_content=False"
        assert find_events(events, "model.invoke"), "model.invoke dropped under capture_content=False"
        assert find_events(events, "cost.record"), "cost.record dropped under capture_content=False"
        assert find_events(events, "agent.output"), "agent.output dropped under capture_content=False"

        # ... but no content leaks.
        # 1) SENTINEL sweep over the serialized trace.
        assert SENTINEL not in json.dumps(events), "PRIVACY LEAK: SENTINEL survived capture_content=False"
        # 2) The content key is absent from the payload that would carry it.
        assert "output" not in find_event(events, "agent.output")["payload"], "agent.output leaked 'output'"


# ---------------------------------------------------------------------------
# Cost floor — a REAL SK run must attribute AND price the model call
# ---------------------------------------------------------------------------
class TestCostFloor:
    """On a REAL ``ChatCompletionAgent`` run the model id is stranded on
    ``message.ai_model_id`` (the message-level ``metadata`` dict carries no
    ``model``/``model_id`` key — see the recorded-shape probe), so the adapter
    must recover it from the message to attribute and price the call. Without
    that recovery ``model.invoke`` / ``cost.record`` ship ``model=None`` and the
    shared price-on-emit hook (``_price_cost_record``) short-circuits on the
    falsy model, silently under-reporting spend on every real run. The sibling
    ``SemanticKernelAdapter`` reads the configured ``ai_model_id`` and DOES
    price, so this was a source-level divergence, not an intrinsic limitation of
    the trace."""

    def test_real_run_prices_model_invoke_and_cost(self, mock_client):
        uploaded, result = _drive(mock_client, _recorded_transport(), CaptureConfig.full())
        # Sanity: the real provider response flowed through SK end to end.
        assert str(result[0].message.content) == "pong"

        events = uploaded["events"]

        # The configured model id (stranded on the message, absent from metadata)
        # must reach cost.record AND be priced by the shared hook — bite: lost if
        # the adapter reads only the empty message-level metadata model key, which
        # leaves model=None and cost_usd absent (a tokens-only cost.record).
        cost = find_event(events, "cost.record")["payload"]
        assert cost["model"] == _MODEL, f"cost.record model not recovered: {cost.get('model')!r}"
        assert cost.get("cost_usd") is not None, "cost.record shipped tokens-only (no cost_usd)"
        assert cost["cost_usd"] > 0
        # Token telemetry off the real CompletionUsage object is preserved.
        assert cost["tokens_prompt"] == 12
        assert cost["tokens_completion"] == 1
        assert cost["tokens_total"] == 13

        # model.invoke carries the same recovered model id + detected provider so
        # the model column fills honestly (not None) on a real run.
        invoke = find_event(events, "model.invoke")["payload"]
        assert invoke["model"] == _MODEL
        assert invoke["provider"] == "openai"


# ---------------------------------------------------------------------------
# Streaming multiplicity floor — one model.invoke / cost.record per real call
# ---------------------------------------------------------------------------
class TestStreamingModelInvokeMultiplicity:
    """A REAL ``invoke_stream`` run must emit exactly ONE ``model.invoke`` and
    ONE ``cost.record`` for the single underlying model call.

    SK's ``ChatCompletionAgent.invoke_stream`` yields many partial
    ``StreamingChatMessageContent`` chunks per model call — each carrying the
    same ``ai_model_id`` but no usage — followed by a terminal chunk carrying the
    real ``CompletionUsage`` (SK forces ``stream_options={"include_usage":
    True}``). The adapter's wrapper processes every yielded chunk, so the
    model-alone ``model.invoke`` branch used to fire once per fragment: N
    token-less ``model.invoke`` events plus one token-bearing one — inflating the
    model-call count and shipping phantom tokens-less invocations on every
    streamed run. The honest accounting is a single ``model.invoke`` carrying the
    model id AND the real token counts (the non-streaming ``invoke`` path, one
    consolidated message, was always single — this proves the streamed path now
    matches it). Bite: on the pre-fix adapter ``model.invoke`` count is N+1 > 1.
    """

    def test_stream_emits_single_priced_model_invoke(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        agent = _agent_over_transport(_streaming_transport())
        adapter = MSAgentFrameworkAdapter(mock_client, capture_config=CaptureConfig(capture_content=False))
        adapter.instrument_chat(agent)
        result = _drain_stream(agent)
        adapter.disconnect()

        # Sanity: the streamed deltas reconstruct the real assistant reply, so the
        # multiple partial chunks the adapter processed are genuine stream fragments
        # (not a single consolidated message masquerading as a stream).
        assert len(result) >= 2, "the streamed body must yield multiple partial chunks"
        text = "".join(str(item.message.content) for item in result)
        assert "pong" in text

        events = uploaded["events"]

        # BITE: the fragments must collapse to a SINGLE model.invoke. On the
        # pre-fix adapter each streamed chunk emitted its own model.invoke
        # (N token-less fragments + 1 token-bearing terminal), so this is > 1.
        invokes = find_events(events, "model.invoke")
        assert len(invokes) == 1, (
            f"expected exactly one model.invoke for one streamed model call, saw "
            f"{len(invokes)}: {[i['payload'] for i in invokes]}"
        )
        invoke = invokes[0]["payload"]
        # The surviving model.invoke is the complete accounting — the model id AND
        # the real token counts off the terminal usage chunk, not a phantom
        # tokens-less fragment.
        assert invoke["model"] == _MODEL
        assert invoke["provider"] == "openai"
        assert invoke["tokens_prompt"] == 12
        assert invoke["tokens_completion"] == 2
        assert invoke["tokens_total"] == 14

        # cost.record was already single (only the terminal usage chunk carries
        # usage) — lock it so the fix can't regress it into per-fragment
        # duplicates, and confirm it is still priced by the shared hook.
        costs = find_events(events, "cost.record")
        assert len(costs) == 1, f"expected exactly one cost.record, saw {len(costs)}"
        assert costs[0]["payload"]["model"] == _MODEL
        assert costs[0]["payload"].get("cost_usd") is not None
        assert costs[0]["payload"]["cost_usd"] > 0
