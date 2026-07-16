"""Offline error + attestation + cost + redaction + capture-tier floor for the
OpenAI Agents SDK adapter.

Closes the W2 census ◑/gap OFFLINE cells that the existing ``test_openai_agents.py``
leaves open, so a regression fails in plain CI with no credentials and no network.
The adapter *is* an ``agents.TracingProcessor``; every object here is a REAL Agents
SDK type (``TraceImpl`` / ``SpanImpl`` / the real ``*SpanData`` classes) driven
through the adapter's real ``on_trace_start`` / ``on_span_end`` / ``on_trace_end``
processor contract — nothing about the SDK or the adapter is mocked, and no
transport is involved on this path.

* Error-paths (``error`` cell — was partial, not floor-gated) — a REAL provider
                SDK exception (``openai.RateLimitError``) is wrapped in the EXACT
                ``agents.tracing.SpanError`` shape the real SDK builds for a
                failing tool (see ``agents/tool.py`` / ``agents/_run_impl.py``:
                ``message="Error running tool", data={"tool_name": ..., "error":
                str(e)}``) and attached the real way via the SDK's own
                ``agents.util._error_tracing.attach_error_to_span``. It surfaces
                as ``agent.error`` with the adapter's honest OA classification
                (``error_type == "span_error"`` — the SDK hands the adapter an
                opaque ``SpanError`` dict, so the real class name is genuinely not
                recoverable) and the real exception text flowing through verbatim.
* Attestation (``attest`` cell — was a gap: no attestation test anywhere) — a
                real multi-span trace (agent + generation + tool) is flushed and
                its attestation chain reconstructs and ``verify_chain(...)``
                returns valid; a TAMPER control breaks an interior link and
                proves the check is not vacuous.
* Cost        (``cost`` cell — was partial: token counts asserted, ``cost_usd``
                value never) — a real ``gpt-4o`` token shape (100 / 25) yields a
                ``cost.record`` whose ``cost_usd`` equals the value the pricing
                path computes (Group-B: GREEN on current code, so the cell closes).
* Redaction   (``redaction`` cell — reinforced + floor-gated) — with
                ``capture_content=False`` the SAME real span lifecycle keeps its
                structure but drops model messages / tool io — and a SENTINEL
                sweep over ``json.dumps(events)`` stays clean — with a
                ``capture_content=True`` vacuity control proving the path DOES
                carry the content otherwise.
* Capture-tier (``params`` cell — was partial: not floor-gated) — the framework
                analog of a params allowlist: ``CaptureConfig.minimal()`` keeps
                L1 (agent.input/output) + cost.record and drops L3 (model.invoke)
                and L5 (tool.call), floor-gated with a ``.full()`` vacuity control.
"""

from __future__ import annotations

import sys
import json

import pytest

if sys.version_info < (3, 10):
    pytest.skip("openai-agents requires Python >= 3.10", allow_module_level=True)
try:
    import agents  # noqa: F401
except (ImportError, Exception):  # pragma: no cover - env guard
    pytest.skip("openai-agents not installed or incompatible", allow_module_level=True)

import httpx  # noqa: E402
from agents.tracing import SpanError, TracingProcessor, set_trace_processors  # noqa: E402
from agents.tracing.spans import SpanImpl  # noqa: E402
from agents.tracing.traces import TraceImpl  # noqa: E402
from agents.tracing.span_data import (  # noqa: E402
    AgentSpanData,
    FunctionSpanData,
    GenerationSpanData,
)
from agents.util._error_tracing import attach_error_to_span  # noqa: E402

import openai  # noqa: E402
from layerlens.attestation._verify import verify_chain  # noqa: E402
from layerlens.attestation._envelope import HashScope, AttestationEnvelope  # noqa: E402
from layerlens.instrument._capture_config import CaptureConfig  # noqa: E402
from layerlens.instrument.adapters.providers.pricing import PRICING, calculate_cost  # noqa: E402
from layerlens.instrument.adapters.providers.token_usage import NormalizedTokenUsage  # noqa: E402
from layerlens.instrument.adapters.frameworks.openai_agents import OpenAIAgentsAdapter  # noqa: E402

from .conftest import find_event, find_events, capture_framework_trace  # noqa: E402

SENTINEL = "LL-SENTINEL-7f3a9c2e"


# ---------------------------------------------------------------------------
# Real Agents SDK span helpers (mirrors test_openai_agents.py — do not import
# private helpers across test modules).
# ---------------------------------------------------------------------------
class _NoOpProcessor(TracingProcessor):
    """Minimal processor so ``span.start()/finish()`` don't double-fire our
    adapter — tests call ``adapter.on_span_end()`` explicitly."""

    def on_trace_start(self, trace):
        pass

    def on_trace_end(self, trace):
        pass

    def on_span_start(self, span):
        pass

    def on_span_end(self, span):
        pass

    def shutdown(self):
        pass

    def force_flush(self):
        pass


_noop = _NoOpProcessor()


def _make_trace(trace_id: str, name: str = "floor_trace") -> TraceImpl:
    return TraceImpl(name=name, trace_id=trace_id, group_id=None, metadata=None, processor=_noop)


def _make_span(trace_id, span_id, span_data, parent_id=None) -> SpanImpl:
    return SpanImpl(
        trace_id=trace_id,
        span_id=span_id,
        parent_id=parent_id,
        processor=_noop,
        span_data=span_data,
    )


def _drive_full_trace(adapter, trace_id: str, sentinel: str = "content") -> None:
    """Drive a realistic single-agent-with-one-tool trace through the REAL
    adapter processor contract: agent span wrapping a generation (gpt-4o, real
    usage) + a function span, all content-bearing. ``on_trace_end`` flushes."""
    trace = _make_trace(trace_id)
    adapter.on_trace_start(trace)

    agent = _make_span(trace_id, "s_agent", AgentSpanData(name="claims_triage", tools=["policy_lookup"]))
    agent.start()
    adapter.on_span_start(agent)

    gen = _make_span(
        trace_id,
        "s_gen",
        GenerationSpanData(
            input=[{"role": "user", "content": f"file a claim: {sentinel}"}],
            output=[{"role": "assistant", "content": f"routing: {sentinel}"}],
            model="gpt-4o",
            model_config={"temperature": 0.2},
            usage={"input_tokens": 100, "output_tokens": 25},
        ),
        parent_id="s_agent",
    )
    gen.start()
    gen.finish()
    adapter.on_span_end(gen)

    tool = _make_span(
        trace_id,
        "s_tool",
        FunctionSpanData(
            name="policy_lookup",
            input='{"policy_id":"' + sentinel + '"}',
            output='{"coverage":"' + sentinel + '"}',
        ),
        parent_id="s_agent",
    )
    tool.start()
    tool.finish()
    adapter.on_span_end(tool)

    agent.finish()
    adapter.on_span_end(agent)

    adapter.on_trace_end(trace)


@pytest.fixture(autouse=True)
def _clean_processors():
    """Reset global OA trace processors after each test (the adapter registers
    itself globally at connect())."""
    yield
    set_trace_processors([])


# ---------------------------------------------------------------------------
# Real error-shape floor — a real openai exception wrapped in the SDK's own
# SpanError shape, attached the real way.
# ---------------------------------------------------------------------------
class TestRealErrorShape:
    def test_real_tool_failure_surfaces_as_agent_error(self, mock_client):
        # A genuine openai SDK exception — the kind a tool that calls an LLM
        # actually raises. Its str() is what the real SDK stores in the tool
        # SpanError's data["error"].
        response = httpx.Response(429, request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"))
        underlying = openai.RateLimitError(
            "Error code: 429 - {'error': {'message': 'Rate limit reached for gpt-4o', "
            "'type': 'rate_limit_error', 'code': 'rate_limit_exceeded'}}",
            response=response,
            body=None,
        )
        # Prove it is the real class, not a hand-rolled stand-in.
        assert type(underlying).__name__ == "RateLimitError"
        assert isinstance(underlying, openai.OpenAIError)
        real_message = str(underlying)

        # The EXACT SpanError the real SDK builds for a failing function tool
        # (agents/tool.py, agents/_run_impl.py). Built from the real SpanError
        # TypedDict, not a bare hand-rolled dict.
        span_error: SpanError = SpanError(
            message="Error running tool",
            data={"tool_name": "policy_lookup", "error": real_message},
        )

        uploaded = capture_framework_trace(mock_client)
        adapter = OpenAIAgentsAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect()
        try:
            trace = _make_trace("t_err")
            adapter.on_trace_start(trace)

            tool = _make_span(
                "t_err",
                "s_tool_err",
                FunctionSpanData(name="policy_lookup", input='{"policy_id":"A-4471"}', output=None),
            )
            tool.start()
            # Attach the error the REAL way — through the SDK's own helper, which
            # is what agents/tool.py invokes on a tool exception.
            attach_error_to_span(tool, span_error)
            tool.finish()
            adapter.on_span_end(tool)

            adapter.on_trace_end(trace)
        finally:
            adapter.disconnect()

        events = uploaded["events"]
        err = find_event(events, "agent.error")

        # Honest OA classification: the adapter only ever receives an opaque
        # SpanError dict, so "span_error" is the honest label (the real
        # RateLimitError class is genuinely not recoverable from the SDK payload).
        # Bite: lost if the adapter stops classifying or stops emitting on a
        # tool-span error.
        assert err["payload"]["error_type"] == "span_error"
        assert err["payload"]["status"] == "error"
        assert err["payload"]["tool_name"] == "policy_lookup"
        assert err["payload"]["framework"] == "openai-agents"

        # The real exception text flows through verbatim inside the preserved
        # SpanError structure (bite: dropped/mangled error text fails here).
        serialized = err["payload"]["error"]
        assert serialized["message"] == "Error running tool"
        assert serialized["data"]["tool_name"] == "policy_lookup"
        assert serialized["data"]["error"] == real_message
        assert "429" in json.dumps(serialized)


# ---------------------------------------------------------------------------
# Offline attestation-chain verification over a real flushed trace.
# ---------------------------------------------------------------------------
class TestAttestationOffline:
    def test_attestation_chain_verifies_over_real_trace(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = OpenAIAgentsAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect()
        try:
            _drive_full_trace(adapter, "t_attest")
        finally:
            adapter.disconnect()

        events = uploaded["events"]
        assert events, "real trace flush must produce a non-empty event list"

        chain = (uploaded["attestation"] or {}).get("chain") or {}
        raw = chain.get("events") or []
        envelopes = [
            AttestationEnvelope(hash=e["hash"], scope=HashScope(e["scope"]), previous_hash=e.get("previous_hash"))
            for e in raw
        ]
        assert envelopes, "no attestation envelopes captured for the real trace"
        assert len(envelopes) == len(events), (
            f"attestation chain has {len(envelopes)} envelopes for {len(events)} events"
        )
        assert (uploaded["attestation"] or {}).get("root_hash") is not None

        result = verify_chain(envelopes)
        assert result.valid, f"attestation chain invalid: {result.error}"

        # Vacuity control: verify_chain must REJECT a broken interior link,
        # proving the pass above is not trivially true.
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
# Cost floor — cost_usd VALUE on a real gpt-4o token shape (Group-B: GREEN).
# ---------------------------------------------------------------------------
class TestCostFloor:
    def test_cost_record_carries_priced_cost_usd(self, mock_client):
        # The value the shared framework pricing path (FrameworkAdapter._emit ->
        # _price_cost_record) must reproduce for this exact token shape.
        expected = calculate_cost(
            "gpt-4o",
            NormalizedTokenUsage(prompt_tokens=100, completion_tokens=25, total_tokens=125),
            PRICING,
        )
        assert expected is not None and expected > 0

        uploaded = capture_framework_trace(mock_client)
        adapter = OpenAIAgentsAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect()
        try:
            _drive_full_trace(adapter, "t_cost")
        finally:
            adapter.disconnect()

        cost = find_event(uploaded["events"], "cost.record")
        assert cost["payload"]["model"] == "gpt-4o"
        assert cost["payload"]["tokens_prompt"] == 100
        assert cost["payload"]["tokens_completion"] == 25
        # Bite: the token counts alone were already asserted; this pins the
        # PRICED value, which was never asserted offline before. Regresses to a
        # KeyError if _emit stops augmenting cost_usd on cost.record.
        assert "cost_usd" in cost["payload"], "cost.record dropped cost_usd — pricing augmentation regressed"
        assert cost["payload"]["cost_usd"] == expected
        assert cost["payload"]["cost_usd"] > 0


# ---------------------------------------------------------------------------
# Redaction floor — content absence + SENTINEL sweep, with a vacuity control.
# ---------------------------------------------------------------------------
class TestRedactionFloor:
    def test_content_present_when_capturing(self, mock_client):
        """Vacuity control: capture_content=True carries the SENTINEL and the
        content keys it rides on."""
        uploaded = capture_framework_trace(mock_client)
        adapter = OpenAIAgentsAdapter(mock_client, capture_config=CaptureConfig(capture_content=True))
        adapter.connect()
        try:
            _drive_full_trace(adapter, "t_redact_on", sentinel=SENTINEL)
        finally:
            adapter.disconnect()

        events = uploaded["events"]
        assert SENTINEL in json.dumps(events), "control run must carry the SENTINEL when capturing content"
        model_invoke = find_event(events, "model.invoke")
        assert "messages" in model_invoke["payload"]
        assert "output_message" in model_invoke["payload"]
        assert "input" in find_event(events, "tool.call")["payload"]
        assert "output" in find_event(events, "tool.result")["payload"]

    def test_content_absent_when_not_capturing(self, mock_client):
        """capture_content=False keeps the structure but drops model messages /
        tool io — and the SENTINEL — from every stored event."""
        uploaded = capture_framework_trace(mock_client)
        adapter = OpenAIAgentsAdapter(mock_client, capture_config=CaptureConfig(capture_content=False))
        adapter.connect()
        try:
            _drive_full_trace(adapter, "t_redact_off", sentinel=SENTINEL)
        finally:
            adapter.disconnect()

        events = uploaded["events"]
        assert events, "the lifecycle must still emit structural events without content"

        # 1) SENTINEL sweep over the serialized trace.
        assert SENTINEL not in json.dumps(events), "PRIVACY LEAK: SENTINEL survived capture_content=False"

        # 2) Content keys absent from the payloads that would carry them (the
        # structural keys — model, tokens, tool_name — remain).
        model_invoke = find_event(events, "model.invoke")
        assert "messages" not in model_invoke["payload"]
        assert "output_message" not in model_invoke["payload"]
        assert model_invoke["payload"]["model"] == "gpt-4o"

        assert "input" not in find_event(events, "tool.call")["payload"]
        assert "output" not in find_event(events, "tool.result")["payload"]
        for e in find_events(events, "tool.call"):
            assert e["payload"]["tool_name"] == "policy_lookup"


# ---------------------------------------------------------------------------
# Capture-tier gating — the framework analog of a params allowlist, floor-gated.
# ---------------------------------------------------------------------------
class TestCaptureTierGating:
    def test_minimal_config_gates_l3_l5_keeps_cost(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = OpenAIAgentsAdapter(mock_client, capture_config=CaptureConfig.minimal())
        adapter.connect()
        try:
            _drive_full_trace(adapter, "t_min")
        finally:
            adapter.disconnect()

        types = [e["event_type"] for e in uploaded["events"]]
        # L1 identity/agent tier kept.
        assert "agent.input" in types
        assert "agent.output" in types
        # L3 model invocation + L5 tool call gated OUT under minimal.
        assert "model.invoke" not in types
        assert "tool.call" not in types
        # cost.record is always enabled (billing must never be gated away).
        assert "cost.record" in types

    def test_full_config_emits_l3_l5(self, mock_client):
        """Vacuity control: the SAME lifecycle DOES emit L3/L5 under .full()."""
        uploaded = capture_framework_trace(mock_client)
        adapter = OpenAIAgentsAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect()
        try:
            _drive_full_trace(adapter, "t_full")
        finally:
            adapter.disconnect()

        types = [e["event_type"] for e in uploaded["events"]]
        assert "model.invoke" in types
        assert "tool.call" in types
        assert "cost.record" in types
