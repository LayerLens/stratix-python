"""Offline error + attestation + redaction + cost floor for the PydanticAI adapter.

Closes the W2 census ◑/gap cells that ``test_pydantic_ai.py`` proves only via the
fake ``TestModel`` happy path or a synthetic tool ``ValueError`` — so a regression
fails in plain CI with no credentials and no network:

* Error-paths — a REAL provider HTTP failure driven through the instrumented
                ``_InstrumentedModel.request`` seam surfaces as ``agent.error``
                with the REAL ``pydantic_ai.exceptions.ModelHTTPError`` class name
                and the real error text verbatim. The existing suite only fires a
                synthetic tool ``ValueError`` and never asserts ``error_type``,
                and the model-level ``_emit_model_error`` path (the except branch
                inside ``_InstrumentedModel.request``) was untested offline. We
                isolate the model-level ``agent.error`` (the one WITHOUT an
                ``agent_name`` — the run-level ``_finish_run_error`` adds it) so
                the assertion bites that specific handler.
* Attestation — a REAL ``pydantic_ai.Agent`` (real ``OpenAIModel`` over a mocked
                transport serving the recorded OpenAI body) flushes a trace whose
                attestation chain reconstructs and ``verify_chain(...)`` returns
                valid; a tamper control breaking link 1 proves the check is not
                vacuous. The existing ``test_attestation_present`` only asserts a
                non-None trace id.
* Redaction   — a REAL agent tool-loop with ``capture_content=False`` keeps the
                structural events but strips every content field — and a SENTINEL
                sweep over ``json.dumps(events)`` proves the planted secret never
                reaches the stored trace — with a ``capture_content=True`` vacuity
                control proving the same path DOES carry it otherwise.
* Cost        — the run's ``cost.record`` carries a real ``cost_usd`` priced off
                the real recorded token shape (``gpt-4o-mini``); the existing
                suite only asserts ``tokens_total`` so a broken pricing lookup
                fails no test today.
* Field-hygiene — under ``capture_content=False`` every emitted payload's keys
                stay within a per-event allowlist and no raw content key leaks,
                closing the ``params`` gap (this adapter exposes no model-param
                allowlist surface, so the hygiene guarantee is "no unexpected/raw
                field escapes the emitter").

The only mock is the network boundary (``httpx.MockTransport`` for the real
pydantic-ai OpenAI provider, or the offline ``TestModel``); every pydantic-ai
object, the real agent loop, and the adapter's own parser are real.
"""

from __future__ import annotations

import os
import json
import asyncio
from dataclasses import dataclass

import pytest

pydantic_ai = pytest.importorskip("pydantic_ai")

import httpx  # noqa: E402
from pydantic_ai import Agent  # noqa: E402
from pydantic_ai.exceptions import ModelHTTPError  # noqa: E402
from pydantic_ai.models.test import TestModel  # noqa: E402
from pydantic_ai.models.openai import OpenAIModel  # noqa: E402
from pydantic_ai.providers.openai import OpenAIProvider  # noqa: E402

from layerlens.attestation._verify import verify_chain  # noqa: E402
from layerlens.attestation._envelope import HashScope, AttestationEnvelope  # noqa: E402
from layerlens.instrument._capture_config import CaptureConfig  # noqa: E402
from layerlens.instrument.adapters.frameworks.pydantic_ai import PydanticAIAdapter  # noqa: E402

from .conftest import find_event, find_events, capture_framework_trace  # noqa: E402
from ..._recorded import load_recorded, mock_transport  # noqa: E402

SENTINEL = "LL-SENTINEL-7f3a9c2e"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _recorded_agent(fixture, name: str) -> Agent:
    """A real ``Agent`` whose real ``OpenAIModel`` is backed by a MockTransport
    serving the recorded OpenAI ChatCompletion — the proven seam from
    ``test_pydantic_ai_recorded.py`` (no key, no network)."""
    transport, _ = mock_transport(fixture)
    provider = OpenAIProvider(api_key="test-key", http_client=httpx.AsyncClient(transport=transport))
    model = OpenAIModel("gpt-4o-mini", provider=provider)
    return Agent(model, name=name)


def _sentinel_tool(city: str) -> str:
    """A tool whose return value carries the SENTINEL (drives tool.result content)."""
    return f"weather for {city}: {SENTINEL}"


# ---------------------------------------------------------------------------
# Real error-shape floor (real provider HTTP failure through _emit_model_error)
# ---------------------------------------------------------------------------
class TestRealErrorShape:
    def test_real_model_http_error_surfaces_as_agent_error(self, mock_client):
        # A real 404 from the provider makes the REAL pydantic-ai OpenAIModel raise
        # a REAL ``ModelHTTPError`` inside ``super().request(...)`` — caught by the
        # instrumented ``_InstrumentedModel.request`` except branch. NOT a synthetic
        # string, NOT a hand-built exception.
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                404,
                json={
                    "error": {
                        "message": "The model `gpt-4o-mini-ghost` does not exist",
                        "type": "invalid_request_error",
                        "code": "model_not_found",
                    }
                },
            )

        transport = httpx.MockTransport(handler)
        provider = OpenAIProvider(api_key="test-key", http_client=httpx.AsyncClient(transport=transport))
        model = OpenAIModel("gpt-4o-mini", provider=provider)
        agent = Agent(model, name="billing_agent")

        uploaded = capture_framework_trace(mock_client)
        adapter = PydanticAIAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect(target=agent)
        with pytest.raises(ModelHTTPError):
            agent.run_sync("hi")
        adapter.disconnect()

        events = uploaded["events"]
        errors = find_events(events, "agent.error")
        assert errors, "a real model failure must emit agent.error"

        # The model-level ``_emit_model_error`` path emits an agent.error WITHOUT
        # an ``agent_name`` (the run-level ``_finish_run_error`` adds the declared
        # name). Isolating it bites that specific handler: if
        # ``_InstrumentedModel.request`` stops emitting on failure, this drops to 0.
        model_level = [e for e in errors if "agent_name" not in e["payload"]]
        assert len(model_level) == 1, (
            f"expected exactly one model-level agent.error from _emit_model_error; saw {[e['payload'] for e in errors]}"
        )
        payload = model_level[0]["payload"]

        # The REAL SDK exception class name — not a synthetic RuntimeError/str.
        assert payload["error_type"] == ModelHTTPError.__name__ == "ModelHTTPError"
        assert payload["framework"] == "pydantic-ai"
        assert "latency_ms" in payload
        # The real exception text flows through verbatim (bite: dropped/mangled
        # error text fails here). Tied to the real HTTP status the class carries.
        assert "status_code: 404" in payload["error"]
        assert "does not exist" in payload["error"]


# ---------------------------------------------------------------------------
# Offline attestation-chain verification over a real recorded run
# ---------------------------------------------------------------------------
class TestAttestationOffline:
    def test_attestation_chain_verifies_over_real_run(self, mock_client):
        os.environ.pop("OPENAI_API_KEY", None)
        fixture = load_recorded("openai", "default")
        uploaded = capture_framework_trace(mock_client)

        agent = _recorded_agent(fixture, "attest_agent")
        adapter = PydanticAIAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect(target=agent)
        result = agent.run_sync("Reply with exactly: pong")
        adapter.disconnect()

        assert result.output == "pong"

        events = uploaded["events"]
        assert events, "real recorded run must flush a non-empty trace"
        chain = (uploaded["attestation"] or {}).get("chain") or {}
        raw = chain.get("events") or []
        envelopes = [
            AttestationEnvelope(hash=e["hash"], scope=HashScope(e["scope"]), previous_hash=e.get("previous_hash"))
            for e in raw
        ]
        assert envelopes, "no attestation envelopes captured for the real pydantic-ai trace"
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
# Cost floor — real cost_usd priced off the real recorded token shape
# ---------------------------------------------------------------------------
class TestCostFloor:
    def test_cost_usd_present_on_real_token_shape(self, mock_client):
        os.environ.pop("OPENAI_API_KEY", None)
        fixture = load_recorded("openai", "default")
        uploaded = capture_framework_trace(mock_client)

        agent = _recorded_agent(fixture, "cost_agent")
        adapter = PydanticAIAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect(target=agent)
        agent.run_sync("Reply with exactly: pong")
        adapter.disconnect()

        cost = find_event(uploaded["events"], "cost.record")
        payload = cost["payload"]
        # Real recorded token shape (gpt-4o-mini: 12 prompt / 1 completion).
        assert payload["model"] == "gpt-4o-mini"
        assert payload["tokens_prompt"] == 12
        assert payload["tokens_completion"] == 1
        assert payload["tokens_total"] == 13
        # The dollar figure the existing suite never checks: base ``_price_cost_record``
        # must price the declared model. Bite: a broken pricing lookup (model not in
        # PRICING, or the augmentation removed) drops cost_usd and fails HERE.
        assert "cost_usd" in payload, "cost.record carried no cost_usd — pricing lookup broke"
        assert payload["cost_usd"] > 0
        # 12 prompt + 1 completion at the gpt-4o-mini rate.
        assert payload["cost_usd"] == pytest.approx(2.4e-06, rel=1e-3)


# ---------------------------------------------------------------------------
# Redaction content-absence over a real tool-loop run (offline TestModel)
# ---------------------------------------------------------------------------
class TestRedactionFloor:
    @staticmethod
    def _run(mock_client, config: CaptureConfig):
        uploaded = capture_framework_trace(mock_client)
        agent = Agent(model=TestModel(custom_output_text=f"done {SENTINEL}"), name="claims_agent")
        agent.tool_plain(_sentinel_tool)
        adapter = PydanticAIAdapter(mock_client, capture_config=config)
        adapter.connect(target=agent)
        agent.run_sync(f"secret prompt {SENTINEL}")
        adapter.disconnect()
        return uploaded["events"]

    def test_content_present_when_capturing(self, mock_client):
        """Vacuity control: capture_content=True DOES carry the SENTINEL and the
        content keys it rides on across the SAME tool-loop run."""
        events = self._run(mock_client, CaptureConfig.full())
        assert SENTINEL in json.dumps(events), "control run must carry the SENTINEL when capturing content"
        assert find_event(events, "agent.input")["payload"]["input"] == f"secret prompt {SENTINEL}"
        assert SENTINEL in str(find_event(events, "agent.output")["payload"]["output"])
        assert "input" in find_event(events, "tool.call")["payload"]
        assert SENTINEL in str(find_event(events, "tool.result")["payload"]["output"])

    def test_content_absent_when_not_capturing(self, mock_client):
        """capture_content=False keeps the structural events but strips every
        content field — and the SENTINEL — from the stored trace."""
        events = self._run(mock_client, CaptureConfig(capture_content=False))
        assert events, "the tool-loop must still emit structural events without content"
        # Structure survives: the tool round-trip still produced call + result.
        assert find_events(events, "tool.call"), "tool.call dropped under capture_content=False"
        assert find_events(events, "tool.result"), "tool.result dropped under capture_content=False"

        # 1) SENTINEL sweep over the serialized trace.
        assert SENTINEL not in json.dumps(events), "PRIVACY LEAK: SENTINEL survived capture_content=False"

        # 2) The content keys must be absent from every payload that would carry them.
        for e in find_events(events, "agent.input"):
            assert "input" not in e["payload"], "agent.input leaked 'input' under capture_content=False"
        for e in find_events(events, "agent.output"):
            assert "output" not in e["payload"], "agent.output leaked 'output' under capture_content=False"
        for e in find_events(events, "tool.call"):
            assert "input" not in e["payload"], "tool.call leaked 'input' under capture_content=False"
        for e in find_events(events, "tool.result"):
            assert "output" not in e["payload"], "tool.result leaked 'output' under capture_content=False"


# ---------------------------------------------------------------------------
# Deps privacy — deps_summary must be SHAPE-ONLY, never raw dependency values
# ---------------------------------------------------------------------------
class TestDepsPrivacyFloor:
    def test_deps_values_never_leak_shape_only(self, mock_client):
        """Dependencies are request-scoped secrets (tokens, db handles). Under
        capture_content=True the adapter records a deps *shape* summary — key
        names + value TYPES — and must NEVER store a raw dependency value, even
        when ``safe_serialize(deps)`` falls back to ``str(deps)`` (the dataclass /
        arbitrary-object case, whose repr embeds the values verbatim)."""
        os.environ.pop("OPENAI_API_KEY", None)
        fixture = load_recorded("openai", "default")
        uploaded = capture_framework_trace(mock_client)

        @dataclass
        class _Deps:
            api_token: str
            user_id: int

        transport, _ = mock_transport(fixture)
        provider = OpenAIProvider(api_key="test-key", http_client=httpx.AsyncClient(transport=transport))
        agent = Agent(OpenAIModel("gpt-4o-mini", provider=provider), name="deps_agent", deps_type=_Deps)
        adapter = PydanticAIAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect(target=agent)
        agent.run_sync("Reply with exactly: pong", deps=_Deps(api_token=SENTINEL, user_id=7))
        adapter.disconnect()

        events = uploaded["events"]
        # 1) The raw secret must not appear ANYWHERE in the stored trace.
        assert SENTINEL not in json.dumps(events), "deps_summary leaked a raw dependency value"

        # 2) deps_summary is captured and is a SHAPE-ONLY dict (never the raw repr string).
        inp = find_event(events, "agent.input")["payload"]
        assert "deps_summary" in inp, "deps_summary should be captured (shape-only) under capture_content=True"
        summary = inp["deps_summary"]
        assert isinstance(summary, dict), (
            f"deps_summary must be a shape-only dict, got {type(summary).__name__}: {summary!r}"
        )
        assert summary.get("type") == "_Deps"
        # Field NAMES and TYPES only — never the values.
        assert summary.get("fields") == {"api_token": "str", "user_id": "int"}
        assert SENTINEL not in json.dumps(summary)


# ---------------------------------------------------------------------------
# Streamed agent.output content floor (real StreamedRunResult, offline TestModel)
# ---------------------------------------------------------------------------
class TestStreamingOutputFloor:
    """A streamed run must emit ``agent.output`` WITH the run's real resolved
    output content under ``capture_content=True`` — and stay HONEST (no ``output``
    key, no content leak) under ``capture_content=False``.

    pydantic-ai's ``StreamedRunResult`` exposes its result ONLY via the
    ``await get_output()`` coroutine — it has no ``.output`` attribute the way the
    non-streaming ``AgentRunResult`` does. So a run wrapper that reads ``.output``
    drops the streamed output entirely. This floor drives the REAL agent loop over
    the offline ``TestModel`` (no key, no network) and, crucially, the consumer
    here consumes only the text stream and does NOT itself call ``get_output()`` —
    so the resolved content can only reach ``agent.output`` if the *wrapper*
    awaits it. A ``capture_content=False`` control proves the same path stays
    redacted (structural ``streaming=True`` event, no ``output``, no SENTINEL).
    """

    @staticmethod
    def _run_stream(mock_client, config: CaptureConfig):
        uploaded = capture_framework_trace(mock_client)
        agent = Agent(model=TestModel(custom_output_text=f"streamed {SENTINEL}"), name="stream_out_agent")
        adapter = PydanticAIAdapter(mock_client, capture_config=config)
        adapter.connect(target=agent)

        async def _go() -> None:
            # Consume the text stream the way a production caller would, but do
            # NOT call get_output() here — the wrapper must resolve it itself.
            async with agent.run_stream("stream me") as stream:
                async for _ in stream.stream_text(delta=True):
                    pass

        asyncio.get_event_loop().run_until_complete(_go())
        adapter.disconnect()
        return uploaded["events"]

    def test_streamed_output_carries_resolved_content(self, mock_client):
        events = self._run_stream(mock_client, CaptureConfig.full())
        out = find_event(events, "agent.output")
        assert out["payload"]["status"] == "ok"
        assert out["payload"]["streaming"] is True
        # The heart of the floor: the streamed output content must be present and
        # be the run's REAL resolved output (from await get_output()), not dropped.
        assert "output" in out["payload"], (
            "streamed agent.output dropped the resolved output content "
            "(StreamedRunResult has no .output attr — the wrapper must await get_output())"
        )
        assert out["payload"]["output"] == f"streamed {SENTINEL}"
        # And the content genuinely reaches the stored trace.
        assert SENTINEL in json.dumps(events), "resolved streamed output never reached the trace"

    def test_streamed_output_honest_when_not_capturing(self, mock_client):
        events = self._run_stream(mock_client, CaptureConfig(capture_content=False))
        out = find_event(events, "agent.output")
        # Structural event survives and is still marked streamed…
        assert out["payload"]["streaming"] is True
        assert out["payload"]["status"] == "ok"
        # …but resolving the output for cost/usage must NOT leak it into the trace.
        assert "output" not in out["payload"], "streamed agent.output leaked 'output' under capture_content=False"
        assert SENTINEL not in json.dumps(events), "PRIVACY LEAK: streamed output SENTINEL survived redaction"


# ---------------------------------------------------------------------------
# Field-hygiene / allowlist sweep (closes the params gap)
# ---------------------------------------------------------------------------
class TestFieldHygiene:
    #: Keys each event type may honestly carry on a redacted (capture_content=False)
    #: non-streaming run. A NEW raw field escaping the emitter fails this sweep.
    _ALLOWED = {
        "agent.input": {"framework", "agent_name", "model", "result_type", "deps_type"},
        "model.invoke": {
            "framework",
            "agent_name",
            "model",
            "latency_ms",
            "response_id",
            "tokens_prompt",
            "tokens_completion",
            "tokens_total",
        },
        "agent.output": {
            "framework",
            "agent_name",
            "model",
            "status",
            "latency_ms",
            "model_requests",
            "tokens_prompt",
            "tokens_completion",
            "tokens_total",
        },
        "cost.record": {
            "framework",
            "model",
            "model_requests",
            "cost_usd",
            "tokens_prompt",
            "tokens_completion",
            "tokens_total",
        },
        "agent.identity": {"framework", "agent_name", "source"},
    }
    #: Raw content keys that must NEVER appear on any event under capture_content=False.
    _CONTENT_KEYS = {"input", "output", "context", "deps_summary", "messages", "system_prompt", "prompt"}

    def test_no_unexpected_or_raw_fields_under_redaction(self, mock_client):
        os.environ.pop("OPENAI_API_KEY", None)
        fixture = load_recorded("openai", "default")
        uploaded = capture_framework_trace(mock_client)

        agent = _recorded_agent(fixture, "hygiene_agent")
        adapter = PydanticAIAdapter(mock_client, capture_config=CaptureConfig(capture_content=False))
        adapter.connect(target=agent)
        agent.run_sync("Reply with exactly: pong")
        adapter.disconnect()

        events = uploaded["events"]
        assert events, "run must emit events"
        for e in events:
            et = e["event_type"]
            keys = set(e["payload"].keys())
            allowed = self._ALLOWED.get(et)
            assert allowed is not None, f"unexpected event type {et!r} emitted by the pydantic-ai floor run"
            extra = keys - allowed
            assert not extra, (
                f"{et} leaked unexpected/raw payload key(s) {sorted(extra)} under capture_content=False "
                f"(a deliberate schema addition must update this allowlist)"
            )
            # No raw content key may ride on any event when content capture is off.
            assert not (keys & self._CONTENT_KEYS), (
                f"{et} leaked raw content key(s) {sorted(keys & self._CONTENT_KEYS)} under capture_content=False"
            )
