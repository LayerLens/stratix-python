"""Offline error + attestation + cost + redaction floor for the Google ADK adapter.

Closes the W2 census cells that ``test_google_adk.py`` proves only via direct
``adapter._on_*`` handler calls with hand-built ``Mock`` contexts and *synthetic*
``RuntimeError``/``ValueError`` errors. Every test here drives a **real ADK
``Runner``** (real ``LlmAgent`` + ``InMemorySessionService`` + the adapter's real
``BasePlugin``) whose ``Gemini`` model is backed by an ``httpx.MockTransport`` —
the network is the ONLY mock. A regression fails in plain CI with no credentials
and no spend:

* Error-shape — a REAL ``google.genai.errors.ClientError`` (an ``APIError``
                subclass) raised by the *real* genai client from a real ``400``
                transport response, propagated through the real ADK model flow
                (``base_llm_flow`` catches it and runs ``on_model_error`` on the
                plugin manager). It surfaces as ``agent.error`` with the honest
                ``error_type == "ClientError"`` (the real class name, NOT the
                synthetic ``RuntimeError`` the existing suite feeds) and the real
                exception message flowing through verbatim.
* Attestation — a real ``Runner`` run over the recorded ``google_genai`` body
                flushes a trace whose attestation chain reconstructs and
                ``verify_chain(...)`` returns valid; a tamper control proves the
                check is not vacuous.
* Cost        — the same real run's ``cost.record`` carries a numeric
                ``cost_usd`` priced from the real Gemini token shape
                (``gemini-2.5-flash`` in PRICING); the value is recomputed from
                the pricing table so a dropped/mis-priced cost bites.
* Redaction   — a real ``Runner`` run with ``capture_content=False`` keeps the
                structural fields but strips the agent instruction/description
                (system prompt) and user input — and a SENTINEL sweep over
                ``json.dumps(events)`` — from the stored trace, with a
                ``capture_content=True`` vacuity control proving the same path
                DOES carry the content otherwise.

The ADK adapter never captures model/agent *output* text (only tokens /
finish_reason), so — unlike the crewai floor — the redaction controls ride on
the instruction/description (``environment.config``) and the user message
(``agent.input``), the content slots this adapter actually emits.
"""

from __future__ import annotations

import sys
import json
import asyncio
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

if sys.version_info < (3, 10):
    pytest.skip("google-adk requires Python >= 3.10", allow_module_level=True)

pytest.importorskip("google.adk")
pytest.importorskip("google.genai")

import google.genai.errors as genai_errors  # noqa: E402
from google import genai  # noqa: E402
from google.genai import types  # noqa: E402
from google.adk.agents import LlmAgent  # noqa: E402
from google.adk.runners import Runner  # noqa: E402
from google.adk.sessions import InMemorySessionService  # noqa: E402
from google.adk.models.google_llm import Gemini  # noqa: E402

from layerlens.attestation._verify import verify_chain  # noqa: E402
from layerlens.attestation._envelope import HashScope, AttestationEnvelope  # noqa: E402
from layerlens.instrument._capture_config import CaptureConfig  # noqa: E402
from layerlens.instrument.adapters.providers.pricing import PRICING, calculate_cost  # noqa: E402
from layerlens.instrument.adapters.frameworks.google_adk import GoogleADKAdapter  # noqa: E402
from layerlens.instrument.adapters.providers.token_usage import NormalizedTokenUsage  # noqa: E402

from .conftest import find_event, find_events, capture_framework_trace  # noqa: E402
from ..._recorded import load_recorded, mock_transport  # noqa: E402

SENTINEL = "LL-SENTINEL-7f3a9c2e"


# ---------------------------------------------------------------------------
# Real-ADK-Runner helpers (the proven seam from test_google_adk_recorded.py)
# ---------------------------------------------------------------------------
def _recorded_gemini(transport: httpx.MockTransport) -> Gemini:
    """A real ADK ``Gemini`` whose ``api_client`` is routed through a
    MockTransport (the documented subclass-and-override injection seam). ADK
    calls the async client, so both sync/async client args carry the transport."""

    class _RecordedGemini(Gemini):
        @property
        def api_client(self) -> Any:
            return genai.Client(
                api_key="test-key",
                http_options=types.HttpOptions(
                    client_args={"transport": transport},
                    async_client_args={"transport": transport},
                ),
            )

    return _RecordedGemini(model="gemini-2.5-flash")


async def _drive_runner(
    adapter: GoogleADKAdapter,
    transport: httpx.MockTransport,
    *,
    instruction: str = "Reply with exactly: pong",
    description: str = "",
    user_text: str = "Reply with exactly: pong",
) -> None:
    """Run a real ADK ``Runner`` end-to-end over the recorded body. The plugin
    fires on the real Runner lifecycle; ``after_run`` flushes the trace."""
    agent = LlmAgent(
        name="concierge",
        model=_recorded_gemini(transport),
        instruction=instruction,
        description=description,
    )
    session_service = InMemorySessionService()
    await session_service.create_session(app_name="app", user_id="u", session_id="s")
    runner = Runner(app_name="app", agent=agent, session_service=session_service, plugins=[adapter.plugin])
    async for _ in runner.run_async(
        user_id="u",
        session_id="s",
        new_message=types.Content(role="user", parts=[types.Part(text=user_text)]),
    ):
        pass


def _error_transport(status: int, message: str) -> httpx.MockTransport:
    """A transport that returns a Gemini-shaped API error for every request, so
    the real genai client raises a real ``ClientError``/``ServerError``."""
    body = {"error": {"code": status, "message": message, "status": "INVALID_ARGUMENT"}}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=body)

    return httpx.MockTransport(handler)


async def _drive_runner_failing(
    adapter: GoogleADKAdapter, transport: httpx.MockTransport
) -> BaseException:
    """Run a real ``Runner`` whose model call fails. ADK re-raises the model
    error out of ``run_async`` BEFORE its Step-4 ``after_run`` callback, so the
    run never auto-flushes — we drive the same real plugin ``after_run_callback``
    the Runner would have called, so the ``agent.error`` trace is uploaded."""
    agent = LlmAgent(
        name="concierge",
        model=_recorded_gemini(transport),
        instruction="Act as a travel concierge.",
    )
    session_service = InMemorySessionService()
    await session_service.create_session(app_name="app", user_id="u", session_id="s")
    runner = Runner(app_name="app", agent=agent, session_service=session_service, plugins=[adapter.plugin])
    raised: BaseException | None = None
    try:
        async for _ in runner.run_async(
            user_id="u",
            session_id="s",
            new_message=types.Content(role="user", parts=[types.Part(text="Plan a trip to Kyoto")]),
        ):
            pass
    except Exception as exc:  # noqa: BLE001 - we assert the real class below
        raised = exc
    await adapter.plugin.after_run_callback(invocation_context=SimpleNamespace(agent=agent))
    assert raised is not None, "the failing model transport must raise out of run_async"
    return raised


# ---------------------------------------------------------------------------
# Real error-shape floor (real google.genai exception via the real ADK flow)
# ---------------------------------------------------------------------------
class TestRealErrorShape:
    def test_real_genai_error_surfaces_as_agent_error(self, mock_client):
        message = "google_adk floor: the model `gemini-ghost` does not exist"
        uploaded = capture_framework_trace(mock_client)

        adapter = GoogleADKAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect()
        try:
            raised = asyncio.run(_drive_runner_failing(adapter, _error_transport(400, message)))
        finally:
            adapter.disconnect()

        # The real genai client raised a genuine google.genai error class — the
        # shape a real ADK Gemini call raises — NOT a hand-rolled stand-in.
        assert isinstance(raised, genai_errors.APIError), f"expected a real google.genai APIError, got {type(raised)}"
        assert type(raised).__name__ == "ClientError"  # 4xx -> ClientError
        real_message = str(raised)

        events = uploaded["events"]
        errors = find_events(events, "agent.error")
        assert len(errors) == 1, f"expected exactly one agent.error, saw {[e['payload'] for e in errors]}"
        payload = errors[0]["payload"]

        # Honest classification: the error_type is the REAL google.genai class
        # name lifted off the exception (bite: lost if the adapter hard-codes a
        # label or mis-reads type(error).__name__).
        assert payload["error_type"] == type(raised).__name__ == "ClientError"
        # The REAL exception message flows through VERBATIM (bite: dropped/mangled
        # text fails here). Tied to the real exception object, not a literal.
        assert payload["error"] == real_message
        assert message in payload["error"]
        assert "400" in payload["error"]
        assert payload["framework"] == "google_adk"
        # The request model is carried onto the error (honest, non-fabricated).
        assert payload["model"] == "gemini-2.5-flash"


# ---------------------------------------------------------------------------
# Offline attestation-chain verification over a real Runner run
# ---------------------------------------------------------------------------
class TestAttestationOffline:
    def test_attestation_chain_verifies_over_real_runner(self, mock_client):
        fixture = load_recorded("google_genai", "default")
        transport, _ = mock_transport(fixture)
        uploaded = capture_framework_trace(mock_client)

        adapter = GoogleADKAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect()
        try:
            asyncio.run(_drive_runner(adapter, transport))
        finally:
            adapter.disconnect()

        events = uploaded["events"]
        assert events, "real Runner run must flush a non-empty trace"
        chain = (uploaded["attestation"] or {}).get("chain") or {}
        raw = chain.get("events") or []
        envelopes = [
            AttestationEnvelope(hash=e["hash"], scope=HashScope(e["scope"]), previous_hash=e.get("previous_hash"))
            for e in raw
        ]
        assert envelopes, "no attestation envelopes captured for the real ADK trace"
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
# Cost floor — numeric cost_usd priced from the real Gemini token shape
# ---------------------------------------------------------------------------
class TestCostFloor:
    def test_cost_usd_priced_on_real_gemini_tokens(self, mock_client):
        fixture = load_recorded("google_genai", "default")
        transport, _ = mock_transport(fixture)
        uploaded = capture_framework_trace(mock_client)

        adapter = GoogleADKAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect()
        try:
            asyncio.run(_drive_runner(adapter, transport))
        finally:
            adapter.disconnect()

        cost = find_event(uploaded["events"], "cost.record")
        payload = cost["payload"]
        # Real token accounting off the recorded usageMetadata (prompt+completion).
        assert payload["model"] == "gemini-2.5-flash"
        assert payload["tokens_prompt"] == 6
        assert payload["tokens_completion"] == 1

        # cost_usd must be present + numeric + positive (bite: None/absent if the
        # adapter stops pricing framework cost.records or drops the model id).
        cost_usd = payload.get("cost_usd")
        assert cost_usd is not None, "cost.record shipped WITHOUT cost_usd — framework cost pricing broke"
        assert isinstance(cost_usd, (int, float)) and cost_usd > 0

        # And it equals the pricing-table computation for these exact tokens (bite:
        # a mis-priced value — wrong model/rate/token wiring — fails here).
        expected = calculate_cost(
            "gemini-2.5-flash",
            NormalizedTokenUsage(prompt_tokens=6, completion_tokens=1, total_tokens=7),
            PRICING,
        )
        assert expected is not None and expected > 0
        assert cost_usd == expected


# ---------------------------------------------------------------------------
# Redaction content-absence over a real Runner run
# ---------------------------------------------------------------------------
class TestRedactionFloor:
    def test_content_present_when_capturing(self, mock_client):
        """Vacuity control: with capture_content=True the SAME real Runner run
        DOES carry the SENTINEL on the content keys it rides on (agent
        instruction/description + the user input)."""
        fixture = load_recorded("google_genai", "default")
        transport, _ = mock_transport(fixture)
        uploaded = capture_framework_trace(mock_client)

        adapter = GoogleADKAdapter(mock_client, capture_config=CaptureConfig(capture_content=True))
        adapter.connect()
        try:
            asyncio.run(
                _drive_runner(
                    adapter,
                    transport,
                    instruction=f"System policy: {SENTINEL}",
                    description=f"Confidential brief: {SENTINEL}",
                    user_text=f"User asks: {SENTINEL}",
                )
            )
        finally:
            adapter.disconnect()

        events = uploaded["events"]
        assert SENTINEL in json.dumps(events), "control run must carry the SENTINEL when capturing content"
        config = find_event(events, "environment.config")
        assert SENTINEL in config["payload"]["instruction"], "instruction must carry content when capturing"
        assert SENTINEL in config["payload"]["description"], "description must carry content when capturing"
        assert any(SENTINEL in json.dumps(e["payload"].get("input")) for e in find_events(events, "agent.input")), (
            "at least one agent.input must carry the user content when capturing"
        )

    def test_content_absent_when_not_capturing(self, mock_client):
        """capture_content=False keeps the structural events but strips the agent
        instruction/description (system prompt) and user input — and the SENTINEL
        — from every stored event."""
        fixture = load_recorded("google_genai", "default")
        transport, _ = mock_transport(fixture)
        uploaded = capture_framework_trace(mock_client)

        adapter = GoogleADKAdapter(mock_client, capture_config=CaptureConfig(capture_content=False))
        adapter.connect()
        try:
            asyncio.run(
                _drive_runner(
                    adapter,
                    transport,
                    instruction=f"System policy: {SENTINEL}",
                    description=f"Confidential brief: {SENTINEL}",
                    user_text=f"User asks: {SENTINEL}",
                )
            )
        finally:
            adapter.disconnect()

        events = uploaded["events"]
        assert events, "the lifecycle must still emit structural events without content"

        # 1) SENTINEL sweep over the serialized trace.
        assert SENTINEL not in json.dumps(events), "PRIVACY LEAK: SENTINEL survived capture_content=False"

        # 2) The content keys must be absent from every payload that would carry them.
        config = find_event(events, "environment.config")
        assert "instruction" not in config["payload"], "environment.config leaked 'instruction' (system prompt)"
        assert "description" not in config["payload"], "environment.config leaked 'description'"
        for e in find_events(events, "agent.input"):
            assert "input" not in e["payload"], "agent.input leaked user 'input' under capture_content=False"

        # 3) The structural, non-content fields still ship (proves the strip is
        # scoped to content, not the whole event).
        assert config["payload"]["agent_name"] == "concierge"
        assert "model" in config["payload"]
