"""Offline error + attestation + redaction + cost floor for the AWS Strands adapter.

Closes the W2 census cells that the existing ``test_strands.py`` proves only via
hand-rolled exceptions (``RuntimeError``/``ValueError``), a ``root_hash is not
None`` smoke, per-key content gating, and a tokens-only cost assertion — by
driving the *real* Strands runtime so a regression fails in plain CI with no
credentials and no network:

* Error-paths — a REAL ``openai`` SDK exception is raised the way a real Strands
                model call raises it: a real ``strands.Agent`` runs its real
                event loop over an ``httpx.MockTransport`` that returns a 404, so
                the real ``openai`` client raises ``openai.NotFoundError``,
                Strands' event loop constructs the real
                ``AfterModelCallEvent(exception=...)`` (event_loop.py) and the
                adapter surfaces it on ``model.invoke`` with the honest
                ``error_type == "NotFoundError"`` (the real class name, NOT a
                hand-built ``RuntimeError``) and the real message verbatim.
* Attestation — a real ``strands.Agent`` run over a recorded OpenAI SSE stream
                (the proven ``test_strands_recorded`` seam) flushes a trace whose
                attestation chain reconstructs and ``verify_chain(...)`` returns
                valid; a tamper control proves the check is not vacuous.
* Cost        — the SAME real run emits a ``cost.record`` carrying a *priced*
                ``cost_usd`` on the real per-cycle token shape Strands lifted off
                the recorded stream (Group-B adjudication: the adapter's
                ``pricing_table=BEDROCK_PRICING`` AUGMENTS the default ``PRICING``,
                so its OpenAI-backed real path — ``gpt-4o-mini`` — still prices).
* Redaction   — a full hook lifecycle driven through a real
                ``strands.hooks.HookRegistry`` (the real dispatch the adapter
                registers on) with ``capture_content=False`` keeps the structural
                events — and a SENTINEL sweep over ``json.dumps(events)`` — free
                of input/output/system_prompt/tool content, with a
                ``capture_content=True`` vacuity control proving the same path
                DOES carry the content otherwise.

The only mock is the network boundary (``httpx.MockTransport``); every Strands
object, event, hook dispatch and the adapter's own parser are real. A real
Strands ``Agent`` registry cannot be dispatched synchronously (its built-in
retry strategy registers async callbacks), so the redaction lane uses a clean
real ``HookRegistry`` with only the adapter registered — the same real dispatch
seam the solid ``test_strands.py`` uses.
"""

from __future__ import annotations

import json
from typing import Any, Dict
from unittest.mock import Mock

import httpx
import pytest

strands_mod = pytest.importorskip("strands")
pytest.importorskip("openai")

from strands import Agent  # noqa: E402
from strands.hooks import HookRegistry  # noqa: E402
from strands.hooks.events import (  # noqa: E402
    AfterToolCallEvent,
    AfterModelCallEvent,
    BeforeToolCallEvent,
    AfterInvocationEvent,
    BeforeModelCallEvent,
    BeforeInvocationEvent,
)
from strands.models.openai import OpenAIModel  # noqa: E402

import openai  # noqa: E402
from layerlens.attestation._verify import verify_chain  # noqa: E402
from layerlens.attestation._envelope import HashScope, AttestationEnvelope  # noqa: E402
from layerlens.instrument._capture_config import CaptureConfig  # noqa: E402
from layerlens.instrument.adapters.frameworks.strands import StrandsAdapter  # noqa: E402

from .conftest import find_event, capture_framework_trace  # noqa: E402
from ..._recorded import load_recorded, mock_transport  # noqa: E402

SENTINEL = "LL-SENTINEL-7f3a9c2e"


# ---------------------------------------------------------------------------
# Real-runtime model seams (network is the ONLY mock)
# ---------------------------------------------------------------------------
def _error_model(status: int, body: Dict[str, Any], *, model_id: str) -> OpenAIModel:
    """A real ``OpenAIModel`` whose AsyncOpenAI is served an error status over a
    MockTransport — the real openai SDK raises the genuine SDK exception."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=body)

    return OpenAIModel(
        client_args={"api_key": "test-key", "http_client": httpx.AsyncClient(transport=httpx.MockTransport(handler))},
        model_id=model_id,
    )


def _recorded_model(fixture: Dict[str, Any]) -> OpenAIModel:
    """A real ``OpenAIModel`` backed by the recorded OpenAI SSE stream — the real
    openai client parses the recorded body over the injected MockTransport."""
    transport, _ = mock_transport(fixture)
    return OpenAIModel(
        client_args={"api_key": "test-key", "http_client": httpx.AsyncClient(transport=transport)},
        model_id="gpt-4o-mini",
    )


# ---------------------------------------------------------------------------
# Real hook-dispatch seam (real HookRegistry + real Strands event dataclasses)
# ---------------------------------------------------------------------------
def _data_agent(*, name: str, system_prompt: str) -> Mock:
    """A passive data-container agent carrying a REAL ``strands.hooks.HookRegistry``.

    A real ``strands.Agent`` registry cannot be driven synchronously with
    hand-built events (its built-in retry strategy registers async callbacks and
    ``HookRegistry.invoke_callbacks`` refuses to run async callbacks); a clean
    real ``HookRegistry`` with only the adapter registered is the real dispatch
    seam ``test_strands.py`` already relies on. The adapter reads simple
    attributes off the agent; every hook event + the dispatch are real Strands.
    """
    agent = Mock()
    agent.name = name
    type(agent).__name__ = "Agent"
    agent.model = Mock()
    agent.model.config = {"model_id": "gpt-4o-mini"}
    agent.tool_names = ["search"]
    agent.system_prompt = system_prompt
    agent.event_loop_metrics = Mock()
    agent.event_loop_metrics.agent_invocations = []
    agent.hooks = HookRegistry()
    return agent


def _drive_hook_lifecycle(agent: Any, sentinel: str) -> None:
    """Drive a full agent lifecycle THROUGH THE REAL HookRegistry with content
    in every slot (system_prompt/input/tool-input/tool-output/output), each
    carrying ``sentinel``."""
    hk = agent.hooks
    hk.invoke_callbacks(
        BeforeInvocationEvent(
            agent=agent,
            invocation_state={},
            messages=[{"role": "user", "content": [{"text": f"question {sentinel}"}]}],
        )
    )
    hk.invoke_callbacks(BeforeModelCallEvent(agent=agent, invocation_state={}))
    hk.invoke_callbacks(
        AfterModelCallEvent(
            agent=agent,
            invocation_state={},
            stop_response=AfterModelCallEvent.ModelStopResponse(message=Mock(), stop_reason="tool_use"),
        )
    )
    tool_use = {"name": "search", "toolUseId": "t1", "input": {"q": f"find {sentinel}"}}
    hk.invoke_callbacks(BeforeToolCallEvent(agent=agent, selected_tool=Mock(), tool_use=tool_use, invocation_state={}))
    hk.invoke_callbacks(
        AfterToolCallEvent(
            agent=agent,
            selected_tool=Mock(),
            tool_use=tool_use,
            invocation_state={},
            result={"toolUseId": "t1", "status": "success", "content": [{"text": f"result {sentinel}"}]},
        )
    )
    result = Mock()
    result.stop_reason = "end_turn"
    result.message = {"role": "assistant", "content": [{"text": f"answer {sentinel}"}]}
    hk.invoke_callbacks(AfterInvocationEvent(agent=agent, invocation_state={}, result=result))


# ---------------------------------------------------------------------------
# Real error-shape floor (real openai exception, raised the real way)
# ---------------------------------------------------------------------------
class TestRealErrorShape:
    def test_real_openai_error_surfaces_on_model_invoke(self, mock_client):
        # A real Strands Agent run whose model 404s — the real openai client
        # raises the genuine SDK exception, Strands' event loop constructs the
        # real AfterModelCallEvent(exception=...), and the adapter surfaces it.
        body = {
            "error": {
                "message": "The model `gpt-4o-mini-ghost` does not exist",
                "type": "invalid_request_error",
                "code": "model_not_found",
            }
        }
        uploaded = capture_framework_trace(mock_client)
        adapter = StrandsAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect()
        raised: BaseException | None = None
        try:
            agent = Agent(model=_error_model(404, body, model_id="gpt-4o-mini-ghost"), hooks=[adapter], name="err_agent")
            try:
                agent("hello")
            except Exception as exc:  # the real model error propagates out of the run
                raised = exc
        finally:
            adapter.disconnect()

        # Prove it is the real openai class, not a hand-rolled stand-in.
        assert isinstance(raised, openai.OpenAIError), f"expected a real openai SDK error, got {type(raised)!r}"
        assert type(raised).__name__ == "NotFoundError"
        real_message = str(raised)

        events = uploaded["events"]
        # AfterInvocationEvent fires in a ``finally`` even on error, so the trace
        # flushes with the model error captured.
        model_evt = find_event(events, "model.invoke")
        payload = model_evt["payload"]
        # Honest adapter classification carries the REAL class name (bite: lost if
        # the adapter stops reading event.exception or hard-codes a type).
        assert payload["error_type"] == "NotFoundError"
        # The REAL exception message flows through verbatim (bite: dropped/mangled
        # error text fails here), tied to the real HTTP status of the class.
        assert payload["error"] == real_message
        assert "404" in payload["error"]
        assert payload["framework"] == "strands"


# ---------------------------------------------------------------------------
# Offline attestation-chain verification + cost over a real Agent run
# ---------------------------------------------------------------------------
class TestAttestationOffline:
    def test_attestation_chain_verifies_over_real_agent_run(self, mock_client):
        fixture = load_recorded("openai", "stream")
        uploaded = capture_framework_trace(mock_client)

        adapter = StrandsAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect()
        try:
            agent = Agent(model=_recorded_model(fixture), hooks=[adapter], name="pong_agent")
            result = agent("Reply with exactly: pong")
        finally:
            adapter.disconnect()

        # The real Strands event loop consumed the recorded SSE deltas.
        assert str(result).strip() == "pong"

        events = uploaded["events"]
        assert events, "real agent run must flush a non-empty trace"
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

    def test_cost_usd_priced_on_real_token_shape(self, mock_client):
        """Group-B cost adjudication: the OpenAI-backed real path prices.

        Strands sets ``pricing_table=BEDROCK_PRICING`` but its recorded/live/sample
        model is ``gpt-4o-mini``. The base ``_price_cost_record`` merges
        ``{**PRICING, **BEDROCK_PRICING}`` (AUGMENT, not replace), so the real
        per-cycle token shape Strands lifts off the recorded stream (12/1/13) is
        priced. Bite: a regression that dropped the default table (BEDROCK-only)
        would leave ``cost_usd is None`` here.
        """
        fixture = load_recorded("openai", "stream")
        uploaded = capture_framework_trace(mock_client)

        adapter = StrandsAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect()
        try:
            agent = Agent(model=_recorded_model(fixture), hooks=[adapter], name="cost_agent")
            agent("Reply with exactly: pong")
        finally:
            adapter.disconnect()

        cost = find_event(uploaded["events"], "cost.record")["payload"]
        assert cost["framework"] == "strands"
        assert cost["model"] == "gpt-4o-mini"
        # Real per-cycle usage parsed off the recorded stream's terminal usage chunk.
        assert cost["tokens_prompt"] == 12
        assert cost["tokens_completion"] == 1
        assert cost["tokens_total"] == 13
        # The Group-B assertion: a positive USD price, not None (cell closes).
        assert isinstance(cost["cost_usd"], float) and cost["cost_usd"] > 0, (
            f"cost_usd not priced on the OpenAI-backed real path: {cost.get('cost_usd')!r}"
        )


# ---------------------------------------------------------------------------
# Redaction content-absence over a real hook-dispatch lifecycle
# ---------------------------------------------------------------------------
class TestRedactionFloor:
    def test_content_present_when_capturing(self, mock_client):
        """Vacuity control: with capture_content=True the SAME real hook
        lifecycle DOES carry the SENTINEL and the content keys it rides on."""
        uploaded = capture_framework_trace(mock_client)
        adapter = StrandsAdapter(mock_client, capture_config=CaptureConfig(capture_content=True))
        agent = _data_agent(name="Helper", system_prompt=f"You are helpful {SENTINEL}")
        adapter.connect(target=agent)
        _drive_hook_lifecycle(agent, SENTINEL)
        adapter.disconnect()

        events = uploaded["events"]
        assert SENTINEL in json.dumps(events), "control run must carry the SENTINEL when capturing content"
        assert SENTINEL in find_event(events, "environment.config")["payload"]["system_prompt"]
        assert "input" in find_event(events, "agent.input")["payload"]
        assert "output" in find_event(events, "agent.output")["payload"]
        assert "input" in find_event(events, "tool.call")["payload"]
        assert "output" in find_event(events, "tool.result")["payload"]

    def test_content_absent_when_not_capturing(self, mock_client):
        """capture_content=False keeps the structural events — and strips every
        content slot plus the SENTINEL — from the stored trace."""
        uploaded = capture_framework_trace(mock_client)
        adapter = StrandsAdapter(mock_client, capture_config=CaptureConfig(capture_content=False))
        agent = _data_agent(name="Helper", system_prompt=f"You are helpful {SENTINEL}")
        adapter.connect(target=agent)
        _drive_hook_lifecycle(agent, SENTINEL)
        adapter.disconnect()

        events = uploaded["events"]
        assert events, "the lifecycle must still emit structural events without content"

        # 1) SENTINEL sweep over the serialized trace.
        assert SENTINEL not in json.dumps(events), "PRIVACY LEAK: SENTINEL survived capture_content=False"

        # 2) Every content slot is absent from the payload that would carry it...
        config = find_event(events, "environment.config")["payload"]
        assert "system_prompt" not in config, "environment.config leaked system_prompt under capture_content=False"
        assert "input" not in find_event(events, "agent.input")["payload"], "agent.input leaked 'input'"
        assert "output" not in find_event(events, "agent.output")["payload"], "agent.output leaked 'output'"
        assert "input" not in find_event(events, "tool.call")["payload"], "tool.call leaked 'input'"
        assert "output" not in find_event(events, "tool.result")["payload"], "tool.result leaked 'output'"

        # 3) ...while the STRUCTURE survives (bite: a redaction that nuked the
        # events entirely would pass the sweep but fail here).
        assert config["framework"] == "strands"
        assert find_event(events, "agent.input")["payload"]["agent_name"] == "Helper"
        assert find_event(events, "tool.call")["payload"]["tool_name"] == "search"
        assert find_event(events, "model.invoke")["payload"]["model"] == "gpt-4o-mini"
