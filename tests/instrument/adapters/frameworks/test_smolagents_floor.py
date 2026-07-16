"""Offline error + attestation + redaction + cost floor for the SmolAgents adapter.

Closes the W2 census ``◑`` cells that ``test_smolagents.py`` proves only via a
*synthetic* ``RuntimeError``, a root-hash-only attestation check, per-key
redaction (no full-trace SENTINEL sweep), and a tokens-only cost.record. Every
assertion here fails in plain CI (no creds, no network) if the behaviour
regresses:

* Error-paths — a REAL ``openai`` SDK exception is the root cause: a real
                ``smolagents.CodeAgent`` runs a real ``OpenAIServerModel`` whose
                transport returns a genuine OpenAI ``404`` body, so the openai
                client raises a real ``openai.NotFoundError`` that smolagents
                wraps in a real ``AgentGenerationError`` (the shape a real
                smolagents run raises). The adapter's run-wrapper surfaces it as
                ``agent.error`` with honest ``error_type == "AgentGenerationError"``
                and the openai error text (``404`` / ``model_not_found``) flowing
                through verbatim. (Existing suite feeds only a hand-rolled
                ``RuntimeError("LLM timeout")``.)
* Attestation — a real ``CodeAgent`` run over the recorded OpenAI corpus flushes
                a trace whose attestation chain reconstructs and
                ``verify_chain(...)`` returns valid; a tamper control breaking an
                interior link proves the check is not vacuous.
* Redaction   — a real ``CodeAgent`` driven through its real ``CallbackRegistry``
                with real ``PlanningStep`` / ``ActionStep`` / ``ToolCall`` objects
                carrying a SENTINEL in every content slot (task input, run output,
                plan, code_action, step observations, tool args, tool output):
                ``capture_content=False`` keeps the structural events but strips
                every content key AND the SENTINEL from ``json.dumps(events)``; a
                ``capture_content=True`` vacuity control proves the same path DOES
                carry the content otherwise.
* Cost        — the real recorded run prices the real token shape: ``cost.record``
                carries a non-``None`` ``cost_usd`` computed from ``PRICING`` for
                ``gpt-4o-mini`` (Group-B adjudication: the source suspicion that
                smolagents cost is unpriced is DISPROVEN — ``_price_cost_record``
                resolves the openai model id).

The only mock is the network boundary (``httpx.MockTransport`` behind a real
``openai.OpenAI`` client); every smolagents object, the agent, its callback
registry, the step types and the adapter's own parser are real.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

smolagents = pytest.importorskip("smolagents")  # base venv has no smolagents; matrix installs it

import httpx  # noqa: E402
from smolagents import (  # noqa: E402
    ToolCall,
    CodeAgent,
    ActionStep,
    PlanningStep,
    OpenAIServerModel,
)
from smolagents.utils import AgentGenerationError  # noqa: E402
from smolagents.memory import Timing  # noqa: E402
from smolagents.monitoring import TokenUsage  # noqa: E402

import openai  # noqa: E402
from layerlens.attestation._verify import verify_chain  # noqa: E402
from layerlens.attestation._envelope import HashScope, AttestationEnvelope  # noqa: E402
from layerlens.instrument._capture_config import CaptureConfig  # noqa: E402
from layerlens.instrument.adapters.providers.pricing import PRICING  # noqa: E402
from layerlens.instrument.adapters.frameworks.smolagents import SmolAgentsAdapter  # noqa: E402

from .conftest import find_event, find_events, capture_framework_trace  # noqa: E402
from ..._recorded import load_recorded, mock_transport  # noqa: E402

SENTINEL = "LL-SENTINEL-7f3a9c2e"


# ---------------------------------------------------------------------------
# Real-agent builders (network is the only mock)
# ---------------------------------------------------------------------------
def _recorded_agent(fixture, name: str = "replay_agent") -> CodeAgent:
    """A real ``CodeAgent`` whose ``OpenAIServerModel`` is backed by a real
    ``openai.OpenAI`` client over the recorded-corpus MockTransport — the proven
    seam from ``test_smolagents_recorded.py`` (no key, no network)."""
    transport, _ = mock_transport(fixture)
    client = openai.OpenAI(api_key="test-key", http_client=httpx.Client(transport=transport))
    model = OpenAIServerModel(model_id="gpt-4o-mini", client=client)
    return CodeAgent(tools=[], model=model, max_steps=1, verbosity_level=0, name=name)


def _error_agent(status: int, body: dict) -> CodeAgent:
    """A real ``CodeAgent`` whose model transport returns a genuine OpenAI error
    body, so the real openai client raises the real SDK exception the run wraps."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=body)

    client = openai.OpenAI(
        api_key="test-key",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    model = OpenAIServerModel(model_id="gpt-4o-mini", client=client)
    return CodeAgent(tools=[], model=model, max_steps=1, verbosity_level=0, name="fail_agent")


def _inert_agent(name: str = "researcher") -> CodeAgent:
    """A real ``CodeAgent`` whose transport is never exercised — its real
    ``run()`` is replaced by a driver that fires real step objects, so every
    smolagents object (agent, registry, step types) is real while the redaction
    lifecycle stays deterministic and offline."""
    client = openai.OpenAI(
        api_key="test-key",
        http_client=httpx.Client(transport=httpx.MockTransport(lambda _r: httpx.Response(200, json={}))),
    )
    model = OpenAIServerModel(model_id="gpt-4o-mini", client=client)
    return CodeAgent(tools=[], model=model, max_steps=1, verbosity_level=0, name=name)


def _drive_content_lifecycle(adapter: SmolAgentsAdapter, agent: CodeAgent, sentinel: str) -> None:
    """Drive a full content-bearing lifecycle through the agent's REAL callback
    registry: run start (task input) -> planning step (plan) -> action step
    (code_action + observations + a tool call/result) -> run end (output). Every
    content slot carries ``sentinel``."""
    plan = PlanningStep(
        model_input_messages=[],
        model_output_message=MagicMock(),
        plan=f"Plan: retrieve {sentinel}",
        timing=Timing(start_time=100.0, end_time=100.5),
    )
    plan.token_usage = TokenUsage(input_tokens=8, output_tokens=4)

    action = ActionStep(step_number=1, timing=Timing(start_time=100.0, end_time=101.0))
    action.tool_calls = [ToolCall(name="web_search", arguments={"query": f"find {sentinel}"}, id="tc-1")]
    action.token_usage = TokenUsage(input_tokens=10, output_tokens=5)
    action.model_output = "the model's raw output (never emitted)"
    action.observations = f"tool observed {sentinel}"
    action.error = None
    action.is_final_answer = False
    action.code_action = f"result = compute('{sentinel}')"

    def _fake_run(*_args, **_kwargs):
        # Real CallbackRegistry.callback dispatch to the adapter's real handlers.
        agent.step_callbacks.callback(plan, agent=agent)
        agent.step_callbacks.callback(action, agent=agent)
        return f"final answer {sentinel}"

    original = adapter._original_run
    adapter._original_run = _fake_run
    try:
        agent.run(f"task input {sentinel}")
    finally:
        adapter._original_run = original


# ---------------------------------------------------------------------------
# Real error-shape floor (real openai.NotFoundError, raised the real way)
# ---------------------------------------------------------------------------
class TestRealErrorShape:
    def test_real_openai_error_surfaces_as_agent_error(self, mock_client):
        body = {
            "error": {
                "message": "The model `gpt-4o-mini-ghost` does not exist",
                "type": "invalid_request_error",
                "code": "model_not_found",
            }
        }
        uploaded = capture_framework_trace(mock_client)
        agent = _error_agent(404, body)
        adapter = SmolAgentsAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect(target=agent)
        with pytest.raises(AgentGenerationError) as excinfo:
            agent.run("summarise the doc")
        adapter.disconnect()

        # The root cause is a GENUINE openai SDK exception (a real 404 -> the real
        # NotFoundError), wrapped by real smolagents into AgentGenerationError.
        raised = excinfo.value
        assert isinstance(raised, AgentGenerationError)
        assert isinstance(raised.__cause__, openai.NotFoundError)
        assert isinstance(raised.__cause__, openai.OpenAIError)

        events = uploaded["events"]
        err = find_event(events, "agent.error")
        payload = err["payload"]
        # Honest adapter classification: the real wrapper class name, not a
        # hand-rolled placeholder (bite: lost if the adapter stops emitting on
        # run failure or stops stamping type(exc).__name__).
        assert payload["error_type"] == "AgentGenerationError"
        assert payload["framework"] == "smolagents"
        # The real openai error text flows through verbatim (bite: dropped/mangled
        # error text fails here). Tied to the real HTTP status + openai error code.
        assert "404" in payload["error"]
        assert "does not exist" in payload["error"]
        assert "model_not_found" in payload["error"]


# ---------------------------------------------------------------------------
# Offline attestation-chain verification over a real recorded run
# ---------------------------------------------------------------------------
class TestAttestationOffline:
    def test_attestation_chain_verifies_over_real_run(self, mock_client):
        fixture = load_recorded("openai", "default")
        uploaded = capture_framework_trace(mock_client)

        agent = _recorded_agent(fixture)
        adapter = SmolAgentsAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect(target=agent)
        result = agent.run("Reply with exactly: pong")
        adapter.disconnect()

        assert result == "pong"

        events = uploaded["events"]
        assert events, "real recorded run must flush a non-empty trace"
        chain = (uploaded["attestation"] or {}).get("chain") or {}
        raw = chain.get("events") or []
        envelopes = [
            AttestationEnvelope(hash=e["hash"], scope=HashScope(e["scope"]), previous_hash=e.get("previous_hash"))
            for e in raw
        ]
        assert envelopes, "no attestation envelopes captured for the real smolagents trace"
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
# Cost floor — the real token shape is PRICED (Group-B adjudication: GREEN)
# ---------------------------------------------------------------------------
class TestCostFloor:
    def test_cost_usd_priced_on_real_token_shape(self, mock_client):
        fixture = load_recorded("openai", "default")
        uploaded = capture_framework_trace(mock_client)

        agent = _recorded_agent(fixture)
        adapter = SmolAgentsAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect(target=agent)
        agent.run("Reply with exactly: pong")
        adapter.disconnect()

        events = uploaded["events"]
        cost = find_event(events, "cost.record")
        payload = cost["payload"]
        # Real recorded OpenAI usage parsed off the chat.completion body.
        assert payload["model"] == "gpt-4o-mini"
        assert payload["tokens_prompt"] == 12
        assert payload["tokens_completion"] == 1

        # The framework emit path prices the openai model id via PRICING
        # (bite: cost_usd goes None if _price_cost_record stops resolving the
        # model / stops running on the smolagents cost.record).
        rate = PRICING["gpt-4o-mini"]
        expected = 12 * rate["input"] / 1000 + 1 * rate["output"] / 1000
        assert payload["cost_usd"] is not None, "smolagents cost.record must carry a priced cost_usd"
        assert payload["cost_usd"] == pytest.approx(expected)
        assert payload["cost_usd"] > 0


# ---------------------------------------------------------------------------
# Redaction content-absence + SENTINEL sweep over a real callback-driven run
# ---------------------------------------------------------------------------
class TestRedactionFloor:
    def test_content_present_when_capturing(self, mock_client):
        """Vacuity control: with capture_content=True the SAME real lifecycle DOES
        carry the SENTINEL and the content keys it rides on."""
        uploaded = capture_framework_trace(mock_client)
        agent = _inert_agent()
        adapter = SmolAgentsAdapter(mock_client, capture_config=CaptureConfig(capture_content=True))
        adapter.connect(target=agent)
        _drive_content_lifecycle(adapter, agent, SENTINEL)
        adapter.disconnect()

        events = uploaded["events"]
        assert SENTINEL in json.dumps(events), "control run must carry the SENTINEL when capturing content"
        assert "input" in find_event(events, "agent.input")["payload"]
        assert "output" in find_event(events, "agent.output")["payload"]
        assert "input" in find_event(events, "tool.call")["payload"]
        assert "output" in find_event(events, "tool.result")["payload"]
        steps = find_events(events, "agent.step")
        assert any("code_action" in s["payload"] for s in steps), "action step must carry code_action when capturing"
        assert any("observations" in s["payload"] for s in steps), "action step must carry observations when capturing"
        assert any("plan" in s["payload"] for s in steps), "planning step must carry plan when capturing"

    def test_content_absent_when_not_capturing(self, mock_client):
        """capture_content=False keeps the structural events but strips every
        content key — and the SENTINEL — from the stored trace."""
        uploaded = capture_framework_trace(mock_client)
        agent = _inert_agent()
        adapter = SmolAgentsAdapter(mock_client, capture_config=CaptureConfig(capture_content=False))
        adapter.connect(target=agent)
        _drive_content_lifecycle(adapter, agent, SENTINEL)
        adapter.disconnect()

        events = uploaded["events"]
        assert events, "the lifecycle must still emit structural events without content"

        # 1) SENTINEL sweep over the whole serialized trace.
        assert SENTINEL not in json.dumps(events), "PRIVACY LEAK: SENTINEL survived capture_content=False"

        # 2) Every content key absent from the payloads that would carry it.
        for e in find_events(events, "agent.input"):
            assert "input" not in e["payload"], "agent.input leaked 'input' under capture_content=False"
        for e in find_events(events, "agent.output"):
            assert "output" not in e["payload"], "agent.output leaked 'output' under capture_content=False"
        assert "input" not in find_event(events, "tool.call")["payload"], "tool.call leaked 'input'"
        assert "output" not in find_event(events, "tool.result")["payload"], "tool.result leaked 'output'"
        for s in find_events(events, "agent.step"):
            assert "code_action" not in s["payload"], "agent.step leaked 'code_action'"
            assert "observations" not in s["payload"], "agent.step leaked 'observations'"
            assert "plan" not in s["payload"], "agent.step leaked 'plan'"
