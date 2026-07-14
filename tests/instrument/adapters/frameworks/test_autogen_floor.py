"""Offline error + attestation + redaction + cost floor for the AutoGen adapter.

Closes the W2 census ◑/gap cells for ``autogen`` so a regression fails in plain
CI with no credentials and no network. Every event is a *real* ``autogen_core``
event class dispatched through the adapter's *real* production chokepoint — the
module-global ``EVENT_LOGGER_NAME`` logging handler (``logger.info(event)``) — or,
for the cost cell, a *real* ``OpenAIChatCompletionClient.create()`` over a mocked
transport serving the recorded OpenAI ``chat.completions`` body. The network is
the only thing mocked.

* Error-paths (``error`` cell — hardened) — the three REAL autogen error events
                (``MessageDroppedEvent`` / ``MessageHandlerExceptionEvent`` /
                ``AgentConstructionExceptionEvent``) are fired the real way, on
                the real event logger, and each surfaces as ``agent.error`` with
                the uniform ``status == "error"`` shape, the adapter's honest
                ``message_dropped`` classification for a drop, the honest
                per-agent identity on the exception paths, and the real exception
                message flowing through verbatim. (Real autogen stringifies the
                exception in the event's kwargs, so ``error_type`` honestly falls
                back to ``"Exception"`` on the exception paths — we assert the
                verbatim message + status + honest name, never a class name the
                adapter cannot recover.)
* Attestation (``attest`` cell — new) — a real multi-agent autogen flow
                (router↔fulfillment handoffs + an LLM call) flushed as its own
                trace produces an attestation chain that ``verify_chain(...)``
                accepts (one envelope per event, ``root_hash`` present); a tamper
                control proves the check is not vacuous.
* Redaction   (``redaction`` cell — new SENTINEL sweep) — a real content-bearing
                lifecycle with ``capture_content=False`` keeps message / tool /
                model content — and a SENTINEL sweep over ``json.dumps(events)`` —
                out of every stored event, with a ``capture_content=True`` vacuity
                control proving the SAME path DOES carry the content otherwise.
* Cost        (``cost`` cell — new ``cost_usd`` assertion) — the ``cost.record``
                emitted off a REAL OpenAI usage block (recorded ``12/1`` tokens on
                the dated ``gpt-4o-mini-2024-07-18`` id) carries a real
                ``cost_usd`` filled by ``_price_cost_record`` — the dollar figure
                the existing suite never asserted — equal to an independently
                recomputed ``calculate_cost`` and strictly positive.

Requires autogen-core >= 0.4 (Python >= 3.10); CI-gated via the ``autogen`` row
of ``tests/matrix/frameworks.toml``.
"""

from __future__ import annotations

import sys
import json
import asyncio
import logging

import pytest

# Mirror test_autogen.py's env guards: autogen-core needs py>=3.10, and its
# import can raise TypeError on incompatible versions. importorskip only catches
# ImportError, so guard explicitly.
if sys.version_info < (3, 10):
    pytest.skip("autogen-core requires Python >= 3.10", allow_module_level=True)
try:
    import autogen_core  # noqa: F401
except (ImportError, TypeError):  # pragma: no cover - env guard
    pytest.skip("autogen-core not installed or incompatible", allow_module_level=True)
try:
    from autogen_core.models import UserMessage
    from autogen_ext.models.openai import OpenAIChatCompletionClient
except (ImportError, Exception):  # pragma: no cover - env guard
    pytest.skip("autogen not installed or incompatible", allow_module_level=True)

import httpx  # noqa: E402
from autogen_core import EVENT_LOGGER_NAME, AgentId  # noqa: E402
from autogen_core.logging import (  # noqa: E402
    MessageKind,
    LLMCallEvent,
    MessageEvent,
    DeliveryStage,
    ToolCallEvent,
    MessageDroppedEvent,
    MessageHandlerExceptionEvent,
    AgentConstructionExceptionEvent,
)

from layerlens.attestation._verify import verify_chain  # noqa: E402
from layerlens.attestation._envelope import HashScope, AttestationEnvelope  # noqa: E402
from layerlens.instrument._capture_config import CaptureConfig  # noqa: E402
from layerlens.instrument.adapters.providers.pricing import PRICING, calculate_cost  # noqa: E402
from layerlens.instrument.adapters.frameworks.autogen import AutoGenAdapter  # noqa: E402
from layerlens.instrument.adapters.providers.token_usage import NormalizedTokenUsage  # noqa: E402

from .conftest import find_event, find_events, capture_framework_trace  # noqa: E402
from ..._recorded import load_recorded, mock_transport  # noqa: E402

SENTINEL = "LL-SENTINEL-7f3a9c2e"


# ---------------------------------------------------------------------------
# Real-event-logger helpers (the proven pattern from test_autogen.py)
# ---------------------------------------------------------------------------
def _logger() -> logging.Logger:
    return logging.getLogger(EVENT_LOGGER_NAME)


def _drive_content_lifecycle(logger: logging.Logger, sentinel: str) -> None:
    """Drive a full content-bearing autogen lifecycle through the REAL logger.

    A DIRECT message (agent.input content) -> an LLM call (model.invoke messages
    + output_message) -> a tool call (tool.call input + output) -> a RESPOND
    message (agent.output content). Every content slot carries ``sentinel``.
    """
    logger.info(
        MessageEvent(
            payload=f"please handle order {sentinel}",
            sender=AgentId("router", "default"),
            receiver=AgentId("fulfillment", "default"),
            kind=MessageKind.DIRECT,
            delivery_stage=DeliveryStage.SEND,
        )
    )
    logger.info(
        LLMCallEvent(
            messages=[{"role": "user", "content": f"prompt {sentinel}"}],
            response={
                "model": "gpt-4o-mini",
                "choices": [{"message": {"content": f"reply {sentinel}"}}],
            },
            prompt_tokens=20,
            completion_tokens=8,
        )
    )
    logger.info(
        ToolCallEvent(
            tool_name="inventory_lookup",
            arguments={"sku": f"sku-{sentinel}"},
            result=f"in stock {sentinel}",
        )
    )
    logger.info(
        MessageEvent(
            payload=f"order shipped {sentinel}",
            sender=AgentId("fulfillment", "default"),
            receiver=AgentId("router", "default"),
            kind=MessageKind.RESPOND,
            delivery_stage=DeliveryStage.SEND,
        )
    )


# ---------------------------------------------------------------------------
# Real error-shape floor (the 3 real autogen error events, fired the real way)
# ---------------------------------------------------------------------------
class TestRealErrorShape:
    def test_real_error_events_surface_as_agent_error(self, mock_client):
        # Genuine Python exceptions carried on the REAL autogen error events —
        # the shapes a real autogen runtime logs (a dropped message, a handler
        # crash, a construction failure). Fired on the REAL event logger, not via
        # direct ``adapter._on_*`` calls.
        handler_exc = RuntimeError(f"inventory lookup crashed {SENTINEL}")
        construction_exc = TypeError(f"missing required param {SENTINEL}")
        assert type(handler_exc).__name__ == "RuntimeError"
        assert type(construction_exc).__name__ == "TypeError"

        uploaded = capture_framework_trace(mock_client)
        adapter = AutoGenAdapter(mock_client, capture_config=CaptureConfig(capture_content=True))
        adapter.connect()
        logger = _logger()
        logger.info(
            MessageDroppedEvent(
                payload="blocked",
                sender=AgentId("router", "default"),
                receiver=AgentId("fulfillment", "default"),
                kind=MessageKind.DIRECT,
            )
        )
        logger.info(
            MessageHandlerExceptionEvent(
                payload="bad message",
                handling_agent=AgentId("fulfillment", "default"),
                exception=handler_exc,
            )
        )
        logger.info(
            AgentConstructionExceptionEvent(
                agent_id=AgentId("broken", "default"),
                exception=construction_exc,
            )
        )
        adapter.disconnect()

        events = uploaded["events"]
        errors = find_events(events, "agent.error")
        assert len(errors) == 3, f"all 3 real error events must surface as agent.error, saw {[e['payload'] for e in errors]}"
        # Uniform agent.error shape across all 3 paths (bite: lost if any path
        # stops classifying status or stops tagging the framework).
        assert all(e["payload"]["status"] == "error" for e in errors)
        assert all(e["payload"]["framework"] == "autogen" for e in errors)

        # Dropped path — the adapter's honest classification (bite: lost if the
        # drop stops being tagged or stops carrying its endpoints).
        dropped = [e for e in errors if e["payload"].get("error_type") == "message_dropped"]
        assert len(dropped) == 1
        assert dropped[0]["payload"]["dropped"] is True
        assert dropped[0]["payload"]["sender"] == "router/default"
        assert dropped[0]["payload"]["receiver"] == "fulfillment/default"

        # Handler-exception path — the REAL exception message flows through
        # verbatim, carrying the SENTINEL, on the honest per-agent identity (bite:
        # dropped/mangled error text or a lost agent binding fails here).
        handler_errs = [e for e in errors if "inventory lookup crashed" in (e["payload"].get("error") or "")]
        assert len(handler_errs) == 1
        assert SENTINEL in handler_errs[0]["payload"]["error"]
        assert handler_errs[0]["payload"]["agent_id"] == "fulfillment/default"
        assert handler_errs[0]["payload"]["agent_name"] == "fulfillment"

        # Construction-exception path — same verbatim + honest-identity contract.
        constr_errs = [e for e in errors if "missing required param" in (e["payload"].get("error") or "")]
        assert len(constr_errs) == 1
        assert SENTINEL in constr_errs[0]["payload"]["error"]
        assert constr_errs[0]["payload"]["agent_id"] == "broken/default"
        assert constr_errs[0]["payload"]["agent_name"] == "broken"


# ---------------------------------------------------------------------------
# Offline attestation-chain verification over a real multi-agent autogen flow
# ---------------------------------------------------------------------------
class TestAttestationOffline:
    def test_attestation_chain_verifies_over_real_flow(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        adapter = AutoGenAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect()
        logger = _logger()
        # A real multi-agent autogen turn: router hands the request to
        # fulfillment, fulfillment invokes the model and responds.
        logger.info(
            MessageEvent(
                payload="route the order",
                sender=AgentId("router", "default"),
                receiver=AgentId("fulfillment", "default"),
                kind=MessageKind.DIRECT,
                delivery_stage=DeliveryStage.SEND,
            )
        )
        logger.info(
            LLMCallEvent(
                messages=[{"role": "user", "content": "route the order"}],
                response={"model": "gpt-4o-mini"},
                prompt_tokens=40,
                completion_tokens=12,
            )
        )
        logger.info(
            MessageEvent(
                payload="order shipped",
                sender=AgentId("fulfillment", "default"),
                receiver=AgentId("router", "default"),
                kind=MessageKind.RESPOND,
                delivery_stage=DeliveryStage.SEND,
            )
        )
        adapter.disconnect()

        events = uploaded["events"]
        assert events, "the real autogen flow must flush a non-empty trace"
        chain = (uploaded["attestation"] or {}).get("chain") or {}
        raw = chain.get("events") or []
        envelopes = [
            AttestationEnvelope(hash=e["hash"], scope=HashScope(e["scope"]), previous_hash=e.get("previous_hash"))
            for e in raw
        ]
        assert envelopes, "no attestation envelopes captured for the real autogen trace"
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
# Redaction content-absence over a real bus-driven autogen lifecycle
# ---------------------------------------------------------------------------
class TestRedactionFloor:
    def test_content_present_when_capturing(self, mock_client):
        """Vacuity control: with capture_content=True the SAME real lifecycle
        DOES carry the SENTINEL and the content keys it rides on."""
        uploaded = capture_framework_trace(mock_client)
        adapter = AutoGenAdapter(mock_client, capture_config=CaptureConfig(capture_content=True))
        adapter.connect()
        _drive_content_lifecycle(_logger(), SENTINEL)
        adapter.disconnect()

        events = uploaded["events"]
        assert SENTINEL in json.dumps(events), "control run must carry the SENTINEL when capturing content"
        assert any("content" in e["payload"] for e in find_events(events, "agent.input"))
        assert any("content" in e["payload"] for e in find_events(events, "agent.output"))
        mi = find_event(events, "model.invoke")
        assert "messages" in mi["payload"]
        assert "output_message" in mi["payload"]
        tc = find_event(events, "tool.call")
        assert "input" in tc["payload"]
        assert "output" in tc["payload"]

    def test_content_absent_when_not_capturing(self, mock_client):
        """capture_content=False keeps message/tool/model content — and the
        SENTINEL — out of every stored event, structure intact."""
        uploaded = capture_framework_trace(mock_client)
        adapter = AutoGenAdapter(mock_client, capture_config=CaptureConfig(capture_content=False))
        adapter.connect()
        _drive_content_lifecycle(_logger(), SENTINEL)
        adapter.disconnect()

        events = uploaded["events"]
        assert events, "the lifecycle must still emit structural events without content"
        # Structure is intact — the redaction removed CONTENT, not the events.
        assert find_events(events, "agent.input"), "structural agent.input lost under redaction"
        assert find_events(events, "agent.output"), "structural agent.output lost under redaction"
        assert find_events(events, "model.invoke"), "structural model.invoke lost under redaction"
        assert find_events(events, "tool.call"), "structural tool.call lost under redaction"

        # 1) SENTINEL sweep over the whole serialized trace.
        assert SENTINEL not in json.dumps(events), "PRIVACY LEAK: SENTINEL survived capture_content=False"

        # 2) Every content-bearing key is absent from the payloads that would carry it.
        for e in find_events(events, "agent.input"):
            assert "content" not in e["payload"], "agent.input leaked 'content' under capture_content=False"
        for e in find_events(events, "agent.output"):
            assert "content" not in e["payload"], "agent.output leaked 'content' under capture_content=False"
        for e in find_events(events, "model.invoke"):
            assert "messages" not in e["payload"], "model.invoke leaked 'messages' under capture_content=False"
            assert "output_message" not in e["payload"], "model.invoke leaked 'output_message'"
        for e in find_events(events, "tool.call"):
            assert "input" not in e["payload"], "tool.call leaked 'input' under capture_content=False"
            assert "output" not in e["payload"], "tool.call leaked 'output' under capture_content=False"


# ---------------------------------------------------------------------------
# Cost floor — cost_usd priced on a REAL OpenAI usage block
# ---------------------------------------------------------------------------
class TestCostFloor:
    def test_cost_usd_priced_on_real_token_shape(self, mock_client):
        # Drive a REAL OpenAIChatCompletionClient over the recorded OpenAI
        # chat.completions body: the real OpenAI SDK deserializes the recorded
        # usage block (12 prompt / 1 completion on the dated gpt-4o-mini id) and
        # autogen's real LLMCallEvent carries it. The adapter's cost.record must
        # then carry a real cost_usd filled by _price_cost_record — the dollar
        # figure the existing suite (tokens-only) never asserts.
        fixture = load_recorded("openai", "default")
        uploaded = capture_framework_trace(mock_client)

        transport, _ = mock_transport(fixture)
        client = OpenAIChatCompletionClient(
            model="gpt-4o-mini",
            api_key="test-key",
            http_client=httpx.AsyncClient(transport=transport),
        )
        adapter = AutoGenAdapter(mock_client)
        adapter.connect()
        try:
            result = asyncio.run(
                client.create([UserMessage(content="Reply with exactly: pong", source="user")])
            )
        finally:
            adapter.disconnect()

        # The real provider shape flowed through (dated id + real usage).
        assert result.content == "pong"

        cost = find_event(uploaded["events"], "cost.record")
        assert cost["payload"]["model"] == "gpt-4o-mini-2024-07-18"
        assert cost["payload"]["tokens_prompt"] == 12
        assert cost["payload"]["tokens_completion"] == 1

        # The dollar figure: present, positive, and equal to an independently
        # recomputed cost (bite: RED the moment _price_cost_record stops filling
        # cost_usd on a framework cost.record, or the model stops resolving).
        cost_usd = cost["payload"].get("cost_usd")
        assert cost_usd is not None, "cost.record carried NO cost_usd — pricing did not fill the dollar figure"
        assert cost_usd > 0, f"cost_usd must be strictly positive for a priced model, got {cost_usd}"
        expected = calculate_cost(
            "gpt-4o-mini-2024-07-18",
            NormalizedTokenUsage(prompt_tokens=12, completion_tokens=1, total_tokens=13),
            PRICING,
        )
        assert expected is not None and expected > 0
        assert cost_usd == expected, f"cost_usd {cost_usd} != independently recomputed {expected}"
