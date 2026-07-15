"""Offline privacy + error + attestation + cache-cost floor for the anthropic provider.

Closes the W1 census ◑ cells that were previously proven only in the gated live
lanes (or only via synthetic inputs), so a regression fails in plain CI without
any provider credentials:

* Redaction   — ``capture_content=False`` strips content AND a SENTINEL never
                reaches the stored trace, with a ``True`` vacuity control that
                proves the assertion is not vacuous.
* Error-paths — a REAL ``anthropic.NotFoundError`` (the shape of the shipped
                ``prodready_errors/anthropic_badmodel_404.json`` fixture) is fed
                through ``AnthropicProvider`` and surfaces as ``agent.error``
                with ``error_type == "NotFoundError"`` — not the synthetic
                ``RuntimeError`` the existing suite uses.
* Attestation — the captured trace's attestation chain verifies offline
                (mirrors the live harness ``_assert_attestation``).
* Cost/Tokens — non-zero prompt-cache tokens flow end-to-end into a priced
                ``cost.record`` through the adapter path (not a synthetic emit),
                proven by a strict cost delta vs an identical no-cache response.
"""

from __future__ import annotations

import json
from unittest.mock import Mock

import httpx
from anthropic import NotFoundError
from anthropic.types import Usage, Message, TextBlock

from layerlens.instrument import trace
from layerlens.instrument._capture_config import CaptureConfig
from layerlens.instrument.adapters.providers.anthropic import AnthropicProvider

from ...conftest import find_event

SENTINEL = "LL-SENTINEL-7f3a9c2e"


def _msg(
    text: str,
    *,
    model: str = "claude-3-opus-20240229",
    input_tokens: int = 20,
    output_tokens: int = 10,
    cache_read: int | None = None,
    cache_creation: int | None = None,
) -> Message:
    """A real Anthropic Message, optionally carrying prompt-cache token counts."""
    usage = Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_creation,
    )
    return Message(
        id="msg-test",
        type="message",
        role="assistant",
        model=model,
        content=[TextBlock(type="text", text=text)],
        usage=usage,
        stop_reason="end_turn",
    )


def _run(anthropic_client, mock_client, config, *, prompt="Hi"):
    provider = AnthropicProvider()
    provider.connect(anthropic_client)

    @trace(mock_client, capture_config=config)
    def my_agent():
        r = anthropic_client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return r.content[0].text

    return my_agent()


# ---------------------------------------------------------------------------
# Redaction floor (offline, credential-free)
# ---------------------------------------------------------------------------
class TestRedactionFloor:
    def test_content_stripped_when_capture_content_false(self, mock_client, capture_trace):
        client = Mock()
        client.messages.create = Mock(return_value=_msg("I'm Claude!"))
        _run(client, mock_client, CaptureConfig(capture_content=False))

        model_invoke = find_event(capture_trace["events"], "model.invoke")
        assert "messages" not in model_invoke["payload"]
        assert "output_message" not in model_invoke["payload"]
        # Usage + params still present (redaction removes CONTENT, not metadata).
        assert model_invoke["payload"]["usage"]["output_tokens"] == 10
        assert model_invoke["payload"]["parameters"]["model"] == "claude-3-opus-20240229"

    def test_content_present_when_capture_content_true(self, mock_client, capture_trace):
        """Vacuity control: the redaction assertion above is only meaningful if
        the SAME path DOES carry content when capture is on."""
        client = Mock()
        client.messages.create = Mock(return_value=_msg("I'm Claude!"))
        _run(client, mock_client, CaptureConfig.full())

        model_invoke = find_event(capture_trace["events"], "model.invoke")
        assert model_invoke["payload"]["output_message"]["text"] == "I'm Claude!"

    def test_sentinel_never_leaks_when_redacted(self, mock_client, capture_trace):
        client = Mock()
        client.messages.create = Mock(return_value=_msg(f"Secret is {SENTINEL}"))
        _run(client, mock_client, CaptureConfig(capture_content=False), prompt=f"Remember {SENTINEL}")

        blob = json.dumps(capture_trace["events"])
        assert SENTINEL not in blob, "content redaction leaked the SENTINEL into the stored trace"

    def test_sentinel_present_when_capture_on(self, mock_client, capture_trace):
        """Vacuity control for the sweep above."""
        client = Mock()
        client.messages.create = Mock(return_value=_msg(f"Secret is {SENTINEL}"))
        _run(client, mock_client, CaptureConfig.full(), prompt=f"Remember {SENTINEL}")
        assert SENTINEL in json.dumps(capture_trace["events"])


# ---------------------------------------------------------------------------
# Real error-shape floor (feeds the real 4xx shape through the adapter)
# ---------------------------------------------------------------------------
class TestRealErrorShape:
    def test_real_notfounderror_emits_agent_error(self, mock_client, capture_trace):
        raw = (
            "Error code: 404 - {'type': 'error', 'error': {'type': 'not_found_error', "
            "'message': 'model: claude-3-5-haiku-latest'}}"
        )
        response = httpx.Response(404, request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"))
        err = NotFoundError(raw, response=response, body=None)

        client = Mock()
        client.messages.create = Mock(side_effect=err)
        provider = AnthropicProvider()
        provider.connect(client)

        @trace(mock_client, capture_config=CaptureConfig.full())
        def my_agent():
            try:
                client.messages.create(model="claude-3-5-haiku-latest", max_tokens=8, messages=[])
            except NotFoundError:
                pass
            return "handled"

        my_agent()
        error = find_event(capture_trace["events"], "agent.error")
        # The REAL SDK exception class name — not the synthetic RuntimeError.
        assert error["payload"]["error_type"] == "NotFoundError"
        assert "404" in error["payload"]["error"]
        assert "latency_ms" in error["payload"]


# ---------------------------------------------------------------------------
# Offline attestation-chain verification
# ---------------------------------------------------------------------------
class TestAttestationOffline:
    def test_attestation_chain_verifies_offline(self, mock_client, capture_trace):
        from layerlens.attestation._verify import verify_chain
        from layerlens.attestation._envelope import HashScope, AttestationEnvelope

        client = Mock()
        client.messages.create = Mock(return_value=_msg("I'm Claude!"))
        _run(client, mock_client, CaptureConfig.full())

        events = capture_trace["events"]
        raw = (capture_trace["attestation"].get("chain") or {}).get("events") or []
        envelopes = [
            AttestationEnvelope(hash=e["hash"], scope=HashScope(e["scope"]), previous_hash=e.get("previous_hash"))
            for e in raw
        ]
        assert envelopes, "no attestation envelopes captured"
        assert len(envelopes) == len(events)
        result = verify_chain(envelopes)
        assert result.valid, f"attestation chain invalid: {result.error}"


# ---------------------------------------------------------------------------
# Prompt-cache tokens flow into a priced cost.record end-to-end
# ---------------------------------------------------------------------------
class TestCacheCostEndToEnd:
    def test_cache_tokens_increase_priced_cost(self, mock_client, capture_trace_list):
        client = Mock()
        # Two flushes over capture_trace_list: identical response but the second
        # carries non-zero prompt-cache tokens.
        client.messages.create = Mock(return_value=_msg("hi"))
        _run(client, mock_client, CaptureConfig.full())

        client.messages.create = Mock(return_value=_msg("hi", cache_read=1000, cache_creation=500))
        _run(client, mock_client, CaptureConfig.full())

        assert len(capture_trace_list) == 2
        cost_base = find_event(capture_trace_list[0]["events"], "cost.record")["payload"]
        cost_cache = find_event(capture_trace_list[1]["events"], "cost.record")["payload"]

        # The cache tokens are carried on the model.invoke usage AND priced.
        mi_cache = find_event(capture_trace_list[1]["events"], "model.invoke")["payload"]
        assert mi_cache["usage"].get("cache_creation_input_tokens") == 500
        assert mi_cache["usage"].get("cache_read_input_tokens") == 1000

        assert cost_base["cost_usd"] > 0
        assert cost_cache["cost_usd"] > cost_base["cost_usd"], (
            "non-zero prompt-cache tokens did not increase the priced cost.record "
            f"({cost_cache['cost_usd']} !> {cost_base['cost_usd']})"
        )
