"""Offline privacy + error + attestation + params-allowlist floor for the openai provider.

Closes the W1 census ◑ cells that were previously proven only in the gated live
lanes (or only via synthetic inputs), so a regression fails in plain CI without
any provider credentials:

* Redaction   — ``capture_content=False`` strips ``messages`` AND
                ``output_message`` (usage + parameters remain) with a ``True``
                vacuity control, plus a SENTINEL sweep over the serialized events
                (absent when off, present when on). Driven over the REAL
                ``OpenAIProvider`` wrapping a real openai ``ChatCompletion``.
* Error-paths — a REAL ``openai.AuthenticationError`` (the shape of the shipped
                ``prodready_errors/openai_badkey_401.json`` fixture) and a REAL
                ``openai.BadRequestError`` (``openai_malformed_400.json`` shape)
                are fed through the instrumented ``chat.completions.create`` and
                surface as ``agent.error`` with ``error_type`` == the real SDK
                class name — not the synthetic ``RuntimeError`` the existing
                suite uses.
* Attestation — the captured trace's attestation chain verifies offline
                (mirrors the live harness ``_assert_attestation``); one envelope
                per event.
* Params      — the ``OpenAIProvider`` capture-params allowlist is enforced
                end-to-end: allowed keys that were passed appear in the captured
                ``model.invoke.parameters`` and a non-allowlisted kwarg does not.
"""

from __future__ import annotations

import json
from unittest.mock import Mock

import httpx

from openai import BadRequestError, AuthenticationError
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from layerlens.instrument import trace
from openai.types.chat.chat_completion import Choice
from layerlens.instrument._capture_config import CaptureConfig
from layerlens.instrument.adapters.providers.openai import OpenAIProvider

from .conftest import make_openai_response
from ...conftest import find_event

SENTINEL = "LL-SENTINEL-7f3a9c2e"


def _response(content: str = "Hello!", *, model: str = "gpt-4") -> ChatCompletion:
    """A real OpenAI ChatCompletion with the given assistant content."""
    return ChatCompletion(
        id="chatcmpl-test",
        model=model,
        object="chat.completion",
        created=1700000000,
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(role="assistant", content=content),
            )
        ],
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def _run(openai_client, mock_client, config, *, prompt="Hi"):
    """Drive the REAL OpenAIProvider over the real openai response object."""
    provider = OpenAIProvider()
    provider.connect(openai_client)

    @trace(mock_client, capture_config=config)
    def my_agent():
        r = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
        )
        return r.choices[0].message.content

    return my_agent()


# ---------------------------------------------------------------------------
# Redaction floor (offline, credential-free)
# ---------------------------------------------------------------------------
class TestRedactionFloor:
    def test_content_stripped_when_capture_content_false(self, mock_client, capture_trace):
        client = Mock()
        client.chat.completions.create = Mock(return_value=_response("I am GPT!"))
        _run(client, mock_client, CaptureConfig(capture_content=False))

        model_invoke = find_event(capture_trace["events"], "model.invoke")
        assert "messages" not in model_invoke["payload"]
        assert "output_message" not in model_invoke["payload"]
        # Usage + params still present (redaction removes CONTENT, not metadata).
        assert model_invoke["payload"]["usage"]["completion_tokens"] == 5
        assert model_invoke["payload"]["parameters"]["model"] == "gpt-4"

    def test_content_present_when_capture_content_true(self, mock_client, capture_trace):
        """Vacuity control: the redaction assertion above is only meaningful if
        the SAME path DOES carry content when capture is on."""
        client = Mock()
        client.chat.completions.create = Mock(return_value=_response("I am GPT!"))
        _run(client, mock_client, CaptureConfig.full())

        model_invoke = find_event(capture_trace["events"], "model.invoke")
        assert model_invoke["payload"]["output_message"]["content"] == "I am GPT!"
        assert model_invoke["payload"]["messages"][0]["content"] == "Hi"

    def test_sentinel_never_leaks_when_redacted(self, mock_client, capture_trace):
        client = Mock()
        client.chat.completions.create = Mock(return_value=_response(f"Secret is {SENTINEL}"))
        _run(client, mock_client, CaptureConfig(capture_content=False), prompt=f"Remember {SENTINEL}")

        blob = json.dumps(capture_trace["events"])
        assert SENTINEL not in blob, "content redaction leaked the SENTINEL into the stored trace"

    def test_sentinel_present_when_capture_on(self, mock_client, capture_trace):
        """Vacuity control for the sweep above."""
        client = Mock()
        client.chat.completions.create = Mock(return_value=_response(f"Secret is {SENTINEL}"))
        _run(client, mock_client, CaptureConfig.full(), prompt=f"Remember {SENTINEL}")
        assert SENTINEL in json.dumps(capture_trace["events"])


# ---------------------------------------------------------------------------
# Real error-shape floor (feeds the real 4xx SDK exceptions through the adapter)
# ---------------------------------------------------------------------------
class TestRealErrorShape:
    def test_real_authentication_error_emits_agent_error(self, mock_client, capture_trace):
        # Shape of prodready_errors/openai_badkey_401.json (raw SDK message).
        raw = (
            "Error code: 401 - {'error': {'message': 'Incorrect API key provided: "
            "sk-proj-****nary.', 'type': 'invalid_request_error', "
            "'code': 'invalid_api_key', 'param': None}, 'status': 401}"
        )
        response = httpx.Response(401, request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"))
        err = AuthenticationError(raw, response=response, body=None)

        client = Mock()
        client.chat.completions.create = Mock(side_effect=err)
        provider = OpenAIProvider()
        provider.connect(client)

        @trace(mock_client, capture_config=CaptureConfig.full())
        def my_agent():
            try:
                client.chat.completions.create(model="gpt-4", messages=[])
            except AuthenticationError:
                pass
            return "handled"

        my_agent()
        error = find_event(capture_trace["events"], "agent.error")
        # The REAL SDK exception class name — not the synthetic RuntimeError.
        assert error["payload"]["error_type"] == "AuthenticationError"
        assert "401" in error["payload"]["error"]
        assert "latency_ms" in error["payload"]

    def test_real_badrequest_error_emits_agent_error(self, mock_client, capture_trace):
        # Shape of prodready_errors/openai_malformed_400.json.
        raw = (
            "Error code: 400 - {'error': {'message': \"Missing required parameter: "
            "'messages'.\", 'type': 'invalid_request_error', 'param': 'messages', "
            "'code': 'missing_required_parameter'}}"
        )
        response = httpx.Response(400, request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"))
        err = BadRequestError(raw, response=response, body=None)

        client = Mock()
        client.chat.completions.create = Mock(side_effect=err)
        provider = OpenAIProvider()
        provider.connect(client)

        @trace(mock_client, capture_config=CaptureConfig.full())
        def my_agent():
            try:
                client.chat.completions.create(model="gpt-4")
            except BadRequestError:
                pass
            return "handled"

        my_agent()
        error = find_event(capture_trace["events"], "agent.error")
        assert error["payload"]["error_type"] == "BadRequestError"
        assert "400" in error["payload"]["error"]
        assert "latency_ms" in error["payload"]


# ---------------------------------------------------------------------------
# Offline attestation-chain verification
# ---------------------------------------------------------------------------
class TestAttestationOffline:
    def test_attestation_chain_verifies_offline(self, mock_client, capture_trace):
        from layerlens.attestation._verify import verify_chain
        from layerlens.attestation._envelope import HashScope, AttestationEnvelope

        client = Mock()
        client.chat.completions.create = Mock(return_value=make_openai_response())
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
# Params allowlist enforced end-to-end through the real adapter path
# ---------------------------------------------------------------------------
class TestParamsAllowlist:
    def test_allowlisted_params_kept_unknown_dropped(self, mock_client, capture_trace):
        client = Mock()
        client.chat.completions.create = Mock(return_value=make_openai_response())
        provider = OpenAIProvider()
        provider.connect(client)

        @trace(mock_client, capture_config=CaptureConfig.full())
        def my_agent():
            client.chat.completions.create(
                model="gpt-4",
                temperature=0.7,
                max_tokens=256,
                top_p=0.9,
                messages=[{"role": "user", "content": "Hi"}],
                # non-allowlisted / unknown kwargs — must NOT reach parameters.
                user="tenant-42",
                metadata={"secret": SENTINEL},
            )
            return "done"

        my_agent()
        params = find_event(capture_trace["events"], "model.invoke")["payload"]["parameters"]
        # Several of the 12 allowlisted keys survive with their passed values.
        assert params["model"] == "gpt-4"
        assert params["temperature"] == 0.7
        assert params["max_tokens"] == 256
        assert params["top_p"] == 0.9
        # The allowlist is a positive filter: unknown kwargs are excluded, and
        # `messages` is captured separately (never as a parameter).
        assert "user" not in params
        assert "metadata" not in params
        assert "messages" not in params
        # And the unknown kwarg's SENTINEL value never leaks via the params path.
        assert SENTINEL not in json.dumps(params)
