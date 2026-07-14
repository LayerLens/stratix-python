"""Offline privacy + error + attestation + params-allowlist + cost floor for ollama.

Closes the W2 census ◑/gap cells that were previously proven only in the gated
live lane (or only via synthetic ``dict`` inputs), so a regression fails in plain
CI without any provider host / network:

* Redaction   — ``capture_content=False`` strips ``messages`` AND
                ``output_message`` (usage + parameters remain) with a ``True``
                vacuity control, plus a SENTINEL sweep over the serialized events
                (absent when off, present when on). Driven over the REAL
                ``OllamaProvider`` wrapping a REAL ollama ``ChatResponse`` object
                — the modern pydantic shape the live client actually returns,
                not the plain ``dict`` the unit suite feeds. Closes ``redaction``
                (was: gap — SENTINEL only lived in the live _scenarios).
* Error-paths — a REAL ``ollama.ResponseError`` (the shape of the shipped
                ``prodready_errors/ollama_badmodel_404.json`` and
                ``ollama_badjson_400.json`` fixtures) is fed through the
                instrumented ``chat`` and surfaces as ``agent.error`` with
                ``error_type`` == ``"ResponseError"`` (the real SDK class), the
                message verbatim, and ``latency_ms`` — wiring the orphaned error
                corpus THROUGH the adapter (was: partial — only a synthetic
                ``ConnectionError`` was exercised; the 404/400 corpus was wired
                only through the secret-leak ``sdk_surface`` invariant).
* Attestation — the captured trace's attestation chain verifies offline (one
                envelope per event) with a TAMPER control that breaks link 1.
                Closes ``attest`` (was: gap).
* Params      — the ``OllamaProvider`` capture-params allowlist is enforced
                end-to-end: allowlisted keys survive, an unknown kwarg does not,
                and the prompt-content keys (``messages`` / ``prompt``) never
                reach ``parameters``. Closes ``params`` (was: partial — only
                ``model``-present was asserted, no allowlist / prompt exclusion).
* Cost        — a real token shape produces a ``cost.record`` whose ``cost_usd``
                is honestly ``None`` (ollama runs local models: zero API cost,
                no PRICING row). Bites both regressions: a vanished
                ``cost.record`` AND a fabricated non-null price. Closes ``cost``
                (was: partial — only ``infra_cost_usd`` was asserted).

Everything runs offline over real ollama SDK objects; the only mock is the
ollama client method (the network boundary) and the upload client.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import Mock

from ollama import ChatResponse, ResponseError, GenerateResponse

from layerlens.instrument import trace
from layerlens.instrument._capture_config import CaptureConfig
from layerlens.instrument.adapters.providers.ollama import OllamaProvider

from ...conftest import find_event

SENTINEL = "LL-SENTINEL-7f3a9c2e"


def _chat_response(
    content: str = "pong",
    *,
    model: str = "llama3",
    prompt_tokens: int = 15,
    completion_tokens: int = 2,
    done_reason: str = "stop",
) -> ChatResponse:
    """A REAL modern ollama ``ChatResponse`` object (the pydantic shape the live
    client returns — LAY-3614), matching ``fixtures/recorded/ollama/default.json``."""
    return ChatResponse(
        model=model,
        created_at="2026-06-22T23:00:07.684943Z",
        done=True,
        done_reason=done_reason,
        total_duration=231132791,
        load_duration=99185250,
        prompt_eval_count=prompt_tokens,
        prompt_eval_duration=92657750,
        eval_count=completion_tokens,
        eval_duration=35099667,
        message={"role": "assistant", "content": content},
    )


def _generate_response(text: str = "generated", *, model: str = "llama3") -> GenerateResponse:
    return GenerateResponse(
        model=model,
        response=text,
        done=True,
        done_reason="stop",
        prompt_eval_count=8,
        eval_count=4,
    )


def _run(ollama_client: Any, mock_client: Any, config: CaptureConfig, *, prompt: str = "Hi") -> None:
    """Drive the REAL OllamaProvider over the real ollama response object."""
    provider = OllamaProvider()
    provider.connect(ollama_client)

    @trace(mock_client, capture_config=config)
    def my_agent() -> str:
        r = ollama_client.chat(
            model="llama3",
            messages=[{"role": "user", "content": prompt}],
        )
        return r["message"]["content"]

    my_agent()


# ---------------------------------------------------------------------------
# Redaction floor (offline, host-free)
# ---------------------------------------------------------------------------
class TestRedactionFloor:
    def test_content_stripped_when_capture_content_false(self, mock_client, capture_trace):
        client = Mock()
        client.chat = Mock(return_value=_chat_response("I am a local llama!"))
        _run(client, mock_client, CaptureConfig(capture_content=False))

        model_invoke = find_event(capture_trace["events"], "model.invoke")
        assert "messages" not in model_invoke["payload"]
        assert "output_message" not in model_invoke["payload"]
        # Usage + params still present (redaction removes CONTENT, not metadata).
        assert model_invoke["payload"]["usage"]["completion_tokens"] == 2
        assert model_invoke["payload"]["parameters"]["model"] == "llama3"

    def test_content_present_when_capture_content_true(self, mock_client, capture_trace):
        """Vacuity control: the redaction assertion above is only meaningful if
        the SAME path DOES carry content when capture is on."""
        client = Mock()
        client.chat = Mock(return_value=_chat_response("I am a local llama!"))
        _run(client, mock_client, CaptureConfig.full())

        model_invoke = find_event(capture_trace["events"], "model.invoke")
        assert model_invoke["payload"]["output_message"]["content"] == "I am a local llama!"
        assert model_invoke["payload"]["messages"][0]["content"] == "Hi"

    def test_sentinel_never_leaks_when_redacted(self, mock_client, capture_trace):
        client = Mock()
        client.chat = Mock(return_value=_chat_response(f"Secret is {SENTINEL}"))
        _run(client, mock_client, CaptureConfig(capture_content=False), prompt=f"Remember {SENTINEL}")

        blob = json.dumps(capture_trace["events"])
        assert SENTINEL not in blob, "content redaction leaked the SENTINEL into the stored trace"

    def test_sentinel_present_when_capture_on(self, mock_client, capture_trace):
        """Vacuity control for the sweep above."""
        client = Mock()
        client.chat = Mock(return_value=_chat_response(f"Secret is {SENTINEL}"))
        _run(client, mock_client, CaptureConfig.full(), prompt=f"Remember {SENTINEL}")
        assert SENTINEL in json.dumps(capture_trace["events"])


# ---------------------------------------------------------------------------
# Real error-shape floor (feeds the real ollama.ResponseError through the adapter)
# ---------------------------------------------------------------------------
class TestRealErrorShape:
    def test_real_response_error_404_emits_agent_error(self, mock_client, capture_trace):
        # Shape of prodready_errors/ollama_badmodel_404.json (raw SDK message).
        err = ResponseError("model 'definitely-not-a-real-model-xyz' not found", 404)
        client = Mock()
        client.chat = Mock(side_effect=err)
        provider = OllamaProvider()
        provider.connect(client)

        @trace(mock_client, capture_config=CaptureConfig.full())
        def my_agent() -> str:
            try:
                client.chat(model="definitely-not-a-real-model-xyz", messages=[])
            except ResponseError:
                pass
            return "handled"

        my_agent()
        error = find_event(capture_trace["events"], "agent.error")
        # The REAL SDK exception class name — not the synthetic ConnectionError.
        assert error["payload"]["error_type"] == "ResponseError"
        assert "404" in error["payload"]["error"]
        assert "not found" in error["payload"]["error"]
        assert "latency_ms" in error["payload"]

    def test_real_response_error_400_emits_agent_error(self, mock_client, capture_trace):
        # Shape of prodready_errors/ollama_badjson_400.json.
        err = ResponseError(
            "invalid character 'n' looking for beginning of object key string", 400
        )
        client = Mock()
        client.chat = Mock(side_effect=err)
        provider = OllamaProvider()
        provider.connect(client)

        @trace(mock_client, capture_config=CaptureConfig.full())
        def my_agent() -> str:
            try:
                client.chat(model="llama3", messages=[])
            except ResponseError:
                pass
            return "handled"

        my_agent()
        error = find_event(capture_trace["events"], "agent.error")
        assert error["payload"]["error_type"] == "ResponseError"
        assert "400" in error["payload"]["error"]
        assert "latency_ms" in error["payload"]


# ---------------------------------------------------------------------------
# Offline attestation-chain verification
# ---------------------------------------------------------------------------
class TestAttestationOffline:
    def _envelopes(self, capture_trace):
        from layerlens.attestation._envelope import HashScope, AttestationEnvelope

        raw = (capture_trace["attestation"].get("chain") or {}).get("events") or []
        return [
            AttestationEnvelope(
                hash=e["hash"], scope=HashScope(e["scope"]), previous_hash=e.get("previous_hash")
            )
            for e in raw
        ]

    def test_attestation_chain_verifies_offline(self, mock_client, capture_trace):
        from layerlens.attestation._verify import verify_chain

        client = Mock()
        client.chat = Mock(return_value=_chat_response())
        _run(client, mock_client, CaptureConfig.full())

        events = capture_trace["events"]
        envelopes = self._envelopes(capture_trace)
        assert envelopes, "no attestation envelopes captured"
        assert len(envelopes) == len(events)
        result = verify_chain(envelopes)
        assert result.valid, f"attestation chain invalid: {result.error}"

    def test_tampered_chain_is_rejected(self, mock_client, capture_trace):
        """TAMPER control: verify_chain must FAIL when link 1's hash is broken —
        otherwise the pass above proves nothing."""
        from layerlens.attestation._verify import verify_chain
        from layerlens.attestation._envelope import AttestationEnvelope

        client = Mock()
        client.chat = Mock(return_value=_chat_response())
        _run(client, mock_client, CaptureConfig.full())

        envelopes = self._envelopes(capture_trace)
        assert len(envelopes) >= 2, "need >=2 links to tamper a non-terminal one"
        envelopes[0] = AttestationEnvelope(
            hash="deadbeef" * 8, scope=envelopes[0].scope, previous_hash=envelopes[0].previous_hash
        )
        result = verify_chain(envelopes)
        assert not result.valid, "tampered attestation chain verified as valid"


# ---------------------------------------------------------------------------
# Params allowlist enforced end-to-end through the real adapter path
# ---------------------------------------------------------------------------
class TestParamsAllowlist:
    def test_allowlisted_params_kept_unknown_dropped(self, mock_client, capture_trace):
        client = Mock()
        client.chat = Mock(return_value=_chat_response())
        provider = OllamaProvider()
        provider.connect(client)

        @trace(mock_client, capture_config=CaptureConfig.full())
        def my_agent() -> str:
            client.chat(
                model="llama3",
                options={"temperature": 0.7, "num_ctx": 4096},
                keep_alive="5m",
                format="json",
                messages=[{"role": "user", "content": "Hi"}],
                # non-allowlisted / unknown kwargs — must NOT reach parameters.
                user="tenant-42",
                metadata={"secret": SENTINEL},
            )
            return "done"

        my_agent()
        params = find_event(capture_trace["events"], "model.invoke")["payload"]["parameters"]
        # Allowlisted keys survive with their passed values.
        assert params["model"] == "llama3"
        assert params["options"] == {"temperature": 0.7, "num_ctx": 4096}
        assert params["keep_alive"] == "5m"
        assert params["format"] == "json"
        # The allowlist is a positive filter: unknown kwargs are excluded, and
        # `messages` is prompt CONTENT captured separately (never as a parameter).
        assert "user" not in params
        assert "metadata" not in params
        assert "messages" not in params
        # And the unknown kwarg's SENTINEL value never leaks via the params path.
        assert SENTINEL not in json.dumps(params)

    def test_generate_prompt_never_reaches_parameters(self, mock_client, capture_trace):
        """The ``prompt`` content key is excluded from ``parameters`` (LAY-3567 B1)
        exactly like ``messages`` — otherwise capture_content=False would leak it."""
        client = Mock()
        client.generate = Mock(return_value=_generate_response())
        provider = OllamaProvider()
        provider.connect(client)

        @trace(mock_client, capture_config=CaptureConfig.full())
        def my_agent() -> str:
            client.generate(model="llama3", prompt=f"remember {SENTINEL}")
            return "done"

        my_agent()
        params = find_event(capture_trace["events"], "model.invoke")["payload"]["parameters"]
        assert "prompt" not in params
        assert SENTINEL not in json.dumps(params)


# ---------------------------------------------------------------------------
# Cost floor — ollama is local: cost.record is emitted with an HONEST null price
# ---------------------------------------------------------------------------
class TestCostRecord:
    def test_cost_record_emitted_with_null_cost(self, mock_client, capture_trace):
        client = Mock()
        client.chat = Mock(return_value=_chat_response())
        _run(client, mock_client, CaptureConfig.full())

        cost = find_event(capture_trace["events"], "cost.record")
        # ollama has no PRICING row and runs local models: API cost is honestly $0.
        # cost_usd is None (NOT fabricated, NOT dropped). This bites both ways:
        # a vanished cost.record fails find_event; a fabricated price fails here.
        assert cost["payload"]["cost_usd"] is None
        assert cost["payload"]["provider"] == "ollama"
        # The token shape that priced the (free) call is still recorded honestly.
        assert cost["payload"]["total_tokens"] == 17
