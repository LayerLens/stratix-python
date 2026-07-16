"""Offline privacy + error + attestation + cost/params floor for the litellm provider.

Closes the W2 census ◑/gap cells that ``test_litellm.py`` proves only in the
gated live lanes (``redaction``/``attest``) or only via a *synthetic*
``RuntimeError`` (``error``), so a regression fails in plain CI with no
credentials and no network:

* Redaction   — ``capture_content=False`` strips ``messages`` AND
                ``output_message`` (usage + parameters remain) with a ``True``
                vacuity control, plus a SENTINEL sweep over the serialized events
                (absent when off, present when on). Driven over the REAL
                ``LiteLLMProvider`` wrapping ``litellm.completion``. (census
                ``redaction: gap``.)
* Error-paths — a REAL ``litellm.exceptions.RateLimitError`` and a REAL
                ``litellm.exceptions.APIError`` (not the synthetic
                ``RuntimeError`` the existing suite feeds) are raised through the
                instrumented ``litellm.completion`` and surface as
                ``agent.error`` with ``error_type`` == the real litellm SDK class
                name and the real message flowing through verbatim. (census
                ``error: partial``.)
* Attestation — the captured trace's attestation chain verifies offline, one
                envelope per event, with a TAMPER control proving the check is
                not vacuous. (census ``attest: gap``.)
* Cost/Params — regression pins that ``cost.record`` carries a real ``cost_usd``
                priced from the emitted usage, and that the capture-params
                allowlist keeps allowlisted keys / drops unknown kwargs (census
                marks both ``solid`` — kept green here so the floor is a single
                gate).

The litellm module is the only mock (its module object is injected into
``sys.modules`` via the proven ``test_litellm._install_mock_litellm`` seam, then
the REAL ``LiteLLMProvider`` wraps it); every response object, extractor, emit
path, attestation chain and the real litellm exception classes are real.
"""

from __future__ import annotations

import json
from unittest.mock import Mock

from litellm.exceptions import APIError, RateLimitError

from layerlens.instrument import trace
from layerlens.instrument._capture_config import CaptureConfig
from layerlens.instrument.adapters.providers.litellm import instrument_litellm

from .conftest import make_openai_response
from ...conftest import find_event, find_events
from .test_litellm import (
    _expected_cost,
    _remove_mock_litellm,
    _install_mock_litellm,
)

SENTINEL = "LL-SENTINEL-7f3a9c2e"


def _drive(
    mock_client,
    config,
    *,
    response=None,
    side_effect=None,
    prompt: str = "Hi",
    model: str = "gpt-4",
    extra_kwargs=None,
):
    """Drive the REAL ``LiteLLMProvider`` over an injected litellm module.

    Installs a fake ``litellm`` (carrying a real OpenAI-shaped response, or a
    ``side_effect`` exception), wraps it with the real ``instrument_litellm()``,
    and runs a ``@trace``-decorated call that flushes into ``mock_client``.
    """
    mod = _install_mock_litellm(response if response is not None else make_openai_response(model=model))
    if side_effect is not None:
        mod.completion = Mock(side_effect=side_effect)
    kw = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    if extra_kwargs:
        kw.update(extra_kwargs)
    try:
        instrument_litellm()

        @trace(mock_client, capture_config=config)
        def my_agent():
            import litellm

            try:
                r = litellm.completion(**kw)
            except Exception:
                return "handled"
            return r.choices[0].message.content

        return my_agent()
    finally:
        _remove_mock_litellm()


# ---------------------------------------------------------------------------
# Redaction floor (offline, credential-free) — census ``redaction: gap``
# ---------------------------------------------------------------------------
class TestRedactionFloor:
    def test_content_stripped_when_capture_content_false(self, mock_client, capture_trace):
        _drive(
            mock_client,
            CaptureConfig(capture_content=False),
            response=make_openai_response("I am the model!"),
            prompt="my secret prompt",
        )

        model_invoke = find_event(capture_trace["events"], "model.invoke")
        assert "messages" not in model_invoke["payload"]
        assert "output_message" not in model_invoke["payload"]
        # Redaction removes CONTENT, not metadata: usage + params survive.
        assert model_invoke["payload"]["usage"]["completion_tokens"] == 5
        assert model_invoke["payload"]["parameters"]["model"] == "gpt-4"

    def test_content_present_when_capture_content_true(self, mock_client, capture_trace):
        """Vacuity control: the SAME path DOES carry content when capture is on."""
        _drive(
            mock_client,
            CaptureConfig.full(),
            response=make_openai_response("I am the model!"),
            prompt="my secret prompt",
        )

        model_invoke = find_event(capture_trace["events"], "model.invoke")
        assert model_invoke["payload"]["output_message"]["content"] == "I am the model!"
        assert model_invoke["payload"]["messages"][0]["content"] == "my secret prompt"

    def test_sentinel_never_leaks_when_redacted(self, mock_client, capture_trace):
        _drive(
            mock_client,
            CaptureConfig(capture_content=False),
            response=make_openai_response(f"Secret is {SENTINEL}"),
            prompt=f"Remember {SENTINEL}",
        )
        blob = json.dumps(capture_trace["events"])
        assert SENTINEL not in blob, "content redaction leaked the SENTINEL into the stored trace"

    def test_sentinel_present_when_capture_on(self, mock_client, capture_trace):
        """Vacuity control for the sweep above."""
        _drive(
            mock_client,
            CaptureConfig.full(),
            response=make_openai_response(f"Secret is {SENTINEL}"),
            prompt=f"Remember {SENTINEL}",
        )
        assert SENTINEL in json.dumps(capture_trace["events"])


# ---------------------------------------------------------------------------
# Real error-shape floor — census ``error: partial`` (was synthetic RuntimeError)
# ---------------------------------------------------------------------------
class TestRealErrorShape:
    def test_real_ratelimit_error_emits_agent_error(self, mock_client, capture_trace):
        # A genuine litellm SDK exception — the shape a real litellm.completion
        # raises on a 429 — NOT the synthetic RuntimeError the existing suite feeds.
        err = RateLimitError(
            message="Error code: 429 - Rate limit reached for gpt-4",
            llm_provider="openai",
            model="gpt-4",
        )
        assert type(err).__name__ == "RateLimitError"
        assert type(err).__module__ == "litellm.exceptions"
        real_message = str(err)

        _drive(mock_client, CaptureConfig.full(), side_effect=err)

        error = find_event(capture_trace["events"], "agent.error")
        # The REAL litellm SDK class name — not "RuntimeError".
        assert error["payload"]["error_type"] == "RateLimitError"
        # The real exception message flows through verbatim (bite: dropped/mangled
        # error text fails here). Tied to the real 429 status of the class.
        assert error["payload"]["error"] == real_message
        assert "429" in error["payload"]["error"]
        assert "latency_ms" in error["payload"]

    def test_real_apierror_emits_agent_error(self, mock_client, capture_trace):
        err = APIError(
            status_code=500,
            message="Error code: 500 - internal server error",
            llm_provider="openai",
            model="gpt-4",
        )
        assert type(err).__name__ == "APIError"
        assert type(err).__module__ == "litellm.exceptions"
        real_message = str(err)

        _drive(mock_client, CaptureConfig.full(), side_effect=err)

        error = find_event(capture_trace["events"], "agent.error")
        assert error["payload"]["error_type"] == "APIError"
        assert error["payload"]["error"] == real_message
        assert "500" in error["payload"]["error"]
        assert "latency_ms" in error["payload"]


# ---------------------------------------------------------------------------
# Offline attestation-chain verification — census ``attest: gap``
# ---------------------------------------------------------------------------
class TestAttestationOffline:
    def test_attestation_chain_verifies_offline(self, mock_client, capture_trace):
        from layerlens.attestation._verify import verify_chain
        from layerlens.attestation._envelope import HashScope, AttestationEnvelope

        _drive(mock_client, CaptureConfig.full())

        events = capture_trace["events"]
        assert events, "the litellm call must flush a non-empty trace"
        chain = (capture_trace["attestation"] or {}).get("chain") or {}
        raw = chain.get("events") or []
        envelopes = [
            AttestationEnvelope(hash=e["hash"], scope=HashScope(e["scope"]), previous_hash=e.get("previous_hash"))
            for e in raw
        ]
        assert envelopes, "no attestation envelopes captured"
        assert len(envelopes) == len(events)
        assert (capture_trace["attestation"] or {}).get("root_hash") is not None

        result = verify_chain(envelopes)
        assert result.valid, f"attestation chain invalid: {result.error}"

        # Vacuity control: verify_chain must REJECT a broken interior link, so the
        # pass above is not trivially true.
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
# Cost floor — census ``cost: solid`` (kept green so the floor is one gate)
# ---------------------------------------------------------------------------
class TestCostFloor:
    def test_cost_usd_present_on_real_token_shape(self, mock_client, capture_trace):
        _drive(mock_client, CaptureConfig.full())

        events = capture_trace["events"]
        model_invoke = find_event(events, "model.invoke")
        costs = find_events(events, "cost.record")
        assert len(costs) == 1, f"expected exactly one cost.record, got {len(costs)}"
        cost = costs[0]["payload"]

        expected = _expected_cost("gpt-4", model_invoke["payload"]["usage"])
        assert expected is not None and expected > 0, "gpt-4 must price to a positive cost"
        assert cost["cost_usd"] == expected, "cost_usd missing or miscomputed on the provider cost.record"
        # LAY-3455: cost is attributed to the routed underlying provider.
        assert cost["provider"] == "openai"
        assert cost["total_tokens"] == 15


# ---------------------------------------------------------------------------
# Params allowlist enforced end-to-end — census ``params: solid`` (kept green)
# ---------------------------------------------------------------------------
class TestParamsAllowlist:
    def test_allowlisted_params_kept_unknown_dropped(self, mock_client, capture_trace):
        _drive(
            mock_client,
            CaptureConfig.full(),
            extra_kwargs={
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 256,
                # non-allowlisted / unknown kwargs — must NOT reach parameters.
                "api_key": "sk-secret-key",
                "user": "tenant-42",
                "metadata": {"secret": SENTINEL},
            },
        )

        params = find_event(capture_trace["events"], "model.invoke")["payload"]["parameters"]
        # Allowlisted keys survive with their passed values.
        assert params["model"] == "gpt-4"
        assert params["temperature"] == 0.7
        assert params["top_p"] == 0.9
        assert params["max_tokens"] == 256
        # The allowlist is a positive filter: unknown kwargs are excluded, and
        # `messages` is captured separately (never as a parameter).
        assert "api_key" not in params
        assert "user" not in params
        assert "metadata" not in params
        assert "messages" not in params
        # The unknown kwarg's SENTINEL value never leaks via the params path.
        assert SENTINEL not in json.dumps(params)
