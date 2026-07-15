"""Offline privacy + error + attestation + params/cost floor for the openrouter provider.

OpenRouter ships SEALED (no ``sk-or-…`` key exists, so the live lane is
deferred). This floor is therefore the ONLY proof for the four contract classes,
and it must hold with no credentials and no network:

* Redaction   — ``capture_content=False`` strips ``messages`` AND
                ``output_message`` (usage/tokens/params remain) with a ``True``
                vacuity control, plus a SENTINEL sweep over the whole serialized
                trace. Driven over the REAL ``OpenRouterProvider`` wrapping a
                real ``openai`` client parsing a real OpenRouter body.
* Error-paths — REAL ``openai.AuthenticationError`` / ``RateLimitError`` shapes
                (a 401 body echoing the ``sk-or-…`` key, and a 429) surface as
                ``agent.error`` with the real SDK class name — and the key is
                scrubbed out of the stored error text.
* Attestation — the captured trace's attestation chain verifies offline, plus a
                TAMPER control that must fail.
* Params/cost — the capture-params allowlist is enforced end-to-end, and the
                PROVIDER-COST-OR-NOTHING rule is proven: no reported charge =>
                no cost.record, never a catalog price and never a 0.0.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import httpx
import pytest

from openai import OpenAI, RateLimitError, AuthenticationError
from layerlens.instrument import trace
from layerlens.instrument._capture_config import CaptureConfig
from layerlens.instrument.adapters.providers.openrouter import (
    OPENROUTER_BASE_URL,
    OpenRouterProvider,
)

from ...conftest import find_event, find_events

SENTINEL = "LL-SENTINEL-0f41d7bc"
_API_KEY = "sk-or-v1-0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
_ROUTED_SLUG = "anthropic/claude-opus-4.8"


def _chat_json(content: str = "Hello!", *, cost: Any = 0.00042) -> Dict[str, Any]:
    usage: Dict[str, Any] = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    if cost is not None:
        usage["cost"] = cost
    return {
        "id": "gen-1770000000-floor",
        "object": "chat.completion",
        "created": 1770000000,
        "model": _ROUTED_SLUG,
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
        "usage": usage,
    }


def _make_client(response_json: Optional[Dict[str, Any]] = None) -> OpenAI:
    payload = response_json if response_json is not None else _chat_json()

    def handler(request: httpx.Request) -> httpx.Response:  # noqa: ARG001
        return httpx.Response(200, json=payload)

    return OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=_API_KEY,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def _erroring_client(exc: Exception) -> OpenAI:
    def handler(request: httpx.Request) -> httpx.Response:  # noqa: ARG001
        raise exc

    return OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=_API_KEY,
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def _run(client: OpenAI, mock_client: Any, config: CaptureConfig, *, prompt: str = "Hi") -> Any:
    """Drive the REAL OpenRouterProvider over the real openai client."""
    OpenRouterProvider().connect(client)

    @trace(mock_client, capture_config=config)
    def my_agent():
        r = client.chat.completions.create(
            model=_ROUTED_SLUG,
            messages=[{"role": "user", "content": prompt}],
            extra_body={"usage": {"include": True}},
        )
        return r.choices[0].message.content

    return my_agent()


# ---------------------------------------------------------------------------
# Redaction floor (offline, credential-free)
# ---------------------------------------------------------------------------
class TestRedactionFloor:
    def test_content_stripped_when_capture_content_false(self, mock_client, capture_trace):
        _run(_make_client(_chat_json("I am Claude via OpenRouter!")), mock_client, CaptureConfig(capture_content=False))

        mi = find_event(capture_trace["events"], "model.invoke")["payload"]
        assert "messages" not in mi
        assert "output_message" not in mi
        # Structure/topology/metadata MUST survive redaction.
        assert mi["usage"]["completion_tokens"] == 5
        assert mi["total_tokens"] == 15
        assert mi["model"] == _ROUTED_SLUG
        assert mi["framework"] == "openrouter"
        assert mi["parameters"]["model"] == _ROUTED_SLUG
        # The billed charge is metadata, not content — it must still be recorded.
        cost = find_event(capture_trace["events"], "cost.record")["payload"]
        assert cost["cost_usd"] == pytest.approx(0.00042)
        assert cost["cost_source"] == "provider"

    def test_content_present_when_capture_content_true(self, mock_client, capture_trace):
        """Vacuity control: the assertion above is only meaningful if the SAME
        path DOES carry content when capture is on."""
        _run(_make_client(_chat_json("I am Claude via OpenRouter!")), mock_client, CaptureConfig.full())

        mi = find_event(capture_trace["events"], "model.invoke")["payload"]
        assert mi["output_message"]["content"] == "I am Claude via OpenRouter!"
        assert mi["messages"][0]["content"] == "Hi"

    def test_sentinel_never_leaks_when_redacted(self, mock_client, capture_trace):
        _run(
            _make_client(_chat_json(f"Secret is {SENTINEL}")),
            mock_client,
            CaptureConfig(capture_content=False),
            prompt=f"Remember {SENTINEL}",
        )
        blob = json.dumps(capture_trace["events"])
        assert SENTINEL not in blob, "content redaction leaked the SENTINEL into the stored trace"

    def test_sentinel_present_when_capture_on(self, mock_client, capture_trace):
        """Vacuity control for the sweep above."""
        _run(
            _make_client(_chat_json(f"Secret is {SENTINEL}")),
            mock_client,
            CaptureConfig.full(),
            prompt=f"Remember {SENTINEL}",
        )
        assert SENTINEL in json.dumps(capture_trace["events"])


# ---------------------------------------------------------------------------
# Real error-shape floor (real SDK exceptions, not synthetic RuntimeError)
# ---------------------------------------------------------------------------
class TestRealErrorShape:
    def test_real_authentication_error_scrubs_the_openrouter_key(self, mock_client, capture_trace):
        # OpenRouter 401 bodies echo the offending key back verbatim.
        raw = (
            "Error code: 401 - {'error': {'message': 'No auth credentials found: "
            f"{_API_KEY}', 'code': 401}}}}"
        )
        response = httpx.Response(401, request=httpx.Request("POST", f"{OPENROUTER_BASE_URL}/chat/completions"))
        err = AuthenticationError(raw, response=response, body=None)
        client = _erroring_client(err)
        OpenRouterProvider().connect(client)

        @trace(mock_client, capture_config=CaptureConfig.full())
        def my_agent():
            try:
                client.chat.completions.create(model=_ROUTED_SLUG, messages=[])
            except AuthenticationError:
                pass
            return "handled"

        my_agent()
        error = find_event(capture_trace["events"], "agent.error")["payload"]
        assert error["error_type"] == "AuthenticationError"
        assert error["name"] == "openrouter.chat.completions.create"
        assert "401" in error["error"]
        assert "latency_ms" in error["error"] or "latency_ms" in error
        # The key must never reach the stored trace.
        assert _API_KEY not in json.dumps(capture_trace["events"])
        # A failed call is not a priced call.
        assert find_events(capture_trace["events"], "cost.record") == []
        assert find_events(capture_trace["events"], "model.invoke") == []

    def test_real_rate_limit_error_is_an_error_not_a_policy_violation(self, mock_client, capture_trace):
        """A rate limit involves no policy — agent.error, never policy.violation."""
        raw = "Error code: 429 - {'error': {'message': 'Rate limit exceeded: free-models-per-day', 'code': 429}}"
        response = httpx.Response(429, request=httpx.Request("POST", f"{OPENROUTER_BASE_URL}/chat/completions"))
        err = RateLimitError(raw, response=response, body=None)
        client = _erroring_client(err)
        OpenRouterProvider().connect(client)

        @trace(mock_client, capture_config=CaptureConfig.full())
        def my_agent():
            try:
                client.chat.completions.create(model=_ROUTED_SLUG, messages=[])
            except RateLimitError:
                pass
            return "handled"

        my_agent()
        events = capture_trace["events"]
        error = find_event(events, "agent.error")["payload"]
        assert error["error_type"] == "RateLimitError"
        assert "429" in error["error"]
        assert find_events(events, "policy.violation") == []


# ---------------------------------------------------------------------------
# Offline attestation-chain verification (+ tamper control)
# ---------------------------------------------------------------------------
class TestAttestationOffline:
    def test_attestation_chain_verifies_offline(self, mock_client, capture_trace):
        from layerlens.attestation._verify import verify_chain
        from layerlens.attestation._envelope import HashScope, AttestationEnvelope

        _run(_make_client(), mock_client, CaptureConfig.full())

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

    def test_tampered_chain_fails_verification(self, mock_client, capture_trace):
        """Tamper control: the verifier above is only meaningful if a broken
        chain actually fails."""
        from layerlens.attestation._verify import verify_chain
        from layerlens.attestation._envelope import HashScope, AttestationEnvelope

        _run(_make_client(), mock_client, CaptureConfig.full())

        raw = (capture_trace["attestation"].get("chain") or {}).get("events") or []
        assert len(raw) >= 2, "need at least two envelopes to break a link"
        envelopes = [
            AttestationEnvelope(hash=e["hash"], scope=HashScope(e["scope"]), previous_hash=e.get("previous_hash"))
            for e in raw
        ]
        tampered = list(envelopes)
        tampered[1] = AttestationEnvelope(
            hash="0" * 64,
            scope=envelopes[1].scope,
            previous_hash=envelopes[1].previous_hash,
        )
        assert not verify_chain(tampered).valid, "a forged event hash verified — the chain proves nothing"


# ---------------------------------------------------------------------------
# Params allowlist + the no-fabricated-cost rule
# ---------------------------------------------------------------------------
class TestParamsAllowlist:
    def test_allowlisted_params_kept_unknown_dropped(self, mock_client, capture_trace):
        client = _make_client()
        OpenRouterProvider().connect(client)

        @trace(mock_client, capture_config=CaptureConfig.full())
        def my_agent():
            client.chat.completions.create(
                model=_ROUTED_SLUG,
                temperature=0.7,
                max_tokens=256,
                top_p=0.9,
                seed=11,
                messages=[{"role": "user", "content": "Hi"}],
                # non-allowlisted kwargs — must NOT reach parameters.
                user="tenant-42",
                extra_body={"usage": {"include": True}, "transforms": [SENTINEL]},
            )
            return "done"

        my_agent()
        params = find_event(capture_trace["events"], "model.invoke")["payload"]["parameters"]
        assert params["model"] == _ROUTED_SLUG
        assert params["temperature"] == 0.7
        assert params["max_tokens"] == 256
        assert params["top_p"] == 0.9
        assert params["seed"] == 11
        assert "user" not in params
        assert "extra_body" not in params
        assert SENTINEL not in json.dumps(params)


class TestCostFloor:
    """PROVIDER COST OR NOTHING — the rule this adapter exists to enforce."""

    def test_provider_reported_cost_is_recorded_with_its_provenance(self, mock_client, capture_trace):
        _run(_make_client(_chat_json(cost=0.00042)), mock_client, CaptureConfig.full())
        cost = find_event(capture_trace["events"], "cost.record")["payload"]
        assert cost["cost_usd"] == pytest.approx(0.00042)
        assert cost["cost_source"] == "provider"

    def test_no_reported_cost_emits_no_cost_record_and_never_zero(self, mock_client, capture_trace):
        _run(_make_client(_chat_json(cost=None)), mock_client, CaptureConfig.full())
        events = capture_trace["events"]
        assert find_events(events, "cost.record") == [], "invented a cost the gateway never reported"
        # And no 0.0 leaked onto any other event either.
        for event in events:
            assert event["payload"].get("cost_usd") != 0.0

    def test_the_bundled_catalog_cannot_price_any_openrouter_slug(self):
        """Pins the premise of the whole design (and would catch a future
        pricing.py change that starts resolving slugs behind our back)."""
        from layerlens.instrument.adapters.providers.pricing import PRICING, calculate_cost
        from layerlens.instrument.adapters.providers.token_usage import NormalizedTokenUsage

        usage = NormalizedTokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        for slug in (
            "anthropic/claude-opus-4.8",
            "openai/gpt-4o",
            "meta-llama/llama-3.1-70b-instruct",
            "openrouter/auto",
        ):
            assert calculate_cost(slug, usage, PRICING) is None, f"{slug} unexpectedly priced"

    def test_provider_cost_only_is_declared(self):
        """The class-level switch that makes the gateway the sole cost authority."""
        assert OpenRouterProvider.provider_cost_only is True

    @pytest.mark.parametrize(
        "usage_obj",
        [None, object()],
        ids=["no-usage", "usage-without-cost"],
    )
    def test_extract_provider_cost_is_none_safe(self, usage_obj):
        class _Response:
            usage = usage_obj

        assert OpenRouterProvider.extract_provider_cost(_Response()) is None

    @pytest.mark.parametrize("bad", ["not-a-number", "", [], {}, object()], ids=repr)
    def test_extract_provider_cost_rejects_a_malformed_value_at_the_source(self, bad):
        """Pinned at the extractor, not just end-to-end.

        End-to-end, ``emit_llm_events`` wraps this probe in a blanket except, so an
        unguarded ``float()`` would raise and still yield "no cost.record" — the
        right outcome for the wrong reason. The extractor must make that decision
        itself, or the guard is untested and the next caller inherits a raiser.
        """

        class _Usage:
            cost = bad

        class _Response:
            usage = _Usage()

        assert OpenRouterProvider.extract_provider_cost(_Response()) is None

    @pytest.mark.parametrize(
        "bad",
        [
            "nan",
            "inf",
            "-inf",
            "Infinity",
            "-Infinity",
            "1e400",  # overflows to inf on coercion
            float("nan"),
            float("inf"),
            float("-inf"),
        ],
        ids=repr,
    )
    def test_extract_provider_cost_rejects_a_non_finite_value(self, bad):
        """NaN/Inf coerce CLEANLY through ``float()`` — the type guard never sees them.

        This is the float-coercible-but-malformed class the type checks above miss:
        ``float("nan")`` raises nothing, so an unguarded extractor returns it as a
        price. A non-finite ``usage.cost`` stamped ``cost_source="provider"`` asserts
        an undefined/infinite charge as a BILLED FACT — the exact fabrication this
        adapter exists to prevent. It also serializes to bare ``NaN``/``Infinity``
        (invalid JSON per RFC 8259) and poisons every downstream rollup it reaches:
        one NaN in a SUM turns an entire org's cost dashboard into NaN.
        """

        class _Usage:
            cost = bad

        class _Response:
            usage = _Usage()

        assert OpenRouterProvider.extract_provider_cost(_Response()) is None

    @pytest.mark.parametrize("bad", [-5.0, -0.00042, "-1", -1], ids=repr)
    def test_extract_provider_cost_rejects_a_negative_charge(self, bad):
        """A gateway cannot bill a NEGATIVE charge for a completion.

        There is no refund we could honestly model from a per-call usage field, so a
        negative value is a malformed field, not a credit. Left unguarded it would
        subtract from an org's real spend — under-reporting a bill is as dishonest
        as inventing one.
        """

        class _Usage:
            cost = bad

        class _Response:
            usage = _Usage()

        assert OpenRouterProvider.extract_provider_cost(_Response()) is None

    def test_a_reported_zero_is_still_a_price(self):
        """Vacuity control for the two rejection tests above.

        ``:free`` slugs genuinely bill $0, so the finite/negative guards must reject
        the malformed class WITHOUT swallowing an honest reported zero — a guard
        written as ``if not cost`` would pass every test above and silently drop it.
        """

        class _Usage:
            cost = 0.0

        class _Response:
            usage = _Usage()

        assert OpenRouterProvider.extract_provider_cost(_Response()) == 0.0

    def test_a_numeric_string_cost_still_counts(self):
        """The gateway is JSON: a quoted number is still a reported charge."""

        class _Usage:
            cost = "0.00042"

        class _Response:
            usage = _Usage()

        assert OpenRouterProvider.extract_provider_cost(_Response()) == pytest.approx(0.00042)

    def test_decided_price_cannot_be_clobbered_by_the_usage_block(self):
        """Regression lock for the fragility this port was warned about.

        ``_emit_cost`` used to spread the raw usage dict LAST, so the price was
        decided by dict ordering: any provider whose usage block carried a
        ``cost_usd`` would silently overwrite the real charge, and no test would
        notice. The decided price is now authoritative.
        """
        from layerlens.instrument.adapters.providers._emit_helpers import _emit_cost

        emitted: List[Any] = []

        class _Collector:
            def emit(self, event_type: str, payload: Dict[str, Any], **kwargs: Any) -> None:
                emitted.append((event_type, payload))

        _emit_cost(
            _Collector(),
            provider="openrouter",
            model=_ROUTED_SLUG,
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost_usd": 99.0},
            pricing_table=None,
            span_id="a" * 16,
            parent_span_id=None,
            provider_cost_usd=0.00042,
            provider_cost_only=True,
        )

        assert len(emitted) == 1
        assert emitted[0][1]["cost_usd"] == pytest.approx(0.00042), "a stray usage key overwrote the billed charge"
