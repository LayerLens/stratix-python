"""Real-behaviour unit suite for the OpenRouter provider adapter.

OpenRouter is credential-gated for us (no ``sk-or-…`` key exists), so these
tests stand in for live verification the way ``test_azure_openai.py`` does for
Azure: a REAL ``openai.OpenAI`` client is pointed at the real OpenRouter base
URL with an ``httpx.MockTransport`` injected through the ``http_client=`` seam.
Request signing, routing, SSE framing and response parsing therefore all run
through the real SDK against a real OpenRouter response body — including the
``usage.cost`` field OpenRouter adds under usage accounting — with no network.

The body shapes mirror the documented OpenRouter contract: a routed
``vendor/model`` slug in ``model``, a ``gen-…`` response id, and (only when the
caller opts in with ``extra_body={"usage": {"include": True}}``) ``usage.cost``
carrying the charge OpenRouter actually billed.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import httpx
import pytest

from openai import OpenAI
from layerlens.instrument import trace
from layerlens.instrument._capture_config import CaptureConfig
from layerlens.instrument.adapters.providers.openrouter import (
    OPENROUTER_BASE_URL,
    OpenRouterProvider,
    build_client,
    instrument_openrouter,
    uninstrument_openrouter,
)

from ...conftest import find_event, find_events

_API_KEY = "sk-or-v1-fake-openrouter-key"
_ROUTED_SLUG = "anthropic/claude-opus-4.8"


def _chat_json(
    content: str = "Hello from OpenRouter!",
    *,
    model: str = _ROUTED_SLUG,
    prompt_tokens: int = 14,
    completion_tokens: int = 7,
    cost: Any = None,
    include_usage: bool = True,
) -> Dict[str, Any]:
    """A realistic OpenRouter chat.completions body.

    ``cost`` is present only when the caller enabled usage accounting, which is
    exactly the switch that decides whether a cost.record can be emitted.
    """
    usage: Optional[Dict[str, Any]] = None
    if include_usage:
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        if cost is not None:
            usage["cost"] = cost
            usage["is_byok"] = False
            usage["cost_details"] = {"upstream_inference_cost": 0.0}
    return {
        "id": "gen-1770000000-Xy7Qw",
        "object": "chat.completion",
        "created": 1770000000,
        "model": model,
        "provider": "Anthropic",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
        "usage": usage,
    }


def _tool_call_json(model: str = _ROUTED_SLUG, cost: Any = 0.0004) -> Dict[str, Any]:
    return {
        "id": "gen-1770000000-tool",
        "object": "chat.completion",
        "created": 1770000000,
        "model": model,
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_or_abc123",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city": "Lisbon"}'},
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 50, "completion_tokens": 12, "total_tokens": 62, "cost": cost},
    }


def _make_client(
    response_json: Optional[Dict[str, Any]] = None,
    *,
    status_code: int = 200,
    base_url: str = OPENROUTER_BASE_URL,
    sse: Optional[str] = None,
    raw_body: Optional[str] = None,
) -> tuple:
    """Real ``openai.OpenAI`` client over httpx.MockTransport. Returns (client, requests).

    ``raw_body`` puts an exact byte sequence on the wire, bypassing httpx's ``json=``
    encoder. That encoder runs with ``allow_nan=False`` and so CANNOT emit the bare
    ``NaN``/``Infinity`` tokens a misbehaving gateway can — while Python's parser on
    the receiving side accepts them and yields ``float('nan')``/``float('inf')``.
    Only a raw body can reproduce that real production path.
    """
    requests: List[httpx.Request] = []
    payload = response_json if response_json is not None else _chat_json()

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if sse is not None:
            return httpx.Response(status_code, content=sse, headers={"content-type": "text/event-stream"})
        if raw_body is not None:
            return httpx.Response(status_code, content=raw_body, headers={"content-type": "application/json"})
        return httpx.Response(status_code, json=payload)

    client = OpenAI(
        base_url=base_url,
        api_key=_API_KEY,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    return client, requests


def _sse(chunks: List[Dict[str, Any]]) -> str:
    return "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# model.invoke — honest naming, routed-slug attribution
# ---------------------------------------------------------------------------
class TestModelInvoke:
    def test_event_name_provider_and_routed_slug(self, mock_client, capture_trace):
        client, requests = _make_client(_chat_json(cost=0.00123456))
        OpenRouterProvider().connect(client)

        @trace(mock_client, capture_config=CaptureConfig.full())
        def my_agent():
            r = client.chat.completions.create(
                model=_ROUTED_SLUG,
                messages=[{"role": "user", "content": "Hello?"}],
                temperature=0.2,
                extra_body={"usage": {"include": True}},
            )
            return r.choices[0].message.content

        assert my_agent() == "Hello from OpenRouter!"

        # The real SDK routed to the real OpenRouter host.
        request = requests[0]
        assert request.url.host == "openrouter.ai"
        assert request.url.path == "/api/v1/chat/completions"
        sent = json.loads(request.content)
        assert sent["usage"] == {"include": True}

        mi = find_event(capture_trace["events"], "model.invoke")["payload"]
        # The event name must name the surface that actually served the call.
        # Inheriting OpenAIProvider's hardcoded name would claim "openai.*".
        assert mi["name"] == "openrouter.chat.completions.create"
        assert mi["framework"] == "openrouter"
        assert mi["model"] == _ROUTED_SLUG
        assert mi["response_model"] == _ROUTED_SLUG
        assert mi["response_id"] == "gen-1770000000-Xy7Qw"
        assert mi["finish_reason"] == "stop"
        assert mi["latency_ms"] > 0
        assert mi["messages"] == [{"role": "user", "content": "Hello?"}]
        assert mi["output_message"] == {"role": "assistant", "content": "Hello from OpenRouter!"}
        assert mi["usage"] == {"prompt_tokens": 14, "completion_tokens": 7, "total_tokens": 21}
        # Flat token fields (S11/F2) — the atlas extractor never reads usage.*.
        assert (mi["prompt_tokens"], mi["completion_tokens"], mi["total_tokens"]) == (14, 7, 21)
        assert mi["parameters"]["model"] == _ROUTED_SLUG
        assert mi["parameters"]["temperature"] == 0.2
        # gen_ai.system falls through to the literal provider — honest.
        assert mi["otel_gen_ai"]["gen_ai.system"] == "openrouter"

    def test_routed_model_wins_over_requested_slug(self, mock_client, capture_trace):
        """``openrouter/auto`` names no real model; the routed slug does."""
        client, _ = _make_client(_chat_json(model=_ROUTED_SLUG, cost=0.0002))
        OpenRouterProvider().connect(client)

        @trace(mock_client, capture_config=CaptureConfig.full())
        def my_agent():
            client.chat.completions.create(
                model="openrouter/auto",
                messages=[{"role": "user", "content": "Hi"}],
            )
            return "done"

        my_agent()
        events = capture_trace["events"]
        mi = find_event(events, "model.invoke")["payload"]
        assert mi["model"] == _ROUTED_SLUG
        assert mi["parameters"]["model"] == "openrouter/auto"
        # Cost attributes to the model that actually served the request.
        assert find_event(events, "cost.record")["payload"]["model"] == _ROUTED_SLUG

    def test_no_events_without_a_collector(self, mock_client):
        """No active trace context => pass straight through, no interference."""
        client, requests = _make_client()
        OpenRouterProvider().connect(client)

        r = client.chat.completions.create(model=_ROUTED_SLUG, messages=[{"role": "user", "content": "Hi"}])

        assert r.choices[0].message.content == "Hello from OpenRouter!"
        assert len(requests) == 1
        mock_client.traces.upload.assert_not_called()


# ---------------------------------------------------------------------------
# Provider re-tagging — the whole point of the adapter
# ---------------------------------------------------------------------------
class TestProviderTagging:
    def test_every_event_claims_openrouter_not_openai(self, mock_client, capture_trace):
        """An Anthropic-served OpenRouter call must never claim provider='openai'.

        The inherited patch surface derives the provider from the event name, so
        without ``classify_provider`` every cost.record / tool.call would lie.
        """
        client, _ = _make_client(_tool_call_json())
        OpenRouterProvider().connect(client)

        @trace(mock_client, capture_config=CaptureConfig.full())
        def my_agent():
            client.chat.completions.create(
                model=_ROUTED_SLUG,
                messages=[{"role": "user", "content": "Weather in Lisbon?"}],
            )
            return "done"

        my_agent()
        events = capture_trace["events"]
        assert find_event(events, "cost.record")["payload"]["provider"] == "openrouter"
        tool = find_event(events, "tool.call")["payload"]
        assert tool["provider"] == "openrouter"
        assert tool["framework"] == "openrouter"
        assert tool["model"] == _ROUTED_SLUG
        assert tool["tool_name"] == "get_weather"
        assert tool["arguments"] == {"city": "Lisbon"}
        assert tool["id"] == "call_or_abc123"

    def test_provider_tag_does_not_depend_on_the_event_name(self):
        """The billing tag is decided explicitly, not by splitting a string.

        ``emit_llm_events`` falls back to ``event_name.split('.')[0]``, so the
        tag would silently follow ``event_prefix`` if this override were dropped
        — coupling who-gets-billed to a cosmetic naming choice. Handed the
        inherited openai event name, the call is still OpenRouter's.
        """
        assert (
            OpenRouterProvider.classify_provider("openai.chat.completions.create", {"model": "gpt-4o"}) == "openrouter"
        )
        assert OpenRouterProvider.classify_provider("openrouter.embeddings.create", {}) == "openrouter"

    def test_adapter_info(self):
        provider = OpenRouterProvider()
        assert provider.is_connected is False
        client, _ = _make_client()
        provider.connect(client)
        info = provider.adapter_info()
        assert (info.name, info.adapter_type, info.connected) == ("openrouter", "provider", True)


# ---------------------------------------------------------------------------
# Provider-reported cost — PROVIDER COST OR NOTHING
# ---------------------------------------------------------------------------
class TestProviderCost:
    def test_reported_cost_is_the_cost(self, mock_client, capture_trace):
        client, _ = _make_client(_chat_json(cost=0.00123456))
        OpenRouterProvider().connect(client)

        @trace(mock_client, capture_config=CaptureConfig.full())
        def my_agent():
            client.chat.completions.create(
                model=_ROUTED_SLUG,
                messages=[{"role": "user", "content": "Hi"}],
                extra_body={"usage": {"include": True}},
            )
            return "done"

        my_agent()
        cost = find_event(capture_trace["events"], "cost.record")["payload"]
        assert cost["cost_usd"] == pytest.approx(0.00123456)
        # The provenance of a billed FACT must be distinguishable from an estimate.
        assert cost["cost_source"] == "provider"
        assert cost["provider"] == "openrouter"
        assert cost["model"] == _ROUTED_SLUG
        assert cost["framework"] == "openrouter"
        assert (cost["prompt_tokens"], cost["completion_tokens"], cost["total_tokens"]) == (14, 7, 21)
        # The gateway's raw usage vocabulary must not leak into the nested block.
        assert "cost" not in cost

    def test_no_cost_record_when_gateway_reports_no_cost(self, mock_client, capture_trace):
        """Usage accounting off => the charge is unknowable => emit NOTHING.

        No catalog we ship holds OpenRouter's per-slug rates, so any number here
        would be invented. Tokens still ride model.invoke.
        """
        client, _ = _make_client(_chat_json(cost=None))
        OpenRouterProvider().connect(client)

        @trace(mock_client, capture_config=CaptureConfig.full())
        def my_agent():
            client.chat.completions.create(model=_ROUTED_SLUG, messages=[{"role": "user", "content": "Hi"}])
            return "done"

        my_agent()
        events = capture_trace["events"]
        assert find_events(events, "cost.record") == []
        # Structure/metadata survives the honest cost gap.
        mi = find_event(events, "model.invoke")["payload"]
        assert mi["usage"]["total_tokens"] == 21
        assert mi["total_tokens"] == 21

    def test_bare_model_name_is_never_priced_from_the_openai_catalog(self, mock_client, capture_trace):
        """The fabrication this adapter exists to prevent.

        ``gpt-4o`` DOES resolve in the bundled PRICING table, so a cost.record
        carrying it would be auto-priced at OpenAI list rates by the collector's
        price-on-emit chokepoint — a charge OpenRouter never billed (OpenRouter
        sets its own margin). Emitting no record is the only honest answer.
        """
        from layerlens.instrument.adapters.providers.pricing import PRICING, calculate_cost
        from layerlens.instrument.adapters.providers.token_usage import NormalizedTokenUsage

        # Pin the premise: this model is priceable, so the danger is real.
        assert (
            calculate_cost(
                "gpt-4o", NormalizedTokenUsage(prompt_tokens=14, completion_tokens=7, total_tokens=21), PRICING
            )
            is not None
        )

        client, _ = _make_client(_chat_json(model="gpt-4o", cost=None))
        OpenRouterProvider().connect(client)

        @trace(mock_client, capture_config=CaptureConfig.full())
        def my_agent():
            client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "Hi"}])
            return "done"

        my_agent()
        events = capture_trace["events"]
        assert find_events(events, "cost.record") == [], "priced a routed call from a catalog that never billed it"
        assert find_event(events, "model.invoke")["payload"]["model"] == "gpt-4o"

    @pytest.mark.parametrize("bad", ["not-a-number", "", [], {}, None])
    def test_malformed_gateway_cost_never_becomes_a_price(self, mock_client, capture_trace, bad):
        """A malformed gateway value must not coerce into a fabricated charge."""
        client, _ = _make_client(_chat_json(cost=bad))
        OpenRouterProvider().connect(client)

        @trace(mock_client, capture_config=CaptureConfig.full())
        def my_agent():
            client.chat.completions.create(model=_ROUTED_SLUG, messages=[{"role": "user", "content": "Hi"}])
            return "done"

        my_agent()
        assert find_events(capture_trace["events"], "cost.record") == []

    @pytest.mark.parametrize("bad", ["nan", "inf", "-inf", "Infinity", -5.0, -0.00042], ids=repr)
    def test_non_finite_or_negative_gateway_cost_never_becomes_a_price(self, mock_client, capture_trace, bad):
        """The float-coercible malformed class, proven END-TO-END over the real SDK.

        The parametrize above this one only covers values that FAIL ``float()``.
        These pass it cleanly, so an unguarded extractor emits a real cost.record
        carrying ``cost_usd=nan``/``inf``/negative stamped ``cost_source="provider"``
        — an undefined or impossible charge asserted as a billed fact.
        """
        client, _ = _make_client(_chat_json(cost=bad))
        OpenRouterProvider().connect(client)

        @trace(mock_client, capture_config=CaptureConfig.full())
        def my_agent():
            client.chat.completions.create(model=_ROUTED_SLUG, messages=[{"role": "user", "content": "Hi"}])
            return "done"

        my_agent()
        events = capture_trace["events"]
        assert find_events(events, "cost.record") == [], f"gateway usage.cost={bad!r} was emitted as a billed charge"
        # The call itself is still recorded — rejecting the price must not cost us
        # the tokens, which are honest regardless of the malformed cost field.
        assert find_event(events, "model.invoke")["payload"]["model"] == _ROUTED_SLUG

    @pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"], ids=repr)
    def test_a_bare_non_finite_wire_token_never_becomes_a_price(self, mock_client, capture_trace, token):
        """The REAL non-finite path: a bare ``NaN`` token on the wire.

        RFC 8259 has no NaN/Infinity literal, but Python's ``json.loads`` accepts
        them by default — so a gateway that emits ``"cost": NaN`` hands the SDK a
        genuine ``float('nan')``, which sails through ``float()`` and every type
        check. This drives that exact byte sequence through the real ``openai``
        parser (httpx's ``json=`` encoder runs ``allow_nan=False`` and cannot
        produce it, which is why the body is raw here).
        """
        body = json.dumps(_chat_json(cost=0.0)).replace('"cost": 0.0', f'"cost": {token}')
        assert f'"cost": {token}' in body, "the bare token was not planted on the wire"

        client, _ = _make_client(raw_body=body)
        OpenRouterProvider().connect(client)

        @trace(mock_client, capture_config=CaptureConfig.full())
        def my_agent():
            client.chat.completions.create(model=_ROUTED_SLUG, messages=[{"role": "user", "content": "Hi"}])
            return "done"

        my_agent()
        events = capture_trace["events"]
        assert find_events(events, "cost.record") == [], f"wire token {token} was emitted as a billed charge"
        # Anti-vacuity: proves the body PARSED and the call was really recorded, so
        # "no cost.record" is the guard's doing and not a swallowed transport error.
        assert find_event(events, "model.invoke")["payload"]["model"] == _ROUTED_SLUG

    def test_boolean_cost_is_not_a_price(self):
        """``float(True)`` is 1.0 — a $1 charge invented from a flag."""

        class _Usage:
            cost = True

        class _Response:
            usage = _Usage()

        assert OpenRouterProvider.extract_provider_cost(_Response()) is None

    def test_free_model_zero_cost_is_reported_not_suppressed(self, mock_client, capture_trace):
        """OpenRouter ``:free`` slugs genuinely bill $0. A reported zero is a FACT.

        This is the one honest zero: it is distinguished from a fabricated 0.0 by
        having been reported by the gateway (``cost_source='provider'``).
        """
        client, _ = _make_client(_chat_json(model="meta-llama/llama-3.1-8b-instruct:free", cost=0))
        OpenRouterProvider().connect(client)

        @trace(mock_client, capture_config=CaptureConfig.full())
        def my_agent():
            client.chat.completions.create(
                model="meta-llama/llama-3.1-8b-instruct:free",
                messages=[{"role": "user", "content": "Hi"}],
                extra_body={"usage": {"include": True}},
            )
            return "done"

        my_agent()
        cost = find_event(capture_trace["events"], "cost.record")["payload"]
        assert cost["cost_usd"] == 0.0
        assert cost["cost_source"] == "provider"

    def test_no_cost_record_without_usage(self, mock_client, capture_trace):
        """No usage block at all => tokens unknown => no cost.record (ateam parity)."""
        client, _ = _make_client(_chat_json(include_usage=False))
        OpenRouterProvider().connect(client)

        @trace(mock_client, capture_config=CaptureConfig.full())
        def my_agent():
            client.chat.completions.create(model=_ROUTED_SLUG, messages=[{"role": "user", "content": "Hi"}])
            return "done"

        my_agent()
        events = capture_trace["events"]
        assert find_events(events, "cost.record") == []
        assert find_event(events, "model.invoke")["payload"]["model"] == _ROUTED_SLUG


# ---------------------------------------------------------------------------
# Streaming — inherited aggregation, provider cost on the usage-bearing chunk
# ---------------------------------------------------------------------------
class TestStreaming:
    def test_stream_emits_one_invoke_with_ttft_and_provider_cost(self, mock_client, capture_trace):
        chunks = [
            {
                "id": "gen-stream-1",
                "object": "chat.completion.chunk",
                "created": 1770000000,
                "model": _ROUTED_SLUG,
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hel"}, "finish_reason": None}],
            },
            {
                "id": "gen-stream-1",
                "object": "chat.completion.chunk",
                "created": 1770000000,
                "model": _ROUTED_SLUG,
                "choices": [{"index": 0, "delta": {"content": "lo!"}, "finish_reason": "stop"}],
            },
            # OpenRouter puts usage (and the billed cost) on a final usage-only chunk.
            {
                "id": "gen-stream-1",
                "object": "chat.completion.chunk",
                "created": 1770000000,
                "model": _ROUTED_SLUG,
                "choices": [],
                "usage": {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12, "cost": 0.000456},
            },
        ]
        client, _ = _make_client(sse=_sse(chunks))
        OpenRouterProvider().connect(client)

        @trace(mock_client, capture_config=CaptureConfig.full())
        def my_agent():
            stream = client.chat.completions.create(
                model=_ROUTED_SLUG,
                messages=[{"role": "user", "content": "Hi"}],
                stream=True,
                stream_options={"include_usage": True},
                extra_body={"usage": {"include": True}},
            )
            return "".join(c.choices[0].delta.content or "" for c in stream if c.choices)

        assert my_agent() == "Hello!"

        events = capture_trace["events"]
        invokes = find_events(events, "model.invoke")
        assert len(invokes) == 1, "a stream must emit exactly one consolidated model.invoke"
        mi = invokes[0]["payload"]
        assert mi["name"] == "openrouter.chat.completions.create"
        assert mi["model"] == _ROUTED_SLUG
        assert mi["output_message"]["content"] == "Hello!"
        assert mi["ttft_ms"] is not None and mi["streaming_duration_ms"] is not None
        assert mi["usage"]["total_tokens"] == 12

        cost = find_event(events, "cost.record")["payload"]
        assert cost["cost_usd"] == pytest.approx(0.000456)
        assert cost["cost_source"] == "provider"
        assert cost["provider"] == "openrouter"


# ---------------------------------------------------------------------------
# build_client / connect scope / lifecycle
# ---------------------------------------------------------------------------
class TestBuildClient:
    def test_sets_openrouter_base_url_and_attribution_headers(self):
        client = build_client(_API_KEY, http_referer="https://myapp.example", x_title="My App")
        assert str(client.base_url).rstrip("/") == OPENROUTER_BASE_URL
        assert client.default_headers["HTTP-Referer"] == "https://myapp.example"
        assert client.default_headers["X-Title"] == "My App"

    def test_caller_headers_are_merged_not_collided(self):
        """ateam forwards **client_kwargs alongside its own ``default_headers=``,
        so a caller passing their own headers gets a TypeError. Merge instead."""
        client = build_client(_API_KEY, x_title="My App", default_headers={"X-Tenant": "acme"})
        assert client.default_headers["X-Tenant"] == "acme"
        assert client.default_headers["X-Title"] == "My App"

    def test_no_headers_requested(self):
        client = build_client(_API_KEY)
        assert str(client.base_url).rstrip("/") == OPENROUTER_BASE_URL


class TestConnectScope:
    def test_warns_when_client_is_not_pointed_at_openrouter(self, caplog):
        """Instrumenting a plain OpenAI client would silently re-tag every real
        OpenAI call as provider='openrouter' — a mislabel, so say so."""
        client, _ = _make_client(base_url="https://api.openai.com/v1")
        with caplog.at_level(logging.WARNING):
            OpenRouterProvider().connect(client)
        assert any("openrouter" in r.message.lower() for r in caplog.records)

    def test_no_warning_for_a_real_openrouter_client(self, caplog):
        client, _ = _make_client()
        with caplog.at_level(logging.WARNING):
            OpenRouterProvider().connect(client)
        assert caplog.records == []


class TestLifecycle:
    def test_instrument_registers_and_uninstrument_restores(self, mock_client, capture_trace):
        client, _ = _make_client(_chat_json(cost=0.0001))
        original = client.chat.completions.create

        provider = instrument_openrouter(client)
        assert provider.is_connected is True
        assert client.chat.completions.create != original

        from layerlens.instrument.adapters._registry import get

        assert get("openrouter") is provider

        @trace(mock_client, capture_config=CaptureConfig.full())
        def my_agent():
            client.chat.completions.create(model=_ROUTED_SLUG, messages=[{"role": "user", "content": "Hi"}])
            return "done"

        my_agent()
        assert find_event(capture_trace["events"], "model.invoke")["payload"]["framework"] == "openrouter"

        uninstrument_openrouter()
        assert client.chat.completions.create == original
        assert get("openrouter") is None
