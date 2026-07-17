"""Recorded-real-response replay for the OpenRouter provider — SEALED corpus.

No OpenRouter API key (``sk-or-…``) exists on any machine, so — exactly like
``test_azure_openai_recorded`` — the gateway hop cannot be made live. What is
replayed here is nonetheless a **real model response**: each fixture body was
captured from a genuine live inference (local ollama ``llama3:8b`` over its
OpenAI-compatible endpoint for the free route; a real billed OpenAI
``gpt-4o-mini`` call for the paid route) and re-enveloped into OpenRouter's
documented wire shape. See each fixture's ``provenance.sealed_reason`` /
``real_body_source`` for exactly which fields are real and which are sealed.

The corpus rule still holds — **record UPSTREAM of the parser, assert DOWNSTREAM
of it**: the fixture is the raw gateway body (the adapter's *input*), the
assertions are the adapter's emitted *events*. A real ``openai.OpenAI`` client
pointed at the real OpenRouter base URL does its real routing + deserialization
over ``httpx.MockTransport``, and the real ``OpenRouterProvider`` parses it.

Distinct from ``test_openrouter.py``, whose bodies are hand-built doubles with
fabricated token counts: every token count asserted below is a **real
tokenizer output** from a real inference, so this is the gate on "does the
adapter still parse a REAL response".

The two scenarios pin the adapter's two cost branches, which are the whole
reason this adapter exists (``provider_cost_only`` — OpenRouter's rates are in
no table we ship, so the gateway is the sole authority):

* ``free_route`` — usage accounting ON, ``usage.cost = 0.0``. A ``:free`` slug
  genuinely bills $0.00 and that **zero is a fact**, so it must survive to a
  ``cost.record`` stamped ``cost_source="provider"`` (a truthy-check bug would
  silently drop it).
* ``paid_route_no_accounting`` — usage accounting OFF, no ``usage.cost``. The
  adapter must emit real tokens on ``model.invoke`` and **NO** ``cost.record``:
  pricing the routed ``openai/gpt-4o-mini`` slug from our own catalog would
  attach a charge OpenRouter never billed.
"""

from __future__ import annotations

import httpx

from openai import OpenAI
from layerlens.instrument import trace
from layerlens.instrument._capture_config import CaptureConfig
from layerlens.instrument.adapters.providers.openrouter import (
    OPENROUTER_BASE_URL,
    OpenRouterProvider,
)

from ...conftest import find_event, find_events
from ..._recorded import load_recorded, mock_transport


def _client(fixture):
    transport, requests = mock_transport(fixture)
    client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key="sk-or-v1-sealed-no-openrouter-credential",
        http_client=httpx.Client(transport=transport),
    )
    return client, requests


class TestOpenRouterRecorded:
    def test_free_route_real_shape_keeps_reported_zero_cost(self, mock_client, capture_trace):
        """A ``:free`` route: real llama-3-8b body, real tokens, honest $0.00."""
        fixture = load_recorded("openrouter", "free_route")
        client, requests = _client(fixture)
        provider = OpenRouterProvider()
        provider.connect(client)

        @trace(mock_client, capture_config=CaptureConfig.full())
        def agent():
            r = client.chat.completions.create(
                model="meta-llama/llama-3-8b-instruct:free",
                messages=[{"role": "user", "content": "What is the ingest rate limit on the Growth plan?"}],
                extra_body={"usage": {"include": True}},
            )
            return r.choices[0].message.content

        answer = agent()
        # The real SDK really routed at the real OpenRouter base URL.
        assert requests[0].url.host == "openrouter.ai"
        assert requests[0].url.path == "/api/v1/chat/completions"

        events = capture_trace["events"]
        mi = find_event(events, "model.invoke")
        # The ROUTED slug the gateway reports wins — not the requested alias.
        assert mi["payload"]["model"] == "meta-llama/llama-3-8b-instruct:free"
        assert mi["payload"]["response_model"] == "meta-llama/llama-3-8b-instruct:free"
        assert mi["payload"]["finish_reason"] == "stop"
        assert mi["payload"]["framework"] == "openrouter"

        # REAL tokenizer counts from the real llama3:8b inference (not authored).
        usage = mi["payload"]["usage"]
        assert usage["prompt_tokens"] == 95
        assert usage["completion_tokens"] == 98
        assert usage["total_tokens"] == 193

        # The real model output survived the round-trip intact.
        out = mi["payload"]["output_message"]
        assert out["role"] == "assistant"
        assert out["content"] == answer
        assert "429" in answer

        # THE branch this fixture exists for: a reported ZERO is a billed fact and
        # must reach cost.record stamped as the gateway's own, not be dropped by a
        # truthiness check nor recomputed from our catalog.
        cost = find_event(events, "cost.record")
        assert cost["payload"]["cost_usd"] == 0.0
        assert cost["payload"]["cost_source"] == "provider"
        assert cost["payload"]["provider"] == "openrouter"
        assert cost["payload"]["model"] == "meta-llama/llama-3-8b-instruct:free"
        assert cost["payload"]["total_tokens"] == 193
        provider.disconnect()

    def test_paid_route_without_accounting_emits_no_cost(self, mock_client, capture_trace):
        """No reported charge -> real tokens, but NO invented price."""
        fixture = load_recorded("openrouter", "paid_route_no_accounting")
        client, _ = _client(fixture)
        provider = OpenRouterProvider()
        provider.connect(client)

        @trace(mock_client, capture_config=CaptureConfig.full())
        def agent():
            r = client.chat.completions.create(
                model="openai/gpt-4o-mini",
                messages=[{"role": "user", "content": "Why are we getting 429s during backfill?"}],
            )
            return r.choices[0].message.content

        answer = agent()
        events = capture_trace["events"]

        mi = find_event(events, "model.invoke")
        assert mi["payload"]["model"] == "openai/gpt-4o-mini"
        assert mi["payload"]["framework"] == "openrouter"
        # REAL counts from the real billed gpt-4o-mini call.
        assert mi["payload"]["usage"]["prompt_tokens"] == 128
        assert mi["payload"]["usage"]["completion_tokens"] == 132
        assert mi["payload"]["usage"]["total_tokens"] == 260
        # Real OpenAI-shape members OpenRouter passes through.
        assert mi["payload"]["usage"]["cached_tokens"] == 0
        assert mi["payload"]["usage"]["reasoning_tokens"] == 0
        assert mi["payload"]["output_message"]["content"] == answer

        # THE refusal branch: the gateway reported no charge, so we record none.
        # gpt-4o-mini resolves in our PRICING table, so a regression here would
        # cheerfully attach OpenAI list-rate dollars that OpenRouter never billed.
        assert find_events(events, "cost.record") == []
        provider.disconnect()

    def test_sealed_provenance_is_flagged(self):
        """The gap stays visible: these are sealed, and say so."""
        for scenario in ("free_route", "paid_route_no_accounting"):
            prov = load_recorded("openrouter", scenario)["provenance"]
            assert prov["provider"] == "openrouter"
            assert prov["scenario"] == scenario
            assert prov["sealed"] is True
            assert "No OpenRouter API key" in prov["sealed_reason"]
            # The body is a REAL captured inference — say where it came from.
            assert prov["real_body_source"]
