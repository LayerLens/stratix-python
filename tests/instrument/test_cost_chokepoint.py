"""Centralized price-on-emit chokepoint (A1 / LAY-3626).

Pricing was copy-pasted across ~10 emit paths; the one true chokepoint every
event passes — ``TraceCollector.emit`` — priced nothing, so a new ``_fire``
override that forgot the hook shipped a tokens-only ``cost.record`` and
tier/cache-write pricing (A2/A3) only applied where each site remembered to pass
them. The collector now recomputes ``cost_usd`` for every ``cost.record`` from
its own payload (model + usage + service_tier + provider) — one place, every
path priced identically. These tests bite: drop the chokepoint and the
unpriced/tier/cache cases go RED.
"""

from __future__ import annotations

import pytest

from layerlens.instrument._collector import TraceCollector
from layerlens.instrument._capture_config import CaptureConfig
from layerlens.instrument.adapters.providers.pricing import calculate_cost
from layerlens.instrument.adapters.providers.token_usage import NormalizedTokenUsage

pytestmark = pytest.mark.invariant


def _cost_event(collector: TraceCollector) -> dict:
    return [e for e in collector.events if e["event_type"] == "cost.record"][0]


def _collector() -> TraceCollector:
    # capture_content=True so redaction never interferes with the cost fields.
    return TraceCollector(object(), CaptureConfig(capture_content=True))


def test_chokepoint_fills_missing_cost_usd() -> None:
    """A1/A11: a cost.record emitted WITHOUT cost_usd but with a priced model is
    priced centrally — no emit path can ship a tokens-only record."""
    c = _collector()
    c.emit(
        "cost.record",
        {
            "provider": "openai",
            "model": "gpt-4o",
            "prompt_tokens": 1000,
            "completion_tokens": 1000,
            "total_tokens": 2000,
        },
        span_id="s1",
    )
    ev = _cost_event(c)
    expected = calculate_cost(
        "gpt-4o", NormalizedTokenUsage(prompt_tokens=1000, completion_tokens=1000, total_tokens=2000)
    )
    assert expected and expected > 0
    assert ev["payload"]["cost_usd"] == expected, "chokepoint did not price a tokens-only cost.record"


def test_chokepoint_applies_service_tier_centrally() -> None:
    """A2 at the chokepoint: a priority-tier cost.record emitted WITHOUT cost_usd
    is filled at ~2x standard — tier pricing flows through the central fill."""
    c = _collector()
    c.emit(
        "cost.record",
        {
            "provider": "openai",
            "model": "gpt-4o",
            "prompt_tokens": 1000,
            "completion_tokens": 1000,
            "total_tokens": 2000,
            "service_tier": "priority",
        },
        span_id="s1",
    )
    ev = _cost_event(c)
    std = calculate_cost("gpt-4o", NormalizedTokenUsage(prompt_tokens=1000, completion_tokens=1000, total_tokens=2000))
    assert ev["payload"]["cost_usd"] == pytest.approx(std * 2.0), "service_tier not applied at the chokepoint"


def test_chokepoint_does_not_clobber_preset_cost_usd() -> None:
    """Fill-when-absent: an adapter-computed cost_usd (langfuse vendor cost,
    bedrock_agents, the _fire helpers) is PRESERVED, never overwritten."""
    c = _collector()
    c.emit(
        "cost.record",
        {
            "provider": "langfuse",
            "model": "gpt-4o",
            "prompt_tokens": 1000,
            "completion_tokens": 1000,
            "total_tokens": 2000,
            "cost_usd": 0.123,
        },
        span_id="s1",
    )
    ev = _cost_event(c)
    assert ev["payload"]["cost_usd"] == 0.123, "chokepoint clobbered an adapter-supplied cost_usd"


def test_chokepoint_prices_cache_creation_centrally() -> None:
    """A3 at the chokepoint: cache_creation (write) tokens add cost even when the
    payload uses the raw ``cache_creation_input_tokens`` wire key."""
    c = _collector()
    c.emit(
        "cost.record",
        {
            "provider": "anthropic",
            "model": "claude-3-5-sonnet",
            "prompt_tokens": 1000,
            "completion_tokens": 100,
            "total_tokens": 1100,
            "cache_creation_input_tokens": 1000,
        },
        span_id="s1",
    )
    ev = _cost_event(c)
    base = calculate_cost(
        "claude-3-5-sonnet", NormalizedTokenUsage(prompt_tokens=1000, completion_tokens=100, total_tokens=1100)
    )
    assert ev["payload"]["cost_usd"] > base, "cache_creation not priced at the chokepoint"


def test_chokepoint_leaves_unpriced_model_unpriced() -> None:
    """A model that resolves to no rate stays cost_usd=None (don't fabricate)."""
    c = _collector()
    c.emit(
        "cost.record",
        {
            "provider": "ollama",
            "model": "some-local-model-xyz",
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
        span_id="s1",
    )
    ev = _cost_event(c)
    assert ev["payload"].get("cost_usd") is None


def test_chokepoint_uses_azure_table_for_azure_provider() -> None:
    """Provider routing: azure gpt-4o is priced from AZURE_PRICING, not PRICING
    (different rate) — the chokepoint must not mis-route to the OpenAI table."""
    from layerlens.instrument.adapters.providers.pricing import AZURE_PRICING

    c = _collector()
    c.emit(
        "cost.record",
        {"provider": "azure", "model": "gpt-4o", "prompt_tokens": 1000, "completion_tokens": 0, "total_tokens": 1000},
        span_id="s1",
    )
    ev = _cost_event(c)
    azure_expected = calculate_cost(
        "gpt-4o", NormalizedTokenUsage(prompt_tokens=1000, completion_tokens=0, total_tokens=1000), AZURE_PRICING
    )
    assert ev["payload"]["cost_usd"] == azure_expected, "azure cost.record not priced from AZURE_PRICING"
