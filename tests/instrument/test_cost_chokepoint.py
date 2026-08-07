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


class TestNeverFabricatesAZero:
    """The chokepoint must not turn an unpriceable payload into ``cost_usd: 0.0``.

    FILL-WHEN-ABSENT was written for the *dropped-price* case (an emit path forgot
    the hook). It also fires for a payload the formula cannot price at all — a
    totals-only usage — and the shared formula answers ``0.0`` there, because it
    sums four zeroes. The chokepoint then ships that zero as a computed price: a
    real, billed 1500-token gpt-4o call presented to the customer as free. This is
    the exact failure the chokepoint was built to prevent ("no path can ship a
    tokens-only cost.record"), inverted.

    The honest outcome is a cost that is ABSENT plus an explicit reason, so a
    fail-closed reader can tell "unknowable from this payload" apart from "a priced
    model's cost was dropped" (the A11 lock's job).

    Bite proof: revert the unpriceable-shape check and these go RED with 0.0.
    """

    def test_a_totals_only_record_ships_no_fabricated_zero(self) -> None:
        c = _collector()
        c.emit(
            "cost.record",
            {"provider": "openai", "model": "gpt-4o", "total_tokens": 1500},
            span_id="s1",
        )
        payload = _cost_event(c)["payload"]
        assert payload.get("cost_usd") != 0.0, (
            "the chokepoint fabricated $0.00 for a real billed 1500-token gpt-4o call"
        )
        assert payload.get("cost_usd") is None
        # the honest token count must survive — we suppress the price, not the usage
        assert payload["total_tokens"] == 1500

    def test_an_unpriceable_shape_is_marked_so_it_is_not_read_as_a_dropped_price(self) -> None:
        c = _collector()
        c.emit(
            "cost.record",
            {"provider": "openai", "model": "gpt-4o", "total_tokens": 1500},
            span_id="s1",
        )
        payload = _cost_event(c)["payload"]
        assert payload.get("cost_status") == "unpriceable_token_shape", (
            "a priced model with no priceable dimension must say WHY it has no cost, "
            "or it is indistinguishable from the A11 dropped-price bug"
        )

    def test_the_flat_framework_vocabulary_reaches_the_same_verdict(self) -> None:
        c = _collector()
        c.emit(
            "cost.record",
            {"provider": "openai", "model": "gpt-4o", "tokens_total": 1500},
            span_id="s1",
        )
        payload = _cost_event(c)["payload"]
        assert payload.get("cost_usd") != 0.0
        assert payload.get("cost_status") == "unpriceable_token_shape"

    def test_an_unpriced_model_is_not_marked_unpriceable(self) -> None:
        # BOUNDARY: a local/custom model has no rate at all. That is the
        # pre-existing, legal tokens-only case (A11 allows a None cost there) and
        # must NOT acquire the unpriceable-shape marker, which is specifically
        # about a PRICED model we could not price from this payload.
        c = _collector()
        c.emit(
            "cost.record",
            {"provider": "ollama", "model": "my-local-llama", "total_tokens": 1500},
            span_id="s1",
        )
        payload = _cost_event(c)["payload"]
        assert payload.get("cost_usd") is None
        assert "cost_status" not in payload

    def test_a_priceable_record_is_untouched(self) -> None:
        # BOUNDARY: the fix must not disturb the path the chokepoint exists for.
        c = _collector()
        c.emit(
            "cost.record",
            {"provider": "openai", "model": "gpt-4o", "prompt_tokens": 1000, "completion_tokens": 500},
            span_id="s1",
        )
        payload = _cost_event(c)["payload"]
        assert payload["cost_usd"] == calculate_cost(
            "gpt-4o", NormalizedTokenUsage(prompt_tokens=1000, completion_tokens=500)
        )
        assert payload["cost_usd"] > 0
        assert "cost_status" not in payload


class TestPartialTokenShape:
    """The DETECTABLE under-report (LAY-3622 / F4): a real cost that understates.

    Cluster A killed the fabricated ``0.0``. The remaining case is the opposite
    shape: we DO price the record, but the provider reported more billed tokens than
    any rate was applied to. The canonical producer is Gemini, which reports
    ``thoughtsTokenCount`` OUTSIDE ``candidatesTokenCount`` while
    ``total_token_count`` includes it — thinking tokens bill at the output rate and
    none of them were priced.

    The chokepoint MARKS this and changes no money. Inventing a rate for tokens we
    cannot attribute would be a guess billed to a customer, which is the same class
    of dishonesty as A4b's derived zero, just in the other direction. Reading
    ``reasoning_tokens`` into the formula instead would DOUBLE-bill OpenAI (whose
    ``reasoning_tokens`` is a breakdown of ``completion_tokens``) and bill Anthropic
    from a ``chars // 4`` estimate — see ``pricing._cost_from_rates``.
    """

    #: The real Gemini thinking shape: total_token_count includes the 42 thought
    #: tokens, candidates_token_count does not.
    GEMINI_THINKING = {
        "provider": "google",
        "model": "gemini-2.5-pro",
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 57,
        "reasoning_tokens": 42,
    }

    def test_a_reported_total_beyond_the_priced_sides_is_MARKED(self) -> None:
        c = _collector()
        c.emit("cost.record", dict(self.GEMINI_THINKING), span_id="s1")
        payload = _cost_event(c)["payload"]
        assert payload["cost_status"] == "partial_token_shape"
        assert payload["unpriced_tokens"] == 42, "the magnitude must ride along, not be re-derivable only"
        assert payload["cost_usd"] > 0, "the cost we CAN compute is still reported"

    def test_the_marker_changes_NO_money(self) -> None:
        # The whole point of mark-only. If a future change starts pricing the
        # residual, this is the assertion that makes it a deliberate act.
        c = _collector()
        c.emit("cost.record", dict(self.GEMINI_THINKING), span_id="s1")
        marked = _cost_event(c)["payload"]["cost_usd"]
        assert marked == calculate_cost(
            "gemini-2.5-pro", NormalizedTokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=57)
        )

    def test_a_complete_token_shape_is_NOT_marked(self) -> None:
        # VACUITY CONTROL: if everything were marked, the marker would mean nothing.
        c = _collector()
        c.emit(
            "cost.record",
            {"provider": "openai", "model": "gpt-4o", "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            span_id="s1",
        )
        payload = _cost_event(c)["payload"]
        assert "cost_status" not in payload
        assert "unpriced_tokens" not in payload

    def test_a_fully_cached_turn_is_NOT_marked(self) -> None:
        # THE FALSE POSITIVE the obvious arithmetic produces. A turn reported as
        # fully cached carries prompt_tokens=0 with cached_tokens=1500, and
        # `prompt + completion + cache_creation` reads that as 1500 unpriced tokens —
        # when in fact every one of them was priced through the cached leg. This
        # shape is real: test_pricing.py's `[cached-only]` case.
        c = _collector()
        c.emit(
            "cost.record",
            {
                "provider": "openai",
                "model": "gpt-4o",
                "prompt_tokens": 0,
                "cached_tokens": 1500,
                "total_tokens": 1500,
            },
            span_id="s1",
        )
        payload = _cost_event(c)["payload"]
        assert payload["cost_usd"] > 0
        assert "cost_status" not in payload, "a fully-cached turn was wrongly reported as a partial under-price"

    def test_a_VENDOR_reported_cost_is_never_marked(self) -> None:
        # A gateway's own charge (langfuse billing figure, OpenRouter usage
        # accounting) is a billed FACT, not our estimate of one, so a token gap
        # against it says nothing about its accuracy.
        c = _collector()
        c.emit(
            "cost.record",
            {**self.GEMINI_THINKING, "cost_usd": 0.99, "cost_source": "langfuse"},
            span_id="s1",
        )
        payload = _cost_event(c)["payload"]
        assert payload["cost_usd"] == 0.99
        assert "cost_status" not in payload

    def test_a_totals_only_shape_stays_UNPRICEABLE_not_partial(self) -> None:
        # The two markers must not blur: this one has no cost at all, so calling it
        # "partial" would turn an honestly-withheld cost into an apparently-real one.
        c = _collector()
        c.emit("cost.record", {"provider": "openai", "model": "gpt-4o", "total_tokens": 1500}, span_id="s1")
        payload = _cost_event(c)["payload"]
        assert payload["cost_status"] == "unpriceable_token_shape"
        assert payload.get("cost_usd") is None
        assert "unpriced_tokens" not in payload

    def test_an_adapter_supplied_cost_status_is_not_clobbered(self) -> None:
        c = _collector()
        c.emit("cost.record", {**self.GEMINI_THINKING, "cost_usd": 0.5, "cost_status": "vendor_says_so"}, span_id="s1")
        assert _cost_event(c)["payload"]["cost_status"] == "vendor_says_so"
