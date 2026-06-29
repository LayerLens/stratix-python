"""cost_usd must be computed for cost.record emitted via _fire() (not just _emit).

The central pricing hook lived only in BaseFrameworkAdapter._emit, but 6 adapters
emit cost.record through their own _fire() -> collector.emit, bypassing it — so
the headline "framework traces now have cost" fix silently missed the busiest
agent frameworks. The hook is now also applied inside each _fire(); these tests
drive the REAL per-adapter _fire and assert cost_usd is filled for a priced model.

Bite proof: remove the `_price_cost_record` call from a _fire() and the matching
test fails (cost_usd None).
"""

from __future__ import annotations

import importlib

import pytest

from layerlens.instrument._context import RunState, _current_run
from layerlens.instrument._collector import TraceCollector
from layerlens.instrument._capture_config import CaptureConfig
from layerlens.instrument.adapters.providers.pricing import PRICING, calculate_cost
from layerlens.instrument.adapters.providers.token_usage import NormalizedTokenUsage

# Adapters #23 moved to per-run collector isolation (LAY-3576): their _fire no
# longer reads ``self._collector``. crewai takes the run as an explicit arg;
# google_adk/smolagents/strands resolve it from the ``_current_run`` ContextVar.
# autogen/llamaindex still emit via the instance collector.
_PER_RUN_EXPLICIT = {"crewai"}
_PER_RUN_CONTEXTVAR = {"google_adk", "smolagents", "strands"}


def _drive_fire(adapter: object, module_name: str, collector: TraceCollector, payload: dict) -> None:
    """Invoke the adapter's real ``_fire("cost.record", ...)`` so the emitted
    event lands in *collector*, honouring each adapter's collector mechanics."""
    if module_name in _PER_RUN_EXPLICIT:
        run = RunState(collector=collector, root_span_id="s1")
        adapter._fire(run, "cost.record", payload, span_id="s1")  # type: ignore[attr-defined]
    elif module_name in _PER_RUN_CONTEXTVAR:
        token = _current_run.set(RunState(collector=collector, root_span_id="s1"))
        try:
            adapter._fire("cost.record", payload, span_id="s1")  # type: ignore[attr-defined]
        finally:
            _current_run.reset(token)
    else:
        adapter._fire("cost.record", payload, span_id="s1")  # type: ignore[attr-defined]


# Run as a required CI gate via `-m invariant` (see .github/workflows/invariants.yaml).
pytestmark = pytest.mark.invariant

_ADAPTERS = [
    ("strands", "StrandsAdapter"),
    ("google_adk", "GoogleADKAdapter"),
    ("crewai", "CrewAIAdapter"),
    ("llamaindex", "LlamaIndexAdapter"),
    ("autogen", "AutoGenAdapter"),
    ("smolagents", "SmolAgentsAdapter"),
]


def _expected_cost() -> float:
    usage = NormalizedTokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
    cost = calculate_cost("gpt-4", usage, PRICING)
    assert cost is not None and cost > 0, "gpt-4 must be priced for this test to be meaningful"
    return cost


@pytest.mark.parametrize("module_name,class_name", _ADAPTERS)
def test_fire_prices_cost_record(module_name: str, class_name: str) -> None:
    mod = importlib.import_module(f"layerlens.instrument.adapters.frameworks.{module_name}")
    cls = getattr(mod, class_name)

    # Bypass __init__ — drive the real _fire() with only the attrs it needs.
    adapter = cls.__new__(cls)
    adapter._config = CaptureConfig.standard()
    adapter._root_span_id = None  # autogen._fire reads this
    collector = TraceCollector(object(), CaptureConfig.standard())
    adapter._collector = collector
    if module_name == "llamaindex":  # _fire resolves the collector via the span tree
        adapter._collector_for = lambda _sid: collector  # type: ignore[attr-defined]
        adapter._parent_of = lambda _sid: None  # type: ignore[attr-defined]

    payload = {
        "framework": module_name,
        "model": "gpt-4",
        "tokens_prompt": 100,
        "tokens_completion": 50,
        "tokens_total": 150,
    }
    _drive_fire(adapter, module_name, collector, payload)

    records = [e for e in collector.events if e["event_type"] == "cost.record"]
    assert records, f"{module_name}: _fire emitted no cost.record"
    assert records[0]["payload"].get("cost_usd") == _expected_cost(), (
        f"{module_name}: cost_usd not computed through _fire (bypassed the pricing hook)"
    )


def test_strands_prices_real_bedrock_model() -> None:
    """strands runs Bedrock model ids absent from the default PRICING; its
    pricing_table=BEDROCK_PRICING must make them resolve (the gpt-4 cases above
    mask this). Bite: drop StrandsAdapter.pricing_table -> cost_usd None."""
    mod = importlib.import_module("layerlens.instrument.adapters.frameworks.strands")
    from layerlens.instrument.adapters.providers.pricing import BEDROCK_PRICING

    model = "amazon.nova-lite-v1:0"
    usage = NormalizedTokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
    assert calculate_cost(model, usage, PRICING) is None, (
        "test premise broken: the Bedrock model is unexpectedly in the default PRICING"
    )

    adapter = mod.StrandsAdapter.__new__(mod.StrandsAdapter)
    adapter._config = CaptureConfig.standard()
    collector = TraceCollector(object(), CaptureConfig.standard())
    adapter._collector = collector
    _drive_fire(
        adapter,
        "strands",
        collector,
        {"framework": "strands", "model": model, "tokens_prompt": 100, "tokens_completion": 50, "tokens_total": 150},
    )
    rec = [e for e in collector.events if e["event_type"] == "cost.record"][0]
    expected = calculate_cost(model, usage, {**PRICING, **BEDROCK_PRICING})
    assert expected and expected > 0
    assert rec["payload"].get("cost_usd") == expected, "strands did not price its real Bedrock model"


def test_fire_leaves_cost_usd_alone_when_already_set() -> None:
    """Adapters that already compute cost_usd (bedrock_agents/langfuse) must not be
    overwritten — the hook only fills when absent."""
    mod = importlib.import_module("layerlens.instrument.adapters.frameworks.strands")
    adapter = mod.StrandsAdapter.__new__(mod.StrandsAdapter)
    adapter._config = CaptureConfig.standard()
    collector = TraceCollector(object(), CaptureConfig.standard())
    adapter._collector = collector
    _drive_fire(adapter, "strands", collector, {"model": "gpt-4", "tokens_total": 150, "cost_usd": 0.123})
    rec = [e for e in collector.events if e["event_type"] == "cost.record"][0]
    assert rec["payload"]["cost_usd"] == 0.123, "pre-set cost_usd was overwritten"
