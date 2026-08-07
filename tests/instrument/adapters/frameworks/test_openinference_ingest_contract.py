"""Offline ingest-acceptance oracle for the OpenInference adapter (LAY-3622 C4).

"Survives ingest" had no bare-CI proof. The one live lane that claimed it asserted
nothing server-side, and nothing anywhere pinned the per-event-type required-field
contract — even though that contract is load-bearing in prose across at least four
adapters. This lane closes it: every event the adapter emits from the shared 24-span
corpus must carry the fields its type requires at ingest.

WHERE THE CONTRACT COMES FROM (it is not invented here, and no longer lives here):

The table itself now lives in ONE place in ``src`` —
:mod:`layerlens.instrument._ingest_contract` — which records its provenance (atlas
delegates it; ateam's ``SCHEMA_REGISTRY`` enforces it), its split into an enforced
HARD core and an ADVISORY tier, and the measured non-compliance behind that split.
Read that module first. This lane is a CONSUMER of it: it holds ``openinference``
to the strict HARD + ADVISORY bar, which is the bar the adapter was authored
against.

It used to hold its own hand-copied copy of the table, which is how the contract
came to contradict itself across the tree (LAY-3622 F1).

HONEST LIMITS — read both before citing this lane:

1. Because our server does no per-event validation, this proves the events are
   WELL-FORMED against the stated contract. It does not prove a server rejects a
   malformed one — no server does. The live lane's server-side read-back
   (``tests/e2e/live/_framework_harness.py``) proves the complementary half: that
   what we send is actually stored.

2. **Most of what this lane asserts is ADVISORY platform-wide, and the lane is
   scoped to openinference on purpose.** 717 of 905 committed events — 79% — do not
   satisfy the strict bar (measured 2026-08-03; the per-type breakdown lives in
   ``_ingest_contract.MEASURED_NON_COMPLIANCE``). openinference is the compliant
   OUTLIER. Only ``_ingest_contract.HARD_REQUIRED`` is enforced anywhere, by the A11
   cost lock; everything else this lane checks is ateam's bar, which we hold
   openinference to deliberately.

   So: a green run here means THIS adapter is portable to a strict consumer. It is
   not evidence that the platform is ingest-valid, and it never was.
"""

from __future__ import annotations

import os
import json
from typing import Any, Dict, List

import pytest

from layerlens.instrument._capture_config import CaptureConfig
from layerlens.instrument._ingest_contract import (
    HARD_REQUIRED,
    REGISTERED_EVENT_TYPES,
    describe,
    unsatisfied,
    requirements_for,
)
from layerlens.instrument.adapters.frameworks.openinference import (
    OpenInferenceAdapter,
    span_to_events,
    otlp_json_to_span_records,
)

_CORPUS_PATH = os.path.join(os.path.dirname(__file__), "oi_conformance", "spans.otlp.json")


def _strict_requirements(event_type: str) -> tuple:
    """HARD + ADVISORY — the strict (ateam) bar this adapter is held to.

    Sourced from ``_ingest_contract`` rather than restated here: a second copy of
    this table is what let the contract contradict itself (LAY-3622 F1).
    """
    return requirements_for(event_type, include_advisory=True)


def _corpus_records() -> List[Dict[str, Any]]:
    with open(_CORPUS_PATH) as fh:
        return otlp_json_to_span_records(json.load(fh))


def _adapter_events(*, capture_content: bool) -> List[Dict[str, Any]]:
    """Every event the REAL adapter emits over the corpus, via ingest_span.

    Goes through ``ingest_span`` rather than ``span_to_events`` so ``cost.record``
    is included — it is emitted from the ingest path, outside the conformance
    lane's pinned boundary, which is exactly how Cluster A's defect stayed hidden.
    """
    captured: List[Dict[str, Any]] = []

    class _Client:
        class traces:
            @staticmethod
            def upload(*args: Any, **kwargs: Any) -> Any:
                return type("R", (), {"trace_ids": ["t1"]})()

    adapter = OpenInferenceAdapter(
        _Client(), capture_config=CaptureConfig.full() if capture_content else CaptureConfig()
    )
    adapter.connect()
    for record in _corpus_records():
        adapter.ingest_span(record)
    for collector in adapter._collectors.values():
        captured.extend(collector.events)
    return captured


CORPUS_EVENTS = _adapter_events(capture_content=True)


def test_the_corpus_actually_produced_events() -> None:
    # VACUITY GUARD: every assertion below is over CORPUS_EVENTS, so an empty or
    # truncated corpus would make this whole module pass while proving nothing.
    assert len(_corpus_records()) == 24, "the shared corpus is not the expected 24 spans"
    assert len(CORPUS_EVENTS) >= 24


def test_every_emitted_type_is_in_the_contract() -> None:
    """An event type with no registry entry is REJECTED outright by ateam's
    normalizer ("Unknown event_type"), so an unlisted type is a real ingest
    failure, not a gap in this test. If the adapter grows a type, this fails until
    the contract is extended deliberately."""
    emitted = {e["event_type"] for e in CORPUS_EVENTS}
    unlisted = sorted(emitted - REGISTERED_EVENT_TYPES)
    assert not unlisted, f"emitted event types absent from the ingest contract: {unlisted}"


@pytest.mark.parametrize("event_type", sorted(REGISTERED_EVENT_TYPES))
def test_required_fields_present_for_every_event_of_each_type(event_type: str) -> None:
    events = [e for e in CORPUS_EVENTS if e["event_type"] == event_type]
    if not events:
        pytest.skip(f"the corpus emits no {event_type}")
    for event in events:
        payload = event["payload"]
        for missing in unsatisfied(_strict_requirements(event_type), payload):
            raise AssertionError(
                f"{event_type} is missing ingest-required {describe(missing)!r} — a strict "
                f"consumer rejects the event and the record is lost. payload keys: {sorted(payload)}"
            )


def test_the_adapter_keeps_ingest_required_text_present_but_empty_under_redaction() -> None:
    """The adapter's own privacy design, asserted at the layer that implements it.

    ``_agent_input_output`` documents that ``input_text``/``output_text`` are
    ingest-REQUIRED and are therefore "present-but-EMPTY" under
    ``capture_content=False`` — recording the turn with empty text rather than
    losing it. This proves the normaliser really does that.
    """
    emitted = 0
    for record in _corpus_records():
        for event_type, payload in span_to_events(record, capture_content=False):
            for missing in unsatisfied(_strict_requirements(event_type), payload):
                raise AssertionError(
                    f"span_to_events dropped ingest-required {describe(missing)!r} from "
                    f"{event_type} under capture_content=False"
                )
            if event_type in ("agent.input", "agent.output"):
                emitted += 1
    assert emitted, "the corpus emits no agent.input/agent.output — this would be vacuous"


def test_the_collector_backstop_preserves_ingest_required_text() -> None:
    """FIXED in LAY-3622 F2 — this was a strict xfail, removed deliberately.

    The defect this lane found: the adapter sets ``input_text``/``output_text`` to
    ``""`` so the turn stays valid for a strict consumer under
    ``capture_content=False``, and the COLLECTOR-tier backstop then deleted both keys
    outright (they are in ``_CONTENT_KEYS``) — so the DEFAULT privacy configuration
    emitted agent turns a strict consumer would reject, defeating the invariant
    ``openinference.py`` documents. Nothing broke on our platform (atlas does no
    per-event validation by design), which is exactly why it went unnoticed.

    Fixed at the backstop rather than per adapter: ``_is_content_free``
    (``_capture_config.py``) keeps a content key whose value is already empty.
    Privacy-neutral by construction, and the privacy guarantee for POPULATED values
    is unchanged — ``tests/instrument/test_redaction_backstop.py`` asserts both
    halves, including that a non-empty value is still deleted.

    The xfail was strict precisely so it would FAIL on being fixed rather than rot
    into a passing test nobody re-read.
    """
    for event in _adapter_events(capture_content=False):
        for missing in unsatisfied(_strict_requirements(event["event_type"]), event["payload"]):
            raise AssertionError(
                f"capture_content=False stripped ingest-required {describe(missing)!r} from "
                f"{event['event_type']} — the default config would lose this event"
            )


class TestCostRecordContract:
    """``cost.record`` requires ``cost_usd``, which collides with Cluster A.

    ateam's registry requires ``cost_usd`` on every ``cost.record``. Cluster A
    established that a totals-only span CANNOT be priced, so there are only two
    contract-clean options: suppress the event, or declare why the cost is absent.

    This adapter SUPPRESSES (``_emit_cost_record`` returns before emitting when
    pricing yields nothing), so every ``cost.record`` it emits carries a real
    ``cost_usd`` — the strictest reading of the contract. The shared chokepoint
    instead marks ``cost_status="unpriceable_token_shape"`` for adapters that build
    the record for their own reasons, which is a deliberate, documented relaxation:
    the tokens are worth keeping and the marker makes the absence explicit rather
    than silent. Both are asserted here so a future change to either is visible.
    """

    def test_every_emitted_cost_record_carries_a_real_cost(self) -> None:
        records = [e["payload"] for e in CORPUS_EVENTS if e["event_type"] == "cost.record"]
        assert records, "the corpus emits no cost.record — this test would be vacuous"
        for payload in records:
            assert payload.get("cost_usd") is not None, (
                "a cost.record without cost_usd is rejected at ingest; the adapter must "
                "suppress an unpriceable record, not emit a bare one"
            )
            assert payload["cost_usd"] > 0

    def test_a_totals_only_span_emits_no_cost_record_so_the_contract_holds(self) -> None:
        # The Cluster A shape, checked against the INGEST contract rather than the
        # pricing arithmetic: suppression is what keeps every emitted cost.record
        # ingest-valid. A "leave cost_usd unset but still emit" fix would have
        # produced an ingest-INVALID event instead.
        events = span_to_events(
            {
                "span_kind": "LLM",
                "name": "openai.chat",
                "attributes": {
                    "openinference.span.kind": "LLM",
                    "llm.model_name": "gpt-4o",
                    "llm.token_count.total": 1500,
                },
                "trace_id": "aa" * 16,
                "span_id": "bb" * 8,
            },
            capture_content=True,
        )
        assert [t for t, _ in events] == ["model.invoke"]
        invoke = events[0][1]
        # model_name still present (ingest-required) AND the honest token total kept.
        assert invoke["model_name"] == "gpt-4o"
        assert invoke["total_tokens"] == 1500

    def test_the_shared_chokepoints_marker_satisfies_the_contract(self) -> None:
        """The two artifacts this branch added must not contradict each other.

        openinference SUPPRESSES an unpriceable cost.record, so its own events always
        carry a real cost. But the shared chokepoint takes the other route for the
        ~20 adapters that build a cost.record for their own reasons: it emits the
        record with ``cost_status="unpriceable_token_shape"`` and no cost. Stating
        the requirement as bare ``cost_usd`` declared that output ingest-invalid —
        i.e. this lane would have condemned the fix in the same PR that shipped it.

        The contract is the disjunction the A11 lock actually enforces: a real cost
        OR an explicit reason there is none. Both are asserted here so the two can
        never drift apart again.
        """
        from layerlens.instrument._collector import TraceCollector
        from layerlens.instrument._capture_config import CaptureConfig as _CC

        collector = TraceCollector(object(), _CC(capture_content=True))
        collector.emit(
            "cost.record",
            {"provider": "openai", "model": "gpt-4o", "total_tokens": 1500},
            span_id="s1",
        )
        payload = [e for e in collector.events if e["event_type"] == "cost.record"][0]["payload"]
        assert payload.get("cost_usd") is None
        assert payload["cost_status"] == "unpriceable_token_shape"
        for missing in unsatisfied(HARD_REQUIRED["cost.record"], payload):
            raise AssertionError(
                f"the chokepoint's own marker output violates the ENFORCED core of the "
                f"contract ({describe(missing)!r}) — the two would contradict each other"
            )
