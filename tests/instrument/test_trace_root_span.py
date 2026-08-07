"""Every trace must carry a real, captured root span (LAY-364x / trace-root).

Companion to atlas-app PR #2042 (the universal FE span-tree). The FE had to
SYNTHESIZE a root for ~55% of real traces because the SDK routinely emits leaf
events (``model.invoke``/``cost.record``) whose ``parent_span_id`` points at an
AMBIENT span — the ``trace_context`` / ``_begin_run`` / bare-provider root — that
the SDK never emitted an event for. That ambient span is therefore a "dangling
parent": no owning event, so the trace has no captured root.

The ``@trace`` decorator does NOT have this problem: it emits ``agent.input`` /
``agent.output`` ON its root span, so the root is captured. The gap is every
OTHER collector-establishing path (``trace_context``, framework ``_begin_run``,
bare provider usage) where only child/leaf events land.

The fix: at ``flush()``, if the events reference exactly ONE dangling parent
span (and no captured root already exists), the collector emits ONE lightweight,
content-free ``trace.root`` marker (a dedicated structural event type, NOT an
agent.lifecycle event) ON that dangling span so every leaf event's parent
resolves to a captured span and the FE never needs a synthetic root. It must NOT
fire for decorator traces (would double-root the clean 36.5%), must carry no
PII/content, must emit regardless of the L1 layer / capture_content (the tree
must always have a root), and must flow through the attestation chain.

Bite: revert the ``flush()`` hook and every "root is captured" assertion here
goes RED again.
"""

from __future__ import annotations

from typing import Any, Set, Dict, List

import pytest

from layerlens.instrument import trace, trace_context
from layerlens.instrument._collector import TraceCollector
from layerlens.instrument._capture_config import CaptureConfig

from ._event_schema import KNOWN_EVENT_TYPES

# The synthetic root is a real, uploaded event — it participates in the schema
# lock like everything else.
pytestmark = pytest.mark.invariant


# ---------------------------------------------------------------------------
# FE-mirroring root resolver — the exact "dangling parent" test the frontend
# buildSpanTree performs (atlas-app trace-spans.ts). A trace has a CAPTURED
# root iff there is at least one event whose span is a root (parent is null,
# self-referential, or itself captured up the chain) — i.e. NOT every top span
# hangs off a parent that has no owning event.
# ---------------------------------------------------------------------------


def _captured_span_ids(events: List[Dict[str, Any]]) -> Set[str]:
    return {e["span_id"] for e in events if e.get("span_id")}


def _dangling_parents(events: List[Dict[str, Any]]) -> Set[str]:
    """Parent span_ids referenced by an event but never captured as a span
    (and not self-referential). These are the FE's synthesized-root triggers."""
    captured = _captured_span_ids(events)
    dangling: Set[str] = set()
    for e in events:
        parent = e.get("parent_span_id")
        if parent is None:
            continue
        if parent == e.get("span_id"):
            continue  # self-parent == a real root marker
        if parent not in captured:
            dangling.add(parent)
    return dangling


def _has_captured_root(events: List[Dict[str, Any]]) -> bool:
    """True iff the trace has a real root the FE would NOT need to synthesize:
    at least one captured span whose parent is null/self/captured, AND no
    orphaned leaf hanging off a dangling parent."""
    if not events:
        return False
    return len(_dangling_parents(events)) == 0


# ---------------------------------------------------------------------------
# A provider-style leaf emitter that mimics emit_llm_events: it emits on a
# FRESH span whose parent is the AMBIENT span (the current span id) — exactly
# the shape the real provider adapters produce.
# ---------------------------------------------------------------------------


def _emit_provider_leaf(model: str = "gpt-4o") -> None:
    # Mirrors emit_llm_events: model.invoke + cost.record on a new span,
    # parented to the ambient span via the public emit() path... but emit()
    # uses the current span as its OWN span_id. Provider helpers instead mint
    # a fresh child span. Reproduce that precisely with the collector API.
    from layerlens.instrument._context import _current_span_id, _current_collector

    collector = _current_collector.get()
    assert collector is not None, "test must run inside a collector context"
    parent = _current_span_id.get()  # the ambient/root span (no event emitted on it)
    import uuid

    span_id = uuid.uuid4().hex[:16]
    collector.emit(
        "model.invoke",
        {"provider": "openai", "model": model, "latency_ms": 5.0},
        span_id=span_id,
        parent_span_id=parent,
    )
    collector.emit(
        "cost.record",
        {
            "provider": "openai",
            "model": model,
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "cost_usd": 0.0001,
        },
        span_id=span_id,
        parent_span_id=parent,
    )


ROOT_TYPE = "trace.root"


def _root_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The synthesized root marker: a dedicated ``trace.root`` structural event
    whose payload flags it synthesized. It is NOT an agent.lifecycle event — the
    root marker has its own content-free type so it doesn't pollute the real
    agent lifecycle stream."""
    return [e for e in events if e["event_type"] == ROOT_TYPE and (e["payload"] or {}).get("synthesized") is True]


# ===========================================================================
# 1. The RED case: provider-only usage under trace_context is dangling today.
#    After the fix, a real captured root exists.
# ===========================================================================


class TestProviderOnlyTraceGetsRealRoot:
    def test_trace_context_provider_only_has_captured_root(self, mock_client: Any, capture_trace_list: Any) -> None:
        # A user wraps a shared context (or uses a bare provider adapter) WITHOUT
        # @stratix.trace — the ambient root span emits no event of its own.
        with trace_context(mock_client):
            _emit_provider_leaf()

        assert len(capture_trace_list) == 1
        events = capture_trace_list[0]["events"]
        # Before the fix: the two leaf events hang off the ambient root span,
        # which has no owning event -> exactly one dangling parent -> the FE
        # would synthesize a root. After the fix: a captured root exists.
        assert _has_captured_root(events), (
            f"trace has no captured root — dangling parents {_dangling_parents(events)}; "
            "the FE would have to synthesize a root (the bug this fixes)"
        )

    def test_a_real_root_event_is_emitted_on_the_dangling_span(self, mock_client: Any, capture_trace_list: Any) -> None:
        with trace_context(mock_client):
            _emit_provider_leaf()

        events = capture_trace_list[0]["events"]
        roots = _root_events(events)
        assert len(roots) == 1, f"expected exactly one synthesized root marker, got {len(roots)}"
        root = roots[0]
        # It must own the span the leaves pointed at, so their parent resolves.
        leaf_parents = {e["parent_span_id"] for e in events if e["event_type"] in ("model.invoke", "cost.record")}
        assert root["span_id"] in leaf_parents, "root marker must sit on the span the leaves parent to"
        # The root is itself a root: parent null or self.
        assert root["parent_span_id"] in (None, root["span_id"])

    def test_no_event_is_dropped_by_the_root_synthesis(self, mock_client: Any, capture_trace_list: Any) -> None:
        # Two provider calls, each minting a fresh child span parented to the
        # ambient (never-captured) root — the real provider-adapter shape.
        with trace_context(mock_client):
            _emit_provider_leaf("gpt-4o")
            _emit_provider_leaf("gpt-4o-mini")

        events = capture_trace_list[0]["events"]
        types = sorted(e["event_type"] for e in events)
        # Both calls' model.invoke + cost.record survive (4) + exactly one root.
        assert types.count("model.invoke") == 2
        assert types.count("cost.record") == 2
        assert len(_root_events(events)) == 1
        assert len(events) == 5
        # And the root de-dangles both leaves.
        assert _has_captured_root(events)

    def test_bare_collector_provider_only_gets_root(self, mock_client: Any, capture_trace_list: Any) -> None:
        """The lowest-level path: a collector with leaf events whose common
        parent was never captured (a bare provider call). flush() must root it."""
        collector = TraceCollector(mock_client, CaptureConfig.standard())
        ambient = "ambientspan00000"
        collector.emit("model.invoke", {"model": "gpt-4o", "latency_ms": 1.0}, span_id="leaf1", parent_span_id=ambient)
        collector.emit(
            "cost.record",
            # A coherent priceable shape: this test is about root synthesis, but a
            # hand-written ``cost_usd: 0.0`` on a priced model with real tokens is a
            # state no honest emit path can produce (LAY-3622 / A4b), and the
            # fabricated-cost invariant rightly rejects it. Let the chokepoint price it.
            {"provider": "openai", "model": "gpt-4o", "prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            span_id="leaf2",
            parent_span_id=ambient,
        )
        collector.flush()

        events = capture_trace_list[0]["events"]
        assert _has_captured_root(events)
        roots = _root_events(events)
        assert len(roots) == 1 and roots[0]["span_id"] == ambient


# ===========================================================================
# 2. No regression: decorator traces already have a captured root and must NOT
#    get a second synthesized one.
# ===========================================================================


class TestDecoratorTracesAreNotDoubleRooted:
    def test_trace_decorator_has_captured_root_without_synthesis(
        self, mock_client: Any, capture_trace_list: Any
    ) -> None:
        @trace(mock_client)
        def run() -> str:
            _emit_provider_leaf()
            return "done"

        run()

        events = capture_trace_list[0]["events"]
        assert _has_captured_root(events), "decorator trace should already have a captured root"
        # agent.input/output ARE the root; no synthesized marker should be added.
        assert _root_events(events) == [], "decorator trace must not be double-rooted"

    def test_framework_root_event_is_not_double_rooted(self, mock_client: Any, capture_trace_list: Any) -> None:
        """A path that already emits an event ON its own root span (e.g. an
        adapter lifecycle) must not get a second synthesized root."""
        collector = TraceCollector(mock_client, CaptureConfig.standard())
        root = "frameworkroot000"
        # An event captured ON the root span (self-parented, as adapters do).
        collector.emit("agent.lifecycle", {"lifecycle_action": "start"}, span_id=root, parent_span_id=root)
        collector.emit("model.invoke", {"model": "gpt-4o", "latency_ms": 1.0}, span_id="child1", parent_span_id=root)
        collector.flush()

        events = capture_trace_list[0]["events"]
        assert _has_captured_root(events)
        assert _root_events(events) == [], "a captured root already exists — no synthesis"


# ===========================================================================
# 3. The root marker is content-free and layer-independent.
# ===========================================================================


class TestRootMarkerIsHonestAndAlwaysPresent:
    def test_root_carries_no_content_or_fabricated_agent(self, mock_client: Any, capture_trace_list: Any) -> None:
        with trace_context(mock_client):
            _emit_provider_leaf()

        root = _root_events(capture_trace_list[0]["events"])[0]
        payload = root["payload"] or {}
        # No fabricated agent identity (would falsely populate the FE Agent col).
        assert "agent_name" not in payload
        # No content keys.
        for banned in ("input", "output", "messages", "content", "prompt"):
            assert banned not in payload
        # It IS registered vocabulary.
        assert root["event_type"] in KNOWN_EVENT_TYPES

    def test_root_emitted_even_when_l1_disabled(self, mock_client: Any, capture_trace_list: Any) -> None:
        """The tree must ALWAYS have a root — even under a config that turns off
        L1 agent-io. ``trace.root`` is a dedicated STRUCTURAL type (always-enabled,
        not gated to any capture layer, unlike agent.*), so it survives an L1-off
        config: a synthesized root is structure, not content."""
        cfg = CaptureConfig(l1_agent_io=False)
        collector = TraceCollector(mock_client, cfg)
        ambient = "ambientspan00000"
        collector.emit("model.invoke", {"model": "gpt-4o", "latency_ms": 1.0}, span_id="leaf1", parent_span_id=ambient)
        collector.flush()

        events = capture_trace_list[0]["events"]
        roots = _root_events(events)
        assert len(roots) == 1, "root must be emitted regardless of the L1 layer toggle"
        assert _has_captured_root(events)

    def test_root_emitted_under_capture_content_false(self, mock_client: Any, capture_trace_list: Any) -> None:
        cfg = CaptureConfig(capture_content=False)
        collector = TraceCollector(mock_client, cfg)
        ambient = "ambientspan00000"
        collector.emit("model.invoke", {"model": "gpt-4o", "latency_ms": 1.0}, span_id="leaf1", parent_span_id=ambient)
        collector.flush()
        assert len(_root_events(capture_trace_list[0]["events"])) == 1


# ===========================================================================
# 4. Attestation still verifies with the injected root event.
# ===========================================================================


class TestAttestationHoldsWithSynthesizedRoot:
    def test_chain_verifies_end_to_end(self, mock_client: Any, capture_trace_list: Any) -> None:
        from layerlens.attestation._chain import HashChain
        from layerlens.attestation._verify import verify_trial
        from layerlens.attestation._envelope import HashScope, AttestationEnvelope

        with trace_context(mock_client):
            _emit_provider_leaf()

        payload = capture_trace_list[0]
        att = payload["attestation"]
        # The trace uploaded (not quarantined) -> it has a root_hash.
        assert att.get("root_hash"), "synthesized-root trace must still be attested"

        # Rebuild the chain over the UPLOADED events (root included) and confirm
        # the root_hash matches — i.e. the synthesized root is inside the chain.
        rebuilt = HashChain()
        for e in payload["events"]:
            rebuilt.add_event(e)
        envelopes = rebuilt.envelopes
        trial = AttestationEnvelope(hash=att["root_hash"], scope=HashScope.TRIAL, previous_hash=envelopes[-1].hash)
        result = verify_trial(envelopes, trial)
        assert result.trial_hash_valid, f"root hash does not cover the synthesized root: {result.errors}"

    def test_number_of_chained_events_matches_uploaded_events(self, mock_client: Any, capture_trace_list: Any) -> None:
        with trace_context(mock_client):
            _emit_provider_leaf()
        payload = capture_trace_list[0]
        chain_events = payload["attestation"]["chain"]["events"]
        assert len(chain_events) == len(payload["events"]), "every uploaded event (root included) must be chained"
