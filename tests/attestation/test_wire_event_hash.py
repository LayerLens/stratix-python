"""Per-event attestation hash on the wire (OTLP-conformance step 6).

The hash chain historically lived ONLY in the parallel
``attestation.chain.events[]`` array, correlated to ``events[]`` purely by
position. This suite locks the fix: each wire event carries its own ``hash`` +
``previous_hash`` so a consumer (ateam) can verify per-event and record
``origin='sdk'`` without relying on index alignment.

The load-bearing test is ``test_attached_hash_recomputes`` — it replicates
ateam's recompute (strip the attached hash fields, re-inject ``_previous_hash``,
canonicalize, SHA-256) and proves the attached value verifies. If attaching the
hash fed back into its own digest, this test fails.
"""

from __future__ import annotations

from unittest.mock import Mock

from layerlens.attestation._hash import compute_hash
from layerlens.instrument._collector import TraceCollector
from layerlens.instrument._capture_config import CaptureConfig


def _emit_sample_trace() -> TraceCollector:
    c = TraceCollector(Mock(), CaptureConfig.full())
    c.emit("agent.input", {"name": "root", "input": "hello"}, span_id="s1", span_name="root")
    c.emit(
        "model.invoke",
        {"model": "gpt-4o", "provider": "openai", "prompt_tokens": 10, "completion_tokens": 5},
        span_id="s2",
        parent_span_id="s1",
        span_name="llm",
    )
    c.emit("agent.output", {"output": "done"}, span_id="s3", parent_span_id="s1", span_name="root")
    return c


def test_each_wire_event_carries_hash_and_previous_hash():
    events = _emit_sample_trace().to_replay_dict()["events"]
    assert len(events) == 3
    for ev in events:
        assert "hash" in ev, "wire event must carry its own attestation hash"
        assert ev["hash"].startswith("sha256:")
        assert "previous_hash" in ev, "wire event must carry previous_hash for chain linkage"


def test_wire_events_chain_link():
    events = _emit_sample_trace().to_replay_dict()["events"]
    assert events[0]["previous_hash"] is None, "first event has no predecessor"
    for i in range(1, len(events)):
        assert events[i]["previous_hash"] == events[i - 1]["hash"], (
            f"event {i} previous_hash must equal event {i - 1} hash (chain linkage)"
        )


def test_attached_hash_recomputes():
    """The attached hash must verify under ateam's recompute.

    ateam hashes ``{**event_data, '_previous_hash': prev}``. The recompute must
    canonicalize over the ORIGINAL hashed field set — i.e. EXCLUDE the attached
    ``hash``/``previous_hash`` wire fields and re-inject ``_previous_hash`` from
    the attached ``previous_hash`` — or the attached value feeds into its own
    digest and never verifies.
    """
    events = _emit_sample_trace().to_replay_dict()["events"]
    for ev in events:
        original = {k: v for k, v in ev.items() if k not in ("hash", "previous_hash")}
        original["_previous_hash"] = ev["previous_hash"]
        recomputed = compute_hash(original)
        assert recomputed == ev["hash"], (
            "attested hash does not recompute — attaching the hash must not feed "
            "into its own digest (strip hash/previous_hash, re-inject _previous_hash)"
        )


def test_wire_hash_matches_parallel_chain_array():
    """Backward-compat: the parallel attestation.chain.events[] array is kept and
    each attached wire hash mirrors its positional chain entry."""
    replay = _emit_sample_trace().to_replay_dict()
    events = replay["events"]
    chain_events = replay["attestation"]["chain"]["events"]
    assert len(events) == len(chain_events)
    for ev, ce in zip(events, chain_events):
        assert ev["hash"] == ce["hash"]
        assert ev["previous_hash"] == ce["previous_hash"]
