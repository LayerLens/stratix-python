"""Unit tests for the shared handoff metadata helpers.

Covers correctness of :func:`compute_context_hash`,
:func:`make_preview`, :class:`HandoffSequencer`, and
:func:`build_handoff_payload`. Together these power the standardised
``agent.handoff`` contract that the 5 lighter adapters (agno,
ms_agent_framework, openai_agents, llama_index, google_adk) emit.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List

import pytest

from layerlens.instrument.adapters._base.handoff import (
    DEFAULT_PREVIEW_MAX_CHARS,
    HandoffMetadata,
    HandoffSequencer,
    make_preview,
    compute_context_hash,
    build_handoff_payload,
)

# ---------------------------------------------------------------------------
# compute_context_hash
# ---------------------------------------------------------------------------


def test_compute_context_hash_returns_prefixed_sha256() -> None:
    digest = compute_context_hash({"task": "summarise"})
    assert digest.startswith("sha256:")
    # 64 hex chars after the prefix.
    assert len(digest) == len("sha256:") + 64
    int(digest.split(":", 1)[1], 16)  # Validates hex.


def test_compute_context_hash_canonicalises_key_order() -> None:
    """Two semantically-equal contexts must hash to the same value."""
    a = {"task": "x", "agent": "alpha"}
    b = {"agent": "alpha", "task": "x"}
    assert compute_context_hash(a) == compute_context_hash(b)


def test_compute_context_hash_distinguishes_different_contexts() -> None:
    a = compute_context_hash({"task": "x"})
    b = compute_context_hash({"task": "y"})
    assert a != b


def test_compute_context_hash_handles_none() -> None:
    """``None`` and empty dict both hash to the same digest of ``{}``."""
    none_hash = compute_context_hash(None)
    empty_hash = compute_context_hash({})
    assert none_hash == empty_hash
    assert none_hash.startswith("sha256:")


def test_compute_context_hash_coerces_non_jsonable_values() -> None:
    """Non-JSON-native values (sets, custom objects) must not raise."""

    class Custom:
        def __str__(self) -> str:
            return "custom-repr"

    # ``set`` and ``Custom`` would normally break ``json.dumps``; the
    # ``default=str`` hook saves us.
    digest = compute_context_hash({"tags": {"a", "b"}, "obj": Custom()})
    assert digest.startswith("sha256:")


def test_compute_context_hash_is_deterministic_across_calls() -> None:
    state = {"k": 1, "list": [1, 2, 3], "nested": {"x": "y"}}
    h1 = compute_context_hash(state)
    h2 = compute_context_hash(state)
    h3 = compute_context_hash(dict(state))
    assert h1 == h2 == h3


# ---------------------------------------------------------------------------
# make_preview
# ---------------------------------------------------------------------------


def test_make_preview_short_string_returned_as_is() -> None:
    assert make_preview("hello") == "hello"


def test_make_preview_truncates_with_ellipsis() -> None:
    text = "x" * 1000
    out = make_preview(text, max_chars=10)
    assert len(out) == 10
    assert out.endswith("…")
    # First nine chars are content, last char is the ellipsis.
    assert out[:9] == "x" * 9


def test_make_preview_default_cap_is_256() -> None:
    text = "y" * 500
    out = make_preview(text)
    assert len(out) == DEFAULT_PREVIEW_MAX_CHARS == 256
    assert out.endswith("…")


def test_make_preview_none_returns_empty_string() -> None:
    assert make_preview(None) == ""


def test_make_preview_zero_max_returns_empty() -> None:
    assert make_preview("nonempty", max_chars=0) == ""
    assert make_preview("nonempty", max_chars=-1) == ""


def test_make_preview_coerces_non_strings() -> None:
    assert make_preview(42) == "42"
    assert make_preview({"k": "v"}) == "{'k': 'v'}"


def test_make_preview_handles_str_failure() -> None:
    """A faulty ``__str__`` must not propagate."""

    class Broken:
        def __str__(self) -> str:
            raise RuntimeError("nope")

    assert make_preview(Broken()) == "<unrepresentable>"


# ---------------------------------------------------------------------------
# HandoffSequencer
# ---------------------------------------------------------------------------


def test_sequencer_starts_at_one() -> None:
    seq = HandoffSequencer()
    assert seq.current == 0
    assert seq.next() == 1
    assert seq.current == 1


def test_sequencer_is_monotonic() -> None:
    seq = HandoffSequencer()
    values = [seq.next() for _ in range(100)]
    assert values == list(range(1, 101))


def test_sequencer_reset() -> None:
    seq = HandoffSequencer()
    for _ in range(5):
        seq.next()
    assert seq.current == 5
    seq.reset()
    assert seq.current == 0
    assert seq.next() == 1


def test_sequencer_is_thread_safe() -> None:
    """Concurrent ``next()`` calls produce a contiguous, unique set."""
    seq = HandoffSequencer()
    n_threads = 20
    n_per_thread = 200
    results: List[int] = []
    results_lock = threading.Lock()

    def worker() -> None:
        local: List[int] = [seq.next() for _ in range(n_per_thread)]
        with results_lock:
            results.extend(local)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Every value appears exactly once and the full set is contiguous.
    assert len(results) == n_threads * n_per_thread
    assert sorted(results) == list(range(1, n_threads * n_per_thread + 1))
    assert seq.current == n_threads * n_per_thread


def test_sequencer_independence_across_instances() -> None:
    """Two sequencers must not share state."""
    a, b = HandoffSequencer(), HandoffSequencer()
    a.next()
    a.next()
    a.next()
    assert a.current == 3
    assert b.current == 0
    assert b.next() == 1


# ---------------------------------------------------------------------------
# build_handoff_payload
# ---------------------------------------------------------------------------


def test_build_handoff_payload_populates_required_fields() -> None:
    seq = HandoffSequencer()
    payload = build_handoff_payload(
        sequencer=seq,
        from_agent="alpha",
        to_agent="beta",
        context={"task": "summarise"},
    )
    assert payload["from_agent"] == "alpha"
    assert payload["to_agent"] == "beta"
    assert payload["handoff_seq"] == 1
    assert payload["context_hash"].startswith("sha256:")
    assert "context_preview" in payload
    assert "timestamp" in payload


def test_build_handoff_payload_seq_advances() -> None:
    seq = HandoffSequencer()
    p1 = build_handoff_payload(sequencer=seq, from_agent="a", to_agent="b")
    p2 = build_handoff_payload(sequencer=seq, from_agent="b", to_agent="c")
    assert p1["handoff_seq"] == 1
    assert p2["handoff_seq"] == 2


def test_build_handoff_payload_uses_explicit_preview_text() -> None:
    seq = HandoffSequencer()
    payload = build_handoff_payload(
        sequencer=seq,
        from_agent="a",
        to_agent="b",
        context={"task": "x"},
        preview_text="explicit preview wins",
    )
    assert payload["context_preview"] == "explicit preview wins"


def test_build_handoff_payload_truncates_explicit_preview() -> None:
    seq = HandoffSequencer()
    payload = build_handoff_payload(
        sequencer=seq,
        from_agent="a",
        to_agent="b",
        preview_text="z" * 1000,
        preview_max_chars=50,
    )
    assert len(payload["context_preview"]) == 50


def test_build_handoff_payload_preview_falls_back_to_context() -> None:
    seq = HandoffSequencer()
    payload = build_handoff_payload(
        sequencer=seq,
        from_agent="a",
        to_agent="b",
        context={"task": "summarise"},
    )
    # Falls back to stringified context.
    assert "task" in payload["context_preview"]


def test_build_handoff_payload_empty_context_yields_empty_preview() -> None:
    seq = HandoffSequencer()
    payload = build_handoff_payload(
        sequencer=seq,
        from_agent="a",
        to_agent="b",
        context=None,
    )
    assert payload["context_preview"] == ""
    # Hash is still populated (digest of empty dict).
    assert payload["context_hash"].startswith("sha256:")


def test_build_handoff_payload_extra_fields_merged_without_clobbering() -> None:
    seq = HandoffSequencer()
    payload = build_handoff_payload(
        sequencer=seq,
        from_agent="a",
        to_agent="b",
        extra={
            "reason": "delegation",
            "framework": "ms_agent_framework",
            # Standard keys must NOT be overridden by extras.
            "handoff_seq": 999,
            "context_hash": "sha256:00",
        },
    )
    assert payload["reason"] == "delegation"
    assert payload["framework"] == "ms_agent_framework"
    assert payload["handoff_seq"] == 1  # Standard wins.
    assert payload["context_hash"] != "sha256:00"  # Standard wins.


def test_build_handoff_payload_same_context_yields_same_hash() -> None:
    """Same context → same hash, even with different from/to/seq."""
    seq = HandoffSequencer()
    p1 = build_handoff_payload(
        sequencer=seq, from_agent="a", to_agent="b", context={"k": "v"}
    )
    p2 = build_handoff_payload(
        sequencer=seq, from_agent="x", to_agent="y", context={"k": "v"}
    )
    assert p1["context_hash"] == p2["context_hash"]
    assert p1["handoff_seq"] != p2["handoff_seq"]


# ---------------------------------------------------------------------------
# HandoffMetadata.to_payload
# ---------------------------------------------------------------------------


def test_handoff_metadata_to_payload_round_trip() -> None:
    md = HandoffMetadata(
        seq=7,
        context_hash="sha256:abc",
        preview="preview-text",
        from_agent="alpha",
        to_agent="beta",
    )
    payload: Dict[str, Any] = md.to_payload()
    assert payload["handoff_seq"] == 7
    assert payload["context_hash"] == "sha256:abc"
    assert payload["context_preview"] == "preview-text"
    assert payload["from_agent"] == "alpha"
    assert payload["to_agent"] == "beta"
    # Timestamp is ISO 8601.
    assert "T" in payload["timestamp"]


def test_handoff_metadata_to_payload_returns_fresh_dict() -> None:
    md = HandoffMetadata(
        seq=1,
        context_hash="sha256:xx",
        preview="",
        from_agent="a",
        to_agent="b",
    )
    p1 = md.to_payload()
    p2 = md.to_payload()
    assert p1 is not p2
    p1["mutated"] = True
    assert "mutated" not in p2


# ---------------------------------------------------------------------------
# Module surface
# ---------------------------------------------------------------------------


def test_public_constants_exposed() -> None:
    assert DEFAULT_PREVIEW_MAX_CHARS == 256


def test_helpers_importable_from_base_namespace() -> None:
    """Re-exports from ``_base/__init__.py`` are wired correctly."""
    from layerlens.instrument.adapters._base import (
        DEFAULT_PREVIEW_MAX_CHARS as DPC,
        HandoffMetadata as HM,
        HandoffSequencer as HS,
        make_preview as mp,
        compute_context_hash as cch,
        build_handoff_payload as bhp,
    )

    assert DPC == DEFAULT_PREVIEW_MAX_CHARS
    assert HM is HandoffMetadata
    assert HS is HandoffSequencer
    assert bhp is build_handoff_payload
    assert cch is compute_context_hash
    assert mp is make_preview


# ---------------------------------------------------------------------------
# Pytest hygiene
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "context,expected_substr",
    [
        ({"a": 1}, "a"),
        ({"long": "x" * 1000}, "x"),
        ({}, ""),
    ],
)
def test_build_handoff_payload_parametrised(
    context: Dict[str, Any], expected_substr: str
) -> None:
    seq = HandoffSequencer()
    payload = build_handoff_payload(
        sequencer=seq, from_agent="a", to_agent="b", context=context
    )
    if expected_substr:
        assert expected_substr in payload["context_preview"]
    else:
        assert payload["context_preview"] == ""
