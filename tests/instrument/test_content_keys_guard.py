"""L6 — structural keys-must-match guard (the backstop that keeps L1-L5 fixed).

The privacy machinery is three maps that must stay in lock-step:

* ``_CONTENT_KEYS``   — what redaction strips under ``capture_content=False``
* ``_EVENT_TYPE_MAP`` (+ ``_ALWAYS_ENABLED``) — what layer gates an event
* ``KNOWN_EVENT_TYPES`` — the schema-lock vocabulary

A content-bearing event type that is missing from any of these silently
reintroduces a leak (exactly the regression class that made the schema lock
toothless and left 13/14 protocol types fail-open). This guard ties them
together so a renamed / added event type fails loudly in the same PR.

Bite check: revert any of the L2/L4 ``_EVENT_TYPE_MAP`` additions, or add a
content key for a type that isn't registered, and a test here fails.
"""

from __future__ import annotations

import os
import re

import pytest

import layerlens.instrument as _instrument_pkg
import layerlens.instrument._events as events
from layerlens.instrument._capture_config import (
    _CONTENT_KEYS,
    _ALWAYS_ENABLED,
    _EVENT_TYPE_MAP,
    known_event_types,
)

from ._event_schema import KNOWN_EVENT_TYPES

# Run as a required CI gate via `-m invariant` (see .github/workflows/invariants.yaml).
pytestmark = pytest.mark.invariant

# Event types that carry no content payload by design (ids / counts / status /
# hashes / lifecycle only) — explicitly exempt from the "must have a content-key
# strip list" reverse guard. Adding a type here is a deliberate act.
_CONTENT_FREE_TYPES = frozenset(
    {
        # Synthesized structural trace root (LAY-364x) — {"synthesized": true}
        # only, no content, no agent name.
        "trace.root",
        "agent.identity",
        "agent.lifecycle",
        "cost.record",
        "environment.config",
        "environment.metrics",
        "conversation.started",
        "conversation.ended",
        "tool.logic",
        "tool.environment",
        "policy.violation",  # bedrock emits structural metadata only (verified)
        "protocol.agent_card",
        # server identity: name/version/protocol_version are identifiers, not
        # free-text content — surfaced as protocol-discovery metadata (S14/F7).
        "mcp.server.connected",
        "protocol.lifecycle",
        "protocol.task.submitted",
        "protocol.task.completed",
        "protocol.async_task",
        "commerce.checkout.started",
        "commerce.ui.surface_created",
        "commerce.ui.user_action",  # keyed-HMAC hash only; see TestA2UIHashing
        "a2a.agent.card",
        "a2a.agent.card.served",
        "a2a.task.completed",
    }
)


def _instrument_dir() -> str:
    return os.path.dirname(_instrument_pkg.__file__)


# Matches an event-type STRING LITERAL passed as the first arg to an emit-style
# call (.emit / ._emit / ._fire / .emit_async / collector.emit) — the inline
# literals where the historical leaks actually lived.
_EMIT_LITERAL = re.compile(r"(?:\b_?emit(?:_async)?|\b_fire)\(\s*[\"']([a-z][a-z0-9_]*(?:\.[a-z0-9_]+)+)[\"']")


def test_every_content_key_is_schema_registered() -> None:
    """Every type we redact must be a known event type (else it's a typo or an
    unregistered type that the schema lock would never validate)."""
    missing = sorted(et for et in _CONTENT_KEYS if et not in KNOWN_EVENT_TYPES)
    assert not missing, f"_CONTENT_KEYS entries not in KNOWN_EVENT_TYPES (register them): {missing}"


def test_every_content_key_is_layer_gated() -> None:
    """Every content-bearing type must be gatable — mapped to an L-layer or
    explicitly always-enabled. An unmapped content type fails OPEN (L2)."""
    fail_open = sorted(et for et in _CONTENT_KEYS if et not in _EVENT_TYPE_MAP and et not in _ALWAYS_ENABLED)
    assert not fail_open, (
        f"content-bearing types missing from _EVENT_TYPE_MAP and _ALWAYS_ENABLED (fail-open, L2): {fail_open}"
    )


def test_every_mapped_type_is_schema_registered() -> None:
    missing = sorted(et for et in _EVENT_TYPE_MAP if et not in KNOWN_EVENT_TYPES)
    assert not missing, f"_EVENT_TYPE_MAP keys not in KNOWN_EVENT_TYPES: {missing}"


def test_every_always_enabled_type_is_schema_registered() -> None:
    missing = sorted(et for et in _ALWAYS_ENABLED if et not in KNOWN_EVENT_TYPES)
    assert not missing, f"_ALWAYS_ENABLED entries not in KNOWN_EVENT_TYPES: {missing}"


def test_runtime_known_event_types_matches_schema_lock() -> None:
    """The RUNTIME event-type registry (``known_event_types``, used by the
    replay fail-closed gate in ``layerlens.replay.snapshot``) must equal the
    test-side schema-lock vocabulary (``KNOWN_EVENT_TYPES``).

    If they drift, replay either OVER-blocks a legitimately-registered type (a
    real recorded trace is wrongly rejected) or UNDER-blocks a type the CI lock
    forbids. ``src/`` cannot import the test lock, so this guard is the seam that
    keeps the runtime registry and the CI lock in lock-step — register a new type
    in BOTH ``_capture_config`` and ``_event_schema`` in the same PR.

    BITE: drop any single type from ``_EVENT_TYPE_MAP``/``_ALWAYS_ENABLED``/
    ``_CONTENT_KEYS`` (or from ``KNOWN_EVENT_TYPES``) and this fails.
    """
    runtime = known_event_types()
    only_runtime = sorted(runtime - KNOWN_EVENT_TYPES)
    only_lock = sorted(KNOWN_EVENT_TYPES - runtime)
    assert runtime == KNOWN_EVENT_TYPES, (
        "runtime known_event_types() drifted from the schema lock KNOWN_EVENT_TYPES — "
        f"only in runtime: {only_runtime}; only in lock: {only_lock} (register new "
        "types in BOTH src/layerlens/instrument/_capture_config.py and "
        "tests/instrument/_event_schema.py)"
    )


def test_every_emitted_event_constant_is_schema_registered() -> None:
    """Every event-name constant in _events.py must be registered in the schema
    lock."""
    emitted = {
        value
        for name, value in vars(events).items()
        if not name.startswith("_") and isinstance(value, str) and "." in value
    }
    missing = sorted(et for et in emitted if et not in KNOWN_EVENT_TYPES)
    assert not missing, f"_events.py constants not registered in KNOWN_EVENT_TYPES: {missing}"


def test_every_emitted_literal_is_schema_registered() -> None:
    """Scan the REAL emit call-sites for inline event-type string literals and
    assert each is schema-registered. Most adapters emit string literals (e.g.
    "agent.input", "commerce.checkout.started", "mcp.tools.listed") rather than
    an _events.py constant — and that is exactly where the historical leaks were
    introduced. Closes the 'new inline-literal type silently unlocked' class.
    """
    found: dict = {}
    for root, _dirs, files in os.walk(_instrument_dir()):
        for fname in files:
            if not fname.endswith(".py"):
                continue
            path = os.path.join(root, fname)
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
            for et in _EMIT_LITERAL.findall(text):
                found.setdefault(et, path)
    # Sanity: the scan must actually find emit literals (else the regex rotted
    # and the guard is silently vacuous).
    assert len(found) >= 15, f"emit-literal scan found only {len(found)} types — regex likely broken"
    missing = sorted(et for et in found if et not in KNOWN_EVENT_TYPES)
    detail = {et: found[et] for et in missing}
    assert not missing, f"emitted event-type literals not registered in KNOWN_EVENT_TYPES: {detail}"


def test_content_surface_types_have_strip_lists() -> None:
    """REVERSE guard: a content-bearing event type that LANDS ON a content layer
    must have a _CONTENT_KEYS strip list (or be explicitly content-free). This is
    the '13/14 protocol types fail-open' regression class — a new content type
    with NO strip list would otherwise pass every other guard here."""
    # ALL layers that can carry content (LAY-3572 / B19): the original guard only
    # inspected l1/l5a/l6b/l6c, so a new content-bearing type landing on
    # l2_agent_code / l3_model_metadata / l5b_tool_logic / l5c_tool_environment
    # with no strip list would fail-open undetected.
    content_layers = {
        "l1_agent_io",
        "l2_agent_code",
        "l3_model_metadata",
        "l5a_tool_calls",
        "l5b_tool_logic",
        "l5c_tool_environment",
        "l6b_protocol_streams",
        "l6c_protocol_lifecycle",
    }
    offenders = sorted(
        et
        for et, layer in _EVENT_TYPE_MAP.items()
        if layer in content_layers and et not in _CONTENT_KEYS and et not in _CONTENT_FREE_TYPES
    )
    assert not offenders, (
        "content-layer event types with no _CONTENT_KEYS strip list (add a strip list, "
        f"or allowlist as content-free in _CONTENT_FREE_TYPES): {offenders}"
    )
