"""``LangGraphStateAdapter.get_hash`` returns a ``sha256:``-prefixed digest.

Regression coverage for the hash-format mismatch that was
found-but-deferred during PR #137 (L2 ``AgentCodeEvent`` work).

Bug
---

Before the fix, ``LangGraphStateAdapter.get_hash()`` returned a bare 64-
character hex digest. ``AgentStateChangeEvent.create()`` (vendored from
``stratix.core.events`` per spec ``02-event-schema-spec.md``) requires
``before_hash`` / ``after_hash`` to be in the canonical form
``sha256:<64-hex>`` — its Pydantic field validator rejects anything
else with ``ValueError``.

The result was that :meth:`NodeTracer._emit_state_change` constructed
``AgentStateChangeEvent.create(...)`` with bare hex, Pydantic raised
``ValidationError``, the ``except Exception`` wrapper swallowed it as
``logger.debug("Typed event emission failed, falling back to legacy")``,
and the typed-event path **silently** never fired. Only the legacy
dict event ever made it to the consumer, defeating the typed-event
emission contract.

These tests pin the contract three ways:

1. :meth:`LangGraphStateAdapter.get_hash` returns ``sha256:``-prefixed
   form.
2. :meth:`AgentStateChangeEvent.create` accepts the result without
   raising ``ValidationError``.
3. The :meth:`NodeTracer._emit_state_change` typed-event path emits
   exactly one typed ``AgentStateChangeEvent`` payload via
   ``adapter.emit_event(...)`` — proving the silent fallback to the
   legacy dict path no longer fires for state-change emission.

A fourth test guards the back-compat helper :meth:`get_hash_unprefixed`
which preserves the bare-hex form for callers that need it.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest
from pydantic import ValidationError

from layerlens.instrument._vendored.events import (
    AgentStateChangeEvent,
    StateType,
)
from layerlens.instrument.adapters._base import CaptureConfig
from layerlens.instrument.adapters.frameworks.langgraph import (
    LayerLensLangGraphAdapter,
    LangGraphStateAdapter,
    NodeTracer,
)


# ---------------------------------------------------------------------------
# Recording stratix client (mirrors the pattern in
# ``tests/instrument/adapters/frameworks/langgraph/test_l2_agent_code_event.py``
# so assertion shape stays consistent across the suite).
# ---------------------------------------------------------------------------


class _RecordingStratix:
    """Minimal STRATIX client double that captures every ``emit()`` call.

    The :class:`NodeTracer._emit_state_change` typed-event path calls
    ``adapter.emit_event(typed_payload)`` which ultimately invokes
    ``stratix.emit(payload)`` (single positional arg). The legacy
    fallback path calls ``stratix.emit(event_type, payload_dict)`` —
    a tuple of (str, dict). We record both shapes so the test can
    distinguish which path actually fired.
    """

    def __init__(self) -> None:
        # Typed-event payloads (Pydantic models passed to emit()).
        self.typed_events: List[Any] = []
        # Legacy dict events captured as (event_type, payload_dict).
        self.legacy_events: List[tuple[str, Dict[str, Any]]] = []

    def emit(self, *args: Any, **kwargs: Any) -> None:
        if len(args) == 1:
            self.typed_events.append(args[0])
            return
        if len(args) == 2 and isinstance(args[0], str):
            self.legacy_events.append((args[0], args[1]))
            return
        # Anything else is unexpected — surface loudly rather than swallow.
        raise AssertionError(
            f"unexpected stratix.emit() call shape: args={args!r} kwargs={kwargs!r}"
        )


# ---------------------------------------------------------------------------
# 1. get_hash() returns sha256:-prefixed form
# ---------------------------------------------------------------------------


class TestGetHashReturnsPrefixedDigest:
    """``LangGraphStateAdapter.get_hash`` always returns ``sha256:<hex>``."""

    def test_returns_sha256_prefixed_form(self) -> None:
        adapter = LangGraphStateAdapter()
        digest = adapter.get_hash({"x": 1, "y": "z"})

        assert digest.startswith("sha256:"), (
            f"get_hash() must return sha256:-prefixed form, got {digest!r}"
        )

    def test_hex_part_is_exactly_64_chars(self) -> None:
        """Spec ``02-event-schema-spec.md``: SHA-256 → 64 hex chars after prefix."""
        adapter = LangGraphStateAdapter()
        digest = adapter.get_hash({"x": 1})

        hex_part = digest.removeprefix("sha256:")
        assert len(hex_part) == 64, (
            f"hex digest must be exactly 64 chars, got {len(hex_part)}: "
            f"{hex_part!r}"
        )
        assert all(c in "0123456789abcdef" for c in hex_part), (
            f"digest must be lowercase hex, got {hex_part!r}"
        )

    def test_deterministic_for_equivalent_state(self) -> None:
        """The fix must not break determinism — same state, same digest."""
        adapter = LangGraphStateAdapter()
        d1 = adapter.get_hash({"a": 1, "b": 2})
        d2 = adapter.get_hash({"b": 2, "a": 1})  # same content, key order varies
        assert d1 == d2

    def test_different_state_yields_different_digest(self) -> None:
        adapter = LangGraphStateAdapter()
        d1 = adapter.get_hash({"x": 1})
        d2 = adapter.get_hash({"x": 2})
        assert d1 != d2


# ---------------------------------------------------------------------------
# 2. AgentStateChangeEvent.create() accepts the result without ValidationError
# ---------------------------------------------------------------------------


class TestAgentStateChangeEventAcceptsGetHashOutput:
    """The typed-event Pydantic model accepts ``get_hash()`` output as-is."""

    def test_create_with_get_hash_output_does_not_raise(self) -> None:
        adapter = LangGraphStateAdapter()
        before = adapter.get_hash({"x": 1})
        after = adapter.get_hash({"x": 2})

        # The bug: this used to raise ValidationError because
        # before/after were bare hex but the validator requires the
        # ``sha256:`` prefix.
        event = AgentStateChangeEvent.create(
            state_type=StateType.INTERNAL,
            before_hash=before,
            after_hash=after,
        )

        assert event.state.before_hash == before
        assert event.state.after_hash == after

    def test_bare_hex_still_rejected_by_validator(self) -> None:
        """Sanity check: the validator we satisfy is the one we think it is.

        If this assertion ever fails it means the ``AgentStateChangeEvent``
        validator was relaxed upstream — at which point the prefix
        contract documented in :meth:`LangGraphStateAdapter.get_hash`
        should be re-evaluated.
        """
        bare_hex = "0" * 64
        with pytest.raises(ValidationError):
            AgentStateChangeEvent.create(
                state_type=StateType.INTERNAL,
                before_hash=bare_hex,
                after_hash=bare_hex,
            )


# ---------------------------------------------------------------------------
# 3. _emit_state_change typed-event path emits successfully (no silent fallback)
# ---------------------------------------------------------------------------


def _state_mutating_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Sample node that mutates state so on_node_exit emits a state change.

    Module-scoped (not a closure) so ``__qualname__`` stays stable —
    this matters because the L2 ``AgentCodeEvent`` machinery in PR #137
    derives identity hashes from ``__qualname__`` and an unstable
    qualname would perturb unrelated assertions in the wider suite.
    """
    out = dict(state)
    out["x"] = state.get("x", 0) + 1
    return out


class TestEmitStateChangeTypedPathFires:
    """:meth:`NodeTracer._emit_state_change` typed-event path now succeeds.

    Before the fix the typed-event path raised ``ValidationError`` and
    silently fell through to the legacy dict path. After the fix the
    typed payload is constructed and forwarded to
    ``BaseAdapter.emit_event`` → ``stratix.emit(typed_payload)``.
    """

    def _build_adapter(self) -> tuple[_RecordingStratix, LayerLensLangGraphAdapter]:
        stratix = _RecordingStratix()
        adapter = LayerLensLangGraphAdapter(
            stratix=stratix,
            capture_config=CaptureConfig.full(),
        )
        adapter.connect()
        return stratix, adapter

    def test_state_change_emits_typed_event_not_legacy_fallback(self) -> None:
        """A node that mutates state emits a typed ``AgentStateChangeEvent``."""
        stratix, adapter = self._build_adapter()
        tracer = NodeTracer(adapter=adapter)

        traced = tracer.decorate(_state_mutating_node)
        traced({"x": 1})

        # Typed-event path: exactly one AgentStateChangeEvent in the
        # typed-event capture buffer. This is the load-bearing assertion
        # — before the fix this was 0 because Pydantic raised on the
        # bare-hex hash and the silent ``except Exception`` swallowed it.
        typed_state_changes = [
            e for e in stratix.typed_events
            if isinstance(e, AgentStateChangeEvent)
        ]
        assert len(typed_state_changes) == 1, (
            "expected exactly 1 typed AgentStateChangeEvent on the typed-event "
            "path; got "
            f"{len(typed_state_changes)} typed_events={stratix.typed_events!r}"
        )

        # Legacy fallback path MUST NOT fire for ``agent.state.change``
        # when an adapter is attached — that would prove the silent
        # fallback path still triggered (the original bug).
        legacy_state_changes = [
            (et, p) for (et, p) in stratix.legacy_events
            if et == "agent.state.change"
        ]
        assert legacy_state_changes == [], (
            "legacy fallback path must not fire for agent.state.change "
            "when typed-event emission succeeds; got "
            f"{legacy_state_changes!r}"
        )

    def test_typed_event_carries_get_hash_output_directly(self) -> None:
        """The before/after hashes in the typed event match ``get_hash()`` exactly."""
        stratix, adapter = self._build_adapter()
        tracer = NodeTracer(adapter=adapter)

        traced = tracer.decorate(_state_mutating_node)
        traced({"x": 1})

        typed_state_changes = [
            e for e in stratix.typed_events
            if isinstance(e, AgentStateChangeEvent)
        ]
        assert len(typed_state_changes) == 1
        evt = typed_state_changes[0]

        # Both hashes must satisfy the schema validator (already proven
        # by Pydantic construction not raising), AND must round-trip the
        # actual state digest — not the ``"sha256:" + "0" * 64`` zero
        # placeholder used by ``_emit_state_change`` for the None case.
        zero_hash = "sha256:" + ("0" * 64)
        assert evt.state.before_hash != zero_hash, (
            "before_hash should be the real state digest, not the zero placeholder"
        )
        assert evt.state.after_hash != zero_hash, (
            "after_hash should be the real state digest, not the zero placeholder"
        )
        assert evt.state.before_hash != evt.state.after_hash, (
            "state mutated, so before/after hashes must differ"
        )


# ---------------------------------------------------------------------------
# 4. Existing get_hash() callers in the langgraph adapter still work
# ---------------------------------------------------------------------------


class TestExistingGetHashCallersStillWork:
    """All in-tree callers of ``get_hash()`` are equality-comparison only.

    The audit (see PR description) found these callers:

    * ``nodes.py``: ``state_hash_before`` / ``state_hash_after`` on
      :class:`NodeExecution` — compared via ``!=``.
    * ``lifecycle.py``: ``initial_state_hash`` / ``final_state_hash`` on
      :class:`GraphExecution` and ``state_hash_before`` /
      ``state_hash_after`` on the node context dict — compared via
      ``!=``.

    The ``langchain/memory.py`` ``get_hash()`` is on a different class
    (:class:`LangChainMemoryAdapter`) and is out of scope for this fix.

    These tests prove equality-comparison semantics survive the format
    change — the comparison still distinguishes equal vs. unequal
    states correctly.
    """

    def test_equality_comparison_unchanged_for_equal_states(self) -> None:
        adapter = LangGraphStateAdapter()
        s1 = {"messages": ["hi"], "step": 0}
        s2 = {"messages": ["hi"], "step": 0}
        assert adapter.get_hash(s1) == adapter.get_hash(s2)

    def test_equality_comparison_unchanged_for_different_states(self) -> None:
        adapter = LangGraphStateAdapter()
        s1 = {"messages": ["hi"], "step": 0}
        s2 = {"messages": ["hi", "there"], "step": 1}
        assert adapter.get_hash(s1) != adapter.get_hash(s2)

    def test_snapshot_has_changed_still_works(self) -> None:
        """``has_changed()`` is the public API consuming the hash internally."""
        adapter = LangGraphStateAdapter()
        before = adapter.snapshot({"x": 1})
        after_same = adapter.snapshot({"x": 1})
        after_diff = adapter.snapshot({"x": 2})

        assert not adapter.has_changed(before, after_same)
        assert adapter.has_changed(before, after_diff)


# ---------------------------------------------------------------------------
# 5. get_hash_unprefixed() back-compat helper
# ---------------------------------------------------------------------------


class TestGetHashUnprefixedHelper:
    """``get_hash_unprefixed`` returns the bare 64-char hex digest.

    Provided for any external caller that genuinely needs the raw hex
    form (no in-tree caller does today, but the helper documents the
    escape hatch and prevents future regressions if we add one).
    """

    def test_returns_bare_hex_no_prefix(self) -> None:
        adapter = LangGraphStateAdapter()
        digest = adapter.get_hash_unprefixed({"x": 1})

        assert not digest.startswith("sha256:"), (
            f"get_hash_unprefixed() must NOT carry the sha256: prefix, got {digest!r}"
        )
        assert len(digest) == 64
        assert all(c in "0123456789abcdef" for c in digest)

    def test_is_consistent_with_prefixed_form(self) -> None:
        """``get_hash() == "sha256:" + get_hash_unprefixed()`` for the same state."""
        adapter = LangGraphStateAdapter()
        state = {"x": 1, "y": "value"}
        assert adapter.get_hash(state) == "sha256:" + adapter.get_hash_unprefixed(state)
