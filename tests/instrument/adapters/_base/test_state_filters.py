"""Tests for the per-key allowlist / denylist / mask filter for adapter state.

Covers :class:`StateFilter`, :func:`filter_state`, :func:`filter_payload_fields`,
:data:`DEFAULT_PII_EXCLUDE_KEYS`, and the ``_filter_payload`` /
``serialize_state_filter_for_replay`` integration on every multi-agent
framework adapter.

Every test asserts behaviour that prevents PII / credentials from leaving
the adapter — the filter is the last line of defence between user state
and a telemetry sink, so failure modes here are CRITICAL severity per
CLAUDE.md ("never silently leak customer data").
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import Mock

import pytest

from layerlens.instrument.adapters._base import (
    REDACTED_PLACEHOLDER,
    DEFAULT_PII_EXCLUDE_KEYS,
    StateFilter,
    filter_state,
    default_state_filter,
    filter_payload_fields,
)
from layerlens.instrument.adapters.frameworks._base_framework import FrameworkAdapter

# ---------------------------------------------------------------------------
# StateFilter dataclass behaviour
# ---------------------------------------------------------------------------


class TestStateFilterConstruction:
    """The filter normalises its inputs so callers can pass any iterable shape."""

    def test_default_is_pii_aware(self) -> None:
        """A ``StateFilter()`` with no args keeps the conservative default
        denylist — the "I forgot to configure" path must still scrub
        common PII out of the box.
        """
        f = StateFilter()
        # The constructor lowercases the frozenset; assert membership
        # rather than identity.
        assert "password" in f.exclude_keys
        assert "api_key" in f.exclude_keys
        assert "ssn" in f.exclude_keys
        assert f.include_keys is None
        assert f.mask_keys == frozenset()

    def test_lowercases_exclude_keys(self) -> None:
        f = StateFilter(exclude_keys=frozenset({"PASSWORD", "Authorization"}))
        assert "password" in f.exclude_keys
        assert "authorization" in f.exclude_keys

    def test_lowercases_include_keys(self) -> None:
        f = StateFilter(include_keys=frozenset({"User_ID", "MODEL"}))
        assert f.include_keys is not None
        assert "user_id" in f.include_keys
        assert "model" in f.include_keys

    def test_permissive_factory(self) -> None:
        """``permissive()`` removes all rules — used in tests / explicit opt-out."""
        f = StateFilter.permissive()
        assert f.exclude_keys == frozenset()
        assert f.mask_keys == frozenset()
        assert f.include_keys is None

    def test_with_extra_excludes_factory(self) -> None:
        f = StateFilter.with_extra_excludes(["custom_secret", "internal_id"])
        assert "password" in f.exclude_keys  # default still present
        assert "custom_secret" in f.exclude_keys
        assert "internal_id" in f.exclude_keys


class TestStateFilterMetadata:
    """``as_metadata`` produces a stable, dashboard-safe snapshot."""

    def test_default_metadata_shape(self) -> None:
        meta = StateFilter().as_metadata()
        assert meta["exclude_keys_count"] == len(DEFAULT_PII_EXCLUDE_KEYS)
        assert meta["mask_keys_count"] == 0
        assert meta["recursive"] is True
        # Allowlist is None → key not surfaced.
        assert "include_keys" not in meta

    def test_allowlist_surfaces_in_metadata(self) -> None:
        f = StateFilter(include_keys=frozenset({"foo", "bar"}))
        meta = f.as_metadata()
        # Allowlists are intentionally short — surface them so customers
        # can verify exactly what they configured.
        assert meta["include_keys"] == ["bar", "foo"]
        assert meta["include_keys_count"] == 2


# ---------------------------------------------------------------------------
# filter_state — exclude / mask / include precedence
# ---------------------------------------------------------------------------


class TestFilterStateExclude:
    def test_excludes_default_pii_keys(self) -> None:
        state = {"username": "alice", "password": "hunter2", "api_key": "sk-..."}
        out, keys = filter_state(state, default_state_filter())
        assert "username" in out
        assert "password" not in out
        assert "api_key" not in out
        # filtered_keys reports BOTH excluded names — auditable trail.
        assert sorted(keys) == ["api_key", "password"]

    def test_substring_match_catches_vendor_variants(self) -> None:
        """``X-Api-Key``, ``stripe_customer_email``, ``USER_API_KEY`` should all match."""
        state = {
            "X-Api-Key": "sk-secret",
            "stripe_customer_email": "alice@example.com",
            "USER_API_KEY": "user-secret",
            "model": "gpt-5",
        }
        out, keys = filter_state(state, default_state_filter())
        assert "model" in out
        assert "X-Api-Key" not in out
        assert "stripe_customer_email" not in out
        assert "USER_API_KEY" not in out

    def test_excludes_nothing_when_disabled(self) -> None:
        state = {"password": "hunter2", "api_key": "sk-..."}
        out, keys = filter_state(state, StateFilter.permissive())
        assert out == state
        assert keys == []


class TestFilterStateMask:
    def test_masks_keys_keeps_field_visible(self) -> None:
        f = StateFilter(exclude_keys=frozenset(), mask_keys=frozenset({"phone"}))
        state = {"name": "Alice", "phone": "555-1234"}
        out, keys = filter_state(state, f)
        # Key remains so dashboards see the field exists, value is REDACTED.
        assert out == {"name": "Alice", "phone": REDACTED_PLACEHOLDER}
        assert keys == ["phone"]

    def test_mask_runs_before_recurse(self) -> None:
        """A masked key's nested structure is NOT walked — the value
        is replaced wholesale so nested PII can't leak through.
        """
        f = StateFilter(exclude_keys=frozenset(), mask_keys=frozenset({"profile"}))
        state = {
            "profile": {"email": "alice@example.com", "phone": "555-1234"},
        }
        out, _ = filter_state(state, f)
        assert out == {"profile": REDACTED_PLACEHOLDER}


class TestFilterStateInclude:
    def test_include_acts_as_allowlist(self) -> None:
        f = StateFilter(
            exclude_keys=frozenset(),
            include_keys=frozenset({"model", "tokens_total"}),
        )
        state = {"model": "gpt-5", "tokens_total": 100, "input": "secret prompt"}
        out, keys = filter_state(state, f)
        assert out == {"model": "gpt-5", "tokens_total": 100}
        assert "input" in keys

    def test_include_runs_after_exclude(self) -> None:
        """Even an allowlisted key is still removed if it matches exclude."""
        f = StateFilter(
            include_keys=frozenset({"password", "model"}),  # allow password
            exclude_keys=frozenset({"password"}),  # but also exclude it
        )
        state = {"password": "hunter2", "model": "gpt-5"}
        out, _ = filter_state(state, f)
        assert "password" not in out  # exclude wins
        assert out == {"model": "gpt-5"}


class TestFilterStateRecursive:
    def test_recurses_into_nested_dicts(self) -> None:
        state = {
            "user": {"name": "Alice", "password": "hunter2"},
            "model": "gpt-5",
        }
        out, keys = filter_state(state, default_state_filter())
        assert out == {"user": {"name": "Alice"}, "model": "gpt-5"}
        assert "password" in keys

    def test_recurses_into_lists_of_dicts(self) -> None:
        state = {
            "messages": [
                {"role": "user", "content": "hi", "api_key": "sk-..."},
                {"role": "assistant", "content": "hello"},
            ],
        }
        out, keys = filter_state(state, default_state_filter())
        assert out["messages"][0] == {"role": "user", "content": "hi"}
        assert out["messages"][1] == {"role": "assistant", "content": "hello"}
        assert "api_key" in keys

    def test_non_recursive_skips_nested(self) -> None:
        f = StateFilter(recursive=False)
        state = {"user": {"password": "hunter2"}}
        out, _ = filter_state(state, f)
        # Top-level only — nested password survives because recursion off.
        assert out == {"user": {"password": "hunter2"}}


class TestFilterStatePassthrough:
    """Non-dict / non-list inputs pass through unchanged."""

    @pytest.mark.parametrize("value", [None, 0, 1.5, "hello", True, b"bytes"])
    def test_primitives_pass_through(self, value: Any) -> None:
        out, keys = filter_state(value, default_state_filter())
        assert out == value
        assert keys == []

    def test_empty_dict_returns_empty(self) -> None:
        out, keys = filter_state({}, default_state_filter())
        assert out == {}
        assert keys == []


# ---------------------------------------------------------------------------
# filter_payload_fields — surgical filter for adapter use
# ---------------------------------------------------------------------------


class TestFilterPayloadFields:
    def test_only_named_fields_are_filtered(self) -> None:
        """Scalar metadata (model, latency_ms) is left alone; only the
        named dict-shaped fields are scrubbed.
        """
        payload: Dict[str, Any] = {
            "model": "gpt-5",
            "latency_ms": 42,
            "input": {"user": "alice", "password": "hunter2"},
        }
        clipped = filter_payload_fields(payload, default_state_filter(), ["input"])
        assert payload["model"] == "gpt-5"  # untouched
        assert payload["latency_ms"] == 42  # untouched
        assert payload["input"] == {"user": "alice"}  # filtered
        assert clipped == ["password"]
        assert payload["_filtered_keys"] == ["password"]

    def test_missing_fields_are_skipped(self) -> None:
        payload: Dict[str, Any] = {"model": "gpt-5"}
        clipped = filter_payload_fields(payload, default_state_filter(), ["input", "deps"])
        assert clipped == []
        assert "_filtered_keys" not in payload

    def test_scalar_field_is_not_filtered(self) -> None:
        """An ``input`` field that's already a string passes through."""
        payload: Dict[str, Any] = {"input": "hello world"}
        clipped = filter_payload_fields(payload, default_state_filter(), ["input"])
        assert payload["input"] == "hello world"
        assert clipped == []

    def test_merges_with_existing_filtered_keys(self) -> None:
        """Multiple filter passes accumulate the filtered-key list."""
        payload: Dict[str, Any] = {
            "_filtered_keys": ["password"],
            "output": {"name": "alice", "ssn": "111-22-3333"},
        }
        filter_payload_fields(payload, default_state_filter(), ["output"])
        # Both old + new are surfaced, sorted, deduped.
        assert payload["_filtered_keys"] == ["password", "ssn"]


# ---------------------------------------------------------------------------
# FrameworkAdapter integration — every adapter must wire the filter
# ---------------------------------------------------------------------------


class _StubAdapter(FrameworkAdapter):
    """Minimal concrete adapter so we can exercise base wiring."""

    name = "stub"

    def _on_connect(self, target: Any = None, **kwargs: Any) -> None:
        pass

    def _on_disconnect(self) -> None:
        pass


class TestFrameworkAdapterStateFilterDefaults:
    def test_default_filter_installed_on_construction(self) -> None:
        a = _StubAdapter(client=Mock())
        # Default filter excludes the PII denylist — verify by snapshot.
        meta = a._state_filter.as_metadata()
        assert meta["exclude_keys_count"] == len(DEFAULT_PII_EXCLUDE_KEYS)

    def test_custom_filter_overrides_default(self) -> None:
        custom = StateFilter.permissive()
        a = _StubAdapter(client=Mock(), state_filter=custom)
        assert a._state_filter is custom

    def test_filter_payload_drops_pii(self) -> None:
        a = _StubAdapter(client=Mock())
        payload: Dict[str, Any] = {
            "model": "gpt-5",
            "input": {"user": "alice", "api_key": "sk-secret"},
        }
        a._filter_payload(payload, "input")
        assert payload["input"] == {"user": "alice"}
        assert "api_key" in payload["_filtered_keys"]

    def test_serialize_state_filter_for_replay(self) -> None:
        """Replay must capture the filter so the replay engine can
        reconstruct an equivalent filter on the other side.
        """
        a = _StubAdapter(client=Mock())
        snap = a.serialize_state_filter_for_replay()
        assert snap["recursive"] is True
        assert snap["exclude_keys_count"] == len(DEFAULT_PII_EXCLUDE_KEYS)

    def test_state_filter_appears_in_adapter_info(self) -> None:
        """``adapter_info().metadata['state_filter']`` lets operators
        verify what's being scrubbed (and detect accidental
        ``StateFilter.permissive()``).
        """
        a = _StubAdapter(client=Mock())
        info = a.adapter_info()
        assert "state_filter" in info.metadata
        assert info.metadata["state_filter"]["exclude_keys_count"] == len(DEFAULT_PII_EXCLUDE_KEYS)


# ---------------------------------------------------------------------------
# Per-adapter constructor wiring — the 6 multi-agent adapters present on
# this base. ms_agent_framework is enumerated in the audit but doesn't
# exist on this branch's history; will be wired when its adapter lands
# on `feat/instrument-callback-resilience` (or its successor).
# ---------------------------------------------------------------------------


_PARAM_ADAPTERS: List[Any] = []
try:
    from layerlens.instrument.adapters.frameworks.agno import AgnoAdapter

    _PARAM_ADAPTERS.append(("agno", AgnoAdapter))
except Exception:
    pass
try:
    from layerlens.instrument.adapters.frameworks.openai_agents import OpenAIAgentsAdapter

    _PARAM_ADAPTERS.append(("openai_agents", OpenAIAgentsAdapter))
except Exception:
    pass
try:
    from layerlens.instrument.adapters.frameworks.llamaindex import LlamaIndexAdapter

    _PARAM_ADAPTERS.append(("llamaindex", LlamaIndexAdapter))
except Exception:
    pass
try:
    from layerlens.instrument.adapters.frameworks.google_adk import GoogleADKAdapter

    _PARAM_ADAPTERS.append(("google_adk", GoogleADKAdapter))
except Exception:
    pass
try:
    from layerlens.instrument.adapters.frameworks.strands import StrandsAdapter

    _PARAM_ADAPTERS.append(("strands", StrandsAdapter))
except Exception:
    pass
try:
    from layerlens.instrument.adapters.frameworks.pydantic_ai import PydanticAIAdapter

    _PARAM_ADAPTERS.append(("pydantic_ai", PydanticAIAdapter))
except Exception:
    pass


@pytest.mark.parametrize(("name", "cls"), _PARAM_ADAPTERS)
class TestPerAdapterStateFilterWiring:
    """Verify every multi-agent adapter accepts ``state_filter`` and wires it correctly."""

    def test_constructor_accepts_state_filter(self, name: str, cls: Any) -> None:
        custom = StateFilter.permissive()
        adapter = cls(client=Mock(), state_filter=custom)
        assert adapter._state_filter is custom

    def test_default_state_filter_is_pii_aware(self, name: str, cls: Any) -> None:
        adapter = cls(client=Mock())
        # Out of the box: every adapter excludes the PII denylist.
        assert "password" in adapter._state_filter.exclude_keys
        assert "api_key" in adapter._state_filter.exclude_keys

    def test_state_filter_surfaces_in_adapter_info(self, name: str, cls: Any) -> None:
        adapter = cls(client=Mock())
        info = adapter.adapter_info()
        assert "state_filter" in info.metadata


# ---------------------------------------------------------------------------
# End-to-end: filter applied at the adapter's emit boundary
# ---------------------------------------------------------------------------


class TestEndToEndAgnoFilter:
    """Use the agno adapter (which doesn't require optional deps for the
    pure ``_filter_payload`` path) to demonstrate that the filter actually
    runs at the emit boundary, not just sits idle on the adapter.
    """

    def test_filter_payload_emits_filtered_keys_metadata(self) -> None:
        from layerlens.instrument.adapters.frameworks.agno import AgnoAdapter

        adapter = AgnoAdapter(client=Mock())
        payload: Dict[str, Any] = {
            "agent_name": "demo",
            "input": {"prompt": "hi", "api_key": "sk-secret"},
        }
        adapter._filter_payload(payload, "input")
        assert payload["input"] == {"prompt": "hi"}
        assert payload["_filtered_keys"] == ["api_key"]

    def test_filter_payload_with_permissive_filter_is_noop(self) -> None:
        from layerlens.instrument.adapters.frameworks.agno import AgnoAdapter

        adapter = AgnoAdapter(client=Mock(), state_filter=StateFilter.permissive())
        payload: Dict[str, Any] = {
            "input": {"prompt": "hi", "api_key": "sk-secret"},
        }
        adapter._filter_payload(payload, "input")
        # Permissive filter touches nothing.
        assert payload["input"] == {"prompt": "hi", "api_key": "sk-secret"}
        assert "_filtered_keys" not in payload
