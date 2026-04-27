"""Unit tests for the field-specific truncation policy.

Verifies the policy correctness, edge cases, and the audit-list
contract relied on by every lighter adapter wired in
cross-pollination audit §2.4.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict

import pytest

from layerlens.instrument.adapters._base.truncation import (
    DROP,
    DEFAULT_POLICY,
    DEFAULT_FIELD_CAPS,
    truncate_field,
    truncate_payload,
)

# ---------------------------------------------------------------------------
# Policy construction
# ---------------------------------------------------------------------------


def test_default_policy_has_expected_caps() -> None:
    """DEFAULT_POLICY must declare the canonical caps from the spec."""
    assert DEFAULT_POLICY.field_caps["prompt"] == 4096
    assert DEFAULT_POLICY.field_caps["completion"] == 4096
    assert DEFAULT_POLICY.field_caps["message"] == 4096
    assert DEFAULT_POLICY.field_caps["tool_input"] == 2048
    assert DEFAULT_POLICY.field_caps["tool_output"] == 2048
    assert DEFAULT_POLICY.field_caps["state_snapshot"] == 8192
    assert DEFAULT_POLICY.field_caps["error_message"] == 1024
    # Drop sentinels for binary fields.
    assert DEFAULT_POLICY.field_caps["screenshot"] == DROP
    assert DEFAULT_POLICY.field_caps["image_data"] == DROP
    # Traceback frame count default.
    assert DEFAULT_POLICY.max_traceback_frames == 8


def test_policy_with_overrides_is_immutable_and_merged() -> None:
    """Overrides return a new policy and merge ``field_caps``."""
    custom = DEFAULT_POLICY.with_overrides(
        field_caps={"my_field": 50},
        max_list_items=5,
    )
    # Merged.
    assert custom.field_caps["my_field"] == 50
    assert custom.field_caps["prompt"] == 4096  # inherited
    assert custom.max_list_items == 5
    # Original untouched.
    assert "my_field" not in DEFAULT_POLICY.field_caps
    assert DEFAULT_POLICY.max_list_items == 100


def test_policy_with_overrides_rejects_non_dict_field_caps() -> None:
    with pytest.raises(TypeError):
        DEFAULT_POLICY.with_overrides(field_caps="not a dict")  # type: ignore[arg-type]


def test_cap_for_is_case_insensitive() -> None:
    """Field-name lookup is case-insensitive (tolerates mixed-case payloads)."""
    assert DEFAULT_POLICY.cap_for("prompt") == 4096
    assert DEFAULT_POLICY.cap_for("PROMPT") == 4096
    assert DEFAULT_POLICY.cap_for("  Prompt  ") == 4096
    # Unknown field returns None when ``apply_to_unknown_fields`` is False.
    assert DEFAULT_POLICY.cap_for("not_a_real_field") is None


def test_cap_for_unknown_field_when_apply_to_unknown_set() -> None:
    """Unknown fields fall back to ``unknown_field_cap`` when enabled."""
    strict = DEFAULT_POLICY.with_overrides(
        apply_to_unknown_fields=True,
        unknown_field_cap=100,
    )
    assert strict.cap_for("custom_field") == 100
    assert strict.cap_for("prompt") == 4096  # explicit cap still wins


# ---------------------------------------------------------------------------
# Char-count truncation
# ---------------------------------------------------------------------------


def test_truncate_field_short_value_unchanged() -> None:
    audit: list[str] = []
    out = truncate_field("hello", "prompt", DEFAULT_POLICY, _truncated=audit)
    assert out == "hello"
    assert audit == []


def test_truncate_field_long_string_clipped_to_cap() -> None:
    audit: list[str] = []
    long_text = "x" * 5000
    out = truncate_field(long_text, "prompt", DEFAULT_POLICY, _truncated=audit)
    assert isinstance(out, str)
    # Capped at 4096 + suffix.
    assert out.startswith("x" * 4096)
    assert "more chars truncated" in out
    assert any("prompt:chars-5000->4096" == entry for entry in audit)


def test_truncate_field_unknown_field_passes_through() -> None:
    """Unknown field name → no truncation under default policy."""
    audit: list[str] = []
    out = truncate_field("y" * 50000, "no_such_field", DEFAULT_POLICY, _truncated=audit)
    assert out == "y" * 50000
    assert audit == []


# ---------------------------------------------------------------------------
# UTF-8 multibyte safety
# ---------------------------------------------------------------------------


def test_truncate_field_multibyte_utf8_safe() -> None:
    """Truncation MUST NOT produce invalid UTF-8 for multi-byte content."""
    # Each emoji = 1 codepoint but multi-byte in UTF-8.
    emoji = "😀"  # 4 bytes in UTF-8, 1 codepoint
    long_text = emoji * 5000
    out = truncate_field(long_text, "prompt", DEFAULT_POLICY)
    assert isinstance(out, str)
    # Re-encoding must succeed (no broken surrogate halves).
    encoded = out.encode("utf-8")
    decoded = encoded.decode("utf-8")
    assert decoded == out


def test_truncate_field_chinese_codepoints_safe() -> None:
    """Multi-byte CJK characters must round-trip after truncation."""
    text = "测试" * 5000  # each char is 3 bytes UTF-8, 1 codepoint
    out = truncate_field(text, "prompt", DEFAULT_POLICY)
    assert isinstance(out, str)
    out.encode("utf-8")  # must not raise
    # Should contain the original first ~4096 codepoints.
    assert out.startswith("测试" * 100)


def test_truncate_field_bytes_input_decoded_safely() -> None:
    """Bytes input is decoded (replace) before char-count truncation."""
    raw = ("a" * 5000).encode("utf-8")
    out = truncate_field(raw, "prompt", DEFAULT_POLICY)
    assert isinstance(out, str)
    assert out.startswith("a" * 4096)
    assert "more chars truncated" in out


# ---------------------------------------------------------------------------
# Drop / hash-reference behaviour
# ---------------------------------------------------------------------------


def test_truncate_field_screenshot_dropped_with_hash() -> None:
    """Screenshots are replaced with deterministic hash references."""
    audit: list[str] = []
    payload_bytes = b"PNG_FAKE_SCREENSHOT_BYTES" * 1000
    out = truncate_field(payload_bytes, "screenshot", DEFAULT_POLICY, _truncated=audit)
    assert isinstance(out, str)
    assert out.startswith("<dropped:screenshot:sha256:")
    assert any("screenshot:dropped" == entry for entry in audit)
    # Same input produces same hash (audit reproducibility).
    out2 = truncate_field(payload_bytes, "screenshot", DEFAULT_POLICY)
    assert out == out2
    expected = hashlib.sha256(payload_bytes).hexdigest()[:16]
    assert expected in out


def test_truncate_field_image_data_dropped() -> None:
    """``image_data`` is also dropped via the DROP sentinel."""
    out = truncate_field("IMAGE_PAYLOAD", "image_data", DEFAULT_POLICY)
    assert isinstance(out, str)
    assert out.startswith("<dropped:image_data:sha256:")


# ---------------------------------------------------------------------------
# Traceback-frame truncation
# ---------------------------------------------------------------------------


_FAKE_TRACEBACK = """Traceback (most recent call last):
  File "/a.py", line 1, in main
    do_thing()
  File "/b.py", line 2, in do_thing
    inner()
  File "/c.py", line 3, in inner
    deeper()
  File "/d.py", line 4, in deeper
    deepest()
  File "/e.py", line 5, in deepest
    raise RuntimeError("boom")
  File "/f.py", line 6, in <module>
    main()
  File "/g.py", line 7, in <module>
    main()
  File "/h.py", line 8, in <module>
    main()
  File "/i.py", line 9, in <module>
    main()
  File "/j.py", line 10, in <module>
    main()
RuntimeError: boom
"""


def test_truncate_field_traceback_keeps_first_n_frames() -> None:
    """Tracebacks are truncated by frame count, not by char count."""
    audit: list[str] = []
    policy = DEFAULT_POLICY.with_overrides(max_traceback_frames=3)
    out = truncate_field(_FAKE_TRACEBACK, "traceback", policy, _truncated=audit)
    assert isinstance(out, str)
    # First 3 frames retained.
    assert "/a.py" in out
    assert "/b.py" in out
    assert "/c.py" in out
    # Frames beyond #3 should NOT appear.
    assert "/d.py" not in out
    assert "/j.py" not in out
    # Truncation note appended.
    assert "more frame" in out
    assert any(entry.startswith("traceback:traceback-frames") for entry in audit)


def test_truncate_field_short_traceback_unchanged() -> None:
    """Tracebacks with fewer frames than the limit are kept verbatim."""
    short = """Traceback (most recent call last):
  File "/x.py", line 1, in main
    raise ValueError("oops")
ValueError: oops
"""
    out = truncate_field(short, "traceback", DEFAULT_POLICY)
    assert out == short


# ---------------------------------------------------------------------------
# Nested dicts and lists
# ---------------------------------------------------------------------------


def test_truncate_payload_nested_dict() -> None:
    """Nested dicts are walked recursively with field-name lookup."""
    payload: Dict[str, Any] = {
        "outer": {
            "prompt": "z" * 10000,
            "metadata": {"label": "ok"},
        },
        "tool_output": "y" * 5000,
    }
    truncated, audit = truncate_payload(payload, DEFAULT_POLICY)
    inner_prompt = truncated["outer"]["prompt"]
    assert isinstance(inner_prompt, str)
    assert inner_prompt.startswith("z" * 4096)
    assert "more chars truncated" in inner_prompt
    tool_out = truncated["tool_output"]
    assert isinstance(tool_out, str)
    assert tool_out.startswith("y" * 2048)
    # Audit lists the dotted path.
    paths = "\n".join(audit)
    assert "outer.prompt" in paths
    assert "tool_output" in paths


def test_truncate_payload_list_of_strings_each_truncated() -> None:
    """List elements inherit the parent field name for cap lookup."""
    payload = {"messages": ["a" * 10000, "b" * 10000]}
    truncated, audit = truncate_payload(payload, DEFAULT_POLICY)
    msgs = truncated["messages"]
    assert isinstance(msgs, list)
    assert all(isinstance(m, str) for m in msgs)
    assert all(m.startswith("a" * 4096) or m.startswith("b" * 4096) for m in msgs)
    # Audit records each element path.
    assert any("messages[0]" in entry for entry in audit)
    assert any("messages[1]" in entry for entry in audit)


def test_truncate_payload_long_list_capped_at_max_items() -> None:
    """Lists beyond ``max_list_items`` are truncated with a sentinel marker."""
    policy = DEFAULT_POLICY.with_overrides(max_list_items=3)
    payload = {"messages": ["short" + str(i) for i in range(10)]}
    truncated, audit = truncate_payload(payload, policy)
    msgs = truncated["messages"]
    assert isinstance(msgs, list)
    assert len(msgs) == 4  # 3 items + truncation marker
    assert "more items truncated" in msgs[-1]
    assert any("list-10->3" in entry for entry in audit)


def test_truncate_payload_recursion_limit_protects_cycles() -> None:
    """Deep nesting beyond the recursion limit is replaced with a sentinel."""
    policy = DEFAULT_POLICY.with_overrides(recursion_limit=3)
    nested: Dict[str, Any] = {"x": "leaf"}
    for _ in range(20):
        nested = {"prompt": "y" * 100, "next": nested}
    truncated, audit = truncate_payload(nested, policy)
    # Walk down to the recursion-limit sentinel.
    found = False
    cur: Any = truncated
    for _ in range(10):
        if isinstance(cur, dict) and "next" in cur:
            cur = cur["next"]
            if cur == "<recursion limit reached>":
                found = True
                break
    assert found, "recursion-limit sentinel not produced"
    assert any("recursion-limit" in entry for entry in audit)


# ---------------------------------------------------------------------------
# Primitive pass-through
# ---------------------------------------------------------------------------


def test_truncate_payload_preserves_primitives() -> None:
    """Numeric/bool/None values are never touched."""
    payload = {
        "prompt": "ok",
        "tokens_prompt": 1234,
        "latency_ms": 12.5,
        "error": None,
        "ok": True,
    }
    truncated, audit = truncate_payload(payload, DEFAULT_POLICY)
    assert audit == []
    assert truncated == payload


def test_truncate_payload_equals_input_when_no_truncation_needed() -> None:
    """Short payloads round-trip through ``truncate_payload`` unchanged.

    The BaseAdapter zero-copy optimisation (skip the rewrite when the
    audit list is empty) is verified by
    ``tests/instrument/test_base_layer.py``; here we only assert
    structural equality.
    """
    payload = {"prompt": "short", "tool_output": "ok"}
    truncated, audit = truncate_payload(payload, DEFAULT_POLICY)
    assert audit == []
    assert truncated == payload


def test_truncate_payload_rejects_non_dict_input() -> None:
    with pytest.raises(TypeError):
        truncate_payload("not a dict", DEFAULT_POLICY)  # type: ignore[arg-type]


def test_truncate_payload_audit_attached_by_caller() -> None:
    """The audit list is returned separately so the caller (BaseAdapter) attaches it."""
    payload = {"prompt": "p" * 5000}
    truncated, audit = truncate_payload(payload, DEFAULT_POLICY)
    assert "_truncated_fields" not in truncated  # caller attaches
    assert audit  # non-empty


# ---------------------------------------------------------------------------
# Empty / boundary inputs
# ---------------------------------------------------------------------------


def test_truncate_field_empty_string_unchanged() -> None:
    out = truncate_field("", "prompt", DEFAULT_POLICY)
    assert out == ""


def test_truncate_field_value_at_exact_cap_unchanged() -> None:
    """A value exactly at ``cap`` chars is NOT truncated."""
    text = "z" * 4096
    out = truncate_field(text, "prompt", DEFAULT_POLICY)
    assert out == text


def test_truncate_field_one_char_over_cap_truncated() -> None:
    """A value of ``cap + 1`` chars IS truncated."""
    text = "z" * 4097
    audit: list[str] = []
    out = truncate_field(text, "prompt", DEFAULT_POLICY, _truncated=audit)
    assert isinstance(out, str)
    assert out.startswith("z" * 4096)
    assert "more chars truncated" in out
    assert audit


def test_default_field_caps_constant_includes_screenshot_drop() -> None:
    """``DEFAULT_FIELD_CAPS`` constant is the policy source of truth."""
    assert DEFAULT_FIELD_CAPS["screenshot"] == DROP
    assert DEFAULT_FIELD_CAPS["image_data"] == DROP
    assert DEFAULT_FIELD_CAPS["html"] == 16384  # browser_use coverage


def test_truncate_field_tuple_truncated_and_returned_as_tuple() -> None:
    """Tuples are normalised through list truncation but coerced back."""
    payload = ("a" * 10000, "b" * 10000)
    out = truncate_field(payload, "messages", DEFAULT_POLICY)
    assert isinstance(out, tuple)
    assert all(isinstance(item, str) for item in out)
    assert out[0].startswith("a" * 4096)
    assert out[1].startswith("b" * 4096)


def test_truncate_payload_does_not_mutate_input() -> None:
    """The original payload dict must NEVER be mutated in place."""
    payload = {"prompt": "p" * 10000, "messages": ["a", "b", "c"]}
    snapshot = dict(payload)
    snapshot_messages = list(payload["messages"])  # type: ignore[arg-type]
    truncate_payload(payload, DEFAULT_POLICY)
    assert payload == snapshot
    assert payload["messages"] == snapshot_messages


def test_truncate_field_existing_truncated_fields_not_overwritten() -> None:
    """Caller-supplied ``_truncated_fields`` list is preserved & extended."""
    # This is an integration with BaseAdapter._apply_truncation — see
    # tests/instrument/test_base_layer.py for the wiring test.
    payload = {
        "_truncated_fields": ["caller-already-truncated:something"],
        "prompt": "p" * 10000,
    }
    # truncate_payload itself does not attach _truncated_fields — that
    # is the BaseAdapter's job. But it must preserve the field through
    # recursion so existing entries survive.
    truncated, audit = truncate_payload(payload, DEFAULT_POLICY)
    # The pre-existing list is in the returned dict (it was a list of
    # short strings, so unchanged).
    assert truncated["_truncated_fields"] == ["caller-already-truncated:something"]
    assert audit  # prompt was truncated


# ---------------------------------------------------------------------------
# Determinism / reproducibility
# ---------------------------------------------------------------------------


def test_truncate_field_deterministic_for_same_input() -> None:
    """Same input + same policy produces identical output."""
    payload = {
        "prompt": "p" * 10000,
        "screenshot": b"\x89PNG\x00\x01\x02\x03" * 100,
    }
    out_a, audit_a = truncate_payload(payload, DEFAULT_POLICY)
    out_b, audit_b = truncate_payload(payload, DEFAULT_POLICY)
    assert out_a == out_b
    assert audit_a == audit_b
