"""Field-specific truncation policy for adapter event payloads.

Lighter framework adapters can emit event payloads that contain
unbounded user data: full prompts, completions, tool outputs, browser
screenshots, large state snapshots, multi-thousand-frame stacktraces.
Untruncated, those payloads:

* Blow past Kafka's 1 MB default record limit.
* Trigger S3 multi-part upload paths inside the ingestion pipeline.
* Inflate observability cost (per-byte indexing in TimescaleDB / OTel).
* Leak high-cardinality identifiers (raw user payloads embedded in
  every span attribute).

Mature adapters (LangChain, LangGraph, AutoGen, Semantic Kernel,
Agentforce) all carry their own ad-hoc truncation: AutoGen uses 500
chars for previews and 10 000 for full messages, Semantic Kernel uses
per-field caps (template 500, prompt 500, query 200, plan 1000),
Agentforce caps responses at 50 000 chars. Each adapter does it
slightly differently — and the lighter adapters (agno, openai_agents,
llama_index, google_adk, strands, pydantic_ai, smolagents,
bedrock_agents, ms_agent_framework, embedding, browser_use) do it
inconsistently or not at all.

This module standardises the policy across all 11 lighter adapters via
:class:`FieldTruncationPolicy`. The defaults match the consensus
ceilings from the mature adapters plus screenshot/image-data drop
behaviour required for the (forthcoming) browser_use adapter — those
payloads must NEVER be embedded in events because a single navigation
step can produce multi-megabyte base64 PNG/WebP blobs.

Design choices
--------------
1. **Pure dataclass, not Pydantic.** ``layerlens`` supports Pydantic
   v1 and v2 simultaneously and the truncation hot path runs once per
   emit. Skipping the validation layer keeps the policy zero-overhead.
2. **UTF-8 safe.** :func:`truncate_string` clips at ``max_chars`` but
   then walks back from the cut point if the cut would split a
   multi-byte codepoint. Adapters can safely pass UTF-8 user content
   (Asian languages, emoji, RTL scripts) without producing invalid
   strings.
3. **Auditable, not silent.** Every truncation is appended to a
   ``_truncated_fields`` list returned alongside the truncated payload.
   :class:`BaseAdapter` attaches that list to the emitted payload as a
   ``_truncated_fields`` metadata key so observability can show what
   was clipped. Silent truncation is forbidden by CLAUDE.md.
4. **Drop, don't redact, screenshots.** ``screenshot`` and
   ``image_data`` fields default to ``max_chars=0`` which causes the
   value to be replaced by a deterministic SHA-256 reference (e.g.
   ``"<dropped:image:sha256:abc123…>"``). Customers retain
   reproducibility (same input → same hash) without paying to ship the
   bytes.
5. **Stack-trace truncation by frames, not chars.** Tracebacks are
   semantically structured — clipping at ``8192`` chars cuts a frame
   in half. The policy keeps the first N frames intact (default 8)
   and appends ``"... (M more frames truncated)"``.

The policy is consumed by :meth:`BaseAdapter.emit_dict_event` /
:meth:`BaseAdapter.emit_event` when an adapter sets its
``_truncation_policy`` attribute — wiring is per-adapter so each one
declares its policy explicitly (per the cross-pollination audit §2.4).
"""

from __future__ import annotations

import re
import hashlib
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import field, dataclass

# ---------------------------------------------------------------------------
# Sentinel for "drop and reference by hash"
# ---------------------------------------------------------------------------

DROP = 0
"""Sentinel ``max_chars`` value: replace value with a hash reference."""


# ---------------------------------------------------------------------------
# Default per-field caps (chars unless noted)
# ---------------------------------------------------------------------------

# Field name → max chars (or DROP). Field-name matching is exact-key
# match within the payload dict, recursively. Nested dicts inherit the
# same policy.
DEFAULT_FIELD_CAPS: Dict[str, int] = {
    # User-visible prompt/response/message text. 4 KiB matches the
    # AutoGen 10_000 ceiling rounded down to a power of two.
    "prompt": 4096,
    "completion": 4096,
    "message": 4096,
    "messages": 4096,  # also caps each list element
    "content": 4096,
    "output": 4096,
    "input": 4096,
    "response": 4096,
    "text": 4096,
    # Tool I/O. Tool inputs/outputs tend to be JSON blobs of structured
    # data — half the prompt cap is plenty.
    "tool_input": 2048,
    "tool_output": 2048,
    "arguments": 2048,
    "result": 2048,
    "parameters": 2048,
    # State snapshots — graph state, agent state, session state.
    "state": 8192,
    "state_snapshot": 8192,
    "context": 8192,
    "history": 8192,
    "memory": 8192,
    # Errors are always short — anything longer is noise.
    "error": 1024,
    "error_message": 1024,
    "exception": 1024,
    "reason": 1024,
    # Tracebacks are handled by frame count, not char count. The value
    # here is a fallback ceiling if the value isn't recognised as a
    # traceback (see :func:`_truncate_traceback`).
    "traceback": 8192,
    "stacktrace": 8192,
    "stack": 8192,
    # Binary-ish fields: drop entirely, replace with hash reference.
    "screenshot": DROP,
    "image_data": DROP,
    "image": DROP,
    "image_b64": DROP,
    "screenshot_b64": DROP,
    "binary_data": DROP,
    # Browser-specific (ports for browser_use adapter). Browser_use
    # navigation captures DOM HTML which can be hundreds of KB per
    # step; 16 KiB lets the schema/structure survive while clipping
    # the bulk.
    "html": 16384,
    "dom": 16384,
    "page_content": 16384,
    # System prompt / instructions — long but bounded at the same cap
    # as a regular prompt.
    "instructions": 4096,
    "system_message": 4096,
    "system_prompt": 4096,
}

# Number of stack frames to retain when truncating a traceback. A
# typical Python traceback frame is ~80-200 chars; 8 frames = ~1-2 KiB
# which fits comfortably under the L4 event budget.
DEFAULT_TRACEBACK_FRAMES = 8

# Default ceiling applied to fields NOT listed in ``DEFAULT_FIELD_CAPS``
# when the policy's ``apply_to_unknown_fields`` flag is set. Defaults
# to None (unknown fields not truncated) so policy adoption is
# strictly additive — adapters opting in to a stricter policy can
# enable it.
DEFAULT_UNKNOWN_FIELD_CAP = 16384


# ---------------------------------------------------------------------------
# Policy dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldTruncationPolicy:
    """Per-field truncation rules for an adapter.

    Instances are immutable; create variants via :meth:`with_overrides`.

    Attributes:
        field_caps: Maps ``field_name`` → max chars. ``0`` means drop
            the value and replace with a hash reference. Lookup is
            case-insensitive and uses exact key match within nested
            dicts.
        max_traceback_frames: Number of frames retained when a value
            is recognised as a traceback (string containing ``"File "``
            and line markers, or list of frame strings).
        unknown_field_cap: Optional ceiling for fields not in
            ``field_caps`` when ``apply_to_unknown_fields`` is True.
        apply_to_unknown_fields: If True, fields not in ``field_caps``
            are still capped at ``unknown_field_cap``. Defaults False
            to keep adoption strictly additive.
        max_list_items: Maximum elements retained in any list-valued
            payload field. Beyond this, the list is truncated and a
            sentinel string ``"... (N more items truncated)"`` is
            appended.
        recursion_limit: Maximum recursion depth for nested dicts /
            lists. Beyond this, the value is replaced with the string
            ``"<recursion limit reached>"`` to prevent infinite loops
            on cyclic structures (which can occur with framework
            objects that hold backreferences).
    """

    field_caps: Dict[str, int] = field(default_factory=lambda: dict(DEFAULT_FIELD_CAPS))
    max_traceback_frames: int = DEFAULT_TRACEBACK_FRAMES
    unknown_field_cap: int = DEFAULT_UNKNOWN_FIELD_CAP
    apply_to_unknown_fields: bool = False
    max_list_items: int = 100
    recursion_limit: int = 8

    def with_overrides(self, **overrides: Any) -> "FieldTruncationPolicy":
        """Return a new policy with selected attributes overridden.

        ``field_caps`` overrides are MERGED into the existing cap
        dict; all other attributes are replaced wholesale.
        """
        new_caps = dict(self.field_caps)
        if "field_caps" in overrides:
            extra_caps = overrides.pop("field_caps")
            if not isinstance(extra_caps, dict):
                raise TypeError("field_caps override must be a dict")
            new_caps.update(extra_caps)
        return FieldTruncationPolicy(
            field_caps=new_caps,
            max_traceback_frames=overrides.get("max_traceback_frames", self.max_traceback_frames),
            unknown_field_cap=overrides.get("unknown_field_cap", self.unknown_field_cap),
            apply_to_unknown_fields=overrides.get(
                "apply_to_unknown_fields", self.apply_to_unknown_fields
            ),
            max_list_items=overrides.get("max_list_items", self.max_list_items),
            recursion_limit=overrides.get("recursion_limit", self.recursion_limit),
        )

    def cap_for(self, field_name: str) -> Optional[int]:
        """Return the configured cap for ``field_name``, or None.

        Lookup is case-insensitive on the normalised field name (lower
        case, no surrounding whitespace).
        """
        if not field_name:
            return None
        key = field_name.strip().lower()
        if key in self.field_caps:
            return self.field_caps[key]
        if self.apply_to_unknown_fields:
            return self.unknown_field_cap
        return None


DEFAULT_POLICY = FieldTruncationPolicy()
"""The default policy used by adapters that opt in without overrides."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Matches a CPython traceback frame line: ``  File "x.py", line N, in fn``.
_TRACEBACK_FRAME_RE = re.compile(r'^\s*File "[^"]+", line \d+', re.MULTILINE)


def _hash_reference(value: Any, kind: str = "value") -> str:
    """Build a deterministic, dropped-payload reference string.

    The reference includes a SHA-256 prefix of the original bytes so
    customers can correlate dropped values across emissions (same
    screenshot → same hash).
    """
    try:
        if isinstance(value, (bytes, bytearray)):
            data = bytes(value)
        elif isinstance(value, str):
            data = value.encode("utf-8", errors="replace")
        else:
            data = repr(value).encode("utf-8", errors="replace")
    except Exception:
        data = b"<unhashable>"
    digest = hashlib.sha256(data).hexdigest()[:16]
    return f"<dropped:{kind}:sha256:{digest}>"


def _utf8_safe_clip(text: str, max_chars: int) -> str:
    """Clip ``text`` to at most ``max_chars`` characters.

    Operates on Python ``str`` (already-decoded UTF-8) so naive
    slicing is codepoint-safe — Python ``str`` indexing is by
    codepoint, not byte. Surrogate pairs only matter on narrow Python
    builds (pre-3.3), which we no longer support. We still validate
    by re-encoding to ensure the result round-trips through UTF-8;
    if encoding fails (lone surrogates, etc.) we replace the offending
    bytes rather than raise.
    """
    if max_chars < 0:
        max_chars = 0
    if len(text) <= max_chars:
        return text
    clipped = text[:max_chars]
    # Re-encode/decode to guarantee a valid UTF-8 sequence even if the
    # source string contained surrogates (e.g. unpaired UTF-16 from a
    # Windows clipboard input).
    try:
        clipped.encode("utf-8")
    except UnicodeEncodeError:
        clipped = clipped.encode("utf-8", errors="replace").decode("utf-8")
    return clipped


def _looks_like_traceback(value: str) -> bool:
    """Heuristic: True if the string looks like a Python traceback."""
    if "Traceback (most recent call last):" in value:
        return True
    # 2+ frame markers also looks like a traceback even without header.
    return len(_TRACEBACK_FRAME_RE.findall(value)) >= 2


def _truncate_traceback(value: str, max_frames: int) -> str:
    """Keep the first ``max_frames`` frames of a traceback string."""
    if max_frames <= 0:
        return ""
    lines = value.splitlines()
    if not lines:
        return value
    frame_indices: List[int] = []
    for i, line in enumerate(lines):
        if _TRACEBACK_FRAME_RE.match(line):
            frame_indices.append(i)
    if len(frame_indices) <= max_frames:
        return value
    # Keep header + first N frames (each frame = "File..." line + the
    # immediately following code line if present). We slice up to the
    # start of the (max_frames+1)th frame.
    cut_at = frame_indices[max_frames]
    kept = "\n".join(lines[:cut_at]).rstrip()
    dropped = len(frame_indices) - max_frames
    return f"{kept}\n  ... ({dropped} more frame{'s' if dropped != 1 else ''} truncated)"


def _stringify(value: Any) -> str:
    """Coerce non-string values to string for char-count truncation."""
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, bytearray)):
        try:
            return bytes(value).decode("utf-8", errors="replace")
        except Exception:
            return repr(value)
    return str(value)


# ---------------------------------------------------------------------------
# Public truncation API
# ---------------------------------------------------------------------------


def truncate_field(
    value: Any,
    field_name: str,
    policy: FieldTruncationPolicy = DEFAULT_POLICY,
    *,
    _path: str = "",
    _depth: int = 0,
    _truncated: Optional[List[str]] = None,
) -> Any:
    """Recursively truncate ``value`` according to ``policy``.

    Args:
        value: Arbitrary payload — string, bytes, dict, list, or
            primitive. Dicts and lists are walked recursively.
        field_name: The name this value is associated with in its
            parent payload. Used for cap lookup. Empty string is
            allowed for top-level invocations.
        policy: The truncation policy to apply.
        _path: Internal — full dotted path of the field for the
            ``_truncated_fields`` audit list.
        _depth: Internal — current recursion depth.
        _truncated: Internal — accumulator for truncated field paths.

    Returns:
        The truncated value. Container types (dict, list) are walked
        recursively; the original is NOT mutated. The
        ``_truncated_fields`` audit list (when supplied) is mutated in
        place to record any truncation that occurred.

    Notes:
        Use :func:`truncate_payload` for a complete payload — it
        manages the audit list and attaches it to the returned dict
        for the caller.
    """
    if _truncated is None:
        _truncated = []

    # Recursion guard.
    if _depth >= policy.recursion_limit:
        _truncated.append(f"{_path or field_name}:recursion-limit")
        return "<recursion limit reached>"

    cap = policy.cap_for(field_name)

    # Dict: walk keys recursively. Cap (if any) is NOT applied to dict
    # itself; it cascades to keys with the same name in nested dicts.
    if isinstance(value, dict):
        new_dict: Dict[str, Any] = {}
        for key, sub_value in value.items():
            sub_path = f"{_path}.{key}" if _path else str(key)
            new_dict[key] = truncate_field(
                sub_value,
                str(key),
                policy,
                _path=sub_path,
                _depth=_depth + 1,
                _truncated=_truncated,
            )
        return new_dict

    # List: cap element count, then truncate each element.
    if isinstance(value, list):
        original_len = len(value)
        items_iter: List[Any] = list(value)
        list_truncated = False
        if original_len > policy.max_list_items:
            items_iter = items_iter[: policy.max_list_items]
            list_truncated = True
        new_list: List[Any] = []
        for idx, sub_value in enumerate(items_iter):
            sub_path = f"{_path}[{idx}]" if _path else f"[{idx}]"
            new_list.append(
                truncate_field(
                    sub_value,
                    field_name,  # list elements inherit the parent field name
                    policy,
                    _path=sub_path,
                    _depth=_depth + 1,
                    _truncated=_truncated,
                )
            )
        if list_truncated:
            dropped = original_len - policy.max_list_items
            new_list.append(f"... ({dropped} more items truncated)")
            _truncated.append(f"{_path or field_name}:list-{original_len}->{policy.max_list_items}")
        return new_list

    # Tuple: treated like list for truncation but coerced back. (Most
    # adapters emit dicts/lists; tuples are rare but appear in
    # serialised tool args.)
    if isinstance(value, tuple):
        as_list = truncate_field(
            list(value),
            field_name,
            policy,
            _path=_path,
            _depth=_depth,
            _truncated=_truncated,
        )
        return tuple(as_list) if isinstance(as_list, list) else as_list

    # Primitives that do not need truncation: bool, int, float, None.
    # IMPORTANT: bool MUST be checked before int because ``bool`` is a
    # subclass of ``int``.
    if value is None or isinstance(value, (bool, int, float)):
        return value

    # No cap configured for this field → leave value as-is. (Subject
    # to ``apply_to_unknown_fields`` via :meth:`cap_for`.)
    if cap is None:
        return value

    # DROP: replace with hash reference.
    if cap == DROP:
        ref = _hash_reference(value, kind=field_name.lower() or "value")
        _truncated.append(f"{_path or field_name}:dropped")
        return ref

    # String / bytes / coerced string truncation.
    text = _stringify(value)

    # Traceback handling: if the field is named like one OR the
    # content matches a traceback signature, truncate by frame count.
    looks_like_tb = field_name.lower() in {"traceback", "stacktrace", "stack"} and _looks_like_traceback(text)
    if looks_like_tb:
        truncated = _truncate_traceback(text, policy.max_traceback_frames)
        if truncated != text:
            _truncated.append(f"{_path or field_name}:traceback-frames")
        return truncated

    # Char-count truncation.
    if len(text) <= cap:
        # Still convert non-str values back? No — preserve original
        # type when no truncation needed.
        if isinstance(value, (bytes, bytearray)):
            return text  # decoded copy
        return value
    clipped = _utf8_safe_clip(text, cap)
    suffix = f"... ({len(text) - cap} more chars truncated)"
    _truncated.append(f"{_path or field_name}:chars-{len(text)}->{cap}")
    return clipped + suffix


def truncate_payload(
    payload: Dict[str, Any],
    policy: FieldTruncationPolicy = DEFAULT_POLICY,
) -> Tuple[Dict[str, Any], List[str]]:
    """Truncate every field in an event payload dict.

    Args:
        payload: The raw event payload. The original dict is NOT
            mutated.
        policy: Truncation policy. Defaults to :data:`DEFAULT_POLICY`.

    Returns:
        A ``(truncated_payload, truncated_fields)`` tuple. The list
        contains dotted field paths describing every truncation that
        occurred (e.g. ``"messages[0].content:chars-12345->4096"``,
        ``"screenshot:dropped"``). Callers can attach this to the
        emitted payload as ``_truncated_fields`` so observability
        surfaces what was clipped.
    """
    if not isinstance(payload, dict):
        raise TypeError(f"payload must be a dict, got {type(payload).__name__}")
    audit: List[str] = []
    truncated = truncate_field(
        payload,
        "",  # top-level has no field name
        policy,
        _path="",
        _depth=0,
        _truncated=audit,
    )
    # truncate_field returns a dict for dict input.
    if not isinstance(truncated, dict):  # pragma: no cover — defensive
        return payload, audit
    return truncated, audit


__all__ = [
    "DEFAULT_FIELD_CAPS",
    "DEFAULT_POLICY",
    "DEFAULT_TRACEBACK_FRAMES",
    "DEFAULT_UNKNOWN_FIELD_CAP",
    "DROP",
    "FieldTruncationPolicy",
    "truncate_field",
    "truncate_payload",
]
