"""Shared handoff metadata helpers for LayerLens framework adapters.

Provides standardised plumbing for ``agent.handoff`` event metadata so
every adapter that emits handoffs surfaces the same downstream contract
to the platform — regardless of which framework produced the event.

The contract has three components:

* ``handoff_seq`` — a monotonically increasing integer per adapter
  instance. Disambiguates events that share a wall-clock timestamp
  (sub-microsecond collisions are common when handoffs fire back-to-back
  in async loops). Must be thread-safe because adapters can run multiple
  agent invocations concurrently.
* ``context_hash`` — a SHA-256 digest of the serialised handoff context.
  Lets the replay engine assert that re-execution received the same
  payload that the original run did. Computed via canonical JSON so
  semantically-equal contexts hash identically across runs.
* ``preview`` — a length-bounded human-readable summary of the
  context. 256 chars by default; an ellipsis (``…``) marks truncation
  so dashboards never silently misrepresent payload size.

Adapter authors should use :func:`build_handoff_payload` to populate
all three fields in a single call. The lower-level helpers
(:func:`compute_context_hash`, :func:`make_preview`,
:class:`HandoffSequencer`) are exposed for cases that need finer
control (e.g. re-using the seq counter across multiple emit sites).

Origin notes — this module consolidates patterns previously duplicated
across:

* CrewAI ``delegation.py`` (delegation_seq + context_hash + 500-char preview)
* LangGraph ``handoff.py`` (canonical-JSON SHA-256)
* AutoGen ``lifecycle.py`` (message_seq)

See ``A:/tmp/adapter-cross-pollination-audit.md`` §2.5 for the full
inventory of adapters that previously lacked one or more of these
fields.
"""

from __future__ import annotations

import json
import hashlib
import threading
from typing import Any, Dict, Mapping, Optional
from datetime import datetime, timezone
from dataclasses import field, dataclass

__all__ = [
    "DEFAULT_PREVIEW_MAX_CHARS",
    "HandoffMetadata",
    "HandoffSequencer",
    "build_handoff_payload",
    "compute_context_hash",
    "make_preview",
]


# Public constant — keep here so adapters can import the same default
# instead of redefining it. Chosen to match the existing OpenTelemetry /
# StreamingSpan convention of "small enough to fit in a Kafka record
# header, large enough to be useful in a dashboard tooltip."
DEFAULT_PREVIEW_MAX_CHARS: int = 256

# Hash prefix discriminator. Using ``sha256:`` instead of a bare hex
# string future-proofs the contract: if we ever rotate to BLAKE3 / SHA-3
# the platform can branch on prefix without reading every emitter.
_HASH_PREFIX: str = "sha256:"

# Truncation marker. Single-character ellipsis (U+2026) keeps the
# truncated string within the requested cap by exactly 1 char of
# overhead — using ``"..."`` would steal three chars instead.
_ELLIPSIS: str = "…"


@dataclass
class HandoffMetadata:
    """Standardised metadata for a single ``agent.handoff`` event.

    Every adapter that emits ``agent.handoff`` should populate one of
    these (typically via :func:`build_handoff_payload`) so the platform
    receives a consistent shape regardless of upstream framework.

    Attributes:
        seq: Monotonically increasing sequence number scoped to the
            emitting adapter instance. Starts at 1.
        context_hash: ``"sha256:" + hex_digest`` of the canonicalised
            handoff context. Empty / ``None`` contexts hash to the
            digest of ``"{}"`` so the field is never absent.
        preview: Truncated, human-readable rendering of the context
            (≤ ``DEFAULT_PREVIEW_MAX_CHARS`` by default).
        from_agent: Identifier of the delegating agent.
        to_agent: Identifier of the receiving agent.
        timestamp: Wall-clock timestamp of the handoff. UTC, populated
            via :func:`datetime.datetime.now` at construction.
    """

    seq: int
    context_hash: str
    preview: str
    from_agent: str
    to_agent: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_payload(self) -> Dict[str, Any]:
        """Render the metadata as a flat dict for ``emit_dict_event``.

        Returns a new dict each call (no aliasing). Uses ``isoformat``
        so the timestamp serialises cleanly through the JSON-based
        event sinks downstream.
        """
        return {
            "handoff_seq": self.seq,
            "context_hash": self.context_hash,
            "context_preview": self.preview,
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "timestamp": self.timestamp.isoformat(),
        }


def compute_context_hash(state: Optional[Mapping[str, Any]]) -> str:
    """Return a deterministic SHA-256 digest of ``state``.

    The state is serialised via canonical JSON (``sort_keys=True``,
    ``default=str`` for non-JSON-native values) so:

    * Two semantically-equal contexts always hash identically across
      machines, Python versions, and adapter instances.
    * Non-serialisable values (datetimes, custom objects, sets) are
      coerced to ``str`` via the JSON ``default=`` hook rather than
      raising — handoff hashing must never break the emitting agent.

    Args:
        state: The handoff context. ``None`` and empty dicts both hash
            to the digest of ``"{}"`` so the contract returns a non-
            empty string in every case.

    Returns:
        A string of the form ``"sha256:<64 hex chars>"``.
    """
    if state is None:
        canonical = "{}"
    else:
        try:
            canonical = json.dumps(
                dict(state),
                sort_keys=True,
                default=str,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        except (TypeError, ValueError):
            # Last-resort fallback: stringify the whole mapping. We
            # still want a stable hash even for pathological inputs.
            canonical = repr(sorted(state.items())) if hasattr(state, "items") else repr(state)

    digest = hashlib.sha256(canonical.encode("utf-8", errors="replace")).hexdigest()
    return _HASH_PREFIX + digest


def make_preview(content: Any, max_chars: int = DEFAULT_PREVIEW_MAX_CHARS) -> str:
    """Render ``content`` as a length-bounded preview string.

    Behaviour:

    * ``None`` returns the empty string.
    * Non-string values are coerced via ``str()``; if coercion raises
      (rare — typically a faulty ``__str__``) the function returns
      ``"<unrepresentable>"`` so callers never see an exception.
    * Strings longer than ``max_chars`` are truncated and a U+2026
      ellipsis is appended. Total length of the returned string never
      exceeds ``max_chars`` — the ellipsis displaces one char of
      content rather than appending to it.
    * ``max_chars <= 0`` returns the empty string (defensive — callers
      that want "no preview" can pass ``0``).

    Args:
        content: The value to summarise.
        max_chars: Maximum length of the returned string, including the
            ellipsis when truncation occurs.

    Returns:
        A string of length ``min(len(stringified_content), max_chars)``.
    """
    if content is None or max_chars <= 0:
        return ""

    if isinstance(content, str):
        text = content
    else:
        try:
            text = str(content)
        except Exception:
            return "<unrepresentable>"

    if len(text) <= max_chars:
        return text

    # Truncate to ``max_chars - 1`` and append the ellipsis so total
    # length is exactly ``max_chars``.
    return text[: max_chars - 1] + _ELLIPSIS


class HandoffSequencer:
    """Thread-safe monotonic counter for adapter-scoped handoff seqs.

    One instance per adapter — typically constructed in the adapter's
    ``__init__`` and held as ``self._handoff_sequencer``. Concurrent
    agent invocations (asyncio gathers, threadpool workers, and
    framework callbacks firing from multiple OS threads) all draw from
    the same instance, so the lock is mandatory.

    The counter is 1-indexed: ``next()`` returns 1 on first call so
    downstream consumers can use ``handoff_seq > 0`` as a "have we
    observed any handoffs yet?" predicate without an extra null check.

    Example::

        class MyAdapter(BaseAdapter):
            def __init__(self) -> None:
                super().__init__()
                self._handoff_sequencer = HandoffSequencer()

            def on_handoff(self, src: str, dst: str, ctx: dict) -> None:
                payload = build_handoff_payload(
                    sequencer=self._handoff_sequencer,
                    from_agent=src,
                    to_agent=dst,
                    context=ctx,
                )
                self.emit_dict_event("agent.handoff", payload)
    """

    __slots__ = ("_counter", "_lock")

    def __init__(self) -> None:
        self._counter: int = 0
        self._lock: threading.Lock = threading.Lock()

    def next(self) -> int:
        """Return the next sequence number (1-indexed, monotonic).

        Holds the lock for the increment + read so two callers cannot
        observe the same value. The lock is uncontended in the
        single-threaded common case, so overhead is a single CAS.
        """
        with self._lock:
            self._counter += 1
            return self._counter

    @property
    def current(self) -> int:
        """The most-recently-issued seq value (0 before any ``next()``).

        Reads do not need the lock because integer assignment is
        atomic in CPython and we tolerate the read returning a value
        that was already stale by the time the caller used it. This
        property exists for diagnostics / dashboards, not for issuing
        new IDs.
        """
        return self._counter

    def reset(self) -> None:
        """Reset the counter back to zero.

        Intended for adapter ``disconnect()`` paths so a subsequent
        ``connect()`` doesn't carry over seqs from the previous
        session.
        """
        with self._lock:
            self._counter = 0


def build_handoff_payload(
    sequencer: HandoffSequencer,
    from_agent: str,
    to_agent: str,
    context: Optional[Mapping[str, Any]] = None,
    preview_text: Optional[str] = None,
    preview_max_chars: int = DEFAULT_PREVIEW_MAX_CHARS,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble a fully-populated handoff event payload.

    Convenience wrapper that:

    1. Allocates the next ``handoff_seq`` from ``sequencer``.
    2. Computes the canonical SHA-256 ``context_hash``.
    3. Builds the preview from ``preview_text`` if supplied, else from
       a stringified ``context``.
    4. Merges ``extra`` (e.g. framework-specific ``reason``,
       ``framework`` tag) on top of the standard fields.

    Args:
        sequencer: The adapter's :class:`HandoffSequencer`.
        from_agent: Source agent identifier.
        to_agent: Destination agent identifier.
        context: Handoff payload (used for hashing and, when
            ``preview_text`` is omitted, for the preview).
        preview_text: Optional explicit preview string. Use this when
            the framework already provides a human-readable summary
            (e.g. CrewAI delegation message) — the helper truncates
            it to ``preview_max_chars``.
        preview_max_chars: Maximum preview length in characters.
        extra: Additional fields to merge into the payload. Keys that
            collide with the standard schema are NOT overridden — the
            standard fields win, so callers can't accidentally
            shadow ``handoff_seq`` etc.

    Returns:
        A dict ready to pass to ``adapter.emit_dict_event(
        "agent.handoff", payload)``. Always contains the keys:
        ``from_agent``, ``to_agent``, ``handoff_seq``,
        ``context_hash``, ``context_preview``, ``timestamp``.
    """
    seq = sequencer.next()
    context_hash = compute_context_hash(context)

    if preview_text is not None:
        preview = make_preview(preview_text, max_chars=preview_max_chars)
    else:
        preview = make_preview(context, max_chars=preview_max_chars) if context else ""

    payload: Dict[str, Any] = {
        "from_agent": from_agent,
        "to_agent": to_agent,
        "handoff_seq": seq,
        "context_hash": context_hash,
        "context_preview": preview,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if extra:
        for key, value in extra.items():
            payload.setdefault(key, value)

    return payload
