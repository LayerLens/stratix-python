"""Shared cross-adapter memory persistence module.

This module ports the **episodic / procedural / semantic** memory
pattern carried ad-hoc by the four mature framework adapters
(LangChain, AutoGen, CrewAI, Semantic Kernel) into a shared,
replay-safe primitive that any framework adapter can plug in to
deliver cross-conversation recall.

Background — Cross-pollination audit §2.1
=========================================

The audit at ``A:/tmp/adapter-cross-pollination-audit.md`` identified
memory persistence as the **highest-value cross-cutting lift** from
the mature adapters to the seven lighter adapters that lack it
(``agno``, ``ms_agent_framework``, ``openai_agents``, ``llama_index``,
``google_adk``, ``bedrock_agents``, ``browser_use``). Without this
plumbing those adapters behave as "goldfish agents" — every run starts
from a blank slate, which is the difference between a usable
production agent and a demo.

The mature adapters all delegate to an external ``AgentMemoryService``
(``stratix.memory.models.MemoryEntry``). That contract works in the
``ateam`` monorepo where the service is available; for the SDK in
``stratix-python`` the service is **not** part of the runtime, so the
shared module here owns the in-process snapshot lifecycle and exposes
a ``MemorySnapshot`` shape that any external service can consume from
:meth:`BaseAdapter.serialize_for_replay` output.

Design contract
===============

Three memory buckets, modelled after the canonical agent-memory
literature (LangChain memory module; CrewAI procedural memory;
AutoGen episodic/semantic split):

* **Episodic** — per-turn input/output pairs, ordered by ``turn_index``.
  Bounded ring (default 200 entries) — drops oldest on overflow.
* **Procedural** — learned recurring patterns derived from the
  episodic stream (e.g. "tool X is called immediately after tool Y").
  Bounded by a per-pattern occurrence cap (default 16 unique patterns).
* **Semantic** — long-lived key/value facts that survive across many
  conversations (e.g. user preferences, conversation summaries).
  Bounded (default 64 entries) — least-recently-set eviction.

Each :class:`MemorySnapshot` is **content-hash addressable** via a
SHA-256 of the canonical-JSON serialization of all three buckets plus
``turn_index`` and ``org_id``. Two snapshots with identical content
produce identical hashes — supports deduplication at the storage
layer.

Replay safety contract
======================

A :class:`MemorySnapshot` is **deterministically restorable**: passing
the same snapshot to :meth:`MemoryRecorder.restore` and emitting the
same input sequence yields the same final snapshot. This means an
adapter that includes its current snapshot in
:meth:`BaseAdapter.serialize_for_replay` output can replay an agent
run with byte-exact memory state.

Multi-tenancy
=============

Every :class:`MemorySnapshot` and every :class:`MemoryRecorder` is
scoped to exactly one ``org_id``. A recorder constructed for tenant
A cannot ingest a snapshot from tenant B — :meth:`restore` raises
``ValueError`` on tenant mismatch. This mirrors the
:class:`BaseAdapter` org_id contract documented in
``docs/adapters/multi-tenancy.md``.
"""

from __future__ import annotations

import copy
import json
import time
import hashlib
import threading
from typing import Any, Dict, List, Tuple, Mapping, Optional
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# Bounded buffers — CLAUDE.md "every cache must be bounded". These caps
# are conservative defaults; callers wanting a different size construct
# the recorder with explicit ``max_*`` kwargs.
DEFAULT_MAX_EPISODIC: int = 200
DEFAULT_MAX_PROCEDURAL: int = 16
DEFAULT_MAX_SEMANTIC: int = 64

# Per-turn truncation — the recorder is *not* the place to enforce
# field-size policy (that's the truncation module from cross-poll #3).
# But to prevent a single oversized turn from blowing past memory caps
# we apply a hard char-cap on individual values. This is a
# defence-in-depth limit, not a policy substitute.
_PER_FIELD_HARD_CAP: int = 8192

# Procedural pattern detection looks at ``_PROCEDURAL_WINDOW`` recent
# turns to find recurring (tool_name, next_action) pairs.
_PROCEDURAL_WINDOW: int = 16


# ---------------------------------------------------------------------------
# Dataclass: MemorySnapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MemorySnapshot:
    """Immutable, content-addressable memory snapshot.

    Snapshots are produced by :meth:`MemoryRecorder.snapshot` and
    consumed by :meth:`MemoryRecorder.restore`. They are designed to:

    * Be **trivially serialisable** to JSON (every field is a primitive
      or a dict/list of primitives).
    * Be **content-addressable** via :attr:`content_hash` for
      deduplication and integrity checks.
    * Be **replay-safe**: restoring an adapter's recorder from a
      snapshot and feeding the same input sequence reproduces the same
      next snapshot.
    * Be **multi-tenant-scoped** via :attr:`org_id` — snapshots from
      tenant A cannot be restored into a tenant-B recorder.

    Attributes:
        turn_index: Monotonic counter of completed turns at the moment
            of the snapshot. Starts at 0 for an empty recorder.
        episodic: Ordered list of recent turn dicts. Each turn carries
            ``turn_index``, ``timestamp_ns``, ``agent_name``, ``input``,
            ``output``, optional ``error``, and optional ``tools``.
        procedural: Ordered list of detected patterns. Each pattern is
            a dict of the form
            ``{"pattern": [...], "count": int, "last_seen_turn": int}``.
        semantic: Long-lived key/value store. Values are strings (or
            stringified) — callers wanting structured semantic memory
            should JSON-encode their value before storing.
        content_hash: SHA-256 hex digest of the canonical JSON
            representation of ``(turn_index, episodic, procedural,
            semantic, org_id)``. Identical content → identical hash.
        org_id: Tenant binding. Mirrors the adapter's bound ``org_id``
            and prevents cross-tenant restore.
    """

    turn_index: int
    episodic: List[Dict[str, Any]]
    procedural: List[Dict[str, Any]]
    semantic: Dict[str, str]
    content_hash: str
    org_id: str

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict view of the snapshot.

        Used by :meth:`BaseAdapter.serialize_for_replay` when including
        the memory state in a :class:`ReplayableTrace`. The shape is
        stable: snapshot reconstruction reads the same keys back.
        """
        return {
            "turn_index": self.turn_index,
            "episodic": copy.deepcopy(self.episodic),
            "procedural": copy.deepcopy(self.procedural),
            "semantic": dict(self.semantic),
            "content_hash": self.content_hash,
            "org_id": self.org_id,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MemorySnapshot":
        """Reconstruct a snapshot from a previously-serialised dict.

        Args:
            data: Mapping produced by :meth:`to_dict`.

        Returns:
            A new :class:`MemorySnapshot` with deep-copied collections
            so callers can mutate ``data`` afterwards without affecting
            the snapshot.

        Raises:
            ValueError: If ``data`` is missing required fields.
        """
        for required in ("turn_index", "episodic", "procedural", "semantic", "content_hash", "org_id"):
            if required not in data:
                raise ValueError(f"MemorySnapshot.from_dict missing required field: {required}")
        return cls(
            turn_index=int(data["turn_index"]),
            episodic=copy.deepcopy(list(data["episodic"])),
            procedural=copy.deepcopy(list(data["procedural"])),
            semantic=dict(data["semantic"]),
            content_hash=str(data["content_hash"]),
            org_id=str(data["org_id"]),
        )


# ---------------------------------------------------------------------------
# Empty-snapshot factory
# ---------------------------------------------------------------------------


def _empty_snapshot(org_id: str) -> MemorySnapshot:
    """Build an empty snapshot bound to ``org_id``.

    Used for the initial state of a fresh :class:`MemoryRecorder` and
    by tests verifying the empty-state hash invariant.
    """
    return _build_snapshot(
        turn_index=0,
        episodic=[],
        procedural=[],
        semantic={},
        org_id=org_id,
    )


def _canonical_json(value: Any) -> str:
    """Return a deterministic JSON encoding suitable for hashing.

    ``sort_keys=True`` + ``separators`` removes whitespace variance.
    ``default=str`` is *not* used — non-JSON-safe inputs are a caller
    bug and should raise so the hash never silently changes shape.
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _compute_content_hash(
    *,
    turn_index: int,
    episodic: List[Dict[str, Any]],
    procedural: List[Dict[str, Any]],
    semantic: Dict[str, str],
    org_id: str,
) -> str:
    """Compute the SHA-256 content hash for a snapshot.

    The exact bucket order and key set is part of the public contract:
    changing it breaks dedup against historical snapshots. If the
    content shape ever needs to grow, do it via a versioned wrapper
    around the raw hash, not by mutating this function.
    """
    payload = {
        "turn_index": turn_index,
        "episodic": episodic,
        "procedural": procedural,
        "semantic": semantic,
        "org_id": org_id,
    }
    encoded = _canonical_json(payload).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _build_snapshot(
    *,
    turn_index: int,
    episodic: List[Dict[str, Any]],
    procedural: List[Dict[str, Any]],
    semantic: Dict[str, str],
    org_id: str,
) -> MemorySnapshot:
    """Construct a :class:`MemorySnapshot` with a freshly-computed hash."""
    content_hash = _compute_content_hash(
        turn_index=turn_index,
        episodic=episodic,
        procedural=procedural,
        semantic=semantic,
        org_id=org_id,
    )
    return MemorySnapshot(
        turn_index=turn_index,
        episodic=copy.deepcopy(episodic),
        procedural=copy.deepcopy(procedural),
        semantic=dict(semantic),
        content_hash=content_hash,
        org_id=org_id,
    )


# ---------------------------------------------------------------------------
# Helper: cap a value at the per-field hard cap
# ---------------------------------------------------------------------------


def _cap_value(value: Any) -> Any:
    """Apply the defence-in-depth per-field char cap.

    Returns the value unchanged for non-string types and short
    strings; truncates long strings with a deterministic suffix that
    makes the truncation visible in downstream tooling.
    """
    if isinstance(value, str) and len(value) > _PER_FIELD_HARD_CAP:
        # The suffix records the original length so reviewers can see
        # how much was elided; the deterministic shape (no timestamps,
        # no random IDs) preserves replay determinism.
        return value[:_PER_FIELD_HARD_CAP] + f"<...truncated:orig_len={len(value)}>"
    return value


def _normalise_turn(
    *,
    turn_index: int,
    timestamp_ns: int,
    agent_name: Optional[str],
    input_data: Any,
    output_data: Any,
    error: Optional[str],
    tools: Optional[List[str]],
    extra: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Build the canonical episodic-turn dict.

    Every recorded turn passes through this function so the schema is
    enforced at one place. Non-JSON-safe inputs are coerced via
    ``str()`` — adapters that want richer fidelity should serialise
    upstream.
    """
    turn: Dict[str, Any] = {
        "turn_index": turn_index,
        "timestamp_ns": timestamp_ns,
        "agent_name": agent_name or "",
        "input": _cap_value(input_data if isinstance(input_data, (str, int, float, bool, type(None))) else str(input_data)),
        "output": _cap_value(output_data if isinstance(output_data, (str, int, float, bool, type(None))) else str(output_data)),
    }
    if error is not None:
        turn["error"] = _cap_value(error)
    if tools:
        # Cap the tool list itself (not the strings inside) at a sane
        # ceiling to prevent runaway tool-name accumulation.
        turn["tools"] = [str(t) for t in tools[:32]]
    if extra:
        # ``extra`` is opt-in metadata. Keys are sorted to keep the
        # hash deterministic even if callers pass dict literals.
        normalised_extra: Dict[str, Any] = {}
        for k in sorted(str(k) for k in extra.keys()):
            normalised_extra[k] = _cap_value(extra[k] if isinstance(extra[k], (str, int, float, bool, type(None))) else str(extra[k]))
        turn["extra"] = normalised_extra
    return turn


# ---------------------------------------------------------------------------
# MemoryRecorder
# ---------------------------------------------------------------------------


class MemoryRecorder:
    """Thread-safe accumulator wired into adapter lifecycle hooks.

    A :class:`MemoryRecorder` lives for the lifetime of an adapter
    instance (one per adapter). The adapter calls
    :meth:`record_turn` after every per-turn callback (typically right
    after emitting ``agent.output``). At any point — most commonly
    inside :meth:`BaseAdapter.serialize_for_replay` — the adapter
    calls :meth:`snapshot` to obtain an immutable, hashable
    :class:`MemorySnapshot` and embeds it in the replay trace.

    The recorder enforces the multi-tenant binding: it is constructed
    with the adapter's ``org_id`` and refuses to restore from a
    snapshot whose ``org_id`` does not match.

    Args:
        org_id: Tenant binding. Must be a non-empty string. Mirrors
            :attr:`BaseAdapter.org_id`.
        max_episodic: Maximum number of episodic turns retained. Older
            turns are dropped FIFO when the cap is reached.
        max_procedural: Maximum number of distinct procedural patterns
            retained.
        max_semantic: Maximum number of semantic key/value entries
            retained. Eviction is least-recently-set.
    """

    def __init__(
        self,
        *,
        org_id: str,
        max_episodic: int = DEFAULT_MAX_EPISODIC,
        max_procedural: int = DEFAULT_MAX_PROCEDURAL,
        max_semantic: int = DEFAULT_MAX_SEMANTIC,
    ) -> None:
        if not isinstance(org_id, str) or not org_id.strip():
            raise ValueError(
                "MemoryRecorder requires a non-empty org_id for multi-tenant "
                "scoping. Pass the adapter's org_id (CLAUDE.md multi-tenancy)."
            )
        if max_episodic < 1 or max_procedural < 1 or max_semantic < 1:
            raise ValueError("MemoryRecorder bounded buffer sizes must be >= 1")

        self._org_id: str = org_id
        self._max_episodic: int = max_episodic
        self._max_procedural: int = max_procedural
        self._max_semantic: int = max_semantic

        self._lock = threading.Lock()
        self._turn_index: int = 0
        self._episodic: List[Dict[str, Any]] = []
        # Procedural store keyed by canonical pattern string so we
        # detect repetition O(1).
        self._procedural: Dict[str, Dict[str, Any]] = {}
        # Semantic store keyed by user-supplied key. Insertion order
        # tracked separately for LRU eviction (3.7+ dicts preserve
        # insertion order, but we re-insert on update so eviction is
        # least-recently-*set*, matching the documented contract).
        self._semantic: Dict[str, str] = {}

    # --- Properties (read-only) ------------------------------------

    @property
    def org_id(self) -> str:
        """The tenant binding fixed at construction."""
        return self._org_id

    @property
    def turn_index(self) -> int:
        """Monotonic count of completed turns. Thread-safe read.

        Returned by value: the underlying counter is locked during
        :meth:`record_turn` and :meth:`restore`, but a read here is a
        plain int load.
        """
        return self._turn_index

    # --- Recording -------------------------------------------------

    def record_turn(
        self,
        *,
        agent_name: Optional[str] = None,
        input_data: Any = None,
        output_data: Any = None,
        error: Optional[str] = None,
        tools: Optional[List[str]] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> int:
        """Record one completed agent turn.

        Adapters wire this into their lifecycle right after the
        ``agent.output`` event emission. The turn data is enrolled in
        the episodic buffer; procedural patterns are detected from the
        recent window; the monotonic counter is incremented and
        returned.

        Args:
            agent_name: The agent that produced this turn (e.g.
                ``"researcher"``).
            input_data: The input the agent received.
            output_data: The output the agent produced.
            error: Optional error message if the turn failed.
            tools: Optional list of tool names invoked during the turn.
            extra: Optional additional metadata. Keys are sorted to
                keep snapshot hashes deterministic.

        Returns:
            The new ``turn_index`` after recording.
        """
        with self._lock:
            self._turn_index += 1
            turn = _normalise_turn(
                turn_index=self._turn_index,
                timestamp_ns=time.time_ns(),
                agent_name=agent_name,
                input_data=input_data,
                output_data=output_data,
                error=error,
                tools=list(tools) if tools else None,
                extra=extra,
            )
            self._episodic.append(turn)
            # Bounded: drop oldest turns FIFO. We use slicing rather
            # than ``deque`` because episodic is also serialised as a
            # plain list and we want zero-cost view semantics.
            if len(self._episodic) > self._max_episodic:
                drop = len(self._episodic) - self._max_episodic
                del self._episodic[:drop]
            self._detect_procedural_patterns()
            return self._turn_index

    def set_semantic(self, key: str, value: str) -> None:
        """Set or overwrite a long-lived semantic memory entry.

        Keys are sorted at snapshot time. Values are coerced to
        strings; structured semantic data should be JSON-encoded by
        the caller.

        Args:
            key: Semantic memory key. Must be non-empty.
            value: Value to store. Coerced via ``str()`` if not
                already a string. Hard-capped at
                :data:`_PER_FIELD_HARD_CAP` characters.

        Raises:
            ValueError: If ``key`` is empty.
        """
        if not isinstance(key, str) or not key.strip():
            raise ValueError("MemoryRecorder.set_semantic requires a non-empty key")
        capped = _cap_value(value if isinstance(value, str) else str(value))
        with self._lock:
            # Re-insert to refresh LRU order (3.7+ dicts).
            if key in self._semantic:
                del self._semantic[key]
            self._semantic[key] = capped
            # Bounded: evict least-recently-set entries.
            if len(self._semantic) > self._max_semantic:
                # ``next(iter(d))`` returns the first (oldest)
                # insertion key — LRU eviction.
                drop = len(self._semantic) - self._max_semantic
                for _ in range(drop):
                    oldest = next(iter(self._semantic))
                    del self._semantic[oldest]

    def clear(self) -> None:
        """Reset all buckets to empty; ``turn_index`` resets to 0.

        Useful for adapters that want a fresh memory state when a new
        conversation/session begins. Multi-tenant binding is
        preserved.
        """
        with self._lock:
            self._turn_index = 0
            self._episodic.clear()
            self._procedural.clear()
            self._semantic.clear()

    # --- Snapshots --------------------------------------------------

    def snapshot(self) -> MemorySnapshot:
        """Return an immutable, content-addressable snapshot.

        Adapters should call this from
        :meth:`BaseAdapter.serialize_for_replay` to embed the memory
        state in the replay trace. The returned snapshot is a deep
        copy — mutating the recorder afterwards never affects a
        previously-returned snapshot (immutability invariant).
        """
        with self._lock:
            procedural_list = self._procedural_as_sorted_list()
            return _build_snapshot(
                turn_index=self._turn_index,
                episodic=self._episodic,
                procedural=procedural_list,
                semantic=self._semantic,
                org_id=self._org_id,
            )

    def restore(self, snapshot: MemorySnapshot) -> None:
        """Replace the recorder's state with a previously-taken snapshot.

        The recorder is rebuilt to byte-exact equivalence: a fresh
        :meth:`snapshot` immediately after a :meth:`restore` returns a
        snapshot with the same :attr:`MemorySnapshot.content_hash`
        (deterministic round-trip). This is the foundation of
        replay-safe memory: the replay engine restores the recorder,
        then the adapter re-runs the agent and produces the same
        next-turn snapshot.

        Args:
            snapshot: The :class:`MemorySnapshot` to restore from.

        Raises:
            ValueError: If ``snapshot.org_id`` does not match the
                recorder's tenant binding (cross-tenant restore is
                prohibited).
            ValueError: If the snapshot's recorded
                :attr:`MemorySnapshot.content_hash` does not match the
                hash recomputed from its content (integrity check).
        """
        if snapshot.org_id != self._org_id:
            raise ValueError(
                f"MemoryRecorder.restore: snapshot org_id={snapshot.org_id!r} "
                f"does not match recorder org_id={self._org_id!r}. "
                "Cross-tenant restore is prohibited (CLAUDE.md multi-tenancy)."
            )
        # Verify the snapshot's stored hash matches its content. Guards
        # against accidentally-mutated dicts in transit.
        recomputed = _compute_content_hash(
            turn_index=snapshot.turn_index,
            episodic=snapshot.episodic,
            procedural=snapshot.procedural,
            semantic=snapshot.semantic,
            org_id=snapshot.org_id,
        )
        if recomputed != snapshot.content_hash:
            raise ValueError(
                "MemoryRecorder.restore: snapshot content_hash mismatch — "
                "snapshot has been tampered with or is corrupted. "
                f"Recorded={snapshot.content_hash} recomputed={recomputed}."
            )
        with self._lock:
            self._turn_index = snapshot.turn_index
            self._episodic = copy.deepcopy(snapshot.episodic)
            self._semantic = dict(snapshot.semantic)
            # Procedural store is keyed internally; rebuild the dict
            # form from the snapshot's list form.
            self._procedural = {}
            for entry in snapshot.procedural:
                pattern_key = _canonical_json(entry["pattern"])
                self._procedural[pattern_key] = {
                    "pattern": list(entry["pattern"]),
                    "count": int(entry["count"]),
                    "last_seen_turn": int(entry["last_seen_turn"]),
                }

    # --- Internal: procedural-pattern detection --------------------

    def _detect_procedural_patterns(self) -> None:
        """Scan recent turns for recurring tool sequences.

        Caller MUST hold ``self._lock``. The detector looks at the
        last :data:`_PROCEDURAL_WINDOW` turns and records any
        ``(prev_turn_tools, current_turn_tools)`` pair that recurs.

        The detection is deliberately simple — pairwise tool-list
        sequences only — to keep the algorithm O(window) per turn.
        Adapters wanting richer pattern detection can layer their own
        analysis on top of the episodic stream returned by
        :meth:`snapshot`.
        """
        if len(self._episodic) < 2:
            return
        window = self._episodic[-_PROCEDURAL_WINDOW:]
        for i in range(1, len(window)):
            prev_tools = window[i - 1].get("tools") or []
            cur_tools = window[i].get("tools") or []
            if not prev_tools and not cur_tools:
                continue
            pattern: List[List[str]] = [list(prev_tools), list(cur_tools)]
            pattern_key = _canonical_json(pattern)
            existing = self._procedural.get(pattern_key)
            if existing is not None:
                existing["count"] += 1
                existing["last_seen_turn"] = window[i]["turn_index"]
            else:
                self._procedural[pattern_key] = {
                    "pattern": pattern,
                    "count": 1,
                    "last_seen_turn": window[i]["turn_index"],
                }
        # Bounded: keep only top-N most-frequent patterns. Ties broken
        # by ``last_seen_turn`` (more recent wins) — deterministic.
        if len(self._procedural) > self._max_procedural:
            ranked: List[Tuple[str, Dict[str, Any]]] = sorted(
                self._procedural.items(),
                key=lambda kv: (-kv[1]["count"], -kv[1]["last_seen_turn"]),
            )
            self._procedural = dict(ranked[: self._max_procedural])

    def _procedural_as_sorted_list(self) -> List[Dict[str, Any]]:
        """Return procedural store as a deterministically-ordered list.

        Caller MUST hold ``self._lock``. Sort key is
        ``(-count, -last_seen_turn, canonical_pattern_json)`` so the
        snapshot's list order is independent of insertion order →
        identical content yields identical hashes.
        """
        # Build a typed (sort_key_tuple, payload_dict) list so mypy --strict
        # can prove the sort lambda inputs are ints / strings (not the
        # ``object`` widening that ``Dict[str, Any]`` produces).
        ranked: List[Tuple[Tuple[int, int, str], Dict[str, Any]]] = []
        for pattern_key, entry in self._procedural.items():
            count = int(entry["count"])
            last_seen = int(entry["last_seen_turn"])
            payload: Dict[str, Any] = {
                "pattern": list(entry["pattern"]),
                "count": count,
                "last_seen_turn": last_seen,
            }
            # Negate count/last_seen so the natural ascending tuple sort
            # produces "highest count first, then most recent first".
            ranked.append(((-count, -last_seen, pattern_key), payload))
        ranked.sort(key=lambda item: item[0])
        return [payload for _, payload in ranked]


# ---------------------------------------------------------------------------
# Public re-exports
# ---------------------------------------------------------------------------


__all__ = [
    "DEFAULT_MAX_EPISODIC",
    "DEFAULT_MAX_PROCEDURAL",
    "DEFAULT_MAX_SEMANTIC",
    "MemoryRecorder",
    "MemorySnapshot",
]
