"""LayerLens LangChain LCEL (LangChain Expression Language) tracing.

LCEL is the dominant LangChain authoring pattern as of langchain-core
0.2+. Pipelines are built by piping ``Runnable`` instances together
(``prompt | llm | output_parser``), and the resulting composition is
exposed to the runtime as a tree of ``RunnableSequence`` /
``RunnableParallel`` / ``RunnableLambda`` / ``RunnablePassthrough`` /
``RunnableBranch`` objects.

Upstream LangChain fires the standard callback protocol
(``on_chain_start`` / ``on_chain_end`` / ``on_chain_error``) for every
runnable in the tree, but the legacy adapter's chain handlers were
written for the legacy ``Chain`` API and ignored these events unless a
``langgraph_node`` marker was present in the metadata. As a result,
LCEL pipelines were observable only at the LLM/tool boundary — the
pipeline topology (which step ran when, in what order, with what
nesting) was lost.

This module fills the gap. It provides:

* :func:`detect_runnable_kind` — classify an ``on_chain_start`` event as
  one of the LCEL primitives (``sequence``, ``parallel``, ``lambda``,
  ``passthrough``, ``branch``) or ``other`` based on the ``name`` kwarg
  that LangChain attaches at runtime.
* :func:`parse_composition_tag` — decode the position tags LangChain
  attaches to child runnables (``seq:step:N``, ``map:key:K``,
  ``branch:N``, ``condition:N``) so the adapter can reconstruct the
  parent's composition graph.
* :class:`LCELRunnableTracker` — per-callback-handler state machine that
  tracks the active runnable hierarchy across nested invocations,
  emits the synthetic ``chain.composition`` graph event when a root
  runnable starts, and emits per-step ``agent.code`` events for every
  intermediate runnable in the tree (capture-config gated to L2).

Reference: ``ateam/docs/incubation-docs/adapter-framework/04-per-framework-specs/04b-langchain-adapter-spec.md``
section 4 (``LCEL Support``).
"""

from __future__ import annotations

import time
import hashlib
from enum import Enum
from typing import Any
from dataclasses import field, dataclass

# Tag prefixes LangChain attaches to runnable children to record their
# position within the parent composition. Source:
# ``langchain_core/runnables/base.py`` (search for ``"seq:step:"``,
# ``"map:key:"``, ``"branch:"``, ``"condition:"``).
_TAG_SEQUENCE = "seq:step:"
_TAG_PARALLEL = "map:key:"
_TAG_BRANCH_BODY = "branch:"
_TAG_BRANCH_COND = "condition:"


class RunnableKind(str, Enum):
    """Classification of an LCEL runnable observed via on_chain_start.

    The five LCEL primitives plus a catch-all for anything that isn't
    one of them (``RunnablePassthrough.assign``-derived hybrids,
    ``ChatPromptTemplate``, ``StrOutputParser``, ``ChatOpenAI``, ...).
    """

    SEQUENCE = "sequence"
    PARALLEL = "parallel"
    LAMBDA = "lambda"
    PASSTHROUGH = "passthrough"
    BRANCH = "branch"
    OTHER = "other"


# Composition position decoded from a child runnable's tags.
@dataclass(frozen=True)
class CompositionPosition:
    """Where a child runnable sits inside its parent's composition."""

    # The parent's RunnableKind (best-effort: derived from the tag prefix).
    parent_kind: RunnableKind
    # For SEQUENCE: 1-based step index. For PARALLEL: branch key.
    # For BRANCH: index of the body or condition predicate.
    label: str
    # Discriminator: "step", "key", "body", "condition".
    role: str


def detect_runnable_kind(name: str | None) -> RunnableKind:
    """Classify a runnable from the ``name`` kwarg of ``on_chain_start``.

    LangChain runtime attaches a ``name`` kwarg to every chain callback
    that identifies the runnable class. For composed runnables the name
    is a class-derived string:

    * ``RunnableSequence``
    * ``RunnableParallel<a,b>`` — angle-bracketed branch keys
    * ``RunnableLambda`` — or the inner function's ``__name__``
    * ``RunnablePassthrough``
    * ``RunnableBranch``

    For non-LCEL runnables (templates, parsers, models) the name is
    typically the class name (``ChatPromptTemplate``, ``StrOutputParser``,
    etc.) — these are classified as :attr:`RunnableKind.OTHER`.

    Args:
        name: The ``name`` kwarg value from ``on_chain_start`` (may be
            ``None`` if upstream LangChain failed to attach one — falls
            back to ``OTHER``).

    Returns:
        The detected :class:`RunnableKind`.
    """
    if not name:
        return RunnableKind.OTHER

    # Order matters: ``RunnableSequence`` and ``RunnableParallel<...>`` are
    # both prefixes, so prefix checks are the safe primary discriminator.
    if name.startswith("RunnableSequence"):
        return RunnableKind.SEQUENCE
    if name.startswith("RunnableParallel"):
        return RunnableKind.PARALLEL
    if name.startswith("RunnableBranch"):
        return RunnableKind.BRANCH
    if name == "RunnablePassthrough":
        return RunnableKind.PASSTHROUGH
    if name == "RunnableLambda":
        return RunnableKind.LAMBDA

    return RunnableKind.OTHER


def parse_composition_tag(tags: list[Any] | None) -> CompositionPosition | None:
    """Decode the first composition tag from a runnable's ``tags`` list.

    Children of LCEL composites carry positional tags assigned by the
    parent runnable. We prefer the most-specific tag (the first one
    matching a known prefix) so multi-level nesting still resolves.

    Args:
        tags: The ``tags`` kwarg value from ``on_chain_start``. Typed
            as ``list[Any]`` rather than ``list[str]`` because some
            third-party callback handlers inject non-string entries
            and we'd rather skip them than crash.

    Returns:
        A :class:`CompositionPosition` if any composition tag is found,
        otherwise ``None``.
    """
    if not tags:
        return None

    for raw in tags:
        if not isinstance(raw, str):
            continue
        if raw.startswith(_TAG_SEQUENCE):
            return CompositionPosition(
                parent_kind=RunnableKind.SEQUENCE,
                label=raw[len(_TAG_SEQUENCE) :] or "?",
                role="step",
            )
        if raw.startswith(_TAG_PARALLEL):
            return CompositionPosition(
                parent_kind=RunnableKind.PARALLEL,
                label=raw[len(_TAG_PARALLEL) :] or "?",
                role="key",
            )
        if raw.startswith(_TAG_BRANCH_COND):
            return CompositionPosition(
                parent_kind=RunnableKind.BRANCH,
                label=raw[len(_TAG_BRANCH_COND) :] or "?",
                role="condition",
            )
        if raw.startswith(_TAG_BRANCH_BODY):
            return CompositionPosition(
                parent_kind=RunnableKind.BRANCH,
                label=raw[len(_TAG_BRANCH_BODY) :] or "?",
                role="body",
            )

    return None


def parse_parallel_branches(name: str) -> list[str]:
    """Extract the branch keys from a ``RunnableParallel<a,b,c>`` name.

    Args:
        name: A runnable name beginning with ``RunnableParallel``.

    Returns:
        List of branch keys in declaration order. Empty list if no
        bracketed key list is present.
    """
    open_bracket = name.find("<")
    close_bracket = name.rfind(">")
    if open_bracket == -1 or close_bracket == -1 or close_bracket <= open_bracket + 1:
        return []
    inside = name[open_bracket + 1 : close_bracket]
    return [k.strip() for k in inside.split(",") if k.strip()]


@dataclass
class LCELNode:
    """One node in the runnable execution tree."""

    run_id: str
    parent_run_id: str | None
    kind: RunnableKind
    # Display name as reported by LangChain (e.g. ``RunnableLambda`` or
    # the inner function's ``__name__``).
    name: str
    # Depth from the root runnable (root = 0).
    depth: int
    # Composition position inside the parent (None for root).
    position: CompositionPosition | None
    # For PARALLEL: the declared branch keys parsed from the name.
    parallel_branches: list[str]
    # When the runnable started (ns since epoch).
    start_time_ns: int
    # Children added as their on_chain_start fires under this run_id.
    child_run_ids: list[str] = field(default_factory=list)
    # Set by on_chain_end / on_chain_error.
    end_time_ns: int | None = None
    status: str = "running"  # "running" | "ok" | "error"
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Serialize for inclusion in a ``chain.composition`` payload."""
        out: dict[str, Any] = {
            "run_id": self.run_id,
            "parent_run_id": self.parent_run_id,
            "kind": self.kind.value,
            "name": self.name,
            "depth": self.depth,
            "status": self.status,
        }
        if self.position is not None:
            out["position"] = {
                "parent_kind": self.position.parent_kind.value,
                "label": self.position.label,
                "role": self.position.role,
            }
        if self.parallel_branches:
            out["parallel_branches"] = list(self.parallel_branches)
        if self.end_time_ns is not None:
            out["duration_ns"] = self.end_time_ns - self.start_time_ns
        if self.error is not None:
            out["error"] = self.error
        if self.child_run_ids:
            out["child_run_ids"] = list(self.child_run_ids)
        return out


def fingerprint_lambda(name: str, depth: int, position: CompositionPosition | None) -> str:
    """Return a stable SHA-256 fingerprint for a ``RunnableLambda`` node.

    Used as the ``artifact_hash`` on the synthetic ``agent.code`` event
    so the same lambda invoked twice produces the same hash. We can't
    fingerprint the inner callable's source (LangChain doesn't surface
    it through the callback path), so the tuple of ``(name, depth,
    position)`` is the best we can offer.

    Args:
        name: The lambda's reported name (function ``__name__`` if known,
            otherwise ``"RunnableLambda"``).
        depth: Depth from the root runnable.
        position: Composition position inside the parent (may be None).

    Returns:
        Hex-encoded SHA-256 digest, truncated to 16 chars for brevity in
        events.
    """
    parts = [name, str(depth)]
    if position is not None:
        parts.extend([position.parent_kind.value, position.role, position.label])
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return digest[:16]


class LCELRunnableTracker:
    """Per-handler state for the active LCEL runnable hierarchy.

    The tracker is owned by the :class:`LayerLensCallbackHandler` and
    consulted from ``on_chain_start`` / ``on_chain_end`` /
    ``on_chain_error``. It does NOT itself emit events — the caller
    decides what to emit, gated by :class:`CaptureConfig`. The tracker
    only maintains the in-memory tree and produces dict payloads.

    Hierarchy is built incrementally as callbacks fire. When the root
    runnable's ``on_chain_end`` fires, the tracker yields the completed
    tree ready for serialization into the ``chain.composition`` event.
    """

    def __init__(self) -> None:
        # All nodes seen this session, indexed by run_id.
        self._nodes: dict[str, LCELNode] = {}
        # Stack of active root run_ids (supports concurrent root runnables).
        self._roots: list[str] = []
        # Roots whose on_chain_start fired and for which we have NOT yet
        # emitted the synthetic chain.composition event. We defer emission
        # to on_chain_end so the snapshot includes the full subtree.
        self._pending_compositions: dict[str, LCELNode] = {}

    def is_root(self, run_id: str) -> bool:
        """Whether the given run_id is currently tracked as a root runnable."""
        return run_id in self._roots and self._nodes.get(run_id) is not None

    def get_node(self, run_id: str) -> LCELNode | None:
        """Retrieve a node by run_id (None if not tracked)."""
        return self._nodes.get(run_id)

    def is_tracked(self, run_id: str) -> bool:
        """Whether we've seen on_chain_start for this run_id."""
        return run_id in self._nodes

    def begin(
        self,
        *,
        run_id: str,
        parent_run_id: str | None,
        name: str | None,
        tags: list[Any] | None,
    ) -> LCELNode:
        """Record a new runnable beginning its execution.

        Returns the constructed :class:`LCELNode`.

        Subsequent reads via :meth:`get_node` and :meth:`is_root` will
        reflect this node until :meth:`end` is called for the same
        ``run_id``.
        """
        kind = detect_runnable_kind(name)
        position = parse_composition_tag(tags)
        parallel_branches = parse_parallel_branches(name) if (kind == RunnableKind.PARALLEL and name) else []

        # Compute depth from the parent.
        parent_node = self._nodes.get(parent_run_id) if parent_run_id else None
        depth = parent_node.depth + 1 if parent_node is not None else 0

        # If parent_run_id refers to a runnable not in our tree
        # (pre-existing legacy chain, or a node created by a different
        # callback handler), treat this as a new root for tracking
        # purposes — we still record the parent_run_id literally so
        # downstream tooling can stitch traces together via run_ids.
        is_root = parent_node is None

        node = LCELNode(
            run_id=run_id,
            parent_run_id=parent_run_id,
            kind=kind,
            name=name or "<unnamed>",
            depth=depth,
            position=position,
            parallel_branches=parallel_branches,
            start_time_ns=time.time_ns(),
        )
        self._nodes[run_id] = node

        if parent_node is not None:
            parent_node.child_run_ids.append(run_id)

        if is_root:
            self._roots.append(run_id)
            self._pending_compositions[run_id] = node

        return node

    def end(self, run_id: str, *, error: str | None = None) -> LCELNode | None:
        """Record a runnable completing.

        Args:
            run_id: The runnable's run_id.
            error: Error message if completion was via ``on_chain_error``.

        Returns:
            The completed node (with ``end_time_ns`` and ``status`` set),
            or None if the run_id was not tracked.
        """
        node = self._nodes.get(run_id)
        if node is None:
            return None
        node.end_time_ns = time.time_ns()
        if error is not None:
            node.status = "error"
            node.error = error
        else:
            node.status = "ok"
        return node

    def consume_root_completion(self, run_id: str) -> LCELNode | None:
        """If ``run_id`` is a completed root, pop and return its node.

        Returns None if the run_id is not a root or has not completed.

        After consumption, the root's subtree remains in the tracker
        until :meth:`reset` is called — keeping the data available for
        the caller's :meth:`composition_payload` call.
        """
        if run_id not in self._roots:
            return None
        node = self._nodes.get(run_id)
        if node is None or node.end_time_ns is None:
            return None
        self._pending_compositions.pop(run_id, None)
        return node

    def composition_payload(self, root_run_id: str) -> dict[str, Any] | None:
        """Build the ``chain.composition`` payload for a completed root.

        Returns the full subtree (root + descendants) as a flat node list
        plus an aggregate summary. Returns ``None`` if the run_id is not
        a tracked root.
        """
        root = self._nodes.get(root_run_id)
        if root is None or root_run_id not in self._roots:
            return None

        # BFS the subtree.
        nodes: list[dict[str, Any]] = []
        stack = [root_run_id]
        seen: set[str] = set()
        kind_counts: dict[str, int] = {}
        max_depth = 0

        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            n = self._nodes.get(current)
            if n is None:
                continue
            nodes.append(n.as_dict())
            kind_counts[n.kind.value] = kind_counts.get(n.kind.value, 0) + 1
            if n.depth > max_depth:
                max_depth = n.depth
            stack.extend(n.child_run_ids)

        return {
            "root_run_id": root_run_id,
            "root_kind": root.kind.value,
            "root_name": root.name,
            "node_count": len(nodes),
            "max_depth": max_depth,
            "kind_counts": kind_counts,
            "nodes": nodes,
            "status": root.status,
        }

    def reset(self) -> None:
        """Drop all tracked state.

        Called from ``LayerLensCallbackHandler.disconnect`` and from
        tests that want a fresh tree.
        """
        self._nodes.clear()
        self._roots.clear()
        self._pending_compositions.clear()

    # --- Convenience helpers consumed by callbacks.py -------------------

    def runnable_input_payload(self, node: LCELNode, inputs: Any) -> dict[str, Any]:
        """Build the ``agent.input`` payload for a runnable's start.

        Includes the composition metadata so dashboards can render the
        runnable's role inside its parent (step number, branch key,
        condition index). The actual ``inputs`` value is included as-is
        and subject to the caller's privacy/truncation policy upstream.
        """
        payload: dict[str, Any] = {
            "run_id": node.run_id,
            "parent_run_id": node.parent_run_id,
            "runnable": {
                "kind": node.kind.value,
                "name": node.name,
                "depth": node.depth,
            },
            "input": inputs,
        }
        if node.position is not None:
            payload["runnable"]["position"] = {
                "parent_kind": node.position.parent_kind.value,
                "label": node.position.label,
                "role": node.position.role,
            }
        if node.parallel_branches:
            payload["runnable"]["parallel_branches"] = list(node.parallel_branches)
        if node.kind == RunnableKind.LAMBDA:
            payload["runnable"]["fingerprint"] = fingerprint_lambda(
                node.name, node.depth, node.position
            )
        return payload

    def runnable_code_payload(self, node: LCELNode) -> dict[str, Any]:
        """Build the ``agent.code`` payload for a completed runnable step.

        Used by the spec's L2 ``AgentCodeEvent`` mapping for LCEL: every
        intermediate runnable in the tree emits one event so the
        pipeline DAG can be reconstructed in the UI.
        """
        duration_ns = (
            (node.end_time_ns - node.start_time_ns) if node.end_time_ns is not None else None
        )
        payload: dict[str, Any] = {
            "run_id": node.run_id,
            "parent_run_id": node.parent_run_id,
            "kind": node.kind.value,
            "name": node.name,
            "depth": node.depth,
            "status": node.status,
            "duration_ns": duration_ns,
        }
        if node.position is not None:
            payload["position"] = {
                "parent_kind": node.position.parent_kind.value,
                "label": node.position.label,
                "role": node.position.role,
            }
        if node.parallel_branches:
            payload["parallel_branches"] = list(node.parallel_branches)
        if node.kind == RunnableKind.PASSTHROUGH:
            payload["passthrough"] = True
        if node.kind == RunnableKind.LAMBDA:
            payload["fingerprint"] = fingerprint_lambda(node.name, node.depth, node.position)
        if node.error is not None:
            payload["error"] = node.error
        return payload


__all__ = [
    "RunnableKind",
    "CompositionPosition",
    "LCELNode",
    "LCELRunnableTracker",
    "detect_runnable_kind",
    "parse_composition_tag",
    "parse_parallel_branches",
    "fingerprint_lambda",
]
