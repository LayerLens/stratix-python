"""LangGraph callback handler.

Builds on :class:`LangChainCallbackHandler` — LangGraph re-uses the langchain-core
callback protocol — but adds **graph-structure** and **node-level state** capture
so traces reflect the graph topology rather than a flat sequence of chains.

Three additions over the base LangChain handler:

* On each chain boundary, inspect ``tags`` and ``metadata`` for LangGraph's
  ``graph:step:N`` and ``langgraph_node`` markers. Emit a dedicated
  ``agent.node.enter`` / ``agent.node.exit`` pair so downstream UIs can render
  the actual graph.
* Surface the node name into the chain span's ``payload["node"]`` so the regular
  LangChain agent/tool callbacks fired inside a node inherit that context.
* After each node exit, emit an ``agent.state.change`` event whose payload
  carries a deterministic ``sha256:<hex>`` digest of the node's output state.
  Downstream tools diff hashes across nodes without needing the raw state.
  Use ``state_include_keys`` / ``state_exclude_keys`` (constructor args) to
  scope the hash to a subset of the state dict; set ``emit_state_hash=False``
  to disable entirely.
* Detect agent-to-agent handoffs by tracking node-name transitions. When
  the active node changes between distinct named agents, emit an
  ``agent.handoff`` event via :class:`HandoffDetector`. Set
  ``detect_handoffs=False`` to disable.
"""

from __future__ import annotations

import time
import logging
from uuid import UUID
from typing import Any, Dict, List, Optional

from ._handoff import HandoffDetector
from .langchain import LangChainCallbackHandler
from ....attestation._hash import compute_hash

log = logging.getLogger(__name__)


class LangGraphCallbackHandler(LangChainCallbackHandler):
    name = "langgraph"

    def __init__(
        self,
        client: Any,
        capture_config: Any = None,
        *,
        emit_state_hash: bool = True,
        state_include_keys: Optional[List[str]] = None,
        state_exclude_keys: Optional[List[str]] = None,
        detect_handoffs: bool = True,
    ) -> None:
        super().__init__(client, capture_config=capture_config)
        # run_id -> node metadata (node_name, step, entered_at_ns)
        self._pending_nodes: Dict[str, Dict[str, Any]] = {}
        self._emit_state_hash = emit_state_hash
        self._state_include_keys = frozenset(state_include_keys) if state_include_keys is not None else None
        self._state_exclude_keys = frozenset(state_exclude_keys) if state_exclude_keys is not None else None
        self._detect_handoffs = detect_handoffs
        self._handoff_detector = HandoffDetector() if detect_handoffs else None

    # ------------------------------------------------------------------
    # Chain callbacks — enrich with node-level detection
    # ------------------------------------------------------------------

    def on_chain_start(
        self,
        serialized: Optional[Dict[str, Any]],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        if parent_run_id is None:
            run = self._begin_run()
            run.data["root_run_id"] = str(run_id)
        serialized = serialized or {}
        node_name = _extract_node_name(serialized, tags, metadata)
        step = _extract_step(tags, metadata)

        if node_name is not None:
            self._pending_nodes[str(run_id)] = {
                "node": node_name,
                "step": step,
                "entered_at_ns": time.time_ns(),
            }
            enter_payload = self._payload(node=node_name, step=step)
            self._set_if_capturing(enter_payload, "input", inputs)
            self._emit(
                "agent.node.enter",
                enter_payload,
                run_id=run_id,
                parent_run_id=parent_run_id,
            )
            if self._handoff_detector is not None:
                self._handoff_detector.detect(node_name, context=inputs)

        name = node_name or serialized.get("name") or serialized.get("id", ["unknown"])[-1]
        payload = self._payload(name=name)
        if node_name is not None:
            payload["node"] = node_name
        if step is not None:
            payload["step"] = step
        self._set_if_capturing(payload, "input", inputs)
        self._emit("agent.input", payload, run_id=run_id, parent_run_id=parent_run_id)

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        node = self._pending_nodes.pop(str(run_id), None)
        if node is not None:
            exit_payload = self._payload(
                node=node["node"],
                step=node.get("step"),
                latency_ms=(time.time_ns() - node["entered_at_ns"]) / 1_000_000,
            )
            self._set_if_capturing(exit_payload, "output", outputs)
            self._emit(
                "agent.node.exit",
                exit_payload,
                run_id=run_id,
                parent_run_id=parent_run_id,
            )
            if self._emit_state_hash:
                self._emit_node_state_change(
                    node_name=node["node"],
                    step=node.get("step"),
                    outputs=outputs,
                    run_id=run_id,
                    parent_run_id=parent_run_id,
                )
        super().on_chain_end(outputs, run_id=run_id, parent_run_id=parent_run_id, **kwargs)

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        node = self._pending_nodes.pop(str(run_id), None)
        if node is not None:
            self._emit(
                "agent.node.exit",
                self._payload(
                    node=node["node"],
                    step=node.get("step"),
                    status="error",
                    error=str(error),
                    latency_ms=(time.time_ns() - node["entered_at_ns"]) / 1_000_000,
                ),
                run_id=run_id,
                parent_run_id=parent_run_id,
            )
        super().on_chain_error(error, run_id=run_id, parent_run_id=parent_run_id, **kwargs)

    # ------------------------------------------------------------------
    # State hashing
    # ------------------------------------------------------------------

    def _select_state(self, outputs: Any) -> Any:
        """Return the subset of *outputs* to hash, honoring include/exclude filters."""
        if not isinstance(outputs, dict):
            return outputs
        if self._state_include_keys is not None:
            return {k: v for k, v in outputs.items() if k in self._state_include_keys}
        if self._state_exclude_keys is not None:
            return {k: v for k, v in outputs.items() if k not in self._state_exclude_keys}
        return outputs

    def _emit_node_state_change(
        self,
        *,
        node_name: str,
        step: Optional[int],
        outputs: Any,
        run_id: UUID,
        parent_run_id: Optional[UUID],
    ) -> None:
        """Emit ``agent.state.change`` with a deterministic sha256: digest of node output."""
        state = self._select_state(outputs)
        try:
            state_hash = compute_hash(state)
        except TypeError:
            # Non-serializable values inside the state — fall back to repr-based
            # hashing so we still emit something stable for the dashboard.
            try:
                state_hash = compute_hash({"_repr": repr(state)})
            except Exception:
                log.debug("layerlens.langgraph: could not hash state for node %s", node_name)
                return

        payload = self._payload(
            node=node_name,
            step=step,
            state_hash=state_hash,
        )
        if isinstance(state, dict):
            payload["state_keys"] = sorted(state.keys())
        self._emit(
            "agent.state.change",
            payload,
            run_id=run_id,
            parent_run_id=parent_run_id,
        )


def _extract_node_name(
    serialized: Dict[str, Any],
    tags: Optional[List[str]],
    metadata: Optional[Dict[str, Any]],
) -> Optional[str]:
    # Priority: explicit metadata.langgraph_node > clean tag > serialized name
    if isinstance(metadata, dict):
        node = metadata.get("langgraph_node")
        if node:
            return str(node)
    if tags:
        for tag in tags:
            if not isinstance(tag, str):
                continue
            if tag.startswith("graph:step:"):
                continue
            if ":" not in tag:
                return tag
    sid = serialized.get("id")
    if isinstance(sid, list) and sid:
        last = sid[-1]
        if isinstance(last, str):
            return last
    return None


def _extract_step(tags: Optional[List[str]], metadata: Optional[Dict[str, Any]]) -> Optional[int]:
    if isinstance(metadata, dict):
        step = metadata.get("langgraph_step")
        if step is not None:
            try:
                return int(step)
            except (TypeError, ValueError):
                pass
    if tags:
        for tag in tags:
            if isinstance(tag, str) and tag.startswith("graph:step:"):
                try:
                    return int(tag.split(":")[-1])
                except (TypeError, ValueError):
                    pass
    return None
