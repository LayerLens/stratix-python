"""LangGraph callback handler.

Builds on :class:`LangChainCallbackHandler` — LangGraph re-uses the langchain-core
callback protocol — but adds **graph-structure** and **node-level state** capture
so traces reflect the graph topology rather than a flat sequence of chains.

Two additions over the base LangChain handler:

* On each chain boundary, inspect ``tags`` and ``metadata`` for LangGraph's
  ``graph:step:N`` and ``langgraph_node`` markers. Emit a dedicated
  ``agent.node.enter`` / ``agent.node.exit`` pair so downstream UIs can render
  the actual graph.
* Surface the node name into the chain span's ``payload["node"]`` so the regular
  LangChain agent/tool callbacks fired inside a node inherit that context.
"""

from __future__ import annotations

import time
from uuid import UUID
from typing import Any, Dict, List, Optional

from .langchain import LangChainCallbackHandler


class LangGraphCallbackHandler(LangChainCallbackHandler):
    name = "langgraph"

    def __init__(self, client: Any, capture_config: Any = None) -> None:
        super().__init__(client, capture_config=capture_config)
        # run_id -> node metadata (node_name, step, entered_at_ns)
        self._pending_nodes: Dict[str, Dict[str, Any]] = {}

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
            self._emit("agent.node.enter", enter_payload, run_id=run_id, parent_run_id=parent_run_id)

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
            self._emit("agent.node.exit", exit_payload, run_id=run_id, parent_run_id=parent_run_id)
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
