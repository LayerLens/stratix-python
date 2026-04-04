from __future__ import annotations

from uuid import UUID
from typing import Any, Dict, List, Optional

from .langchain import LangChainCallbackHandler


class LangGraphCallbackHandler(LangChainCallbackHandler):
    name = "langgraph"

    def on_chain_start(
        self,
        serialized: Optional[Dict[str, Any]],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        if parent_run_id is None:
            run = self._begin_run()
            run.data["root_run_id"] = str(run_id)
        serialized = serialized or {}
        name = serialized.get("name") or serialized.get("id", ["unknown"])[-1]

        # Extract node name from LangGraph tags
        if tags:
            for tag in tags:
                if isinstance(tag, str) and tag.startswith("graph:step:"):
                    continue
                if isinstance(tag, str) and ":" not in tag:
                    name = tag
                    break

        # Check kwargs for langgraph-specific metadata
        metadata = kwargs.get("metadata", {})
        if isinstance(metadata, dict):
            node_name = metadata.get("langgraph_node")
            if node_name:
                name = node_name

        payload = self._payload(name=name)
        self._set_if_capturing(payload, "input", inputs)
        self._emit("agent.input", payload, run_id=run_id, parent_run_id=parent_run_id)
