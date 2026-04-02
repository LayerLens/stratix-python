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

        self._emit_for_run("agent.input", {"name": name, "input": inputs}, run_id, parent_run_id)
