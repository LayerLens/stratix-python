"""
CrewAI Agent Metadata Extraction

Extracts and caches agent metadata for L4a (environment.config) emission.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class AgentMetadataExtractor:
    """Extracts and caches CrewAI agent metadata for L4a emission."""

    def extract(self, agent: Any) -> dict[str, Any]:
        """
        Extract metadata from a CrewAI Agent.

        Args:
            agent: A CrewAI Agent instance

        Returns:
            Dict of agent metadata
        """
        metadata: dict[str, Any] = {}

        for attr in (
            "role", "goal", "backstory", "verbose",
            "allow_delegation", "max_iter", "memory",
        ):
            try:
                val = getattr(agent, attr, None)
                if val is not None:
                    metadata[attr] = val
            except Exception:
                pass

        # Extract tool names
        try:
            tools = getattr(agent, "tools", None)
            if tools:
                metadata["tools"] = [
                    getattr(t, "name", str(t)) for t in tools
                ]
        except Exception:
            pass

        # Extract LLM model info
        try:
            llm = getattr(agent, "llm", None)
            if llm is not None:
                model_name = (
                    getattr(llm, "model_name", None)
                    or getattr(llm, "model", None)
                    or str(llm)
                )
                metadata["llm_model"] = model_name
        except Exception:
            pass

        return metadata
