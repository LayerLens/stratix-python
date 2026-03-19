"""
AutoGen Agent Metadata Extraction

Extracts agent metadata for L4a (environment.config) emission.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class AutoGenAgentMetadataExtractor:
    """Extracts AutoGen agent metadata for environment.config emission."""

    def extract(self, agent: Any) -> dict[str, Any]:
        """
        Extract metadata from an AutoGen ConversableAgent.

        Args:
            agent: An AutoGen ConversableAgent instance

        Returns:
            Dict of agent metadata
        """
        metadata: dict[str, Any] = {}

        # Agent name
        try:
            metadata["name"] = getattr(agent, "name", str(agent))
        except Exception:
            metadata["name"] = "<unknown>"

        # System message
        try:
            system_message = getattr(agent, "system_message", None)
            if system_message is not None:
                metadata["system_message"] = (
                    system_message[:500] if len(system_message) > 500
                    else system_message
                )
        except Exception:
            pass

        # Human input mode
        try:
            him = getattr(agent, "human_input_mode", None)
            if him is not None:
                metadata["human_input_mode"] = him
        except Exception:
            pass

        # LLM config
        try:
            llm_config = getattr(agent, "llm_config", None)
            if llm_config and isinstance(llm_config, dict):
                safe_config: dict[str, Any] = {}
                if "model" in llm_config:
                    safe_config["model"] = llm_config["model"]
                if "config_list" in llm_config:
                    models = []
                    for cfg in llm_config["config_list"]:
                        if isinstance(cfg, dict) and "model" in cfg:
                            models.append(cfg["model"])
                    if models:
                        safe_config["models"] = models
                if "temperature" in llm_config:
                    safe_config["temperature"] = llm_config["temperature"]
                metadata["llm_config"] = safe_config
        except Exception:
            pass

        # Max consecutive auto reply
        try:
            max_reply = getattr(agent, "max_consecutive_auto_reply", None)
            if max_reply is not None:
                metadata["max_consecutive_auto_reply"] = max_reply
        except Exception:
            pass

        # Code execution config
        try:
            code_config = getattr(agent, "code_execution_config", None)
            if code_config and isinstance(code_config, dict):
                safe_code_config: dict[str, Any] = {}
                for key in ("work_dir", "use_docker", "timeout", "last_n_messages"):
                    if key in code_config:
                        safe_code_config[key] = code_config[key]
                metadata["code_execution_config"] = safe_code_config
        except Exception:
            pass

        return metadata
