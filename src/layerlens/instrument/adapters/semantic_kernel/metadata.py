"""
Semantic Kernel Metadata Extraction

Extracts plugin and kernel configuration metadata for environment.config events.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class SKMetadataExtractor:
    """Extract metadata from Semantic Kernel components."""

    def extract_plugin_metadata(self, plugin: Any) -> dict[str, Any]:
        """Extract metadata from a registered plugin."""
        metadata: dict[str, Any] = {}
        try:
            metadata["plugin_name"] = getattr(plugin, "name", str(plugin))
            metadata["description"] = getattr(plugin, "description", None)

            # Extract function names
            functions = getattr(plugin, "functions", None)
            if functions:
                if isinstance(functions, dict):
                    metadata["function_names"] = list(functions.keys())
                elif hasattr(functions, "keys"):
                    metadata["function_names"] = list(functions.keys())
        except Exception:
            logger.debug("Error extracting plugin metadata", exc_info=True)
        return metadata

    def extract_kernel_metadata(self, kernel: Any) -> dict[str, Any]:
        """Extract metadata from a Kernel instance."""
        metadata: dict[str, Any] = {}
        try:
            # Extract registered plugins
            plugins = getattr(kernel, "plugins", None)
            if plugins:
                if isinstance(plugins, dict):
                    metadata["plugin_count"] = len(plugins)
                    metadata["plugin_names"] = list(plugins.keys())
                elif hasattr(plugins, "__len__"):
                    metadata["plugin_count"] = len(plugins)

            # Extract registered services
            services = getattr(kernel, "services", None)
            if services:
                if isinstance(services, dict):
                    metadata["service_count"] = len(services)
                    metadata["service_types"] = [
                        type(s).__name__ for s in services.values()
                    ]

            # Extract memory backend
            memory = getattr(kernel, "memory", None)
            if memory:
                metadata["memory_backend"] = type(memory).__name__

        except Exception:
            logger.debug("Error extracting kernel metadata", exc_info=True)
        return metadata
