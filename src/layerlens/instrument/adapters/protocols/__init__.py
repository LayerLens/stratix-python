"""Protocol adapters for the LayerLens Instrument layer.

Each protocol adapter wraps a wire-format protocol (A2A, MCP, AG-UI, AP2,
A2UI, UCP) so platforms speaking those protocols can stream their lifecycle
events through the LayerLens telemetry pipeline.

Adapters are loaded on demand via :class:`AdapterRegistry`; importing this
package does NOT import any protocol SDK.
"""

from __future__ import annotations
