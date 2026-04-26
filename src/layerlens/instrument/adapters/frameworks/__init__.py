"""Framework adapters for the LayerLens Instrument layer.

Each framework adapter wraps an agent / chain framework's lifecycle to
intercept agent runs, model invocations, tool calls, state changes, and
handoffs, emitting events through the LayerLens telemetry pipeline.

Adapter packages exported by the registry (loaded on demand by
:class:`AdapterRegistry`):

* ``semantic_kernel`` — Microsoft Semantic Kernel (filter API).

Importing this package does NOT import any framework SDK. Each
``frameworks.<name>`` package is loaded only when the user requests it
explicitly, e.g. via ``AdapterRegistry.get("semantic_kernel")``.
"""

from __future__ import annotations
