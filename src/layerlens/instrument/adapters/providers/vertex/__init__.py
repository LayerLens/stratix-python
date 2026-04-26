"""Google Vertex AI provider adapter package.

Public surface::

    from layerlens.instrument.adapters.providers.vertex import VertexAdapter

The package is structured so that importing this ``__init__`` does not
import the ``google-cloud-aiplatform`` SDK — adapter classes are loaded
on demand. Heavy submodules (``adapter``, ``messages``, ``pricing``,
``auth``) are re-exported through this module for convenience but each
remains independently importable.

Backwards-compatible ``LayerLens*`` aliases are exposed for code that
was written against the legacy ``ateam`` package naming. Both forms
point at the same class object — they are not separate types.
"""

from __future__ import annotations

from layerlens.instrument.adapters.providers.vertex.auth import (
    detect_location,
    detect_project_id,
    detect_credential_source,
)
from layerlens.instrument.adapters.providers.vertex.adapter import (
    ADAPTER_CLASS,
    VertexAdapter,
)
from layerlens.instrument.adapters.providers.vertex.pricing import (
    VERTEX_PRICING,
    normalize_vertex_model,
)
from layerlens.instrument.adapters.providers.vertex.messages import (
    extract_usage,
    extract_output_text,
    extract_function_calls,
    normalize_vertex_contents,
)

# --- Backwards-compatible STRATIX-prefixed aliases ----------------------
#
# The ateam source uses ``STRATIX_VERTEX_*`` constant names in some call
# sites. Map them to the LayerLens-prefixed names so downstream callers
# that pin against the legacy identifier keep working without a code
# change. Adding a new constant? Add both forms here.
LayerLensVertexAdapter = VertexAdapter
STRATIX_VERTEX_ADAPTER_CLASS = ADAPTER_CLASS
STRATIX_VERTEX_PRICING = VERTEX_PRICING


__all__ = [
    "ADAPTER_CLASS",
    "LayerLensVertexAdapter",
    "STRATIX_VERTEX_ADAPTER_CLASS",
    "STRATIX_VERTEX_PRICING",
    "VERTEX_PRICING",
    "VertexAdapter",
    "detect_credential_source",
    "detect_location",
    "detect_project_id",
    "extract_function_calls",
    "extract_output_text",
    "extract_usage",
    "normalize_vertex_contents",
    "normalize_vertex_model",
]
