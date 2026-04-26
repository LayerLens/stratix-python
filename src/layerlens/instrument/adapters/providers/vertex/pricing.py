"""Vertex AI pricing table.

Vertex hosts three families of models behind a single endpoint:

* Google's own Gemini models.
* Anthropic Claude models published via the Vertex Model Garden.
* Meta Llama models published via the Vertex Model Garden.

Each family uses Vertex-specific publisher-prefixed model identifiers
(e.g. ``publishers/anthropic/models/claude-opus-4-6``). Direct-API
pricing in :data:`layerlens.instrument.adapters.providers._base.pricing.PRICING`
keys off the bare model name and would miss these. The
:data:`VERTEX_PRICING` table below is passed as ``pricing_table`` when
emitting ``cost.record`` events from a Vertex invocation.

Rates are per-1000 tokens, USD, sourced from Google Cloud's published
Vertex AI pricing page (rates current as of 2026-04). Update hashes are
checked against ``ateam`` in CI to keep the two repos in sync.
"""

from __future__ import annotations

from typing import Dict

# Per-1K-token rates (USD) for models invoked via Vertex AI.
#
# Vertex normalizes its model identifiers to the bare-name form once
# the SDK strips the ``publishers/<vendor>/models/`` prefix, so most
# entries below are aliased to the same key the bare-name pricing
# table uses. The duplication is intentional — keeping a Vertex-only
# table makes the per-route pricing surface explicit and lets us
# diverge if Google ever charges a Vertex premium over the direct API.
VERTEX_PRICING: Dict[str, Dict[str, float]] = {
    # --- Gemini (Google's native Vertex models) ---
    "gemini-2.5-pro": {"input": 0.00125, "output": 0.01},
    "gemini-2.5-flash": {"input": 0.000075, "output": 0.0003},
    "gemini-2.0-flash": {"input": 0.0001, "output": 0.0004},
    "gemini-1.5-pro": {"input": 0.00125, "output": 0.005},
    "gemini-1.5-flash": {"input": 0.000075, "output": 0.0003},
    # --- Anthropic on Vertex (Model Garden) ---
    "claude-opus-4-6": {"input": 0.015, "output": 0.075},
    "claude-opus-4-20250115": {"input": 0.015, "output": 0.075},
    "claude-sonnet-4-5-20250929": {"input": 0.003, "output": 0.015},
    "claude-haiku-4-5-20251001": {"input": 0.0008, "output": 0.004},
    "claude-haiku-3-5-20241022": {"input": 0.0008, "output": 0.004},
    "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
    "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
    "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
    # --- Meta Llama on Vertex (Model Garden) ---
    "llama-3.3-70b-instruct-maas": {"input": 0.00099, "output": 0.00099},
    "llama-3.1-405b-instruct-maas": {"input": 0.005, "output": 0.016},
    "llama-3.1-70b-instruct-maas": {"input": 0.00099, "output": 0.00099},
    "llama-3.1-8b-instruct-maas": {"input": 0.00022, "output": 0.00022},
}


def normalize_vertex_model(model: str) -> str:
    """Strip Vertex publisher prefixes from a model identifier.

    Vertex returns models in fully-qualified form (e.g.
    ``publishers/anthropic/models/claude-opus-4-6`` or
    ``models/gemini-1.5-pro``). The pricing tables key off the bare
    model name, so this helper trims known prefixes.
    """
    if not model:
        return model
    if model.startswith("publishers/"):
        # publishers/<vendor>/models/<name>
        parts = model.split("/")
        if len(parts) >= 4 and parts[2] == "models":
            return parts[3]
    if model.startswith("models/"):
        return model[len("models/") :]
    return model
