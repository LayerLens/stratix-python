"""Bedrock ``modelId`` -> provider family detection.

Bedrock multiplexes many model providers behind a single
``bedrock-runtime`` client. Each family uses a different request and
response body shape, so the adapter must dispatch on the family before
extracting tokens, content, or finish reasons. The family is encoded in
the ``modelId`` prefix (e.g. ``anthropic.claude-...``,
``meta.llama3-...``).

Ported from ``ateam/stratix/sdk/python/adapters/llm_providers/bedrock_adapter.py``.
"""

from __future__ import annotations

from typing import Tuple

# Ordered for fast prefix dispatch — matches the longest-distinct prefix
# Bedrock currently supports. Update in lockstep with the source ateam
# adapter and the ``BEDROCK_PRICING`` table when AWS adds new families.
_FAMILY_PREFIXES: Tuple[Tuple[str, str], ...] = (
    ("anthropic.", "anthropic"),
    ("meta.", "meta"),
    ("cohere.", "cohere"),
    ("amazon.", "amazon"),
    ("ai21.", "ai21"),
    ("mistral.", "mistral"),
)


def detect_provider_family(model_id: str) -> str:
    """Return the provider family name for a Bedrock ``modelId``.

    Args:
        model_id: The Bedrock ``modelId`` (e.g.
            ``"anthropic.claude-3-5-sonnet-20241022-v2:0"``).

    Returns:
        One of ``"anthropic"``, ``"meta"``, ``"cohere"``, ``"amazon"``,
        ``"ai21"``, ``"mistral"``, or ``"unknown"`` (including for empty
        / falsy input).
    """
    if not model_id:
        return "unknown"
    lower = model_id.lower()
    for prefix, family in _FAMILY_PREFIXES:
        if lower.startswith(prefix):
            return family
    return "unknown"


# Backward-compat alias for the original private helper name in ateam.
_detect_provider_family = detect_provider_family
