"""Vertex content normalization helpers.

Vertex AI's ``generate_content`` accepts a wider range of input shapes
than other provider SDKs:

* A bare ``str``.
* A list of ``str``.
* A list of ``Content`` proto objects with ``role`` / ``parts``.
* A list of dicts with ``role`` / ``parts`` keys.

The functions here normalize all of those into the canonical
``[{"role": str, "content": str}]`` shape used by the LayerLens
``model.invoke`` payload, with a per-message length cap to keep traces
small.

Token usage extraction lives here too because Vertex exposes it on a
``usage_metadata`` proto that is shared across Gemini, Anthropic-on-
Vertex, and Llama-on-Vertex responses.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from layerlens.instrument.adapters.providers._base.tokens import NormalizedTokenUsage

logger = logging.getLogger(__name__)

# Per-message cap so a single huge prompt cannot blow up an event payload.
# Matches the value used by every other provider adapter in this package.
_MESSAGE_CONTENT_CAP = 10_000


def normalize_vertex_contents(contents: Any) -> Optional[List[Dict[str, str]]]:
    """Normalize Vertex AI ``contents`` to ``[{role, content}]``.

    Args:
        contents: The first positional argument (or ``contents`` kwarg)
            passed to ``GenerativeModel.generate_content``. May be
            ``None``, a ``str``, a list of ``str``, a list of
            ``Content`` objects (duck-typed via ``role``/``parts``), or
            a list of dicts.

    Returns:
        Normalized list of messages, or ``None`` if the input was empty
        or could not be parsed.
    """
    if contents is None:
        return None
    try:
        messages: List[Dict[str, str]] = []

        if isinstance(contents, str):
            messages.append({"role": "user", "content": contents[:_MESSAGE_CONTENT_CAP]})
            return messages

        if isinstance(contents, list):
            for item in contents:
                if isinstance(item, str):
                    messages.append(
                        {"role": "user", "content": item[:_MESSAGE_CONTENT_CAP]}
                    )
                elif hasattr(item, "role") and hasattr(item, "parts"):
                    role = str(getattr(item, "role", "user"))
                    parts_text: List[str] = []
                    for part in getattr(item, "parts", []):
                        text = getattr(part, "text", None)
                        if text:
                            parts_text.append(str(text))
                    if parts_text:
                        messages.append(
                            {
                                "role": role,
                                "content": "\n".join(parts_text)[:_MESSAGE_CONTENT_CAP],
                            }
                        )
                elif isinstance(item, dict):
                    role = str(item.get("role", "user"))
                    parts = item.get("parts", [])
                    parts_text2: List[str] = []
                    for p in parts:
                        if isinstance(p, str):
                            parts_text2.append(p)
                        elif isinstance(p, dict) and "text" in p:
                            parts_text2.append(str(p["text"]))
                    if parts_text2:
                        messages.append(
                            {
                                "role": role,
                                "content": "\n".join(parts_text2)[:_MESSAGE_CONTENT_CAP],
                            }
                        )
        return messages if messages else None
    except Exception:
        logger.debug("Error normalizing Vertex contents", exc_info=True)
        return None


def extract_output_text(response: Any) -> Optional[Dict[str, str]]:
    """Extract the assistant text from a Vertex generate_content response.

    Returns ``{"role": "model", "content": ...}`` to match Vertex's
    role conventions, or ``None`` if no text could be extracted.
    """
    try:
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return None
        content = getattr(candidates[0], "content", None)
        if not content:
            return None
        parts = getattr(content, "parts", None) or []
        texts: List[str] = []
        for part in parts:
            text = getattr(part, "text", None)
            if text:
                texts.append(str(text))
        if texts:
            return {"role": "model", "content": "\n".join(texts)[:_MESSAGE_CONTENT_CAP]}
    except Exception:
        logger.debug("Error extracting Vertex output text", exc_info=True)
    return None


def extract_function_calls(response: Any) -> List[Dict[str, Any]]:
    """Extract ``function_call`` parts from a Vertex response.

    Vertex represents tool / function calls as ``part.function_call``
    sub-objects with ``name`` and ``args`` fields. Returns one dict per
    function call in the canonical ``{"name", "arguments"}`` shape used
    by ``LLMProviderAdapter._emit_tool_calls``.
    """
    tool_calls: List[Dict[str, Any]] = []
    try:
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return tool_calls
        content = getattr(candidates[0], "content", None)
        if not content:
            return tool_calls
        parts = getattr(content, "parts", None) or []
        for part in parts:
            fn_call = getattr(part, "function_call", None)
            if fn_call:
                tool_calls.append(
                    {
                        "name": getattr(fn_call, "name", "unknown"),
                        "arguments": dict(getattr(fn_call, "args", {}) or {}),
                    }
                )
    except Exception:
        logger.debug("Error extracting Vertex function calls", exc_info=True)
    return tool_calls


def extract_usage(response: Any) -> Optional[NormalizedTokenUsage]:
    """Extract :class:`NormalizedTokenUsage` from a Vertex response.

    Reads from ``response.usage_metadata`` which is consistent across
    Gemini, Anthropic-on-Vertex, and Llama-on-Vertex responses. Returns
    ``None`` if ``usage_metadata`` is absent (e.g. partial stream chunk
    before the final usage frame).
    """
    metadata = getattr(response, "usage_metadata", None)
    if not metadata:
        return None
    prompt = getattr(metadata, "prompt_token_count", 0) or 0
    completion = getattr(metadata, "candidates_token_count", 0) or 0
    total = getattr(metadata, "total_token_count", 0) or (prompt + completion)
    reasoning = getattr(metadata, "thoughts_token_count", None)

    return NormalizedTokenUsage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
        reasoning_tokens=reasoning,
    )
