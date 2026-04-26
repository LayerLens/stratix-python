"""Per-family token, message, and output extraction for Bedrock.

The ``invoke_model`` API speaks the native body shape of each provider
family (Anthropic Messages, Meta prompt-completion, Cohere generations,
Amazon Titan, Mistral, AI21, ...). The ``Converse`` API uses a unified
envelope. This module contains the dispatch logic for both — it is
deliberately stateless so the adapter can call into it without holding
any per-request lock.

Ported from ``ateam/stratix/sdk/python/adapters/llm_providers/bedrock_adapter.py``.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from layerlens.instrument.adapters.providers._base.tokens import NormalizedTokenUsage
from layerlens.instrument.adapters.providers.bedrock.family import detect_provider_family

logger = logging.getLogger(__name__)

# Truncate captured strings to keep telemetry payloads bounded.
_MAX_CONTENT_CHARS = 10_000


def extract_invoke_messages(
    kwargs: Dict[str, Any],
    model_id: str,
) -> Optional[List[Dict[str, str]]]:
    """Extract input messages from an ``invoke_model`` request.

    Args:
        kwargs: The kwargs that were passed to ``invoke_model``. The
            ``body`` field is expected to be a JSON-encoded request body
            (``str``, ``bytes``, or pre-decoded ``dict``).
        model_id: The Bedrock ``modelId`` — used to dispatch on family.

    Returns:
        A normalised list of ``{"role", "content"}`` dicts, or ``None``
        if no messages could be extracted (malformed body, missing
        prompt, etc.).
    """
    try:
        body = kwargs.get("body")
        if not body:
            return None
        if isinstance(body, (str, bytes)):
            body_data: Dict[str, Any] = json.loads(body)
        elif isinstance(body, dict):
            body_data = body
        else:
            return None

        family = detect_provider_family(model_id)
        messages: List[Dict[str, str]] = []

        if family == "anthropic":
            system = body_data.get("system", "")
            if system:
                messages.append(
                    {"role": "system", "content": str(system)[:_MAX_CONTENT_CHARS]}
                )
            for msg in body_data.get("messages", []):
                if isinstance(msg, dict) and "role" in msg:
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        parts = [
                            str(p.get("text", ""))
                            for p in content
                            if isinstance(p, dict) and "text" in p
                        ]
                        content = "\n".join(parts)
                    messages.append(
                        {
                            "role": str(msg["role"]),
                            "content": str(content)[:_MAX_CONTENT_CHARS],
                        }
                    )
        elif family in ("meta", "mistral"):
            prompt = body_data.get("prompt", "")
            if prompt:
                messages.append(
                    {"role": "user", "content": str(prompt)[:_MAX_CONTENT_CHARS]}
                )
        else:
            # ai21 / amazon / cohere / unknown all expose a prompt-style
            # field; fall through to a generic prompt extraction.
            prompt = body_data.get("prompt") or body_data.get("inputText", "")
            if prompt:
                messages.append(
                    {"role": "user", "content": str(prompt)[:_MAX_CONTENT_CHARS]}
                )

        return messages if messages else None
    except Exception:
        logger.debug("Error extracting Bedrock invoke messages", exc_info=True)
        return None


def extract_invoke_output(
    body_data: Dict[str, Any],
    family: str,
) -> Optional[Dict[str, str]]:
    """Extract the assistant output message from an ``invoke_model`` body."""
    try:
        if not body_data:
            return None

        content = ""
        if family == "anthropic":
            content_blocks = body_data.get("content", [])
            if content_blocks and isinstance(content_blocks, list):
                parts: List[str] = []
                for block in content_blocks:
                    if isinstance(block, dict) and "text" in block:
                        parts.append(str(block["text"]))
                content = "\n".join(parts)
        elif family in ("meta", "mistral"):
            content = str(body_data.get("generation", ""))
        elif family == "cohere":
            generations = body_data.get("generations", [])
            if generations and isinstance(generations, list):
                content = str(generations[0].get("text", ""))
        elif family == "amazon":
            results = body_data.get("results", [])
            if results and isinstance(results, list):
                content = str(results[0].get("outputText", ""))
        elif family == "ai21":
            # AI21 Jamba returns a chat-style choices array.
            choices = body_data.get("choices", [])
            if choices and isinstance(choices, list):
                msg = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
                content = str(msg.get("content", ""))
            if not content:
                # AI21 J2 legacy completions.
                completions = body_data.get("completions", [])
                if completions and isinstance(completions, list):
                    data = completions[0].get("data", {}) if isinstance(completions[0], dict) else {}
                    content = str(data.get("text", ""))
        else:
            content = str(
                body_data.get("generation", "")
                or body_data.get("completion", "")
                or body_data.get("outputText", "")
            )

        if content:
            return {"role": "assistant", "content": content[:_MAX_CONTENT_CHARS]}
        return None
    except Exception:
        logger.debug("Error extracting Bedrock invoke output", exc_info=True)
        return None


def extract_converse_output(response: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """Extract the assistant output message from a ``Converse`` response."""
    try:
        output = response.get("output", {})
        message = output.get("message", {})
        if not message:
            return None
        content_blocks = message.get("content", [])
        if not content_blocks:
            return None
        parts: List[str] = []
        for block in content_blocks:
            if isinstance(block, dict) and "text" in block:
                parts.append(str(block["text"]))
        if parts:
            return {
                "role": "assistant",
                "content": "\n".join(parts)[:_MAX_CONTENT_CHARS],
            }
        return None
    except Exception:
        logger.debug("Error extracting Bedrock converse output", exc_info=True)
        return None


def extract_invoke_usage(
    body_data: Dict[str, Any],
    family: str,
) -> Optional[NormalizedTokenUsage]:
    """Extract token usage from an ``invoke_model`` response body.

    Each family reports tokens in a different field; unknown families
    fall through to a generic ``inputTokenCount`` / ``prompt_tokens``
    probe before giving up.
    """
    if not body_data:
        return None

    if family == "anthropic":
        usage = body_data.get("usage", {})
        input_t = int(usage.get("input_tokens", 0) or 0)
        output_t = int(usage.get("output_tokens", 0) or 0)
        return NormalizedTokenUsage(
            prompt_tokens=input_t,
            completion_tokens=output_t,
            total_tokens=input_t + output_t,
        )

    if family == "meta":
        prompt = int(body_data.get("prompt_token_count", 0) or 0)
        completion = int(body_data.get("generation_token_count", 0) or 0)
        return NormalizedTokenUsage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
        )

    if family == "cohere":
        meta = body_data.get("meta", {})
        tokens = meta.get("billed_units", {}) if isinstance(meta, dict) else {}
        prompt = int(tokens.get("input_tokens", 0) or 0)
        completion = int(tokens.get("output_tokens", 0) or 0)
        return NormalizedTokenUsage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
        )

    if family == "ai21":
        usage = body_data.get("usage", {})
        prompt = int(usage.get("prompt_tokens", 0) or 0)
        completion = int(usage.get("completion_tokens", 0) or 0)
        if prompt or completion:
            return NormalizedTokenUsage(
                prompt_tokens=prompt,
                completion_tokens=completion,
                total_tokens=prompt + completion,
            )

    if family == "mistral":
        # Mistral on Bedrock reports tokens via prompt_tokens / completion_tokens
        # at the top level when available.
        prompt = int(body_data.get("prompt_tokens", 0) or 0)
        completion = int(body_data.get("completion_tokens", 0) or 0)
        if prompt or completion:
            return NormalizedTokenUsage(
                prompt_tokens=prompt,
                completion_tokens=completion,
                total_tokens=prompt + completion,
            )

    if family == "amazon":
        # Titan returns tokens at the top level.
        prompt = int(body_data.get("inputTextTokenCount", 0) or 0)
        results = body_data.get("results", [])
        completion = 0
        if results and isinstance(results, list):
            completion = int(results[0].get("tokenCount", 0) or 0)
        if prompt or completion:
            return NormalizedTokenUsage(
                prompt_tokens=prompt,
                completion_tokens=completion,
                total_tokens=prompt + completion,
            )

    # Generic last-chance fallback for unknown / future families.
    prompt = int(body_data.get("inputTokenCount", 0) or body_data.get("prompt_tokens", 0) or 0)
    completion = int(
        body_data.get("outputTokenCount", 0) or body_data.get("completion_tokens", 0) or 0
    )
    if prompt or completion:
        return NormalizedTokenUsage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
        )

    return None


def extract_converse_usage(response: Dict[str, Any]) -> Optional[NormalizedTokenUsage]:
    """Extract token usage from a ``Converse`` response."""
    usage = response.get("usage", {})
    if not usage:
        return None
    prompt = int(usage.get("inputTokens", 0) or 0)
    completion = int(usage.get("outputTokens", 0) or 0)
    return NormalizedTokenUsage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
    )


def build_invoke_metadata(body_data: Dict[str, Any], family: str) -> Dict[str, Any]:
    """Build the ``metadata`` payload for an ``invoke_model`` event."""
    invoke_metadata: Dict[str, Any] = {
        "method": "invoke_model",
        "provider_family": family,
    }
    if family == "anthropic":
        sr = body_data.get("stop_reason")
        if sr is not None:
            invoke_metadata["finish_reason"] = sr
        rid = body_data.get("id")
        if rid is not None:
            invoke_metadata["response_id"] = rid
    elif family in ("meta", "mistral"):
        sr = body_data.get("stop_reason")
        if sr is not None:
            invoke_metadata["finish_reason"] = sr
    elif family == "cohere":
        gens = body_data.get("generations", [])
        if gens and isinstance(gens, list):
            sr = gens[0].get("finish_reason") if isinstance(gens[0], dict) else None
            if sr is not None:
                invoke_metadata["finish_reason"] = sr
    elif family == "ai21":
        choices = body_data.get("choices", [])
        if choices and isinstance(choices, list):
            sr = choices[0].get("finish_reason") if isinstance(choices[0], dict) else None
            if sr is not None:
                invoke_metadata["finish_reason"] = sr
    elif family == "amazon":
        results = body_data.get("results", [])
        if results and isinstance(results, list):
            sr = results[0].get("completionReason") if isinstance(results[0], dict) else None
            if sr is not None:
                invoke_metadata["finish_reason"] = sr
    else:
        sr = body_data.get("stop_reason") or body_data.get("finish_reason")
        if sr is not None:
            invoke_metadata["finish_reason"] = sr
    return invoke_metadata
