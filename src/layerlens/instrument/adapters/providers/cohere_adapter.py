"""Cohere LLM Provider Adapter.

Wraps the Cohere Python SDK (``cohere`` >= 5.x) to intercept ``chat``
and ``embed`` calls. Emits ``model.invoke``, ``cost.record``,
``tool.call``, and ``policy.violation`` events.

This adapter is **fresh-built**, not a port — Cohere did not have an
adapter in ``ateam`` source as of 2026-04-25. It follows the same
contract as the OpenAI / Anthropic adapters:

* Wraps ``client.chat`` (Cohere v1) and ``client.v2.chat`` (Cohere v2)
  with method substitution.
* Wraps ``client.embed`` for embedding telemetry.
* Honors :class:`CaptureConfig` for layer gating.
* Restores originals on :meth:`disconnect`.

Cohere's pricing tier is reused from the canonical
:data:`PRICING` table; Cohere-on-Bedrock uses :data:`BEDROCK_PRICING`.
For models not in either table the ``cost.record`` event sets
``api_cost_usd`` to ``None`` and ``pricing_unavailable`` to ``True``.
"""

from __future__ import annotations

import time
import logging
from typing import Any, Dict, List, Optional

from layerlens.instrument.adapters._base.capture import CaptureConfig
from layerlens.instrument.adapters.providers._base.tokens import NormalizedTokenUsage
from layerlens.instrument.adapters.providers._base.provider import LLMProviderAdapter

logger = logging.getLogger(__name__)

# Parameters captured from request kwargs when present.
_CAPTURE_PARAMS = frozenset(
    {
        "model",
        "temperature",
        "max_tokens",
        "p",
        "k",
        "top_p",
        "top_k",
        "frequency_penalty",
        "presence_penalty",
        "response_format",
        "tool_choice",
    }
)


class CohereAdapter(LLMProviderAdapter):
    """LayerLens adapter for the Cohere Python SDK.

    Usage::

        import cohere
        from layerlens.instrument.adapters.providers.cohere_adapter import CohereAdapter

        adapter = CohereAdapter()
        adapter.connect()

        client = cohere.Client(api_key=os.environ["COHERE_API_KEY"])
        adapter.connect_client(client)

        client.chat(model="command-r-plus", message="Hello")
    """

    FRAMEWORK = "cohere"
    VERSION = "0.1.0"

    def __init__(
        self,
        stratix: Any = None,
        capture_config: Optional[CaptureConfig] = None,
    ) -> None:
        super().__init__(stratix=stratix, capture_config=capture_config)

    def connect_client(self, client: Any) -> Any:
        """Wrap Cohere v1 (``client.chat``) and v2 (``client.v2.chat``) endpoints."""
        self._client = client

        # v1 chat (callable on the client directly).
        if hasattr(client, "chat") and callable(client.chat):
            original_chat = client.chat
            self._originals["chat"] = original_chat
            client.chat = self._wrap_chat(original_chat, version="v1")

        # v2 chat (Cohere SDK 5.x exposes ``client.v2.chat``).
        v2 = getattr(client, "v2", None)
        if v2 is not None and hasattr(v2, "chat") and callable(v2.chat):
            original_v2_chat = v2.chat
            self._originals["v2.chat"] = original_v2_chat
            v2.chat = self._wrap_chat(original_v2_chat, version="v2")

        # Embed.
        if hasattr(client, "embed") and callable(client.embed):
            original_embed = client.embed
            self._originals["embed"] = original_embed
            client.embed = self._wrap_embed(original_embed)

        return client

    def _restore_originals(self) -> None:
        if self._client is None:
            return
        if "chat" in self._originals:
            try:
                self._client.chat = self._originals["chat"]
            except Exception:
                logger.warning("Could not restore chat")
        if "v2.chat" in self._originals:
            try:
                self._client.v2.chat = self._originals["v2.chat"]
            except Exception:
                logger.warning("Could not restore v2.chat")
        if "embed" in self._originals:
            try:
                self._client.embed = self._originals["embed"]
            except Exception:
                logger.warning("Could not restore embed")

    @staticmethod
    def _detect_framework_version() -> Optional[str]:
        try:
            import cohere  # type: ignore[import-not-found,unused-ignore]

            version = getattr(cohere, "__version__", None)
            return str(version) if version is not None else None
        except ImportError:
            return None

    # --- Wrapping ---

    def _wrap_chat(self, original: Any, *, version: str) -> Any:
        adapter = self

        def traced_chat(*args: Any, **kwargs: Any) -> Any:
            model = kwargs.get("model")
            params = {k: kwargs[k] for k in _CAPTURE_PARAMS if k in kwargs}
            params["api_version"] = version
            start_ns = time.time_ns()

            # v1 uses ``message`` (single string), v2 uses ``messages`` (list).
            input_messages: Optional[List[Dict[str, str]]] = None
            if version == "v1":
                msg = kwargs.get("message")
                if msg:
                    input_messages = [{"role": "user", "content": str(msg)[:10_000]}]
                preamble = kwargs.get("preamble")
                if preamble:
                    if input_messages is None:
                        input_messages = []
                    input_messages.insert(
                        0,
                        {"role": "system", "content": str(preamble)[:10_000]},
                    )
            else:
                input_messages = adapter._normalize_messages(kwargs.get("messages"))

            try:
                response = original(*args, **kwargs)
            except Exception as exc:
                elapsed_ms = (time.time_ns() - start_ns) / 1_000_000
                try:
                    adapter._emit_model_invoke(
                        provider="cohere",
                        model=model,
                        parameters=params,
                        latency_ms=elapsed_ms,
                        error=str(exc),
                        input_messages=input_messages,
                    )
                    adapter._emit_provider_error("cohere", str(exc), model=model)
                except Exception:
                    logger.warning("Error emitting Cohere error event", exc_info=True)
                raise

            try:
                elapsed_ms = (time.time_ns() - start_ns) / 1_000_000
                usage = adapter._extract_usage(response, version)
                output_message = adapter._extract_output_message(response, version)

                metadata: Dict[str, Any] = {}
                resp_id = getattr(response, "id", None) or getattr(
                    response, "generation_id", None
                )
                if resp_id is not None:
                    metadata["response_id"] = resp_id
                finish_reason = getattr(response, "finish_reason", None)
                if finish_reason is not None:
                    metadata["finish_reason"] = str(finish_reason)

                adapter._emit_model_invoke(
                    provider="cohere",
                    model=model,
                    parameters=params,
                    usage=usage,
                    latency_ms=elapsed_ms,
                    input_messages=input_messages,
                    output_message=output_message,
                    metadata=metadata if metadata else None,
                )
                adapter._emit_cost_record(
                    model=model,
                    usage=usage,
                    provider="cohere",
                )

                tool_calls = adapter._extract_tool_calls(response, version)
                if tool_calls:
                    adapter._emit_tool_calls(tool_calls, parent_model=model)
            except Exception:
                logger.warning("Error emitting Cohere trace events", exc_info=True)

            return response

        traced_chat._layerlens_original = original  # type: ignore[attr-defined]
        return traced_chat

    def _wrap_embed(self, original: Any) -> Any:
        adapter = self

        def traced_embed(*args: Any, **kwargs: Any) -> Any:
            model = kwargs.get("model")
            start_ns = time.time_ns()

            try:
                response = original(*args, **kwargs)
            except Exception as exc:
                elapsed_ms = (time.time_ns() - start_ns) / 1_000_000
                try:
                    adapter._emit_model_invoke(
                        provider="cohere",
                        model=model,
                        latency_ms=elapsed_ms,
                        error=str(exc),
                        metadata={"request_type": "embedding"},
                    )
                except Exception:
                    logger.warning("Error emitting Cohere embed error", exc_info=True)
                raise

            try:
                elapsed_ms = (time.time_ns() - start_ns) / 1_000_000
                # Cohere embed responses use ``meta.billed_units.input_tokens``.
                meta = getattr(response, "meta", None)
                billed = getattr(meta, "billed_units", None) if meta else None
                if billed is None and isinstance(meta, dict):
                    billed = meta.get("billed_units")
                input_tokens = 0
                if billed is not None:
                    input_tokens = (
                        getattr(billed, "input_tokens", 0)
                        if not isinstance(billed, dict)
                        else billed.get("input_tokens", 0)
                    ) or 0
                usage = NormalizedTokenUsage(
                    prompt_tokens=input_tokens,
                    completion_tokens=0,
                    total_tokens=input_tokens,
                )

                adapter._emit_model_invoke(
                    provider="cohere",
                    model=model,
                    usage=usage,
                    latency_ms=elapsed_ms,
                    metadata={"request_type": "embedding"},
                )
                adapter._emit_cost_record(
                    model=model,
                    usage=usage,
                    provider="cohere",
                )
            except Exception:
                logger.warning("Error emitting Cohere embed events", exc_info=True)

            return response

        traced_embed._layerlens_original = original  # type: ignore[attr-defined]
        return traced_embed

    # --- Token + content extraction ---

    @staticmethod
    def _extract_usage(
        response: Any,
        version: str,  # noqa: ARG004 - kept for callsite symmetry; both versions use the same shape
    ) -> Optional[NormalizedTokenUsage]:
        """Extract token usage from a Cohere chat response.

        Both v1 and v2 expose ``response.meta.billed_units`` and / or
        ``response.usage.tokens`` (varies by SDK version).
        """
        meta = getattr(response, "meta", None)
        if meta is not None:
            billed = getattr(meta, "billed_units", None)
            if billed is None and isinstance(meta, dict):
                billed = meta.get("billed_units")
            if billed is not None:
                input_tokens = (
                    getattr(billed, "input_tokens", 0)
                    if not isinstance(billed, dict)
                    else billed.get("input_tokens", 0)
                ) or 0
                output_tokens = (
                    getattr(billed, "output_tokens", 0)
                    if not isinstance(billed, dict)
                    else billed.get("output_tokens", 0)
                ) or 0
                return NormalizedTokenUsage(
                    prompt_tokens=int(input_tokens),
                    completion_tokens=int(output_tokens),
                    total_tokens=int(input_tokens) + int(output_tokens),
                )

        # v2 sometimes exposes ``usage.tokens.input_tokens`` / ``output_tokens``.
        usage = getattr(response, "usage", None)
        if usage is not None:
            tokens = getattr(usage, "tokens", None)
            if tokens is not None:
                input_tokens = getattr(tokens, "input_tokens", 0) or 0
                output_tokens = getattr(tokens, "output_tokens", 0) or 0
                return NormalizedTokenUsage(
                    prompt_tokens=int(input_tokens),
                    completion_tokens=int(output_tokens),
                    total_tokens=int(input_tokens) + int(output_tokens),
                )

        return None

    @staticmethod
    def _extract_output_message(
        response: Any, version: str
    ) -> Optional[Dict[str, str]]:
        """Extract the assistant output content."""
        try:
            if version == "v1":
                # v1 ``response.text`` contains the generated message.
                text = getattr(response, "text", None)
                if text:
                    return {"role": "assistant", "content": str(text)[:10_000]}
                return None

            # v2: ``response.message.content`` is a list of content blocks.
            message = getattr(response, "message", None)
            if message is None:
                return None
            content = getattr(message, "content", None) or []
            parts: List[str] = []
            for block in content:
                btype = getattr(block, "type", None)
                if btype == "text":
                    text = getattr(block, "text", "")
                    if text:
                        parts.append(str(text))
            if parts:
                return {"role": "assistant", "content": "\n".join(parts)[:10_000]}
        except Exception:
            logger.debug("Error extracting Cohere output message", exc_info=True)
        return None

    @staticmethod
    def _extract_tool_calls(response: Any, version: str) -> List[Dict[str, Any]]:
        """Extract tool calls (function invocations) from the response."""
        calls: List[Dict[str, Any]] = []
        try:
            if version == "v1":
                # v1: ``response.tool_calls`` is a list of {name, parameters}.
                v1_calls = getattr(response, "tool_calls", None) or []
                for tc in v1_calls:
                    name = getattr(tc, "name", "unknown")
                    params = getattr(tc, "parameters", None) or {}
                    calls.append({"name": name, "arguments": params})
                return calls

            # v2: ``response.message.tool_calls`` of {id, function: {name, arguments}}.
            message = getattr(response, "message", None)
            if message is None:
                return calls
            v2_calls = getattr(message, "tool_calls", None) or []
            import json as _json

            for tc in v2_calls:
                fn = getattr(tc, "function", None)
                if fn is None:
                    continue
                args_str = getattr(fn, "arguments", "{}")
                try:
                    args = _json.loads(args_str) if isinstance(args_str, str) else args_str
                except (ValueError, TypeError):
                    args = args_str
                calls.append(
                    {
                        "name": getattr(fn, "name", "unknown"),
                        "arguments": args,
                        "id": getattr(tc, "id", None),
                    }
                )
        except Exception:
            logger.debug("Error extracting Cohere tool calls", exc_info=True)
        return calls


# Registry lazy-loading convention.
ADAPTER_CLASS = CohereAdapter
