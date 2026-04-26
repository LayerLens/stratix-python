"""LiteLLM callback handler that emits LayerLens telemetry events.

LiteLLM exposes a callback registry on ``litellm.callbacks`` and
asynchronous siblings on ``litellm.success_callback`` /
``litellm.failure_callback``. We register an instance of
:class:`LayerLensLiteLLMCallback` (and, for ``acompletion``, the same
instance is invoked through the async helpers) so every call routed
through LiteLLM produces ``model.invoke``, ``cost.record`` and (on
failure) ``policy.violation`` events identical to those emitted by the
direct provider adapters.

Cost is sourced from LiteLLM first (it ships its own pricing manifest
and computes ``litellm.completion_cost``); when LiteLLM cannot price the
call the adapter falls through to the canonical LayerLens pricing
manifest in :mod:`layerlens.instrument.adapters.providers._base.pricing`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

from layerlens.instrument.adapters.providers._base.tokens import NormalizedTokenUsage
from layerlens.instrument.adapters.providers.litellm.routing import detect_provider

if TYPE_CHECKING:  # pragma: no cover - type-only import to avoid cycle
    from layerlens.instrument.adapters.providers.litellm.adapter import LiteLLMAdapter

logger = logging.getLogger(__name__)


class LayerLensLiteLLMCallback:
    """LiteLLM callback handler that emits LayerLens events.

    Implements the LiteLLM logger contract:

    * :meth:`log_success_event` — sync ``completion()`` succeeded.
    * :meth:`log_failure_event` — sync ``completion()`` raised.
    * :meth:`log_stream_event` — streaming ``completion()`` finished.
    * :meth:`async_log_success_event` — async ``acompletion()`` succeeded.
    * :meth:`async_log_failure_event` — async ``acompletion()`` raised.
    * :meth:`async_log_stream_event` — streaming ``acompletion()`` finished.

    The async variants delegate to the sync helpers — LiteLLM serialises
    the callback for both code paths through the same ``kwargs`` /
    ``response_obj`` shape, so no separate async logic is needed inside
    the handler itself.
    """

    def __init__(self, adapter: "LiteLLMAdapter") -> None:
        self._adapter = adapter

    # ------------------------------------------------------------------
    # Sync callbacks (litellm.completion)
    # ------------------------------------------------------------------

    def log_success_event(
        self,
        kwargs: Dict[str, Any],
        response_obj: Any,
        start_time: Any,
        end_time: Any,
    ) -> None:
        """Emit ``model.invoke`` and ``cost.record`` on successful completion."""
        try:
            model = kwargs.get("model", "")
            provider = detect_provider(model)
            latency_ms = self._calc_latency_ms(start_time, end_time)
            usage = self._extract_usage(response_obj)

            input_messages = self._adapter._normalize_messages(kwargs.get("messages"))
            output_message = self._extract_output_message(response_obj)

            metadata: Dict[str, Any] = {}
            if response_obj is not None:
                choices = getattr(response_obj, "choices", None) or []
                if choices:
                    fr = getattr(choices[0], "finish_reason", None)
                    if fr is not None:
                        metadata["finish_reason"] = fr
                resp_id = getattr(response_obj, "id", None)
                if resp_id is not None:
                    metadata["response_id"] = resp_id
                resp_model = getattr(response_obj, "model", None)
                if resp_model is not None:
                    metadata["response_model"] = resp_model

            self._adapter._emit_model_invoke(
                provider=provider,
                model=model,
                parameters=self._extract_params(kwargs),
                usage=usage,
                latency_ms=latency_ms,
                input_messages=input_messages,
                output_message=output_message,
                metadata=metadata if metadata else None,
            )

            cost = self._get_litellm_cost(kwargs, response_obj)
            if cost is not None:
                self._adapter.emit_dict_event(
                    "cost.record",
                    {
                        "provider": provider,
                        "model": model,
                        "prompt_tokens": usage.prompt_tokens if usage else 0,
                        "completion_tokens": usage.completion_tokens if usage else 0,
                        "total_tokens": usage.total_tokens if usage else 0,
                        "api_cost_usd": cost,
                        "cost_source": "litellm",
                    },
                )
            elif usage:
                self._adapter._emit_cost_record(
                    model=model,
                    usage=usage,
                    provider=provider,
                )
        except Exception:
            logger.warning("Error in LiteLLM success callback", exc_info=True)

    def log_failure_event(
        self,
        kwargs: Dict[str, Any],
        response_obj: Any,  # noqa: ARG002 - LiteLLM callback signature requires this arg
        start_time: Any,
        end_time: Any,
    ) -> None:
        """Emit ``model.invoke`` with error and ``policy.violation`` on failure."""
        try:
            model = kwargs.get("model", "")
            provider = detect_provider(model)
            latency_ms = self._calc_latency_ms(start_time, end_time)
            error = kwargs.get("exception", "")

            input_messages = self._adapter._normalize_messages(kwargs.get("messages"))

            self._adapter._emit_model_invoke(
                provider=provider,
                model=model,
                parameters=self._extract_params(kwargs),
                latency_ms=latency_ms,
                error=str(error),
                input_messages=input_messages,
            )
            self._adapter._emit_provider_error(provider, str(error), model=model)
        except Exception:
            logger.warning("Error in LiteLLM failure callback", exc_info=True)

    def log_stream_event(
        self,
        kwargs: Dict[str, Any],
        response_obj: Any,
        start_time: Any,
        end_time: Any,
    ) -> None:
        """Emit ``model.invoke`` (with ``streaming: True``) when the stream completes."""
        try:
            model = kwargs.get("model", "")
            provider = detect_provider(model)
            latency_ms = self._calc_latency_ms(start_time, end_time)
            usage = self._extract_usage(response_obj)

            input_messages = self._adapter._normalize_messages(kwargs.get("messages"))

            stream_meta: Dict[str, Any] = {"streaming": True}
            if response_obj is not None:
                choices = getattr(response_obj, "choices", None) or []
                if choices:
                    fr = getattr(choices[0], "finish_reason", None)
                    if fr is not None:
                        stream_meta["finish_reason"] = fr
                resp_id = getattr(response_obj, "id", None)
                if resp_id is not None:
                    stream_meta["response_id"] = resp_id

            self._adapter._emit_model_invoke(
                provider=provider,
                model=model,
                usage=usage,
                latency_ms=latency_ms,
                metadata=stream_meta,
                input_messages=input_messages,
            )

            if usage:
                self._adapter._emit_cost_record(
                    model=model,
                    usage=usage,
                    provider=provider,
                )
        except Exception:
            logger.warning("Error in LiteLLM stream callback", exc_info=True)

    # ------------------------------------------------------------------
    # Async callbacks (litellm.acompletion)
    # ------------------------------------------------------------------
    #
    # LiteLLM hands the same kwargs / response_obj shape to the async
    # callbacks as it does to the sync ones, so the async variants simply
    # forward to the sync handlers. Marking them ``async`` ensures
    # LiteLLM's async dispatcher schedules them correctly via ``await``.

    async def async_log_success_event(
        self,
        kwargs: Dict[str, Any],
        response_obj: Any,
        start_time: Any,
        end_time: Any,
    ) -> None:
        """Async sibling of :meth:`log_success_event`."""
        self.log_success_event(kwargs, response_obj, start_time, end_time)

    async def async_log_failure_event(
        self,
        kwargs: Dict[str, Any],
        response_obj: Any,
        start_time: Any,
        end_time: Any,
    ) -> None:
        """Async sibling of :meth:`log_failure_event`."""
        self.log_failure_event(kwargs, response_obj, start_time, end_time)

    async def async_log_stream_event(
        self,
        kwargs: Dict[str, Any],
        response_obj: Any,
        start_time: Any,
        end_time: Any,
    ) -> None:
        """Async sibling of :meth:`log_stream_event`."""
        self.log_stream_event(kwargs, response_obj, start_time, end_time)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _calc_latency_ms(start_time: Any, end_time: Any) -> Optional[float]:
        """Compute latency in ms from LiteLLM's ``start_time`` / ``end_time``.

        LiteLLM passes either ``datetime.datetime`` objects (callback path)
        or raw monotonic timestamps (proxy path). Both are handled.
        """
        if start_time is None or end_time is None:
            return None
        try:
            if hasattr(start_time, "timestamp"):
                return float((end_time.timestamp() - start_time.timestamp()) * 1000)
            return float(end_time - start_time) * 1000
        except Exception:
            return None

    @staticmethod
    def _extract_usage(response_obj: Any) -> Optional[NormalizedTokenUsage]:
        """Extract token counts from a LiteLLM ``ModelResponse``."""
        if response_obj is None:
            return None
        usage = getattr(response_obj, "usage", None)
        if usage is None:
            return None
        prompt = getattr(usage, "prompt_tokens", 0) or 0
        completion = getattr(usage, "completion_tokens", 0) or 0
        return NormalizedTokenUsage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
        )

    @staticmethod
    def _extract_output_message(response_obj: Any) -> Optional[Dict[str, str]]:
        """Extract the assistant output message from a LiteLLM response.

        LiteLLM normalises every provider response to the OpenAI
        ``ChatCompletion`` shape, so the same accessor works for OpenAI,
        Anthropic, Bedrock, Vertex, etc.
        """
        try:
            if response_obj is None:
                return None
            choices = getattr(response_obj, "choices", None) or []
            if not choices:
                return None
            message = getattr(choices[0], "message", None)
            if not message:
                return None
            content = getattr(message, "content", None)
            if content:
                return {"role": "assistant", "content": str(content)[:10_000]}
        except Exception:
            return None
        return None

    @staticmethod
    def _extract_params(kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Capture the small set of common sampling params we record."""
        params: Dict[str, Any] = {}
        for key in ("temperature", "max_tokens", "top_p"):
            if key in kwargs:
                params[key] = kwargs[key]
        # LiteLLM nests provider-specific overrides under ``optional_params``.
        opt = kwargs.get("optional_params", {})
        if isinstance(opt, dict):
            for key in ("temperature", "max_tokens", "top_p"):
                if key in opt and key not in params:
                    params[key] = opt[key]
        return params

    @staticmethod
    def _get_litellm_cost(
        kwargs: Dict[str, Any],
        response_obj: Any,
    ) -> Optional[float]:
        """Try LiteLLM's built-in ``completion_cost`` for ground-truth pricing.

        Returns ``None`` if LiteLLM is unavailable, the model is not
        priced, or the helper raises — the caller falls through to the
        canonical LayerLens pricing manifest in that case.
        """
        try:
            import litellm  # type: ignore[import-not-found,import-untyped,unused-ignore]

            cost = litellm.completion_cost(
                model=kwargs.get("model", ""),
                completion_response=response_obj,
            )
            return float(cost) if cost else None
        except Exception:
            return None


__all__ = ["LayerLensLiteLLMCallback"]
