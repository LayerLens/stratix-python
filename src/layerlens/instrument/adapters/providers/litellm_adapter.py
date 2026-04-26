"""LiteLLM Provider Adapter.

Uses the LiteLLM callback handler pattern (not monkey-patch). Registers
:class:`LayerLensLiteLLMCallback` via ``litellm.callbacks``. Auto-detects
the underlying provider from the model-string prefix.

Ported from ``ateam/stratix/sdk/python/adapters/llm_providers/litellm_adapter.py``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from layerlens.instrument.adapters._base.adapter import AdapterStatus
from layerlens.instrument.adapters._base.capture import CaptureConfig
from layerlens.instrument.adapters.providers._base.tokens import NormalizedTokenUsage
from layerlens.instrument.adapters.providers._base.provider import LLMProviderAdapter

logger = logging.getLogger(__name__)

# Model prefix → provider mapping.
_PROVIDER_PREFIXES: Dict[str, str] = {
    "openai/": "openai",
    "anthropic/": "anthropic",
    "azure/": "azure_openai",
    "bedrock/": "aws_bedrock",
    "vertex_ai/": "google_vertex",
    "ollama/": "ollama",
    "cohere/": "cohere",
    "huggingface/": "huggingface",
    "together_ai/": "together_ai",
    "groq/": "groq",
}


def detect_provider(model_str: str) -> str:
    """Detect the underlying provider from a LiteLLM model string."""
    if not model_str:
        return "unknown"
    for prefix, provider in _PROVIDER_PREFIXES.items():
        if model_str.startswith(prefix):
            return provider
    lower = model_str.lower()
    if lower.startswith("gpt-") or lower.startswith("o1") or lower.startswith("o3"):
        return "openai"
    if lower.startswith("claude-"):
        return "anthropic"
    if lower.startswith("gemini-"):
        return "google_vertex"
    if lower.startswith("llama"):
        return "meta"
    if lower.startswith("mistral"):
        return "mistral"
    return "unknown"


class LayerLensLiteLLMCallback:
    """LiteLLM callback handler that emits LayerLens events.

    Registered via ``litellm.callbacks``. Implements
    :meth:`log_success_event`, :meth:`log_failure_event`, and
    :meth:`log_stream_event`.
    """

    def __init__(self, adapter: "LiteLLMAdapter") -> None:
        self._adapter = adapter

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
        """Emit ``model.invoke`` with error on failed completion."""
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
        """Emit ``model.invoke`` when the stream completes."""
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

    @staticmethod
    def _calc_latency_ms(start_time: Any, end_time: Any) -> Optional[float]:
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
        """Extract output message from a LiteLLM response (OpenAI-compatible)."""
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
            pass
        return None

    @staticmethod
    def _extract_params(kwargs: Dict[str, Any]) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        for key in ("temperature", "max_tokens", "top_p"):
            if key in kwargs:
                params[key] = kwargs[key]
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
        """Try to get cost from LiteLLM's built-in cost tracking."""
        try:
            import litellm  # type: ignore[import-not-found,import-untyped,unused-ignore]

            cost = litellm.completion_cost(
                model=kwargs.get("model", ""),
                completion_response=response_obj,
            )
            return float(cost) if cost else None
        except Exception:
            return None


class LiteLLMAdapter(LLMProviderAdapter):
    """LayerLens adapter for LiteLLM.

    Uses LiteLLM's callback handler pattern instead of monkey-patching.
    Auto-detects the underlying provider from the model-string prefix.
    """

    FRAMEWORK = "litellm"
    VERSION = "0.1.0"

    def __init__(
        self,
        stratix: Any = None,
        capture_config: Optional[CaptureConfig] = None,
    ) -> None:
        super().__init__(stratix=stratix, capture_config=capture_config)
        self._callback: Optional[LayerLensLiteLLMCallback] = None

    def connect(self) -> None:
        """Register the LayerLens callback with LiteLLM."""
        self._callback = LayerLensLiteLLMCallback(self)
        try:
            import litellm  # type: ignore[import-not-found,unused-ignore]

            # ``litellm.callbacks`` is typed as ``list[Callable]`` by upstream
            # stubs but accepts handler classes by convention. Cast through
            # ``Any`` at the boundary to satisfy strict type checkers.
            callbacks: Any = getattr(litellm, "callbacks", None)
            if callbacks is None:
                callbacks = []
                litellm.callbacks = callbacks
            callbacks.append(self._callback)
            version = getattr(litellm, "__version__", None)
            self._framework_version = str(version) if version is not None else None
            self._connected = True
            self._status = AdapterStatus.HEALTHY
        except ImportError:
            logger.warning("LiteLLM not installed; adapter in degraded mode")
            self._connected = True
            self._status = AdapterStatus.DEGRADED

    def disconnect(self) -> None:
        """Remove the LayerLens callback from LiteLLM."""
        if self._callback:
            try:
                import litellm  # type: ignore[import-not-found,unused-ignore]

                callbacks: Any = getattr(litellm, "callbacks", None)
                if callbacks is not None and self._callback in callbacks:
                    callbacks.remove(self._callback)
            except ImportError:
                pass
            self._callback = None
        self._connected = False
        self._status = AdapterStatus.DISCONNECTED

    def connect_client(self, client: Any) -> Any:
        """LiteLLM uses callbacks, not client wrapping — no-op."""
        return client

    @staticmethod
    def _detect_framework_version() -> Optional[str]:
        try:
            import litellm  # type: ignore[import-not-found,unused-ignore]

            version = getattr(litellm, "__version__", None)
            return str(version) if version is not None else None
        except ImportError:
            return None


# Backward-compat alias for users coming from ateam.
STRATIXLiteLLMCallback = LayerLensLiteLLMCallback


# Registry lazy-loading convention.
ADAPTER_CLASS = LiteLLMAdapter
