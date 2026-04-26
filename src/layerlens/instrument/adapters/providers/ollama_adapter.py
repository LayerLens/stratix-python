"""Ollama LLM Provider Adapter.

Wraps the Ollama Python SDK to intercept ``chat``, ``generate``, and
``embeddings`` calls. All API costs are $0.00 (local). Optional infra
cost tracking via compute duration when ``cost_per_second`` is set.

Ported from ``ateam/stratix/sdk/python/adapters/llm_providers/ollama_adapter.py``.
"""

from __future__ import annotations

import os
import time
import logging
from typing import Any, Dict, Optional

from layerlens.instrument.adapters._base.adapter import AdapterStatus
from layerlens.instrument.adapters._base.capture import CaptureConfig
from layerlens.instrument.adapters.providers._base.tokens import NormalizedTokenUsage
from layerlens.instrument.adapters.providers._base.provider import LLMProviderAdapter

logger = logging.getLogger(__name__)


class OllamaAdapter(LLMProviderAdapter):
    """LayerLens adapter for the Ollama Python SDK.

    Wraps ``ollama.chat()``, ``ollama.generate()``, and
    ``ollama.embeddings()`` calls. API cost is always $0.00 (local
    inference). Optionally tracks infra cost from compute duration if
    ``cost_per_second`` is configured.
    """

    FRAMEWORK = "ollama"
    VERSION = "0.1.0"

    def __init__(
        self,
        stratix: Any = None,
        capture_config: Optional[CaptureConfig] = None,
        cost_per_second: Optional[float] = None,
    ) -> None:
        super().__init__(stratix=stratix, capture_config=capture_config)
        self._cost_per_second = cost_per_second
        self._endpoint: Optional[str] = None

    def connect(self) -> None:
        """Detect Ollama endpoint and mark as connected."""
        self._endpoint = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self._framework_version = self._detect_framework_version()
        self._connected = True
        self._status = AdapterStatus.HEALTHY

    def connect_client(self, client: Any) -> Any:
        """Wrap Ollama client / module methods with tracing."""
        self._client = client

        if hasattr(client, "chat"):
            original_chat = client.chat
            self._originals["chat"] = original_chat
            client.chat = self._wrap_call(original_chat, "chat")

        if hasattr(client, "generate"):
            original_gen = client.generate
            self._originals["generate"] = original_gen
            client.generate = self._wrap_call(original_gen, "generate")

        if hasattr(client, "embeddings"):
            original_embed = client.embeddings
            self._originals["embeddings"] = original_embed
            client.embeddings = self._wrap_call(original_embed, "embeddings")

        return client

    def _restore_originals(self) -> None:
        if self._client is None:
            return
        for method_name, original in self._originals.items():
            try:
                setattr(self._client, method_name, original)
            except Exception:
                logger.warning("Could not restore %s", method_name)

    @staticmethod
    def _detect_framework_version() -> Optional[str]:
        try:
            import ollama  # type: ignore[import-not-found,unused-ignore]

            version = getattr(ollama, "__version__", None)
            return str(version) if version is not None else None
        except ImportError:
            return None

    def _wrap_call(self, original: Any, method_name: str) -> Any:
        adapter = self

        def traced_call(*args: Any, **kwargs: Any) -> Any:
            model = kwargs.get("model") or (args[0] if args else None)
            start_ns = time.time_ns()

            input_messages = None
            if method_name == "chat":
                input_messages = adapter._normalize_messages(kwargs.get("messages"))
            elif method_name == "generate":
                prompt = kwargs.get("prompt")
                if prompt:
                    input_messages = [
                        {"role": "user", "content": str(prompt)[:10_000]}
                    ]

            try:
                response = original(*args, **kwargs)
            except Exception as exc:
                elapsed_ms = (time.time_ns() - start_ns) / 1_000_000
                try:
                    adapter._emit_model_invoke(
                        provider="ollama",
                        model=model,
                        latency_ms=elapsed_ms,
                        error=str(exc),
                        metadata={
                            "method": method_name,
                            "endpoint": adapter._endpoint,
                        },
                        input_messages=input_messages,
                    )
                    adapter._emit_provider_error("ollama", str(exc), model=model)
                except Exception:
                    logger.warning("Error emitting Ollama error event", exc_info=True)
                raise

            try:
                elapsed_ms = (time.time_ns() - start_ns) / 1_000_000
                usage = adapter._extract_usage(response)
                infra_cost = adapter._calculate_infra_cost(response)
                output_message = adapter._extract_output_message(response, method_name)

                ollama_metadata: Dict[str, Any] = {
                    "method": method_name,
                    "endpoint": adapter._endpoint,
                }
                if isinstance(response, dict):
                    done_reason = response.get("done_reason")
                else:
                    done_reason = getattr(response, "done_reason", None)
                if done_reason is not None:
                    ollama_metadata["finish_reason"] = done_reason

                adapter._emit_model_invoke(
                    provider="ollama",
                    model=model,
                    usage=usage,
                    latency_ms=elapsed_ms,
                    metadata=ollama_metadata,
                    input_messages=input_messages,
                    output_message=output_message,
                )

                cost_meta: Dict[str, Any] = {"api_cost_usd": 0.0}
                if infra_cost is not None:
                    cost_meta["infra_cost_usd"] = infra_cost

                adapter.emit_dict_event(
                    "cost.record",
                    {
                        "provider": "ollama",
                        "model": model,
                        "prompt_tokens": usage.prompt_tokens if usage else 0,
                        "completion_tokens": usage.completion_tokens if usage else 0,
                        "total_tokens": usage.total_tokens if usage else 0,
                        **cost_meta,
                    },
                )
            except Exception:
                logger.warning("Error emitting Ollama trace events", exc_info=True)

            return response

        traced_call._layerlens_original = original  # type: ignore[attr-defined]
        return traced_call

    @staticmethod
    def _extract_usage(response: Any) -> Optional[NormalizedTokenUsage]:
        """Extract token usage from an Ollama response."""
        if response is None:
            return None
        if isinstance(response, dict):
            prompt = response.get("prompt_eval_count", 0) or 0
            completion = response.get("eval_count", 0) or 0
            return NormalizedTokenUsage(
                prompt_tokens=prompt,
                completion_tokens=completion,
                total_tokens=prompt + completion,
            )
        prompt = getattr(response, "prompt_eval_count", 0) or 0
        completion = getattr(response, "eval_count", 0) or 0
        return NormalizedTokenUsage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
        )

    @staticmethod
    def _extract_output_message(
        response: Any, method_name: str
    ) -> Optional[Dict[str, str]]:
        """Extract the output message from an Ollama response."""
        try:
            if response is None:
                return None
            if method_name == "chat":
                msg = (
                    response.get("message", {})
                    if isinstance(response, dict)
                    else getattr(response, "message", None)
                )
                if msg:
                    content = (
                        msg.get("content", "")
                        if isinstance(msg, dict)
                        else getattr(msg, "content", "")
                    )
                    if content:
                        return {"role": "assistant", "content": str(content)[:10_000]}
            elif method_name == "generate":
                text = (
                    response.get("response", "")
                    if isinstance(response, dict)
                    else getattr(response, "response", "")
                )
                if text:
                    return {"role": "assistant", "content": str(text)[:10_000]}
        except Exception:
            pass
        return None

    def _calculate_infra_cost(self, response: Any) -> Optional[float]:
        """Calculate optional infrastructure cost from compute duration."""
        if self._cost_per_second is None:
            return None
        if response is None:
            return None

        total_ns = 0
        if isinstance(response, dict):
            total_ns = (response.get("eval_duration", 0) or 0) + (
                response.get("prompt_eval_duration", 0) or 0
            )
        else:
            total_ns = (getattr(response, "eval_duration", 0) or 0) + (
                getattr(response, "prompt_eval_duration", 0) or 0
            )

        if total_ns > 0:
            total_seconds = total_ns / 1_000_000_000
            return round(total_seconds * self._cost_per_second, 8)
        return None


# Registry lazy-loading convention.
ADAPTER_CLASS = OllamaAdapter
