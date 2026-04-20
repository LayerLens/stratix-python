"""Ollama local LLM provider adapter.

Wraps ``chat``, ``generate``, ``embeddings``. Ollama calls never incur API
cost; an optional ``cost_per_second`` lets callers account for compute time.
"""

from __future__ import annotations

import os
from typing import Any, Dict

from ._base_provider import MonkeyPatchProvider

_CAPTURE_PARAMS = frozenset({"model", "messages", "prompt", "stream", "options", "format", "template", "keep_alive"})


class OllamaProvider(MonkeyPatchProvider):
    name = "ollama"
    capture_params = _CAPTURE_PARAMS
    #: Ollama has no public pricing table; set an override for compute-based billing.
    pricing_table: dict[str, dict[str, float]] | None = None

    def __init__(self, cost_per_second: float | None = None) -> None:
        super().__init__()
        self._cost_per_second = cost_per_second
        self._endpoint = os.environ.get("OLLAMA_HOST")

    @staticmethod
    def extract_output(response: Any) -> Any:
        # ``chat`` returns {"message": {"role", "content"}, ...}
        if isinstance(response, dict):
            msg = response.get("message")
            if isinstance(msg, dict):
                return {"role": msg.get("role", "assistant"), "content": msg.get("content", "")}
            # ``generate`` returns {"response": "..."}
            if "response" in response:
                return {"role": "assistant", "content": response.get("response", "")}
            # ``embeddings`` returns {"embedding": [...]}
            if "embedding" in response:
                return {"type": "embedding", "dim": len(response.get("embedding") or [])}
        return None

    @staticmethod
    def extract_meta(response: Any) -> Dict[str, Any]:
        if not isinstance(response, dict):
            return {}
        meta: Dict[str, Any] = {}
        model = response.get("model")
        if model:
            meta["response_model"] = model
        done_reason = response.get("done_reason")
        if done_reason:
            meta["finish_reason"] = done_reason

        prompt = int(response.get("prompt_eval_count") or 0)
        completion = int(response.get("eval_count") or 0)
        if prompt or completion:
            meta["usage"] = {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": prompt + completion,
            }

        total_ns = response.get("total_duration")
        if total_ns:
            meta["duration_ms"] = total_ns / 1_000_000
        return meta

    def connect(self, target: Any = None, **kwargs: Any) -> Any:  # noqa: ARG002
        self._client = target
        for method in ("chat", "generate", "embeddings", "embed"):
            if hasattr(target, method):
                orig = getattr(target, method)
                self._originals[method] = orig
                setattr(target, method, self._wrap_sync(f"ollama.{method}", orig))
        return target


def instrument_ollama(client: Any, *, cost_per_second: float | None = None) -> OllamaProvider:
    from .._registry import get, register

    existing = get("ollama")
    if existing is not None:
        existing.disconnect()
    provider = OllamaProvider(cost_per_second=cost_per_second)
    provider.connect(client)
    register("ollama", provider)
    return provider


def uninstrument_ollama() -> None:
    from .._registry import unregister

    unregister("ollama")
