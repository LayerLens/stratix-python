"""Ollama local LLM provider adapter.

Wraps ``chat``, ``generate``, ``embeddings``. Ollama calls never incur API
cost; an optional ``cost_per_second`` lets callers account for compute time.
"""

from __future__ import annotations

import os
from typing import Any, Dict

from ._base_provider import MonkeyPatchProvider

_NS_PER_SECOND = 1_000_000_000

_CAPTURE_PARAMS = frozenset(
    {
        "model",
        "stream",
        "options",
        "format",
        "template",
        "keep_alive",
    }
)
# ``messages`` and ``prompt`` are intentionally NOT in _CAPTURE_PARAMS: their
# raw values are prompt content, which must stay out of ``parameters`` so the
# ``capture_content=False`` redaction holds (LAY-3567 B1). Content is captured
# at the payload top level (``messages``/``output_message``), where
# CaptureConfig.redact_payload manages it.


def _as_dict(response: Any) -> Any:
    """Coerce an ollama response object to a dict for parsing.

    Older ollama-python returned plain dicts; modern versions return pydantic
    ``ChatResponse``/``GenerateResponse``/``EmbedResponse`` objects. Both expose
    ``model_dump``; anything else is returned unchanged so non-coercible inputs
    fall through to the dict guards untouched.
    """
    if isinstance(response, dict):
        return response
    dump = getattr(response, "model_dump", None)
    if callable(dump):
        try:
            return dump()
        except Exception:  # pragma: no cover - defensive; never break a trace
            return response
    return response


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
        # Modern ollama-python returns a ``ChatResponse``/``GenerateResponse``
        # pydantic object, not the plain dict the older client returned; coerce
        # it so the real response shape is parsed (LAY-3614).
        response = _as_dict(response)
        # ``chat`` returns {"message": {"role", "content"}, ...}
        if isinstance(response, dict):
            msg = response.get("message")
            if isinstance(msg, dict):
                return {
                    "role": msg.get("role", "assistant"),
                    "content": msg.get("content", ""),
                }
            # ``generate`` returns {"response": "..."}
            if "response" in response:
                return {"role": "assistant", "content": response.get("response", "")}
            # Legacy ``embeddings()`` returns {"embedding": [...]}.
            if "embedding" in response:
                return {
                    "type": "embedding",
                    "dim": len(response.get("embedding") or []),
                }
            # Modern ``embed()`` returns an ``EmbedResponse`` whose dump carries
            # the PLURAL ``embeddings`` key: a list of vectors (one per input).
            # Report the dimensionality of the first vector (shape-only, never
            # the raw floats) so the wrapped ``embed`` method is not output-less.
            if "embeddings" in response:
                vectors = response.get("embeddings") or []
                first = vectors[0] if vectors else []
                return {
                    "type": "embedding",
                    "dim": len(first),
                }
        return None

    @staticmethod
    def extract_meta(response: Any) -> Dict[str, Any]:
        response = _as_dict(response)
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

    @staticmethod
    def aggregate_stream(chunks: list[Any]) -> Any:
        """Merge an ollama stream into one response dict for the model.invoke.

        Ollama's ``chat(stream=True)`` yields a ``ChatResponse`` per token (final
        one carries ``done``/usage); ``generate(stream=True)`` yields ``response``
        deltas. Without this override the base hook returns ``None`` and a stream
        produced a content-less, usage-less model.invoke (the G8 gap). We
        concatenate the deltas and keep the final chunk's metadata (model /
        done_reason / *_eval_count / total_duration) so the existing
        extract_output / extract_meta parse the aggregate unchanged.
        """
        dicts = [d for d in (_as_dict(c) for c in chunks) if isinstance(d, dict)]
        if not dicts:
            return None
        parts: list[str] = []
        has_message = False
        for d in dicts:
            msg = d.get("message")
            if isinstance(msg, dict):
                has_message = True
                piece = msg.get("content")
                if piece:
                    parts.append(piece)
            elif d.get("response"):
                parts.append(d["response"])
        aggregated = dict(dicts[-1])  # final chunk carries the done-metadata + usage
        merged = "".join(parts)
        if has_message:
            aggregated["message"] = {"role": "assistant", "content": merged}
        else:
            aggregated["response"] = merged
        return aggregated

    def connect(self, target: Any = None, **kwargs: Any) -> Any:  # noqa: ARG002
        self._client = target
        for method in ("chat", "generate", "embeddings", "embed"):
            if hasattr(target, method):
                orig = getattr(target, method)
                self._originals[method] = orig
                setattr(target, method, self._wrap_auto(f"ollama.{method}", orig))
        return target

    def _extractors(self) -> "MonkeyPatchProvider._Extractors":  # type: ignore[override]
        # Bind endpoint + (optional) infra-cost calc into meta. Ollama is
        # local-only so API cost is always $0, but `cost_per_second` lets
        # callers attribute compute time as an infra cost on each invoke.
        endpoint = self._endpoint
        cost_per_second = self._cost_per_second
        base_meta = type(self).extract_meta

        def meta_with_extras(response: Any) -> Dict[str, Any]:
            meta = base_meta(response)
            if endpoint:
                meta["endpoint"] = endpoint
            if cost_per_second is not None:
                # Coerce first: modern ollama returns a pydantic ChatResponse/
                # GenerateResponse OBJECT (not the old plain dict), so the
                # eval_duration fields must be read off the coerced dump or the
                # infra cost is silently dropped on every real invoke (LAY-3614).
                coerced = _as_dict(response)
                if isinstance(coerced, dict):
                    total_ns = int(coerced.get("eval_duration") or 0) + int(
                        coerced.get("prompt_eval_duration") or 0
                    )
                    if total_ns > 0:
                        meta["infra_cost_usd"] = round(
                            (total_ns / _NS_PER_SECOND) * cost_per_second, 8
                        )
            return meta

        return MonkeyPatchProvider._Extractors(
            output=type(self).extract_output,
            meta=meta_with_extras,
            tool_calls=type(self).extract_tool_calls,
        )


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
