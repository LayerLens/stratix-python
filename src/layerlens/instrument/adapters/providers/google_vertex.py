from __future__ import annotations

import logging
from typing import Any, Dict

from ._base_provider import MonkeyPatchProvider

log = logging.getLogger(__name__)

_CAPTURE_PARAMS = frozenset(
    {"temperature", "max_output_tokens", "top_p", "top_k", "stream", "generation_config", "tools"}
)


class GoogleVertexProvider(MonkeyPatchProvider):
    """Adapter for google-generativeai / google-cloud-aiplatform GenerativeModel."""

    name = "google_vertex"
    capture_params = _CAPTURE_PARAMS

    @staticmethod
    def extract_output(response: Any) -> Any:
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return None
        content = getattr(candidates[0], "content", None)
        parts = getattr(content, "parts", None) or []
        texts: list[str] = []
        for part in parts:
            text = getattr(part, "text", None)
            if text:
                texts.append(str(text))
        if not texts:
            return None
        return {"role": "model", "content": "\n".join(texts)}

    @staticmethod
    def extract_meta(response: Any) -> Dict[str, Any]:
        meta: Dict[str, Any] = {}
        metadata = getattr(response, "usage_metadata", None)
        if metadata is not None:
            prompt = int(getattr(metadata, "prompt_token_count", 0) or 0)
            completion = int(getattr(metadata, "candidates_token_count", 0) or 0)
            total = int(getattr(metadata, "total_token_count", 0) or (prompt + completion))
            reasoning = getattr(metadata, "thoughts_token_count", None)
            payload: Dict[str, Any] = {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": total,
            }
            if reasoning is not None:
                payload["reasoning_tokens"] = int(reasoning)
            meta["usage"] = payload
        candidates = getattr(response, "candidates", None) or []
        if candidates:
            fr = getattr(candidates[0], "finish_reason", None)
            if fr is not None:
                meta["finish_reason"] = getattr(fr, "name", None) or str(fr)
        return meta

    @staticmethod
    def extract_tool_calls(response: Any) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return out
        content = getattr(candidates[0], "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            fn = getattr(part, "function_call", None)
            if fn is None:
                continue
            out.append(
                {
                    "tool_name": getattr(fn, "name", "unknown"),
                    "arguments": dict(getattr(fn, "args", {}) or {}),
                }
            )
        return out

    @staticmethod
    def aggregate_stream(chunks: list[Any]) -> Any:
        return _AggregatedVertexResponse(chunks) if chunks else None

    def connect(self, target: Any = None, **kwargs: Any) -> Any:  # noqa: ARG002
        self._client = target
        if hasattr(target, "generate_content"):
            orig = target.generate_content
            self._originals["generate_content"] = orig
            target.generate_content = self._wrap_sync("google_vertex.generate_content", orig)
        if hasattr(target, "generate_content_async"):
            async_orig = target.generate_content_async
            self._originals["generate_content_async"] = async_orig
            target.generate_content_async = self._wrap_async("google_vertex.generate_content", async_orig)
        return target


class _AggregatedVertexResponse:
    """Shim that looks like a Vertex response, assembled from streamed chunks."""

    def __init__(self, chunks: list[Any]):
        parts_text: list[str] = []
        tool_calls: list[Any] = []
        usage = None
        finish_reason = None
        for chunk in chunks:
            um = getattr(chunk, "usage_metadata", None)
            if um is not None:
                usage = um
            cands = getattr(chunk, "candidates", None) or []
            if cands:
                fr = getattr(cands[0], "finish_reason", None)
                if fr is not None:
                    finish_reason = fr
                content = getattr(cands[0], "content", None)
                for part in getattr(content, "parts", None) or []:
                    text = getattr(part, "text", None)
                    if text:
                        parts_text.append(text)
                    fn_call = getattr(part, "function_call", None)
                    if fn_call is not None:
                        tool_calls.append(fn_call)

        self.usage_metadata = usage
        self.candidates = [
            _AggregatedCandidate(
                "\n".join(parts_text),
                tool_calls=tool_calls,
                finish_reason=finish_reason,
            )
        ]


class _AggregatedCandidate:
    def __init__(self, text: str, *, tool_calls: list[Any], finish_reason: Any):
        self.content = _AggregatedContent(text, tool_calls)
        self.finish_reason = finish_reason


class _AggregatedContent:
    def __init__(self, text: str, tool_calls: list[Any]):
        parts: list[Any] = []
        if text:
            parts.append(_AggregatedPart(text=text))
        for tc in tool_calls:
            parts.append(_AggregatedPart(function_call=tc))
        self.parts = parts


class _AggregatedPart:
    def __init__(self, *, text: str | None = None, function_call: Any = None):
        self.text = text
        self.function_call = function_call


def instrument_google_vertex(model: Any) -> GoogleVertexProvider:
    from .._registry import get, register

    existing = get("google_vertex")
    if existing is not None:
        existing.disconnect()
    provider = GoogleVertexProvider()
    provider.connect(model)
    register("google_vertex", provider)
    return provider


def uninstrument_google_vertex() -> None:
    from .._registry import unregister

    unregister("google_vertex")
