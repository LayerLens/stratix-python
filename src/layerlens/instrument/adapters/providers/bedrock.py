"""AWS Bedrock LLM provider adapter.

Wraps ``invoke_model``, ``converse``, and their streaming variants.
The ``modelId`` prefix (``anthropic.*``, ``meta.*``, ``cohere.*``, ``amazon.*``,
``ai21.*``, ``mistral.*``) selects the family-specific token/output parser.

Non-streaming responses are fully parsed. Streaming variants emit a
``streaming=True`` model.invoke; fine-grained stream aggregation is handled
by the caller because ``botocore.response.StreamingBody`` is single-read and
we don't want to buffer-swap the user's response.
"""

from __future__ import annotations

import io
import json
import time
import logging
from typing import Any, Dict

from .._base import AdapterInfo, BaseAdapter
from .pricing import BEDROCK_PRICING
from ..._events import AGENT_ERROR, MODEL_INVOKE
from ..._context import _current_span_id, _current_collector
from .token_usage import NormalizedTokenUsage
from ._emit_helpers import _emit_cost  # type: ignore[attr-defined]

log = logging.getLogger(__name__)

_CAPTURE_PARAMS = frozenset({"modelId", "accept", "contentType", "inferenceConfig"})


def _family(model_id: str) -> str:
    lower = (model_id or "").lower()
    for prefix in ("anthropic", "meta", "cohere", "amazon", "ai21", "mistral"):
        if lower.startswith(prefix + "."):
            return prefix
    return "unknown"


class BedrockProvider(BaseAdapter):
    """Monkey-patches ``boto3`` bedrock-runtime client methods."""

    name = "aws_bedrock"

    def __init__(self) -> None:
        self._client: Any = None
        self._originals: Dict[str, Any] = {}

    def adapter_info(self) -> AdapterInfo:
        return AdapterInfo(name=self.name, adapter_type="provider", connected=self._client is not None)

    def connect(self, target: Any = None, **kwargs: Any) -> Any:  # noqa: ARG002
        self._client = target
        if hasattr(target, "invoke_model"):
            orig = target.invoke_model
            self._originals["invoke_model"] = orig
            target.invoke_model = self._wrap_invoke_model(orig)
        if hasattr(target, "converse"):
            orig = target.converse
            self._originals["converse"] = orig
            target.converse = self._wrap_converse(orig)
        if hasattr(target, "invoke_model_with_response_stream"):
            orig = target.invoke_model_with_response_stream
            self._originals["invoke_model_with_response_stream"] = orig
            target.invoke_model_with_response_stream = self._wrap_stream(orig, "invoke_model_with_response_stream")
        if hasattr(target, "converse_stream"):
            orig = target.converse_stream
            self._originals["converse_stream"] = orig
            target.converse_stream = self._wrap_stream(orig, "converse_stream")
        return target

    def disconnect(self) -> None:
        if self._client is None:
            return
        for attr, orig in self._originals.items():
            try:
                setattr(self._client, attr, orig)
            except Exception:
                log.warning("Could not restore %s", attr)
        self._client = None
        self._originals.clear()

    # --- invoke_model ---

    def _wrap_invoke_model(self, original: Any) -> Any:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            if _current_collector.get() is None:
                return original(*args, **kwargs)
            model_id = kwargs.get("modelId", "")
            family = _family(model_id)
            start = time.time()
            input_messages = _extract_invoke_messages(kwargs, family)
            try:
                response = original(*args, **kwargs)
            except Exception as exc:
                _emit_error("aws_bedrock.invoke_model", exc, (time.time() - start) * 1000)
                raise
            latency_ms = (time.time() - start) * 1000

            # Body is a single-read StreamingBody — re-materialize so the caller can still read it.
            body_obj = response.get("body") if isinstance(response, dict) else None
            body_bytes = b""
            if body_obj is not None and hasattr(body_obj, "read"):
                body_bytes = body_obj.read()
                response["body"] = _RereadableBody(body_bytes)

            try:
                body_data = json.loads(body_bytes) if body_bytes else {}
            except (ValueError, TypeError):
                body_data = {}

            output = _extract_invoke_output(body_data, family)
            usage = _extract_invoke_usage(body_data, family)
            _emit_invoke(
                event="aws_bedrock.invoke_model",
                model_id=model_id,
                latency_ms=latency_ms,
                kwargs=kwargs,
                messages=input_messages,
                output=output,
                usage=usage,
                extra={"family": family},
            )
            return response

        return wrapped

    # --- converse ---

    def _wrap_converse(self, original: Any) -> Any:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            if _current_collector.get() is None:
                return original(*args, **kwargs)
            model_id = kwargs.get("modelId", "")
            start = time.time()
            input_messages = _normalize_converse_messages(kwargs.get("messages"))
            try:
                response = original(*args, **kwargs)
            except Exception as exc:
                _emit_error("aws_bedrock.converse", exc, (time.time() - start) * 1000)
                raise
            latency_ms = (time.time() - start) * 1000

            output = _extract_converse_output(response)
            usage = _extract_converse_usage(response)
            metadata_extra: Dict[str, Any] = {}
            stop_reason = response.get("stopReason") if isinstance(response, dict) else None
            if stop_reason:
                metadata_extra["stop_reason"] = stop_reason
            _emit_invoke(
                event="aws_bedrock.converse",
                model_id=model_id,
                latency_ms=latency_ms,
                kwargs=kwargs,
                messages=input_messages,
                output=output,
                usage=usage,
                extra=metadata_extra,
            )
            return response

        return wrapped

    # --- streaming ---

    def _wrap_stream(self, original: Any, method: str) -> Any:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            if _current_collector.get() is None:
                return original(*args, **kwargs)
            model_id = kwargs.get("modelId", "")
            start = time.time()
            try:
                response = original(*args, **kwargs)
            except Exception as exc:
                _emit_error(f"aws_bedrock.{method}", exc, (time.time() - start) * 1000)
                raise
            latency_ms = (time.time() - start) * 1000
            _emit_invoke(
                event=f"aws_bedrock.{method}",
                model_id=model_id,
                latency_ms=latency_ms,
                kwargs=kwargs,
                messages=None,
                output=None,
                usage=None,
                extra={"streaming": True, "method": method},
            )
            return response

        return wrapped


class _RereadableBody:
    """Minimal shim so downstream code can still call ``.read()`` on the body."""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._buf = io.BytesIO(data)

    def read(self, *args: Any, **kwargs: Any) -> bytes:
        return self._buf.read(*args, **kwargs)

    def close(self) -> None:
        self._buf.close()


def _extract_invoke_messages(kwargs: Dict[str, Any], family: str) -> list[dict[str, str]] | None:
    body = kwargs.get("body")
    if not body:
        return None
    try:
        if isinstance(body, (str, bytes, bytearray)):
            data = json.loads(body)
        elif isinstance(body, dict):
            data = body
        else:
            return None
    except (ValueError, TypeError):
        return None

    out: list[dict[str, str]] = []
    if family == "anthropic":
        system = data.get("system")
        if system:
            out.append({"role": "system", "content": str(system)})
        for msg in data.get("messages", []) or []:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content", "")
            if isinstance(content, list):
                content = "\n".join(str(p.get("text", "")) for p in content if isinstance(p, dict) and "text" in p)
            out.append({"role": str(msg.get("role", "user")), "content": str(content)})
    else:
        prompt = data.get("prompt") or data.get("inputText") or ""
        if prompt:
            out.append({"role": "user", "content": str(prompt)})
    return out or None


def _extract_invoke_output(data: Dict[str, Any], family: str) -> dict[str, str] | None:
    if not data:
        return None
    content = ""
    if family == "anthropic":
        parts = [
            str(block.get("text", ""))
            for block in data.get("content", []) or []
            if isinstance(block, dict) and "text" in block
        ]
        content = "\n".join(parts)
    elif family in ("meta", "mistral"):
        content = str(data.get("generation", ""))
    elif family == "cohere":
        generations = data.get("generations") or []
        if generations:
            content = str(generations[0].get("text", ""))
    elif family == "amazon":
        results = data.get("results") or []
        if results:
            content = str(results[0].get("outputText", ""))
    else:
        content = str(data.get("generation") or data.get("completion") or data.get("outputText") or "")
    return {"role": "assistant", "content": content} if content else None


def _extract_invoke_usage(data: Dict[str, Any], family: str) -> NormalizedTokenUsage | None:
    if not data:
        return None
    if family == "anthropic":
        usage = data.get("usage") or {}
        return NormalizedTokenUsage(
            prompt_tokens=int(usage.get("input_tokens") or 0),
            completion_tokens=int(usage.get("output_tokens") or 0),
        )
    # Meta/Mistral/Amazon inline fields
    prompt = int(data.get("prompt_token_count") or data.get("inputTextTokenCount") or 0)
    completion = int(data.get("generation_token_count") or data.get("tokenCount") or 0)
    if prompt or completion:
        return NormalizedTokenUsage(prompt_tokens=prompt, completion_tokens=completion)
    return None


def _extract_converse_output(response: Dict[str, Any]) -> dict[str, str] | None:
    if not isinstance(response, dict):
        return None
    msg = (response.get("output") or {}).get("message") or {}
    parts = [str(b.get("text", "")) for b in msg.get("content", []) or [] if isinstance(b, dict) and "text" in b]
    if not parts:
        return None
    return {"role": str(msg.get("role", "assistant")), "content": "\n".join(parts)}


def _extract_converse_usage(response: Dict[str, Any]) -> NormalizedTokenUsage | None:
    if not isinstance(response, dict):
        return None
    u = response.get("usage") or {}
    if not u:
        return None
    return NormalizedTokenUsage(
        prompt_tokens=int(u.get("inputTokens") or 0),
        completion_tokens=int(u.get("outputTokens") or 0),
        total_tokens=int(u.get("totalTokens") or 0),
    )


def _normalize_converse_messages(messages: Any) -> list[dict[str, str]] | None:
    if not messages:
        return None
    out: list[dict[str, str]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "user"))
        content_blocks = msg.get("content") or []
        parts = [str(b.get("text", "")) for b in content_blocks if isinstance(b, dict) and "text" in b]
        out.append({"role": role, "content": "\n".join(parts)})
    return out or None


def _emit_invoke(
    *,
    event: str,
    model_id: str,
    latency_ms: float,
    kwargs: Dict[str, Any],
    messages: list[dict[str, str]] | None,
    output: dict[str, str] | None,
    usage: NormalizedTokenUsage | None,
    extra: Dict[str, Any],
) -> None:
    import uuid

    collector = _current_collector.get()
    if collector is None:
        return
    span_id = uuid.uuid4().hex[:16]
    parent_span_id = _current_span_id.get()
    payload: Dict[str, Any] = {
        "name": event,
        "model": model_id,
        "latency_ms": latency_ms,
        "parameters": {k: kwargs[k] for k in _CAPTURE_PARAMS if k in kwargs},
        "messages": messages,
        "output_message": output,
    }
    if usage is not None:
        payload["usage"] = usage.as_event_dict()
    payload.update(extra)
    collector.emit(MODEL_INVOKE, payload, span_id=span_id, parent_span_id=parent_span_id)

    if usage is not None:
        _emit_cost(
            collector,
            provider="aws_bedrock",
            model=model_id,
            usage=usage,
            pricing_table=BEDROCK_PRICING,
            span_id=span_id,
            parent_span_id=parent_span_id,
        )


def _emit_error(event: str, exc: Exception, latency_ms: float) -> None:
    import uuid

    collector = _current_collector.get()
    if collector is None:
        return
    collector.emit(
        AGENT_ERROR,
        {"name": event, "error": str(exc), "latency_ms": latency_ms},
        span_id=uuid.uuid4().hex[:16],
        parent_span_id=_current_span_id.get(),
    )


def instrument_bedrock(client: Any) -> BedrockProvider:
    from .._registry import get, register

    existing = get("aws_bedrock")
    if existing is not None:
        existing.disconnect()
    provider = BedrockProvider()
    provider.connect(client)
    register("aws_bedrock", provider)
    return provider


def uninstrument_bedrock() -> None:
    from .._registry import unregister

    unregister("aws_bedrock")
