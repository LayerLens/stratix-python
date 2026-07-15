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

from ..._w3c import gen_ai_attributes
from .._base import AdapterInfo, BaseAdapter
from .pricing import BEDROCK_PRICING
from ..._events import TOOL_CALL, AGENT_ERROR, MODEL_INVOKE
from ..._context import _current_span_id, _current_collector
from .token_usage import NormalizedTokenUsage
from ._emit_helpers import _emit_cost, _flat_token_fields  # type: ignore[attr-defined]
from ..._secret_scrub import safe_error

log = logging.getLogger(__name__)

_CAPTURE_PARAMS = frozenset({"modelId", "accept", "contentType", "inferenceConfig"})


def _family(model_id: str) -> str:
    lower = (model_id or "").lower()
    # Unwrap a Bedrock inference-profile ARN / cross-region prefix so e.g.
    # "us.amazon.nova-lite-v1:0" classifies as the amazon family (LAY-3605).
    if lower.startswith("arn:"):
        lower = lower.rsplit("/", 1)[-1]
    for region in ("us-gov.", "us.", "eu.", "apac."):
        if lower.startswith(region):
            lower = lower[len(region) :]
            break
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
            usage = _extract_invoke_usage(body_data, family, _response_http_headers(response))
            extra: Dict[str, Any] = {"family": family}
            response_id = _bedrock_response_id(response)
            if response_id:
                extra["response_id"] = response_id
            # Family-specific stop_reason from the parsed body.
            stop_reason = _extract_invoke_stop_reason(body_data, family)
            if stop_reason:
                extra["stop_reason"] = stop_reason
            _emit_invoke(
                event="aws_bedrock.invoke_model",
                model_id=model_id,
                latency_ms=latency_ms,
                kwargs=kwargs,
                messages=input_messages,
                output=output,
                usage=usage,
                extra=extra,
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
            tool_calls = _extract_converse_tool_calls(response)
            metadata_extra: Dict[str, Any] = {}
            stop_reason = response.get("stopReason") if isinstance(response, dict) else None
            if stop_reason:
                metadata_extra["stop_reason"] = stop_reason
            response_id = _bedrock_response_id(response)
            if response_id:
                metadata_extra["response_id"] = response_id
            _emit_invoke(
                event="aws_bedrock.converse",
                model_id=model_id,
                latency_ms=latency_ms,
                kwargs=kwargs,
                messages=input_messages,
                output=output,
                usage=usage,
                extra=metadata_extra,
                tool_calls=tool_calls,
            )
            return response

        return wrapped

    # --- streaming ---

    def _wrap_stream(self, original: Any, method: str) -> Any:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            if _current_collector.get() is None:
                return original(*args, **kwargs)
            model_id = kwargs.get("modelId", "")
            family = _family(model_id)
            start = time.time()
            try:
                response = original(*args, **kwargs)
            except Exception as exc:
                _emit_error(f"aws_bedrock.{method}", exc, (time.time() - start) * 1000)
                raise

            # ``invoke_model_with_response_stream`` returns the EventStream at
            # ``body``; ``converse_stream`` at ``stream``. Both are single-read.
            stream_key = "body" if method == "invoke_model_with_response_stream" else "stream"
            source = response.get(stream_key) if isinstance(response, dict) else None
            if source is None or not hasattr(source, "__iter__"):
                # Degenerate/unexpected shape — record the streaming call as a
                # bare marker so it isn't silently dropped, then return untouched.
                _emit_invoke(
                    event=f"aws_bedrock.{method}",
                    model_id=model_id,
                    latency_ms=(time.time() - start) * 1000,
                    kwargs=kwargs,
                    messages=None,
                    output=None,
                    usage=None,
                    extra={"streaming": True, "method": method, "family": family},
                )
                return response

            if method == "converse_stream":
                input_messages = _normalize_converse_messages(kwargs.get("messages"))
            else:
                input_messages = _extract_invoke_messages(kwargs, family)
            response_id = _bedrock_response_id(response)

            def on_complete(chunks: list[Any], first_at: float | None, started: float) -> None:
                latency_ms = (time.time() - started) * 1000
                ttft_ms = (first_at - started) * 1000 if first_at is not None else None
                tool_calls: list[dict[str, Any]] | None = None
                if method == "converse_stream":
                    output, usage, stop_reason, tool_calls = _aggregate_converse_stream(chunks)
                else:
                    output, usage, stop_reason = _aggregate_invoke_stream(chunks, family)
                extra: Dict[str, Any] = {"streaming": True, "method": method, "family": family}
                if stop_reason:
                    extra["stop_reason"] = stop_reason
                if response_id:
                    extra["response_id"] = response_id
                _emit_invoke(
                    event=f"aws_bedrock.{method}",
                    model_id=model_id,
                    latency_ms=latency_ms,
                    kwargs=kwargs,
                    messages=input_messages,
                    output=output,
                    usage=usage,
                    extra=extra,
                    tool_calls=tool_calls,
                    ttft_ms=ttft_ms,
                    streaming_duration_ms=latency_ms,
                )

            def on_error(exc: Exception, _chunks: list[Any], started: float) -> None:
                _emit_error(f"aws_bedrock.{method}", exc, (time.time() - started) * 1000)

            response[stream_key] = _StreamTee(source, start, on_complete, on_error)
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


class _StreamTee:
    """Re-iterable proxy over a Bedrock ``EventStream``.

    Tees every chunk back to the caller (single-read semantics preserved — the
    caller drains the real chunks itself) while accumulating them so the
    adapter can aggregate output text + usage + TTFT. On natural exhaustion the
    ``on_complete`` hook fires (aggregate + emit the streaming ``model.invoke``);
    on a mid-stream failure the ``on_error`` hook fires (emit ``agent.error``).

    Emission is best-effort telemetry: a fault in the hooks is logged, never
    propagated, so instrumentation can't break the user's own iteration.
    """

    def __init__(
        self,
        source: Any,
        start: float,
        on_complete: Any,
        on_error: Any,
    ) -> None:
        self._it = iter(source)
        self._start = start
        self._on_complete = on_complete
        self._on_error = on_error
        self._chunks: list[Any] = []
        self._first_at: float | None = None
        self._closed = False

    def __iter__(self) -> "_StreamTee":
        return self

    def __next__(self) -> Any:
        try:
            chunk = next(self._it)
        except StopIteration:
            self._finish()
            raise
        except Exception as exc:
            if not self._closed:
                self._closed = True
                try:
                    self._on_error(exc, self._chunks, self._start)
                except Exception:  # noqa: BLE001 — telemetry must not break iteration
                    log.warning("bedrock stream on_error hook failed", exc_info=True)
            raise
        if self._first_at is None:
            self._first_at = time.time()
        self._chunks.append(chunk)
        return chunk

    def _finish(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._on_complete(self._chunks, self._first_at, self._start)
        except Exception:  # noqa: BLE001 — telemetry must not break iteration
            log.warning("bedrock stream on_complete hook failed", exc_info=True)

    def close(self) -> None:
        # A caller that closes the stream early is treated as completion so a
        # short-read still records what was consumed.
        self._finish()
        inner_close = getattr(self._it, "close", None)
        if callable(inner_close):
            try:
                inner_close()
            except Exception:  # noqa: BLE001
                log.warning("bedrock stream inner close failed", exc_info=True)


def _decode_invoke_stream_chunk(chunk: Any) -> dict[str, Any] | None:
    """Decode one ``invoke_model_with_response_stream`` event to a JSON dict.

    The wire delivers ``{"chunk": {"bytes": <json-bytes>}}``; some shapes hand
    the parsed dict directly. Returns ``None`` for anything unparseable.
    """
    if not isinstance(chunk, dict):
        return None
    inner = chunk.get("chunk")
    raw: Any = None
    if isinstance(inner, dict):
        raw = inner.get("bytes")
    elif inner is not None and hasattr(inner, "get"):
        raw = inner.get("bytes")
    if raw is None:
        # Already-decoded event dict (no chunk wrapper).
        if any(k in chunk for k in ("type", "contentBlockDelta", "messageStop", "metadata")):
            return chunk
        return None
    if isinstance(raw, (bytes, bytearray, str)):
        try:
            decoded = json.loads(raw)
        except (ValueError, TypeError):
            return None
        return decoded if isinstance(decoded, dict) else None
    return None


def _accumulate_converse_event(
    event: dict[str, Any],
    text_parts: list[str],
    totals: dict[str, int],
    tools: dict[int, dict[str, Any]] | None = None,
) -> str | None:
    """Fold one Converse-shaped stream event into the running aggregate.

    Returns a stop reason when the event carries one, else ``None``. Shared by
    both ``converse_stream`` and Nova ``invoke_model_with_response_stream``
    (whose chunk bytes are Converse-shaped events).

    When ``tools`` is provided, streamed tool-use is captured into it keyed by
    ``contentBlockIndex``: a ``contentBlockStart`` carries ``start.toolUse``'s
    name + id, and subsequent ``contentBlockDelta`` events carry partial JSON
    fragments at ``delta.toolUse.input`` that concatenate into the tool input.
    """
    stop_reason: str | None = None
    if "contentBlockStart" in event and tools is not None:
        cbs = event.get("contentBlockStart") or {}
        start = cbs.get("start") or {}
        tool_use = start.get("toolUse") if isinstance(start, dict) else None
        if isinstance(tool_use, dict):
            idx = cbs.get("contentBlockIndex", 0)
            entry = tools.setdefault(idx, {"tool_name": None, "tool_use_id": None, "input_parts": []})
            entry["tool_name"] = tool_use.get("name")
            entry["tool_use_id"] = tool_use.get("toolUseId")
    if "contentBlockDelta" in event:
        cbd = event.get("contentBlockDelta") or {}
        delta = cbd.get("delta") or {}
        if isinstance(delta, dict):
            if "text" in delta:
                text_parts.append(str(delta.get("text") or ""))
            tool_use = delta.get("toolUse")
            if isinstance(tool_use, dict) and "input" in tool_use and tools is not None:
                idx = cbd.get("contentBlockIndex", 0)
                entry = tools.setdefault(idx, {"tool_name": None, "tool_use_id": None, "input_parts": []})
                entry["input_parts"].append(str(tool_use.get("input") or ""))
    if "messageStop" in event:
        sr = (event.get("messageStop") or {}).get("stopReason")
        if isinstance(sr, str):
            stop_reason = sr
    if "metadata" in event:
        usage = (event.get("metadata") or {}).get("usage") or {}
        if isinstance(usage, dict):
            if usage.get("inputTokens") is not None:
                totals["prompt"] = int(usage.get("inputTokens") or 0)
            if usage.get("outputTokens") is not None:
                totals["completion"] = int(usage.get("outputTokens") or 0)
            if usage.get("totalTokens") is not None:
                totals["total"] = int(usage.get("totalTokens") or 0)
    return stop_reason


def _finalize_stream(
    text_parts: list[str],
    totals: dict[str, int],
    stop_reason: str | None,
) -> tuple[dict[str, str] | None, NormalizedTokenUsage | None, str | None]:
    output = {"role": "assistant", "content": "".join(text_parts)} if text_parts else None
    usage: NormalizedTokenUsage | None = None
    if totals["prompt"] or totals["completion"] or totals["total"]:
        usage = NormalizedTokenUsage(
            prompt_tokens=totals["prompt"],
            completion_tokens=totals["completion"],
            total_tokens=totals["total"],
        )
    return output, usage, stop_reason


def _aggregate_invoke_stream(
    chunks: list[Any], _family: str
) -> tuple[dict[str, str] | None, NormalizedTokenUsage | None, str | None]:
    """Aggregate ``invoke_model_with_response_stream`` chunks.

    Handles both Anthropic Messages SSE (``message_start`` / ``content_block_delta``
    / ``message_delta``) and Nova/Converse-shaped events delivered inside the
    ``chunk.bytes`` blobs, plus the universal ``amazon-bedrock-invocationMetrics``
    token counts carried on the final chunk of most families.
    """
    text_parts: list[str] = []
    totals = {"prompt": 0, "completion": 0, "total": 0}
    stop_reason: str | None = None
    for chunk in chunks:
        payload = _decode_invoke_stream_chunk(chunk)
        if not isinstance(payload, dict):
            continue
        ctype = payload.get("type")
        # --- Anthropic Messages SSE ---
        if ctype == "message_start":
            usage = ((payload.get("message") or {}).get("usage")) or {}
            if isinstance(usage, dict):
                if usage.get("input_tokens") is not None:
                    totals["prompt"] = int(usage.get("input_tokens") or 0)
                if usage.get("output_tokens"):
                    totals["completion"] = int(usage.get("output_tokens") or 0)
        elif ctype == "content_block_delta":
            delta = payload.get("delta") or {}
            if isinstance(delta, dict) and "text" in delta:
                text_parts.append(str(delta.get("text") or ""))
        elif ctype == "message_delta":
            usage = payload.get("usage") or {}
            if isinstance(usage, dict) and usage.get("output_tokens") is not None:
                totals["completion"] = int(usage.get("output_tokens") or 0)
            delta = payload.get("delta") or {}
            if isinstance(delta, dict) and isinstance(delta.get("stop_reason"), str):
                stop_reason = delta.get("stop_reason")
        # --- Nova / Converse-shaped events inside the invoke stream ---
        sr = _accumulate_converse_event(payload, text_parts, totals)
        if sr:
            stop_reason = sr
        # --- Universal invocation metrics (final chunk of most families) ---
        metrics = payload.get("amazon-bedrock-invocationMetrics")
        if isinstance(metrics, dict):
            if metrics.get("inputTokenCount") is not None:
                totals["prompt"] = int(metrics.get("inputTokenCount") or 0)
            if metrics.get("outputTokenCount") is not None:
                totals["completion"] = int(metrics.get("outputTokenCount") or 0)
    return _finalize_stream(text_parts, totals, stop_reason)


def _build_stream_tool_calls(tools: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert accumulated streamed tool-use blocks into tool.call fragments.

    Mirrors the non-streaming ``_extract_converse_tool_calls`` contract
    (``tool_name`` / ``arguments`` / ``tool_use_id``); the streamed ``input``
    arrives as concatenated JSON fragments, so parse them back to the same
    object shape the sync path exposes (falling back to the raw string if the
    fragments don't form valid JSON).
    """
    calls: list[dict[str, Any]] = []
    for idx in sorted(tools):
        entry = tools[idx]
        name = entry.get("tool_name")
        if not name:
            continue
        raw = "".join(entry.get("input_parts") or [])
        arguments: Any = None
        if raw:
            try:
                arguments = json.loads(raw)
            except (ValueError, TypeError):
                arguments = raw
        calls.append(
            {
                "tool_name": name,
                "arguments": arguments,
                "tool_use_id": entry.get("tool_use_id"),
            }
        )
    return calls


def _aggregate_converse_stream(
    chunks: list[Any],
) -> tuple[dict[str, str] | None, NormalizedTokenUsage | None, str | None, list[dict[str, Any]]]:
    """Aggregate ``converse_stream`` events (delivered as dicts, not bytes).

    Also collects streamed tool-use so the streaming path reaches parity with
    the non-streaming ``converse`` path (BUG-3): a tool.call is emitted per
    toolUse block, and a pure-tool turn renders a faithful marker instead of
    collapsing to ``output_message=None``.
    """
    text_parts: list[str] = []
    totals = {"prompt": 0, "completion": 0, "total": 0}
    tools: dict[int, dict[str, Any]] = {}
    stop_reason: str | None = None
    for event in chunks:
        if not isinstance(event, dict):
            continue
        sr = _accumulate_converse_event(event, text_parts, totals, tools)
        if sr:
            stop_reason = sr
    tool_calls = _build_stream_tool_calls(tools)
    # A tool-use turn is real output — render a marker per toolUse block so a
    # pure-tool streaming turn does NOT collapse to output_message=None (parity
    # with _extract_converse_output; the name + input also go out as tool.call).
    for call in tool_calls:
        text_parts.append(f"[tool_use: {call['tool_name']}]")
    output, usage, stop_reason = _finalize_stream(text_parts, totals, stop_reason)
    return output, usage, stop_reason, tool_calls


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
    elif family == "amazon" and isinstance(data.get("messages"), list):
        # Nova (schemaVersion messages-v1) — Converse-shaped body.
        system = data.get("system")
        if isinstance(system, list):
            sys_text = "\n".join(str(s.get("text", "")) for s in system if isinstance(s, dict) and "text" in s)
            if sys_text:
                out.append({"role": "system", "content": sys_text})
        elif system:
            out.append({"role": "system", "content": str(system)})
        out.extend(_normalize_converse_messages(data.get("messages")) or [])
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
    elif family == "meta":
        content = str(data.get("generation", ""))
    elif family == "mistral":
        # Mistral-on-Bedrock wire: {"outputs": [{"text": ..., "stop_reason": ...}]}
        # (NOT the ``generation`` key meta uses).
        outputs = data.get("outputs") or []
        if outputs and isinstance(outputs[0], dict):
            content = str(outputs[0].get("text", ""))
    elif family == "cohere":
        # Command-R / R+ wire: top-level ``text``. Older Command models use
        # ``generations[0].text`` — fall back to keep them working.
        text = data.get("text")
        if isinstance(text, str) and text:
            content = text
        else:
            generations = data.get("generations") or []
            if generations and isinstance(generations[0], dict):
                content = str(generations[0].get("text", ""))
    elif family == "ai21":
        # Jamba wire is OpenAI-chat-shaped: {"choices": [{"message": {"content": ...}}]}.
        choices = data.get("choices") or []
        if choices and isinstance(choices[0], dict):
            message = choices[0].get("message") or {}
            content = str(message.get("content", "") if isinstance(message, dict) else "")
    elif family == "amazon":
        output = data.get("output")
        message = output.get("message") if isinstance(output, dict) else None
        if isinstance(message, dict):  # Nova (Converse-shaped)
            content = "\n".join(
                str(b.get("text", "")) for b in message.get("content", []) or [] if isinstance(b, dict) and "text" in b
            )
        else:  # Titan
            results = data.get("results") or []
            if results:
                content = str(results[0].get("outputText", ""))
    else:
        content = str(data.get("generation") or data.get("completion") or data.get("outputText") or "")
    return {"role": "assistant", "content": content} if content else None


def _extract_invoke_stop_reason(data: Dict[str, Any], family: str) -> str | None:
    """Family-specific stop reason from invoke_model body (TEL-029 / LAY-2883)."""
    if not data:
        return None
    if family == "anthropic":
        val = data.get("stop_reason")
        return val if isinstance(val, str) else None
    if family == "meta":
        val = data.get("stop_reason")
        return val if isinstance(val, str) else None
    if family == "cohere":
        # Command-R / R+: top-level finish_reason. Older Command: generations[0].
        val = data.get("finish_reason")
        if isinstance(val, str):
            return val
        gens = data.get("generations") or []
        if gens and isinstance(gens[0], dict):
            val = gens[0].get("finish_reason")
            return val if isinstance(val, str) else None
    if family == "ai21":
        choices = data.get("choices") or []
        if choices and isinstance(choices[0], dict):
            val = choices[0].get("finish_reason")
            return val if isinstance(val, str) else None
    if family == "amazon":
        stop = data.get("stopReason")  # Nova (Converse-shaped)
        if isinstance(stop, str):
            return stop
        results = data.get("results") or []
        if results and isinstance(results[0], dict):
            val = results[0].get("completionReason")
            return val if isinstance(val, str) else None
    if family == "mistral":
        outputs = data.get("outputs") or []
        if outputs and isinstance(outputs[0], dict):
            val = outputs[0].get("stop_reason")
            return val if isinstance(val, str) else None
    return None


def _bedrock_response_id(response: Any) -> str | None:
    """Pull AWS RequestId — every boto3 Bedrock response has one in
    ``ResponseMetadata.RequestId``."""
    if not isinstance(response, dict):
        return None
    metadata = response.get("ResponseMetadata") or {}
    if not isinstance(metadata, dict):
        return None
    rid = metadata.get("RequestId")
    return rid if isinstance(rid, str) and rid else None


def _extract_invoke_usage(data: Dict[str, Any], family: str, headers: Any = None) -> NormalizedTokenUsage | None:
    """Family-specific token usage from the invoke_model body, with a Bedrock
    response-header fallback.

    Bedrock echoes per-request token counts in the ``X-Amzn-Bedrock-Input/
    Output-Token-Count`` response headers for EVERY family. Some families
    (mistral) carry NO token counts in the body at all, so the header fallback
    is the only honest source there; for others it's a safety net that never
    overrides a real body count.
    """
    prompt = completion = total = 0
    if data:
        if family == "anthropic":
            usage = data.get("usage") or {}
            prompt = int(usage.get("input_tokens") or 0)
            completion = int(usage.get("output_tokens") or 0)
        elif family == "ai21":
            # Jamba: OpenAI-chat-shaped usage block.
            usage = data.get("usage") or {}
            if isinstance(usage, dict):
                prompt = int(usage.get("prompt_tokens") or 0)
                completion = int(usage.get("completion_tokens") or 0)
                total = int(usage.get("total_tokens") or 0)
        elif family == "cohere":
            # Command-R / R+: token counts under meta.billed_units.
            billed = (data.get("meta") or {}).get("billed_units") or {}
            if isinstance(billed, dict):
                prompt = int(billed.get("input_tokens") or 0)
                completion = int(billed.get("output_tokens") or 0)
        elif family == "amazon":
            usage = data.get("usage")
            if isinstance(usage, dict) and ("inputTokens" in usage or "outputTokens" in usage):
                # Nova (Converse-shaped usage block).
                prompt = int(usage.get("inputTokens") or 0)
                completion = int(usage.get("outputTokens") or 0)
                total = int(usage.get("totalTokens") or 0)
            else:
                # Titan: prompt at top-level, completion at results[0].tokenCount.
                prompt = int(data.get("inputTextTokenCount") or 0)
                results = data.get("results") or []
                if results and isinstance(results[0], dict):
                    completion = int(results[0].get("tokenCount") or 0)
        else:
            # Meta and any other inline-field families.
            prompt = int(data.get("prompt_token_count") or data.get("inputTextTokenCount") or 0)
            completion = int(data.get("generation_token_count") or 0)

    # Header fallback fills ONLY what the body didn't supply (never overrides).
    if headers is not None and not (prompt and completion):
        h_prompt, h_completion = _header_token_counts(headers)
        prompt = prompt or h_prompt
        completion = completion or h_completion

    if not (prompt or completion or total):
        return None
    return NormalizedTokenUsage(prompt_tokens=prompt, completion_tokens=completion, total_tokens=total)


def _response_http_headers(response: Any) -> dict[str, Any] | None:
    """The ``HTTPHeaders`` map from a boto3 response's ``ResponseMetadata``."""
    if not isinstance(response, dict):
        return None
    metadata = response.get("ResponseMetadata")
    if not isinstance(metadata, dict):
        return None
    headers = metadata.get("HTTPHeaders")
    return headers if isinstance(headers, dict) else None


def _header_token_counts(headers: Any) -> tuple[int, int]:
    """(prompt, completion) token counts from Bedrock response headers."""
    if not isinstance(headers, dict):
        return 0, 0
    lower = {str(k).lower(): v for k, v in headers.items()}

    def _int(value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    return (
        _int(lower.get("x-amzn-bedrock-input-token-count")),
        _int(lower.get("x-amzn-bedrock-output-token-count")),
    )


def _extract_converse_output(response: Dict[str, Any]) -> dict[str, str] | None:
    if not isinstance(response, dict):
        return None
    msg = (response.get("output") or {}).get("message") or {}
    parts: list[str] = []
    for block in msg.get("content", []) or []:
        if not isinstance(block, dict):
            continue
        if "text" in block:
            parts.append(str(block.get("text", "")))
        elif "toolUse" in block:
            # A tool-use turn is real output — render a faithful marker so a
            # pure-tool assistant turn does NOT collapse to output_message=None
            # (the tool name + input are also emitted as a tool.call event).
            tool_use = block.get("toolUse") or {}
            name = tool_use.get("name")
            if name:
                parts.append(f"[tool_use: {name}]")
    if not parts:
        return None
    return {"role": str(msg.get("role", "assistant")), "content": "\n".join(parts)}


def _extract_converse_tool_calls(response: Dict[str, Any]) -> list[dict[str, Any]]:
    """toolUse blocks from a Converse response → tool.call payload fragments."""
    if not isinstance(response, dict):
        return []
    msg = (response.get("output") or {}).get("message") or {}
    calls: list[dict[str, Any]] = []
    for block in msg.get("content", []) or []:
        if not isinstance(block, dict) or "toolUse" not in block:
            continue
        tool_use = block.get("toolUse") or {}
        name = tool_use.get("name")
        if not name:
            continue
        calls.append(
            {
                "tool_name": name,
                "arguments": tool_use.get("input"),
                "tool_use_id": tool_use.get("toolUseId"),
            }
        )
    return calls


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
        parts: list[str] = []
        for block in msg.get("content") or []:
            if not isinstance(block, dict):
                continue
            if "text" in block:
                parts.append(str(block.get("text", "")))
            elif "toolUse" in block:
                tool_use = block.get("toolUse") or {}
                parts.append(f"[tool_use: {tool_use.get('name')} input={_compact_json(tool_use.get('input'))}]")
            elif "toolResult" in block:
                # Capture the tool's returned content — dropping it discards
                # the entire result of the tool round-trip from the trace.
                parts.append(f"[tool_result: {_tool_result_text(block.get('toolResult'))}]")
        out.append({"role": role, "content": "\n".join(parts)})
    return out or None


def _compact_json(value: Any) -> str:
    """Compact, deterministic JSON rendering (falls back to str)."""
    try:
        return json.dumps(value, separators=(",", ":"), sort_keys=True, default=str)
    except (TypeError, ValueError):
        return str(value)


def _tool_result_text(tool_result: Any) -> str:
    """Flatten a Converse toolResult's content blocks to text."""
    if not isinstance(tool_result, dict):
        return ""
    texts: list[str] = []
    for block in tool_result.get("content") or []:
        if not isinstance(block, dict):
            continue
        if "text" in block:
            texts.append(str(block.get("text", "")))
        elif "json" in block:
            texts.append(_compact_json(block.get("json")))
    rendered = " ".join(t for t in texts if t)
    status = tool_result.get("status")
    if rendered and status:
        return f"{rendered} (status={status})"
    return rendered or (str(status) if status else "")


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
    tool_calls: list[dict[str, Any]] | None = None,
    ttft_ms: float | None = None,
    streaming_duration_ms: float | None = None,
) -> None:
    import uuid

    collector = _current_collector.get()
    if collector is None:
        return
    span_id = uuid.uuid4().hex[:16]
    parent_span_id = _current_span_id.get()
    parameters = {k: kwargs[k] for k in _CAPTURE_PARAMS if k in kwargs}
    payload: Dict[str, Any] = {
        "name": event,
        "model": model_id,
        "latency_ms": latency_ms,
        "parameters": parameters,
        "messages": messages,
        "output_message": output,
        # Integration-name stamp so the framework column shows the integration,
        # not the OTel underlying provider — bedrock's emit path is bespoke (S19/F12).
        "framework": "aws_bedrock",
    }
    if usage is not None:
        payload["usage"] = usage.as_event_dict()
        # Flat token keys beside the nested usage block so the atlas extractor
        # (which reads top-level token keys, never usage.*) fills the tokens
        # column — bedrock's emit path is bespoke, not via emit_llm_events (S11/F2).
        payload.update(_flat_token_fields(usage))
    if ttft_ms is not None:
        payload["ttft_ms"] = ttft_ms
    if streaming_duration_ms is not None:
        payload["streaming_duration_ms"] = streaming_duration_ms
    payload.update(extra)
    # OTel GenAI semantic-convention attributes (TEL-029 / LAY-2883). Bedrock's
    # emit path is bespoke (no _base_provider wrap), so we plumb gen_ai_attributes
    # in directly here using extra + usage dicts.
    response_meta: Dict[str, Any] = {}
    if "response_id" in extra:
        response_meta["response_id"] = extra["response_id"]
    if "stop_reason" in extra:
        response_meta["stop_reason"] = extra["stop_reason"]
    response_meta["response_model"] = model_id
    payload["otel_gen_ai"] = gen_ai_attributes(
        provider="bedrock",
        operation="chat",
        parameters=parameters,
        response_meta=response_meta,
        usage=usage.as_event_dict() if usage is not None else None,
    )
    collector.emit(MODEL_INVOKE, payload, span_id=span_id, parent_span_id=parent_span_id)

    for call in tool_calls or []:
        collector.emit(
            TOOL_CALL,
            {
                "provider": "aws_bedrock",
                "model": model_id,
                "framework": "aws_bedrock",
                **call,
            },
            span_id=uuid.uuid4().hex[:16],
            parent_span_id=span_id,
        )

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
        {
            "name": event,
            "error": safe_error(exc),
            "error_type": type(exc).__name__,
            "status": "error",
            "latency_ms": latency_ms,
        },
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
