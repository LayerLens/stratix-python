"""LLM Provider Base Adapter.

Abstract intermediate class for all LLM provider adapters. Extends
:class:`BaseAdapter` with provider-specific emit helpers for
``model.invoke``, ``cost.record``, ``tool.call``, and
``policy.violation`` events.

Supports W3C Trace Context propagation (``traceparent`` /
``tracestate``) for correlating spans across adapter boundaries.

Ported from ``ateam/stratix/sdk/python/adapters/llm_providers/base_provider.py``.
"""

from __future__ import annotations

import time
import uuid
import logging
from abc import abstractmethod
from typing import Any, Dict, List, Optional

from layerlens._compat.pydantic import model_dump
from layerlens.instrument.adapters._base.adapter import (
    AdapterInfo,
    BaseAdapter,
    AdapterHealth,
    AdapterStatus,
    ReplayableTrace,
    AdapterCapability,
)
from layerlens.instrument.adapters._base.capture import CaptureConfig
from layerlens.instrument.adapters.providers._base.tokens import NormalizedTokenUsage
from layerlens.instrument.adapters.providers._base.pricing import calculate_cost

# W3C Trace Context header names.
_TRACEPARENT_HEADER = "traceparent"
_TRACESTATE_HEADER = "tracestate"

logger = logging.getLogger(__name__)


class LLMProviderAdapter(BaseAdapter):
    """Abstract base class for all LLM provider adapters.

    Provides concrete implementations for:

    * Event emission helpers (:meth:`_emit_model_invoke`,
      :meth:`_emit_cost_record`, :meth:`_emit_tool_calls`,
      :meth:`_emit_provider_error`).
    * Lifecycle methods (:meth:`health_check`,
      :meth:`get_adapter_info`, :meth:`serialize_for_replay`).
    * Client reference management (``_client``, ``_originals``).

    Subclasses must implement:

    * :meth:`connect` — import framework, set HEALTHY.
    * :meth:`disconnect` — restore originals, set DISCONNECTED.
    * :meth:`connect_client` — wrap the provider client.
    """

    adapter_type: str = "llm_provider"

    def __init__(
        self,
        stratix: Any = None,
        capture_config: Optional[CaptureConfig] = None,
    ) -> None:
        super().__init__(stratix=stratix, capture_config=capture_config)
        self._client: Any = None
        self._originals: Dict[str, Any] = {}
        self._framework_version: Optional[str] = None

    # --- Abstract methods subclasses must implement ---

    @abstractmethod
    def connect_client(self, client: Any) -> Any:
        """Wrap or monkey-patch the provider client to intercept API calls.

        Args:
            client: The provider SDK client instance.

        Returns:
            The wrapped client (same object, modified in-place).
        """

    # --- Concrete lifecycle methods ---

    def connect(self) -> None:
        """Verify framework availability and mark as connected."""
        self._framework_version = self._detect_framework_version()
        self._connected = True
        self._status = AdapterStatus.HEALTHY

    def disconnect(self) -> None:
        """Restore all original methods and disconnect."""
        self._restore_originals()
        self._client = None
        self._originals.clear()
        self._connected = False
        self._status = AdapterStatus.DISCONNECTED

    def _restore_originals(self) -> None:
        """Restore original methods on the client. Override for custom logic."""

    def health_check(self) -> AdapterHealth:
        return AdapterHealth(
            status=self._status,
            framework_name=self.FRAMEWORK,
            framework_version=self._framework_version,
            adapter_version=self.VERSION,
            error_count=self._error_count,
            circuit_open=self._circuit_open,
        )

    def get_adapter_info(self) -> AdapterInfo:
        return AdapterInfo(
            name=type(self).__name__,
            version=self.VERSION,
            framework=self.FRAMEWORK,
            framework_version=self._framework_version,
            capabilities=[
                AdapterCapability.TRACE_MODELS,
                AdapterCapability.TRACE_TOOLS,
            ],
            description=f"LayerLens adapter for {self.FRAMEWORK} LLM provider",
        )

    def serialize_for_replay(self) -> ReplayableTrace:
        return ReplayableTrace(
            adapter_name=type(self).__name__,
            framework=self.FRAMEWORK,
            trace_id=str(uuid.uuid4()),
            events=list(self._trace_events),
            state_snapshots=[],
            config={"capture_config": model_dump(self._capture_config)},
        )

    @staticmethod
    def _detect_framework_version() -> Optional[str]:
        """Override in subclasses to detect SDK version."""
        return None

    # --- W3C Trace Context Propagation ---

    def _inject_trace_context(
        self,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        """Inject W3C ``traceparent`` / ``tracestate`` headers for outbound requests.

        If OpenTelemetry is available, uses the OTel propagator. Otherwise
        generates a minimal ``traceparent`` from the current trace / span
        IDs.

        Args:
            headers: Existing headers dict to inject into (mutated in place).

        Returns:
            Headers dict with ``traceparent`` (and optionally ``tracestate``) added.
        """
        if headers is None:
            headers = {}

        try:
            # opentelemetry is an optional dep installed via the
            # `[otel]` extra; fall through to the manual traceparent
            # synthesis below when it is not available.
            from opentelemetry.propagate import inject  # type: ignore[import-not-found,unused-ignore]

            inject(headers)
        except ImportError:
            trace_id = getattr(self, "_current_trace_id", None)
            span_id = getattr(self, "_current_span_id", None)
            if trace_id and span_id:
                headers[_TRACEPARENT_HEADER] = f"00-{trace_id}-{span_id}-01"

        return headers

    def _extract_trace_context(
        self,
        headers: Dict[str, str],
    ) -> Dict[str, str]:
        """Extract W3C ``traceparent`` / ``tracestate`` from inbound headers.

        Args:
            headers: Inbound headers dict.

        Returns:
            Dict with ``trace_id``, ``parent_span_id``, ``trace_flags``,
            and optionally ``tracestate``.
        """
        result: Dict[str, str] = {}

        traceparent = headers.get(_TRACEPARENT_HEADER, "")
        if traceparent:
            parts = traceparent.split("-")
            if len(parts) >= 4:
                result["trace_id"] = parts[1]
                result["parent_span_id"] = parts[2]
                result["trace_flags"] = parts[3]

        tracestate = headers.get(_TRACESTATE_HEADER, "")
        if tracestate:
            result["tracestate"] = tracestate

        return result

    # --- Event emission helpers ---

    def _emit_model_invoke(
        self,
        provider: str,
        model: Optional[str],
        parameters: Optional[Dict[str, Any]] = None,
        usage: Optional[NormalizedTokenUsage] = None,
        latency_ms: Optional[float] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        input_messages: Optional[List[Dict[str, str]]] = None,
        output_message: Optional[Dict[str, str]] = None,
    ) -> None:
        """Emit a ``model.invoke`` (L3) event."""
        payload: Dict[str, Any] = {
            "provider": provider,
            "model": model,
            "timestamp_ns": time.time_ns(),
        }
        if parameters:
            payload["parameters"] = parameters
        if usage:
            payload["prompt_tokens"] = usage.prompt_tokens
            payload["completion_tokens"] = usage.completion_tokens
            payload["total_tokens"] = usage.total_tokens
            if usage.cached_tokens is not None:
                payload["cached_tokens"] = usage.cached_tokens
            if usage.reasoning_tokens is not None:
                payload["reasoning_tokens"] = usage.reasoning_tokens
        if latency_ms is not None:
            payload["latency_ms"] = latency_ms
        if error:
            payload["error"] = error
        if metadata:
            for k, v in metadata.items():
                if k not in payload:
                    payload[k] = v
        if self._capture_config.capture_content:
            if input_messages:
                payload["messages"] = input_messages
            if output_message:
                payload["output_message"] = output_message

        self.emit_dict_event("model.invoke", payload)

    @staticmethod
    def _normalize_messages(
        raw_messages: Any,
        system: Any = None,
    ) -> Optional[List[Dict[str, str]]]:
        """Normalize provider-specific message formats to ``[{role, content}]``.

        Args:
            raw_messages: Messages from the provider SDK kwargs (list of
                dicts, list of objects, or ``None``).
            system: Separate system prompt (e.g. Anthropic's ``system``
                kwarg). May be a string or a list of content blocks.

        Returns:
            Normalized list, or ``None`` if no messages were found.
        """
        if not raw_messages and not system:
            return None

        messages: List[Dict[str, str]] = []

        if system:
            if isinstance(system, str):
                messages.append({"role": "system", "content": system[:10_000]})
            elif isinstance(system, list):
                parts: List[str] = []
                for block in system:
                    if isinstance(block, str):
                        parts.append(block)
                    elif isinstance(block, dict) and "text" in block:
                        parts.append(str(block["text"]))
                if parts:
                    messages.append({"role": "system", "content": "\n".join(parts)[:10_000]})

        if raw_messages:
            for msg in raw_messages:
                role = ""
                content = ""
                if isinstance(msg, dict):
                    role = str(msg.get("role", ""))
                    raw_content = msg.get("content", "")
                    if isinstance(raw_content, str):
                        content = raw_content
                    elif isinstance(raw_content, list):
                        parts2: List[str] = []
                        for part in raw_content:
                            if isinstance(part, str):
                                parts2.append(part)
                            elif isinstance(part, dict):
                                text = part.get("text") or part.get("content", "")
                                if text:
                                    parts2.append(str(text))
                        content = "\n".join(parts2)
                    else:
                        content = str(raw_content) if raw_content else ""
                elif hasattr(msg, "role") and hasattr(msg, "content"):
                    role = str(getattr(msg, "role", ""))
                    raw_content = getattr(msg, "content", "")
                    if isinstance(raw_content, str):
                        content = raw_content
                    elif isinstance(raw_content, list):
                        parts3: List[str] = []
                        for part in raw_content:
                            if isinstance(part, str):
                                parts3.append(part)
                            elif hasattr(part, "text"):
                                parts3.append(str(part.text))
                            elif isinstance(part, dict) and "text" in part:
                                parts3.append(str(part["text"]))
                        content = "\n".join(parts3)
                    else:
                        content = str(raw_content) if raw_content else ""
                else:
                    continue

                if role:
                    messages.append({"role": role, "content": content[:10_000]})

        return messages if messages else None

    def _emit_cost_record(
        self,
        model: Optional[str],
        usage: Optional[NormalizedTokenUsage],
        provider: Optional[str] = None,
        pricing_table: Optional[Dict[str, Dict[str, float]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Emit a ``cost.record`` (cross-cutting) event."""
        payload: Dict[str, Any] = {
            "provider": provider or self.FRAMEWORK,
            "model": model,
        }

        if usage:
            payload["prompt_tokens"] = usage.prompt_tokens
            payload["completion_tokens"] = usage.completion_tokens
            payload["total_tokens"] = usage.total_tokens

            cost = calculate_cost(model or "", usage, pricing_table)
            if cost is not None:
                payload["api_cost_usd"] = cost
            else:
                payload["api_cost_usd"] = None
                payload["pricing_unavailable"] = True

        if metadata:
            for k, v in metadata.items():
                if k not in payload:
                    payload[k] = v

        self.emit_dict_event("cost.record", payload)

    def _emit_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
        parent_model: Optional[str] = None,
    ) -> None:
        """Emit ``tool.call`` (L5a) events for function / tool calls in a response."""
        for tc in tool_calls:
            payload: Dict[str, Any] = {
                "tool_name": tc.get("name", "unknown"),
                "tool_input": tc.get("arguments") or tc.get("input"),
                "provider": self.FRAMEWORK,
            }
            if parent_model:
                payload["model"] = parent_model
            if "id" in tc:
                payload["tool_call_id"] = tc["id"]

            self.emit_dict_event("tool.call", payload)

    def _emit_provider_error(
        self,
        provider: str,
        error: str,
        model: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Emit ``policy.violation`` (cross-cutting) for provider errors."""
        payload: Dict[str, Any] = {
            "provider": provider,
            "error": error,
            "violation_type": "safety",
        }
        if model:
            payload["model"] = model
        if metadata:
            for k, v in metadata.items():
                if k not in payload:
                    payload[k] = v

        self.emit_dict_event("policy.violation", payload)
