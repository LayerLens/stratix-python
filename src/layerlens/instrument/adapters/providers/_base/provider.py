"""LLM Provider Base Adapter.

Abstract intermediate class for all LLM provider adapters. Extends
:class:`BaseAdapter` with provider-specific emit helpers for
``model.invoke``, ``cost.record``, ``tool.call``, and
``policy.violation`` events.

Supports W3C Trace Context propagation (``traceparent`` /
``tracestate``) for correlating spans across adapter boundaries.

Ported from ``ateam/stratix/sdk/python/adapters/llm_providers/base_provider.py``.

Typed-event migration (Bundle #6 — final):
   The four ``self.emit_dict_event(...)`` call sites that previously
   produced legacy ad-hoc dict shapes were migrated to the canonical
   typed envelopes from :mod:`layerlens.instrument._compat.events`.
   Every concrete provider adapter (openai, anthropic, azure_openai,
   aws_bedrock, google_vertex, cohere, mistral, ollama, litellm)
   inherits the new typed emission surface — no provider-specific
   change is required because the helpers (``_emit_model_invoke``,
   ``_emit_cost_record``, ``_emit_tool_calls``,
   ``_emit_provider_error``) keep their public Python signatures.

   Adapter-specific provenance carried in the legacy ``metadata``
   kwarg (``response_id``, ``finish_reason``, ``response_model``,
   ``cache_creation_input_tokens``, ``cache_read_input_tokens``,
   ``request_type``, ``system_fingerprint``, etc.) folds onto the
   canonical :class:`ModelInfo.parameters` slot — the canonical schema
   does not declare these as top-level fields on
   :class:`ModelInvokeEvent`.

   ``tool.call`` emissions: the legacy ``tool_call_id`` and
   ``parent_model`` provenance fold onto :class:`ToolCallEvent.input`
   (``input_data["_tool_call_id"]`` / ``input_data["_parent_model"]``)
   so the canonical input slot remains the single source of truth.

   ``policy.violation`` emissions: the legacy
   ``violation_type="safety"`` default maps to
   :class:`ViolationType.SAFETY`. The provider error string lands on
   ``ViolationInfo.root_cause``; remediation defaults to a generic
   "review provider error and retry" guidance.
"""

from __future__ import annotations

import time
import uuid
import logging
from abc import abstractmethod
from typing import Any, Dict, List, Optional

from layerlens._compat.pydantic import model_dump
from layerlens.instrument._compat.events import (
    ToolCallEvent,
    CostRecordEvent,
    ViolationType,
    IntegrationType,
    ModelInvokeEvent,
    PolicyViolationEvent,
)
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
        *,
        org_id: Optional[str] = None,
    ) -> None:
        super().__init__(stratix=stratix, capture_config=capture_config, org_id=org_id)
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
            from opentelemetry.propagate import inject

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
        """Emit a typed :class:`ModelInvokeEvent` (L3).

        Adapter-specific provenance (``response_id``,
        ``finish_reason``, ``response_model``, ``request_type``,
        ``cache_creation_input_tokens``, ``cache_read_input_tokens``,
        ``system_fingerprint``, ``error``, ``timestamp_ns``) is folded
        onto the canonical :class:`ModelInfo.parameters` slot since the
        canonical schema does not declare these as top-level fields on
        :class:`ModelInvokeEvent`. The original ``parameters`` mapping
        (temperature, max_tokens, has_system, tools_count, etc.) is
        merged in alongside.
        """
        # Build the canonical ``parameters`` payload by merging the
        # invocation kwargs (temperature, max_tokens, …) with adapter-
        # specific provenance (response_id, finish_reason, …) and the
        # always-recorded emission timestamp. Caller-supplied keys win
        # over auto-generated ones (only ``timestamp_ns`` is
        # unconditionally stamped — it is the emission timestamp, not a
        # caller value).
        merged_parameters: Dict[str, Any] = {"timestamp_ns": time.time_ns()}
        if parameters:
            merged_parameters.update(parameters)
        if metadata:
            for k, v in metadata.items():
                merged_parameters.setdefault(k, v)
        if error:
            merged_parameters["error"] = error

        # Token slots map onto ModelInvokeEvent's top-level token
        # fields. ``cached_tokens`` / ``reasoning_tokens`` go into
        # ``parameters`` because the canonical schema only declares
        # ``prompt_tokens`` / ``completion_tokens`` / ``total_tokens``.
        prompt_tokens: Optional[int] = None
        completion_tokens: Optional[int] = None
        total_tokens: Optional[int] = None
        if usage:
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens
            total_tokens = usage.total_tokens
            if usage.cached_tokens is not None:
                merged_parameters["cached_tokens"] = usage.cached_tokens
            if usage.reasoning_tokens is not None:
                merged_parameters["reasoning_tokens"] = usage.reasoning_tokens

        # Content gating: only attach messages / output_message when
        # CaptureConfig opts in. The canonical fields accept
        # ``Optional[list[dict[str, str]]]`` / ``Optional[dict[str,
        # str]]`` so we just leave them ``None`` when content capture
        # is off.
        emit_input_messages: Optional[List[Dict[str, str]]] = None
        emit_output_message: Optional[Dict[str, str]] = None
        if self._capture_config.capture_content:
            if input_messages:
                emit_input_messages = input_messages
            if output_message:
                emit_output_message = output_message

        self.emit_event(
            ModelInvokeEvent.create(
                provider=provider,
                name=model or "unavailable",
                version="unavailable",
                parameters=merged_parameters,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                latency_ms=latency_ms,
                input_messages=emit_input_messages,
                output_message=emit_output_message,
            )
        )

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
        """Emit a typed :class:`CostRecordEvent` (cross-cutting).

        The canonical :class:`CostInfo` slot only carries cost
        primitives (``tokens`` / ``prompt_tokens`` /
        ``completion_tokens`` / ``api_cost_usd`` / ``infra_cost_usd`` /
        ``tool_calls``). Adapter-specific provenance (``provider``,
        ``model``, ``pricing_unavailable``, custom ``metadata`` keys)
        does not have a canonical slot. Per CLAUDE.md ("never silently
        skip failing operations"), we mark unavailable pricing by
        passing ``api_cost_usd="unavailable"`` — the canonical schema
        accepts ``Union[float, str]`` precisely for this case.
        """
        prompt_tokens: Optional[int] = None
        completion_tokens: Optional[int] = None
        total_tokens: Optional[int] = None
        api_cost_usd: Optional[Any] = None

        if usage:
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens
            total_tokens = usage.total_tokens

            cost = calculate_cost(model or "", usage, pricing_table)
            if cost is not None:
                api_cost_usd = cost
            else:
                # Canonical "missing pricing" sentinel: a string union
                # member rather than a side-channel boolean. Mirrors
                # the NORMATIVE rule "Costs must mark unavailable
                # (never omit silently)" from the canonical schema.
                api_cost_usd = "unavailable"

        self.emit_event(
            CostRecordEvent.create(
                tokens=total_tokens,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                api_cost_usd=api_cost_usd,
            )
        )

    def _emit_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
        parent_model: Optional[str] = None,
    ) -> None:
        """Emit typed :class:`ToolCallEvent` (L5a) events for tool / function calls.

        Provider-side ``tool_call_id`` (the Anthropic / OpenAI tool-use
        identifier), ``parent_model`` (the model that requested the
        tool), and the provider name itself fold onto
        :class:`ToolCallEvent.input` as namespaced ``_tool_call_id`` /
        ``_parent_model`` / ``_provider`` keys. The canonical schema
        keeps ``input`` as a free-form ``dict[str, Any]`` so namespaced
        provenance keys do not collide with caller-supplied tool
        arguments. Function-tool integration type is :class:`LIBRARY`
        — provider tool-use is in-process Python, not a remote
        service.
        """
        for tc in tool_calls:
            tool_name = str(tc.get("name", "unknown"))
            raw_args = tc.get("arguments") or tc.get("input")
            input_data: Dict[str, Any]
            if isinstance(raw_args, dict):
                input_data = dict(raw_args)
            elif raw_args is None:
                input_data = {}
            else:
                # Non-dict tool arguments (string JSON, scalar, list)
                # ride on a canonical ``value`` slot — the adapter
                # framework records the raw payload so replay sees the
                # exact byte sequence the provider returned.
                input_data = {"value": raw_args}

            input_data["_provider"] = self.FRAMEWORK
            if parent_model:
                input_data["_parent_model"] = parent_model
            if "id" in tc:
                input_data["_tool_call_id"] = tc["id"]

            self.emit_event(
                ToolCallEvent.create(
                    name=tool_name,
                    version="unavailable",
                    integration=IntegrationType.LIBRARY,
                    input_data=input_data,
                )
            )

    def _emit_provider_error(
        self,
        provider: str,
        error: str,
        model: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Emit a typed :class:`PolicyViolationEvent` for provider errors.

        Provider runtime failures (rate limits, auth failures, network
        errors, model-not-found) ride on the canonical safety
        violation envelope:

        * ``violation_type`` — :class:`ViolationType.SAFETY` (the
          legacy adapter set ``"safety"`` as the dict-shape default).
        * ``root_cause`` — the provider-supplied error string.
        * ``remediation`` — generic guidance to inspect the provider
          response and retry. Subclasses can subclass this method if
          they have provider-specific remediation guidance.
        * ``details`` — carries ``provider`` / ``model`` and any
          caller-supplied metadata so the legacy provenance keys
          remain inspectable for replay.
        """
        details: Dict[str, Any] = {"provider": provider}
        if model:
            details["model"] = model
        if metadata:
            for k, v in metadata.items():
                details.setdefault(k, v)

        self.emit_event(
            PolicyViolationEvent.create(
                violation_type=ViolationType.SAFETY,
                root_cause=error,
                remediation=(
                    "Inspect the provider response and retry — "
                    "consult the provider's status page if the error persists."
                ),
                details=details,
            )
        )
