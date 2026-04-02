from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ._base_framework import FrameworkAdapter
from ._utils import safe_serialize, truncate
from ..._capture_config import CaptureConfig

log = logging.getLogger(__name__)

try:
    import semantic_kernel as _sk  # pyright: ignore[reportMissingImports]  # noqa: F401

    _HAS_SEMANTIC_KERNEL = True
except ImportError:
    _HAS_SEMANTIC_KERNEL = False


class SemanticKernelAdapter(FrameworkAdapter):
    """Semantic Kernel adapter using the SK filter API (semantic-kernel >= 1.0).

    Registers function invocation, prompt rendering, and auto-function
    invocation filters on a Kernel instance to capture plugin calls,
    prompt templates, and LLM-initiated function calls as flat events.

    Usage::

        adapter = SemanticKernelAdapter(client)
        adapter.connect(target=kernel)
        result = await kernel.invoke(my_function, arg1=val1)
        adapter.disconnect()
    """

    name = "semantic_kernel"
    package = "semantic-kernel"

    def __init__(self, client: Any, capture_config: Optional[CaptureConfig] = None) -> None:
        super().__init__(client, capture_config)
        self._kernel: Any = None
        self._filter_ids: List[tuple] = []  # (FilterTypes, filter_id) for removal
        self._seen_plugins: set = set()
        self._patched_services: Dict[str, Any] = {}  # service_id → original method

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _on_connect(self, target: Any = None, **kwargs: Any) -> None:
        self._check_dependency(_HAS_SEMANTIC_KERNEL)
        if target is None:
            raise ValueError("SemanticKernelAdapter requires a target kernel: adapter.connect(target=kernel)")

        from semantic_kernel.filters.filter_types import FilterTypes  # pyright: ignore[reportMissingImports]

        self._kernel = target

        filters = [
            (FilterTypes.FUNCTION_INVOCATION, self._function_invocation_filter),
            (FilterTypes.PROMPT_RENDERING, self._prompt_rendering_filter),
            (FilterTypes.AUTO_FUNCTION_INVOCATION, self._auto_function_invocation_filter),
        ]
        for filter_type, handler in filters:
            target.add_filter(filter_type, handler)
            filter_list = _get_filter_list(target, filter_type)
            if filter_list:
                self._filter_ids.append((filter_type, filter_list[-1][0]))

        # Wrap LLM calls on registered chat services
        self._patch_chat_services(target)

        # Discover existing plugins
        self._discover_plugins(target)

    def _on_disconnect(self) -> None:
        if self._kernel is not None:
            for filter_type, filter_id in self._filter_ids:
                try:
                    self._kernel.remove_filter(filter_type, filter_id=filter_id)
                except Exception:
                    log.debug("layerlens: could not remove SK filter %s/%s", filter_type, filter_id)
        self._unpatch_chat_services()
        self._filter_ids.clear()
        self._seen_plugins.clear()
        self._kernel = None

    # ------------------------------------------------------------------
    # LLM call wrapping
    # ------------------------------------------------------------------

    def _patch_chat_services(self, kernel: Any) -> None:
        """Wrap _inner_get_chat_message_contents on all registered chat services."""
        services = getattr(kernel, "services", None)
        if not services or not isinstance(services, dict):
            return

        for service_id, service in services.items():
            if not hasattr(service, "_inner_get_chat_message_contents"):
                continue
            original = service._inner_get_chat_message_contents
            adapter = self

            async def _traced_inner(chat_history: Any, settings: Any, _orig: Any = original, _svc: Any = service) -> Any:
                span_id = adapter._new_span_id()
                root_span = adapter._get_root_span()
                adapter._start_timer(span_id)
                collector = adapter._ensure_collector()

                model_name = getattr(_svc, "ai_model_id", None)

                try:
                    result = await _orig(chat_history, settings)
                except Exception as exc:
                    latency_ms = adapter._stop_timer(span_id)
                    payload = adapter._payload(
                        error=str(exc),
                        error_type=type(exc).__name__,
                    )
                    if model_name:
                        payload["model"] = model_name
                    if latency_ms is not None:
                        payload["latency_ms"] = latency_ms
                    collector.emit(
                        "agent.error", payload,
                        span_id=span_id, parent_span_id=root_span,
                    )
                    raise

                latency_ms = adapter._stop_timer(span_id)
                tokens = adapter._extract_usage_from_response(result)

                payload = adapter._payload()
                if model_name:
                    payload["model"] = model_name
                if latency_ms is not None:
                    payload["latency_ms"] = latency_ms
                payload.update(tokens)
                collector.emit(
                    "model.invoke", payload,
                    span_id=span_id, parent_span_id=root_span,
                )

                if tokens:
                    cost_payload = adapter._payload()
                    if model_name:
                        cost_payload["model"] = model_name
                    cost_payload.update(tokens)
                    collector.emit(
                        "cost.record", cost_payload,
                        span_id=span_id, parent_span_id=root_span,
                    )

                return result

            service._inner_get_chat_message_contents = _traced_inner
            self._patched_services[service_id] = original

    def _unpatch_chat_services(self) -> None:
        """Restore original _inner_get_chat_message_contents on all patched services."""
        if self._kernel is not None:
            services = getattr(self._kernel, "services", {})
            for service_id, original in self._patched_services.items():
                service = services.get(service_id)
                if service is not None:
                    try:
                        service._inner_get_chat_message_contents = original
                    except Exception:
                        log.debug("layerlens: could not restore SK chat service %s", service_id)
        self._patched_services.clear()

    def _extract_usage_from_response(self, result: Any) -> Dict[str, Any]:
        """Extract token usage from ChatMessageContent list returned by _inner_get_chat_message_contents."""
        if not result:
            return {}
        msg = result[0] if isinstance(result, list) else result
        metadata = getattr(msg, "metadata", None)
        if not metadata or not isinstance(metadata, dict):
            return {}
        return self._normalize_tokens(metadata.get("usage"))

    # ------------------------------------------------------------------
    # Plugin discovery
    # ------------------------------------------------------------------

    def _discover_plugins(self, kernel: Any) -> None:
        try:
            plugins = getattr(kernel, "plugins", None)
            if plugins is None:
                return
            names = list(plugins.keys()) if hasattr(plugins, "keys") else [str(p) for p in plugins]
            collector = self._ensure_collector()
            for name in names:
                if name not in self._seen_plugins:
                    self._seen_plugins.add(name)
                    collector.emit(
                        "environment.config",
                        self._payload(plugin_name=name, event_subtype="plugin_registered"),
                        span_id=self._new_span_id(),
                        parent_span_id=self._get_root_span(),
                    )
        except Exception:
            log.debug("layerlens: error discovering SK plugins", exc_info=True)

    def _maybe_discover_plugin(self, plugin_name: str) -> None:
        if not plugin_name or plugin_name in self._seen_plugins:
            return
        with self._lock:
            if plugin_name in self._seen_plugins:
                return
            self._seen_plugins.add(plugin_name)
        collector = self._ensure_collector()
        collector.emit(
            "environment.config",
            self._payload(plugin_name=plugin_name, event_subtype="plugin_registered"),
            span_id=self._new_span_id(),
            parent_span_id=self._get_root_span(),
        )

    # ------------------------------------------------------------------
    # Shared filter logic
    # ------------------------------------------------------------------

    async def _wrap_invocation(
        self,
        context: Any,
        next: Any,
        *,
        auto_invoked: bool = False,
    ) -> None:
        """Shared wrap-and-emit logic for function and auto-function filters.

        Emits tool.call on start, tool.result on success (or agent.error on failure),
        with timing. The ``auto_invoked`` flag adds LLM-specific metadata.
        """
        plugin_name = _extract_plugin_name(context)
        function_name = _extract_function_name(context)
        tool_name = f"{plugin_name}.{function_name}" if plugin_name else function_name

        self._maybe_discover_plugin(plugin_name)

        span_id = self._new_span_id()
        root_span = self._get_root_span()
        self._start_timer(span_id)
        collector = self._ensure_collector()

        # -- Emit tool.call (start) --
        call_payload = self._payload(
            tool_name=tool_name,
            plugin_name=plugin_name,
            function_name=function_name,
        )
        if auto_invoked:
            call_payload["auto_invoked"] = True
            call_payload["request_sequence_index"] = getattr(context, "request_sequence_index", 0)
            call_payload["function_sequence_index"] = getattr(context, "function_sequence_index", 0)
            # Auto-invoked: args come from the LLM's function_call_content
            call_content = getattr(context, "function_call_content", None)
            if call_content:
                self._set_if_capturing(
                    call_payload, "input",
                    safe_serialize(getattr(call_content, "arguments", None)),
                )
        else:
            # User-invoked: args come from context.arguments
            self._set_if_capturing(
                call_payload, "input",
                safe_serialize(_extract_arguments(context)),
            )

        collector.emit(
            "tool.call", call_payload,
            span_id=span_id, parent_span_id=root_span,
            span_name=f"sk:{tool_name}",
        )

        # -- Execute --
        error = None
        try:
            await next(context)
        except Exception as exc:
            error = exc
            raise
        finally:
            latency_ms = self._stop_timer(span_id)

            if error:
                err_payload = self._payload(
                    tool_name=tool_name,
                    error=str(error),
                    error_type=type(error).__name__,
                )
                if auto_invoked:
                    err_payload["auto_invoked"] = True
                if latency_ms is not None:
                    err_payload["latency_ms"] = latency_ms
                collector.emit(
                    "agent.error", err_payload,
                    span_id=span_id, parent_span_id=root_span,
                )
            else:
                # Extract result from the appropriate field
                if auto_invoked:
                    func_result = getattr(context, "function_result", None)
                else:
                    func_result = getattr(context, "result", None)
                result_value = getattr(func_result, "value", None) if func_result else None

                result_payload = self._payload(
                    tool_name=tool_name,
                    status="ok",
                )
                if auto_invoked:
                    result_payload["auto_invoked"] = True
                if latency_ms is not None:
                    result_payload["latency_ms"] = latency_ms
                self._set_if_capturing(result_payload, "output", safe_serialize(result_value))
                collector.emit(
                    "tool.result", result_payload,
                    span_id=span_id, parent_span_id=root_span,
                    span_name=f"sk:{tool_name}",
                )

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------

    async def _function_invocation_filter(self, context: Any, next: Any) -> None:
        await self._wrap_invocation(context, next, auto_invoked=False)

    async def _prompt_rendering_filter(self, context: Any, next: Any) -> None:
        await next(context)

        function_name = _extract_function_name(context)
        rendered = getattr(context, "rendered_prompt", None)

        payload = self._payload(event_subtype="prompt_render")
        if function_name:
            payload["function_name"] = function_name
        if rendered and self._config.capture_content:
            payload["rendered_prompt"] = truncate(str(rendered), 2000)

        collector = self._ensure_collector()
        collector.emit(
            "agent.code", payload,
            span_id=self._new_span_id(), parent_span_id=self._get_root_span(),
        )

    async def _auto_function_invocation_filter(self, context: Any, next: Any) -> None:
        await self._wrap_invocation(context, next, auto_invoked=True)


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _get_filter_list(kernel: Any, filter_type: Any) -> list:
    name = filter_type.value if hasattr(filter_type, "value") else str(filter_type)
    attr_map = {
        "function_invocation": "function_invocation_filters",
        "prompt_rendering": "prompt_rendering_filters",
        "auto_function_invocation": "auto_function_invocation_filters",
    }
    return getattr(kernel, attr_map.get(name, ""), [])


def _extract_plugin_name(context: Any) -> str:
    fn = getattr(context, "function", None)
    if fn is not None:
        return getattr(fn, "plugin_name", "") or ""
    return getattr(context, "plugin_name", "") or ""


def _extract_function_name(context: Any) -> str:
    fn = getattr(context, "function", None)
    if fn is not None:
        return getattr(fn, "name", "") or ""
    return getattr(context, "function_name", "") or ""


def _extract_arguments(context: Any) -> Optional[Dict[str, Any]]:
    args = getattr(context, "arguments", None)
    if args is None:
        return None
    if isinstance(args, dict):
        return args
    if hasattr(args, "items"):
        return dict(args.items())
    return None
