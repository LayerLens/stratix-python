from __future__ import annotations

import time
import logging
import threading
from typing import Any, Dict, Iterator, Optional
from contextlib import contextmanager

from ._utils import safe_serialize
from ..._identity import _API_METHOD_RE, _is_generic
from ._base_framework import FrameworkAdapter
from ..._capture_config import CaptureConfig

log = logging.getLogger(__name__)

_HAS_HAYSTACK = False
try:
    from haystack import tracing as _hs_tracing  # pyright: ignore[reportMissingImports]

    _HAS_HAYSTACK = True
except ImportError:
    _hs_tracing = None  # type: ignore[assignment]

_GENERATOR_KEYWORDS = ("generator", "chatgenerator", "llm")


class HaystackAdapter(FrameworkAdapter):
    """Haystack 2.x adapter via global tracer replacement.

    Replaces ``haystack.tracing.tracer.actual_tracer`` with a thin
    ``_LayerLensTracer`` that delegates all event emission back to
    this adapter.  Each ``Pipeline.run()`` gets its own collector
    via ``_begin_run`` / ``_end_run``.

    Usage::

        adapter = HaystackAdapter(client)
        adapter.connect()
        result = pipeline.run(...)
        adapter.disconnect()
    """

    name = "haystack"

    def __init__(self, client: Any, capture_config: Optional[CaptureConfig] = None) -> None:
        super().__init__(client, capture_config)
        self._original_tracer: Any = None
        self._tracer: Optional[_LayerLensTracer] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _on_connect(self, target: Any = None, **kwargs: Any) -> None:
        self._check_dependency(_HAS_HAYSTACK)
        self._metadata["framework_version"] = _get_version()
        self._original_tracer = _hs_tracing.tracer.actual_tracer
        self._tracer = _LayerLensTracer(self)
        _hs_tracing.tracer.actual_tracer = self._tracer

    def _on_disconnect(self) -> None:
        if _HAS_HAYSTACK and self._original_tracer is not None:
            try:
                _hs_tracing.tracer.actual_tracer = self._original_tracer
            except Exception:
                log.debug("layerlens: failed to restore Haystack tracer", exc_info=True)
            self._original_tracer = None
        self._tracer = None

    # ------------------------------------------------------------------
    # Span handlers (called by _LayerLensSpan._finish)
    # ------------------------------------------------------------------

    def _on_span_end(self, span: _LayerLensSpan) -> None:
        elapsed_ms = (time.time_ns() - span._start_ns) / 1_000_000
        try:
            if span._is_pipeline:
                self._on_pipeline_end(span, elapsed_ms)
            elif span._operation_name == "haystack.component.run":
                self._on_component_end(span, elapsed_ms)
        except Exception:
            log.warning("layerlens: error emitting Haystack span", exc_info=True)

    def _on_pipeline_end(self, span: _LayerLensSpan, elapsed_ms: float) -> None:
        tags = span._all_tags()
        root = self._get_root_span()

        inp = self._payload()
        self._set_if_capturing(inp, "input", safe_serialize(tags.get("haystack.pipeline.input_data")))
        max_runs = tags.get("haystack.pipeline.max_runs_per_component")
        if max_runs is not None:
            inp["max_runs_per_component"] = max_runs
        self._emit(
            "agent.input",
            inp,
            span_id=root,
            parent_span_id=None,
            span_name="haystack:pipeline",
        )

        # Re-emit the pipeline shape so the server graph engine can render each
        # named component as a node. Only the honest, producer-declared
        # component names observed during this run are listed — an unnamed /
        # generic component is never fabricated into a node.
        components = self._collected_components()
        if components:
            self._emit(
                "environment.config",
                self._payload(components=components),
                span_id=root,
                parent_span_id=None,
                span_name="haystack:pipeline",
            )

        out = self._payload(latency_ms=elapsed_ms)
        self._set_if_capturing(out, "output", safe_serialize(tags.get("haystack.pipeline.output_data")))
        if tags.get("error"):
            out["error"] = str(tags.get("error.message", "unknown"))
        self._emit(
            "agent.output",
            out,
            span_id=root,
            parent_span_id=None,
            span_name="haystack:pipeline",
        )

        self._end_run()

    def _on_component_end(self, span: _LayerLensSpan, elapsed_ms: float) -> None:
        tags = span._all_tags()
        comp_type = str(tags.get("haystack.component.type", ""))
        comp_name = str(tags.get("haystack.component.name", "unknown"))

        # ``component_name`` is the field the server graph engine reads to build
        # a haystack node. The instance name a developer gave the component
        # (``retriever``, ``llm``) is producer-declared and honest; a missing
        # name (the ``"unknown"`` default) or any generic/API-method label is
        # NOT — it must stay blank rather than fabricate a node identity.
        honest_name = _honest_component_name(comp_name)
        if honest_name is not None:
            self._record_component(honest_name, comp_type)

        if any(kw in comp_type.lower() for kw in _GENERATOR_KEYWORDS):
            self._on_generator_end(span, elapsed_ms, tags, comp_name, comp_type, honest_name)
        else:
            self._on_tool_end(span, elapsed_ms, tags, comp_name, comp_type, honest_name)

    def _record_component(self, name: str, comp_type: str) -> None:
        """Accumulate a distinct honest component for this run's ``environment.config``."""
        run = self._get_run()
        if run is None:
            return
        seen = run.data.setdefault("haystack_components", [])
        if not any(c["name"] == name for c in seen):
            entry: Dict[str, Any] = {"name": name}
            if comp_type:
                entry["component_class"] = comp_type
            seen.append(entry)

    def _collected_components(self) -> list:
        run = self._get_run()
        if run is None:
            return []
        return list(run.data.get("haystack_components", []))

    @staticmethod
    def _stamp_component(payload: Dict[str, Any], honest_name: Optional[str], comp_type: str) -> None:
        """Stamp the honest graph-node identity on a component event.

        ``component_name`` is what the server graph engine reads for haystack, so
        it is set ONLY when the developer declared a real component name (never a
        generic/API-method label). ``component_class`` is the component's class
        (``OpenAIChatGenerator``) — carried as metadata, never as the node id.
        """
        if honest_name is not None:
            payload["component_name"] = honest_name
        if comp_type:
            payload["component_class"] = comp_type

    def _on_generator_end(
        self,
        span: _LayerLensSpan,
        elapsed_ms: float,
        tags: Dict[str, Any],
        name: str,
        comp_type: str,
        honest_name: Optional[str] = None,
    ) -> None:
        model = _extract_model(tags)
        output = tags.get("haystack.component.output", {})
        tokens = self._normalize_tokens(_extract_usage(output))

        payload = self._payload(component_type=comp_type, latency_ms=elapsed_ms)
        self._stamp_component(payload, honest_name, comp_type)
        if model:
            payload["model"] = model
        fr = _extract_finish_reason(output)
        if fr:
            payload["finish_reason"] = str(fr)
        payload.update(tokens)
        self._set_if_capturing(payload, "input", safe_serialize(tags.get("haystack.component.input")))
        if isinstance(output, dict) and "replies" in output:
            self._set_if_capturing(payload, "output", safe_serialize(output["replies"]))
        self._emit(
            "model.invoke",
            payload,
            span_id=span.span_id,
            parent_span_id=span._parent_span_id,
            span_name=f"component:{name}",
        )

        if tokens:
            cost = self._payload(**tokens)
            if model:
                cost["model"] = model
            self._emit("cost.record", cost, parent_span_id=span.span_id)

    def _on_tool_end(
        self,
        span: _LayerLensSpan,
        elapsed_ms: float,
        tags: Dict[str, Any],
        name: str,
        comp_type: str,
        honest_name: Optional[str] = None,
    ) -> None:
        call = self._payload(tool_name=name, component_type=comp_type)
        self._stamp_component(call, honest_name, comp_type)
        self._set_if_capturing(call, "input", safe_serialize(tags.get("haystack.component.input")))
        self._emit(
            "tool.call",
            call,
            span_id=span.span_id,
            parent_span_id=span._parent_span_id,
            span_name=f"component:{name}",
        )

        result = self._payload(tool_name=name, component_type=comp_type, latency_ms=elapsed_ms)
        self._stamp_component(result, honest_name, comp_type)
        self._set_if_capturing(result, "output", safe_serialize(tags.get("haystack.component.output")))
        if tags.get("error"):
            result["error"] = str(tags.get("error.message", "unknown"))
        self._emit(
            "tool.result",
            result,
            span_id=span.span_id,
            parent_span_id=span._parent_span_id,
            span_name=f"component:{name}",
        )


# ---------------------------------------------------------------------------
# Thin protocol implementations (Tracer + Span)
# ---------------------------------------------------------------------------


class _LayerLensTracer:
    """Minimal Haystack ``Tracer`` — manages the thread-local span stack
    and delegates all event logic to the adapter."""

    def __init__(self, adapter: HaystackAdapter) -> None:
        self._adapter = adapter
        self._local = threading.local()

    @contextmanager
    def trace(
        self,
        operation_name: str,
        tags: Optional[Dict[str, Any]] = None,
        parent_span: Optional[Any] = None,
    ) -> Iterator[_LayerLensSpan]:
        if parent_span is None:
            parent_span = getattr(self._local, "current_span", None)

        is_pipeline = operation_name == "haystack.pipeline.run"
        if is_pipeline:
            self._adapter._begin_run()

        span = _LayerLensSpan(
            self._adapter,
            operation_name,
            (self._adapter._get_root_span() if is_pipeline else self._adapter._new_span_id()),
            getattr(parent_span, "span_id", None),
            tags or {},
            is_pipeline,
        )

        prev = getattr(self._local, "current_span", None)
        self._local.current_span = span
        try:
            yield span
        except Exception as exc:
            span.set_tag("error", True)
            span.set_tag("error.message", str(exc))
            raise
        finally:
            span._finish()
            self._local.current_span = prev

    def current_span(self) -> Any:
        return getattr(self._local, "current_span", None) or _NullSpan()


class _NullSpan:
    """No-op span returned outside an active trace."""

    def set_tag(self, key: str, value: Any) -> None:
        pass

    def set_content_tag(self, key: str, value: Any) -> None:
        pass

    def raw_span(self) -> None:
        return None

    def get_correlation_data_for_logs(self) -> Dict[str, Any]:
        return {}


class _LayerLensSpan:
    """Tag accumulator implementing the Haystack ``Span`` protocol.
    Delegates to ``adapter._on_span_end`` on finish."""

    def __init__(
        self,
        adapter: HaystackAdapter,
        operation_name: str,
        span_id: str,
        parent_span_id: Optional[str],
        tags: Dict[str, Any],
        is_pipeline: bool,
    ) -> None:
        self._adapter = adapter
        self._operation_name = operation_name
        self.span_id = span_id
        self._parent_span_id = parent_span_id
        self._tags: Dict[str, Any] = dict(tags)
        self._content_tags: Dict[str, Any] = {}
        self._start_ns = time.time_ns()
        self._is_pipeline = is_pipeline

    def set_tag(self, key: str, value: Any) -> None:
        self._tags[key] = value

    def set_content_tag(self, key: str, value: Any) -> None:
        self._content_tags[key] = value

    def raw_span(self) -> None:
        return None

    def get_correlation_data_for_logs(self) -> Dict[str, Any]:
        return {"span_id": self.span_id, "operation_name": self._operation_name}

    def _all_tags(self) -> Dict[str, Any]:
        return {**self._tags, **self._content_tags}

    def _finish(self) -> None:
        self._adapter._on_span_end(self)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _honest_component_name(name: Any) -> Optional[str]:
    """The producer-declared component name, honest-guarded — or None.

    Returns the trimmed component name a developer gave the pipeline component,
    unless it is empty, the ``"unknown"`` default (an unset name), any generic
    class/placeholder label, or a dotted API-method label — none of which is a
    real, producer-declared node identity. Reuses the shared anti-fabrication
    guard so the haystack graph node is never invented.
    """
    if not isinstance(name, str):
        return None
    n = name.strip()
    if not n or _is_generic(n) or _API_METHOD_RE.match(n.lower()):
        return None
    return n


def _extract_model(tags: Dict[str, Any]) -> Optional[str]:
    model = tags.get("haystack.model")
    if model:
        return str(model)
    output = tags.get("haystack.component.output", {})
    if isinstance(output, dict):
        meta_list = output.get("meta")
        if isinstance(meta_list, list) and meta_list:
            m = meta_list[0].get("model") if isinstance(meta_list[0], dict) else None
            if m:
                return str(m)
    return None


def _extract_usage(output: Any) -> Optional[Dict[str, int]]:
    if not isinstance(output, dict):
        return None
    meta_list = output.get("meta")
    if isinstance(meta_list, list) and meta_list and isinstance(meta_list[0], dict):
        return meta_list[0].get("usage")
    return None


def _extract_finish_reason(output: Any) -> Optional[str]:
    if not isinstance(output, dict):
        return None
    meta_list = output.get("meta")
    if isinstance(meta_list, list) and meta_list and isinstance(meta_list[0], dict):
        return meta_list[0].get("finish_reason")
    return None


def _get_version() -> str:
    try:
        import haystack as _mod  # pyright: ignore[reportMissingImports]

        return getattr(_mod, "__version__", "unknown")
    except Exception:
        return "unknown"
