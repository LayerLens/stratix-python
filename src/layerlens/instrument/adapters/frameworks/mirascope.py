"""Mirascope adapter — traced ``Call`` invocations on Mirascope v2's ``llm`` API.

Targets Mirascope 2.x (``mirascope.llm``). The 1.x ``mirascope.core.<provider>``
package the ateam reference patches does not exist on 2.x, and pinning customers
to ``mirascope<2`` to keep that seam is not an option, so the hook is designed
against the current major rather than ported symbol-for-symbol.

Mirascope offers no callback API. ``mirascope.ops`` is a first-party *tracing*
surface but not a usable hook here: it is an OpenTelemetry span emitter gated
behind the heavyweight ``mirascope[ops]`` extra, its ``configure()`` seizes the
global OTel TracerProvider (a one-shot in OTel — it would silently break, or be
broken by, a customer's own pipeline), and it instruments ``Model.call``, which
cannot see the decorated function that ``tool_name``/``agent_name`` are built
from. Wrapping the ``Call`` classes is the honest design point.

The seam is the ``call`` method of the four ``Call`` variants rather than the
``llm.call`` decorator itself, because ``@llm.call`` returns a ``Call`` OBJECT
whose ``__call__`` delegates to ``self.call``. Patching the class therefore
covers both ``fn(...)`` and ``fn.call(...)``, resolves at invocation time (so
functions already decorated at import time are traced — the common case, since
instrumentation usually runs in ``main()``), and cannot be bypassed by
``from mirascope.llm import call``.

Usage::

    adapter = instrument_mirascope(client)


    @llm.call("openai/gpt-4o-mini")
    def recommend_book(genre: str):
        return f"Recommend a {genre} book"


    recommend_book("fantasy")
    uninstrument_mirascope()
"""

from __future__ import annotations

import inspect
import logging
import functools
import threading
from typing import Any, Dict, List, Tuple, Callable, Optional

from ._utils import truncate, safe_serialize
from ..._identity import _API_METHOD_RE, _s, _is_generic
from ._base_framework import FrameworkAdapter
from ..._capture_config import CaptureConfig

log = logging.getLogger(__name__)

try:
    import mirascope.llm as _llm  # pyright: ignore[reportMissingImports]

    _HAS_MIRASCOPE = True
except ImportError:
    _llm = None  # type: ignore[assignment]
    _HAS_MIRASCOPE = False

#: The v2 ``Call`` variants. Each defines its own ``call``; the sync/async split
#: is read off the original rather than assumed by name.
_CALL_CLASS_NAMES: Tuple[str, ...] = ("Call", "AsyncCall", "ContextCall", "AsyncContextCall")

#: Dropped from every captured kwargs mapping — a credential is never telemetry.
_SECRET_KWARGS = frozenset({"api_key"})

#: Cap for embedded content, matching ``_utils.truncate``'s default.
_TEXT_LIMIT = 2000


def _call_classes() -> Tuple[type, ...]:
    """The ``Call`` classes exposed by the installed mirascope."""
    if _llm is None:
        return ()
    found: List[type] = []
    for name in _CALL_CLASS_NAMES:
        cls = getattr(_llm, name, None)
        if isinstance(cls, type):
            found.append(cls)
    return tuple(found)


def _detect_framework_version() -> Optional[str]:
    """The installed mirascope version, or None.

    2.x exposes no ``__version__`` attribute, so the distribution metadata is the
    only honest source; omitted rather than defaulted when unavailable.
    """
    try:
        from importlib.metadata import PackageNotFoundError, version

        return version("mirascope")
    except (ImportError, PackageNotFoundError):
        return None


#: The ONLY colon suffixes a v2 ``ModelId`` uses to select a transport. Mirascope's
#: own normaliser (``providers.openai.model_id.model_name``) strips exactly these
#: two and nothing else, so anything else after a colon belongs to the model name.
_TRANSPORT_SUFFIXES: Tuple[str, ...] = (":responses", ":completions")


def _bare_model_id(model_id: str) -> str:
    """``'openai/gpt-4o-mini:responses'`` -> ``'gpt-4o-mini'``.

    A v2 ``ModelId`` namespaces the provider and may carry a transport suffix.
    Neither is part of the model's name, and LayerLens pricing resolves only the
    bare id — the discarded parts ride ``provider`` and ``model_id``, so nothing
    is lost.

    Only the closed set of transport suffixes is stripped. A colon is NOT a
    generic delimiter here: ollama (and every other OpenAI-compatible provider
    that namespaces by tag) puts the tag after it, so ``ollama/llama3:8b`` names
    the model ``llama3:8b`` — mirascope itself sends that verbatim on the wire and
    the response reports it back. Dropping the tag would report ``llama3``, a
    DIFFERENT model that did not run (``llama3:8b`` vs ``llama3:70b``), and would
    silently mis-price it.
    """
    bare = model_id
    for suffix in _TRANSPORT_SUFFIXES:
        if bare.endswith(suffix):
            bare = bare[: -len(suffix)]
            break
    return bare.rpartition("/")[2] or bare


def _target_model_id(target: Any) -> Optional[str]:
    """The model the call is about to use, honouring a ``with llm.model(...)``
    override. Read BEFORE the call so it survives the error path (where there is
    no response to read ``model_id`` off)."""
    try:
        model = target.model
    except Exception:
        return None
    return _str_or_none(getattr(model, "model_id", None))


def _response_model_id(response: Any) -> Optional[str]:
    """``Response.model_id`` — the model that ACTUALLY ran.

    Never ``Response.model``: that property returns a ``Model`` OBJECT which
    defines no ``__str__``, so stringifying it ships
    ``'<mirascope.llm.models.models.Model object at 0x...>'`` as the model name.
    """
    if response is None:
        return None
    return _str_or_none(getattr(response, "model_id", None))


def _response_provider_id(response: Any) -> Optional[str]:
    """``Response.provider_id`` — honest and per-call.

    Not derived from the ``model_id`` prefix: that prefix is a model namespace,
    not a provider id (``mlx-community/...`` is served by ``mlx``). With no
    response there is no honest provider, so it is omitted.
    """
    if response is None:
        return None
    return _str_or_none(getattr(response, "provider_id", None))


def _str_or_none(value: Any) -> Optional[str]:
    return value if isinstance(value, str) and value else None


def _function_name(target: Any) -> Optional[str]:
    """The decorated function's name, which ``BaseCall`` copies onto itself."""
    return _str_or_none(getattr(target, "__name__", None))


def _honest_agent_name(function_name: Optional[str]) -> Optional[str]:
    """The function name as an Agent-column identity, or None.

    Applies the same guards ``_identity`` applies at flush so a generically named
    function (``def agent(...)``) never fabricates an agent.
    """
    name = _s(function_name)
    if name is None or _is_generic(name) or _API_METHOD_RE.match(name.lower()):
        return None
    return name


def _format_name(target: Any) -> Optional[str]:
    """The ``format=`` spec's name (v2's structured-output spec), or None.

    ``format=`` takes either the formattable type itself (``format=Car``) or a
    wrapper built by ``llm.format(Car, mode=...)`` — the latter is mandatory to
    select a formatting mode, and the only way to get typed output from a model
    without tool support. Mirascope normalises the wrapper into a ``Format`` whose
    ``.name`` is the formattable's name, so that is read FIRST: falling straight
    through to ``type(spec).__name__`` would report the literal ``"Format"`` as
    the customer's response model on every moded call.
    """
    spec = getattr(getattr(target, "prompt", None), "format", None)
    if spec is None:
        return None
    if isinstance(spec, type):
        return spec.__name__
    # ``Format.name`` is the formattable's name; ``__name__`` covers a bare
    # callable/parser spec. Only then fall back to the wrapper's own class.
    return (
        _str_or_none(getattr(spec, "name", None))
        or _str_or_none(getattr(spec, "__name__", None))
        or type(spec).__name__
    )


def _capped(value: Any) -> Any:
    return truncate(safe_serialize(value), _TEXT_LIMIT)


def _render_input(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "args": [_capped(a) for a in args],
        "kwargs": {k: _capped(v) for k, v in kwargs.items() if k not in _SECRET_KWARGS},
    }


def _render_output(response: Any) -> Any:
    if response is None:
        return None
    content = getattr(response, "content", None)
    return _capped(content if content is not None else response)


class MirascopeAdapter(FrameworkAdapter):
    """Mirascope v2 adapter — see module docstring for the instrumentation model."""

    name = "mirascope"
    package = "mirascope"

    def __init__(self, client: Any, capture_config: Optional[CaptureConfig] = None) -> None:
        super().__init__(client, capture_config)
        self._patched: Dict[type, Any] = {}
        self._adapter_lock = threading.Lock()
        self._framework_version: Optional[str] = None
        self._config_emitted = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    # target/kwargs are unused but fixed by the FrameworkAdapter signature:
    # mirascope is patched at its own Call classes, so there is no per-object
    # surface to hand in (the litellm/crewai shape).
    def _on_connect(self, target: Any = None, **kwargs: Any) -> None:  # noqa: ARG002
        self._check_dependency(_HAS_MIRASCOPE)
        self._framework_version = _detect_framework_version()
        patched = self.patch_call_classes()
        if not patched:
            # Silence here would report a healthy adapter that traces nothing.
            log.warning(
                "layerlens: no mirascope Call classes were patched; no events will be emitted "
                "(expected mirascope >= 2.0 exposing llm.Call)"
            )
        self._metadata = {"framework_version": self._framework_version, "patched_calls": patched}

    def _on_disconnect(self) -> None:
        with self._adapter_lock:
            for cls, original in self._patched.items():
                current = cls.__dict__.get("call")
                if not getattr(current, "_layerlens_traced", False):
                    continue  # someone else re-patched; leave their patch alone
                try:
                    cls.call = original
                except (AttributeError, TypeError):
                    log.warning("layerlens: could not restore %s.call", cls.__name__)
            self._patched.clear()
            self._config_emitted = False

    # ------------------------------------------------------------------
    # Public instrumentation API
    # ------------------------------------------------------------------

    def patch_call_classes(self) -> List[str]:
        """Wrap ``call`` on every v2 ``Call`` class. Returns the classes patched."""
        patched: List[str] = []
        with self._adapter_lock:
            for cls in _call_classes():
                if cls in self._patched:
                    continue
                original = cls.__dict__.get("call")
                if original is None or getattr(original, "_layerlens_traced", False):
                    continue  # never double-wrap: latency/tokens would double-count
                wrapped = self._wrap_call_method(original)
                try:
                    cls.call = wrapped
                except (AttributeError, TypeError):
                    log.warning("layerlens: could not patch %s.call", cls.__name__)
                    continue
                # Recorded only AFTER the assignment lands, so patched_calls can
                # never claim a class we did not actually instrument.
                self._patched[cls] = original
                patched.append(cls.__name__)
        return patched

    def traced_call(self, target: Any) -> Any:
        """Trace ONE ``Call`` object without patching mirascope globally.

        ``Call.__call__`` delegates through ``self.call``, so shadowing the
        method on the instance covers both ``fn(...)`` and ``fn.call(...)``::

            adapter.traced_call(recommend_book)
        """
        cls = type(target)
        original = self._pristine_call(cls)
        if original is None:
            raise TypeError(f"{cls.__name__} is not a mirascope Call: it exposes no call() method")
        if inspect.iscoroutinefunction(original):

            async def traced_async(*args: Any, **kwargs: Any) -> Any:
                return await self._invoke_async(original, target, args, kwargs)

            traced_async._layerlens_traced = True  # type: ignore[attr-defined]
            target.call = traced_async
            return target

        def traced_sync(*args: Any, **kwargs: Any) -> Any:
            return self._invoke_sync(original, target, args, kwargs)

        traced_sync._layerlens_traced = True  # type: ignore[attr-defined]
        target.call = traced_sync
        return target

    def _pristine_call(self, cls: type) -> Optional[Callable[..., Any]]:
        """The un-patched ``call`` implementation for *cls*.

        Returns our saved original when the class is already patched, so an
        instance opt-in layered on a global patch still delegates to mirascope
        exactly once.
        """
        for base in cls.__mro__:
            fn = base.__dict__.get("call")
            if fn is None:
                continue
            saved = self._patched.get(base)
            return saved if saved is not None else fn
        return None

    # ------------------------------------------------------------------
    # Wrappers
    # ------------------------------------------------------------------

    def _wrap_call_method(self, original: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap a class-level ``call``; the descriptor protocol binds *target*."""
        adapter = self

        if inspect.iscoroutinefunction(original):

            @functools.wraps(original)
            async def traced_async(target: Any, *args: Any, **kwargs: Any) -> Any:
                return await adapter._invoke_async(original, target, args, kwargs)

            traced_async._layerlens_traced = True  # type: ignore[attr-defined]
            return traced_async

        @functools.wraps(original)
        def traced_sync(target: Any, *args: Any, **kwargs: Any) -> Any:
            return adapter._invoke_sync(original, target, args, kwargs)

        traced_sync._layerlens_traced = True  # type: ignore[attr-defined]
        return traced_sync

    # ------------------------------------------------------------------
    # Invocation core
    # ------------------------------------------------------------------

    def _invoke_sync(
        self,
        original: Callable[..., Any],
        target: Any,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        self._begin_run()
        model_hint = _target_model_id(target)
        error: Optional[BaseException] = None
        response: Any = None
        self._start_timer("call")
        try:
            response = original(target, *args, **kwargs)
            return response
        except BaseException as exc:
            error = exc
            raise
        finally:
            # Emission lives in the finally so a failed call stays visible.
            self._emit_call_events(
                target=target,
                model_hint=model_hint,
                args=args,
                kwargs=kwargs,
                response=response,
                error=error,
            )
            self._end_run()

    async def _invoke_async(
        self,
        original: Callable[..., Any],
        target: Any,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        self._begin_run()
        model_hint = _target_model_id(target)
        error: Optional[BaseException] = None
        response: Any = None
        self._start_timer("call")
        try:
            response = await original(target, *args, **kwargs)
            return response
        except BaseException as exc:
            error = exc
            raise
        finally:
            self._emit_call_events(
                target=target,
                model_hint=model_hint,
                args=args,
                kwargs=kwargs,
                response=response,
                error=error,
            )
            self._end_run()

    # ------------------------------------------------------------------
    # Event emission
    # ------------------------------------------------------------------

    def _emit_call_events(
        self,
        *,
        target: Any,
        model_hint: Optional[str],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        response: Any,
        error: Optional[BaseException],
    ) -> None:
        latency_ms = self._stop_timer("call")
        self._emit_settings_once()

        function_name = _function_name(target)
        response_model = _format_name(target)
        root = self._get_root_span()
        label = function_name or "call"
        span_name = f"mirascope:{label}"
        tool_name = f"mirascope.{label}"
        success = error is None

        call_payload = self._payload(tool_name=tool_name, success=success)
        self._stamp_identity(call_payload, function_name)
        if response_model:
            call_payload["response_model"] = response_model
        if latency_ms is not None:
            call_payload["latency_ms"] = latency_ms
        if error is not None:
            call_payload["error_type"] = type(error).__name__
        self._set_if_capturing(call_payload, "input", _render_input(args, kwargs))
        self._emit(
            "tool.call",
            call_payload,
            span_id=self._new_span_id(),
            parent_span_id=root,
            span_name=span_name,
        )

        result_payload = self._payload(tool_name=tool_name, success=success)
        self._stamp_identity(result_payload, function_name)
        if latency_ms is not None:
            result_payload["latency_ms"] = latency_ms
        if error is not None:
            # error_type is metadata and survives redaction; the message can
            # quote the prompt, so it is gated as content.
            result_payload["error_type"] = type(error).__name__
            self._set_if_capturing(result_payload, "error", truncate(str(error), _TEXT_LIMIT))
        else:
            self._set_if_capturing(result_payload, "output", _render_output(response))
        self._emit(
            "tool.result",
            result_payload,
            span_id=self._new_span_id(),
            parent_span_id=root,
            span_name=span_name,
        )

        model_id = _response_model_id(response) or model_hint
        if not model_id:
            # A placeholder would be a fabricated model name, and the model is
            # required at ingest. The call is still visible via the tool.call
            # above.
            log.debug(
                "layerlens: no real model known for mirascope.%s — skipping model.invoke (tool.call still emitted)",
                label,
            )
            return

        model = _bare_model_id(model_id)
        invoke_payload = self._payload(model=model, model_id=model_id)
        self._stamp_identity(invoke_payload, function_name)
        if function_name:
            invoke_payload["function_name"] = function_name
        provider = _response_provider_id(response)
        if provider:
            invoke_payload["provider"] = provider
        if response_model:
            invoke_payload["response_model"] = response_model
        if latency_ms is not None:
            invoke_payload["latency_ms"] = latency_ms
        tokens = self._normalize_tokens(getattr(response, "usage", None))
        invoke_payload.update(tokens)
        if error is not None:
            invoke_payload["error_type"] = type(error).__name__
            self._set_if_capturing(invoke_payload, "error", truncate(str(error), _TEXT_LIMIT))
        self._emit(
            "model.invoke",
            invoke_payload,
            span_id=self._new_span_id(),
            parent_span_id=root,
            span_name=span_name,
        )

        if tokens:
            cost_payload = self._payload(model=model)
            # A spend row that cannot say who was billed is unattributable, so it
            # carries the same resolved provider as the model.invoke beside it.
            if provider:
                cost_payload["provider"] = provider
            cost_payload.update(tokens)
            self._emit("cost.record", cost_payload)

    def _stamp_identity(self, payload: Dict[str, Any], function_name: Optional[str]) -> None:
        """Stamp the decorated function as the call's identity.

        ``agent_id`` mirrors the reference contract; ``agent_name`` is the key
        the Agent column actually resolves from, and is omitted when the name is
        not an honest identity.
        """
        if not function_name:
            return
        payload["agent_id"] = function_name
        agent_name = _honest_agent_name(function_name)
        if agent_name:
            payload["agent_name"] = agent_name

    def _emit_settings_once(self) -> None:
        """Emit ``environment.config`` once, on the first traced invocation.

        Not emitted from connect(): ``_emit`` routes through the run's collector,
        which only exists inside a ``_begin_run`` scope.
        """
        with self._adapter_lock:
            if self._config_emitted:
                return
            self._config_emitted = True
            # Built under the lock — a concurrent patch would otherwise mutate
            # _patched mid-iteration.
            patched_calls = sorted(cls.__name__ for cls in self._patched)
        if not patched_calls:
            return  # nothing was patched, so claim no configuration
        config: Dict[str, Any] = {"patched_calls": patched_calls}
        if self._framework_version:
            config["framework_version"] = self._framework_version
        self._emit(
            "environment.config",
            self._payload(config=config),
            span_id=self._new_span_id(),
            parent_span_id=self._get_root_span(),
            span_name="mirascope:config",
        )


def instrument_mirascope(
    client: Any,
    *,
    capture_config: Optional[CaptureConfig] = None,
) -> MirascopeAdapter:
    """Trace every Mirascope ``@llm.call`` function in the process.

    Args:
        client: The layerlens client events are uploaded through.
        capture_config: Optional capture configuration.
    """
    from .._registry import get, register

    existing = get("mirascope")
    if existing is not None:
        existing.disconnect()
    adapter = MirascopeAdapter(client, capture_config)
    adapter.connect()
    register("mirascope", adapter)
    return adapter


def uninstrument_mirascope() -> None:
    from .._registry import unregister

    unregister("mirascope")


__all__ = ["MirascopeAdapter", "instrument_mirascope", "uninstrument_mirascope"]
