"""Marvin adapter — traced wrappers monkey-patched onto the marvin module.

Marvin's primitives (``classify`` / ``extract`` / ``cast`` / ``generate`` /
``fn``, plus their ``*_async`` twins) are module-level functions, not class
methods, so there is no instance to wrap and no per-call hook to register: the
module attributes themselves are the only surface a customer reaches. This
adapter swaps them for wrappers that delegate to the originals and restores
them on ``disconnect()``.

``marvin.fn`` is a decorator AND a decorator factory; both forms are handled,
and NOTHING is emitted at decoration time — only calling the decorated function
traces it. A function decorated while connected keeps its traced wrapper for
life (the closure holds the adapter), so ``disconnect()`` cannot un-instrument
it; the wrappers check ``_connected`` and pass straight through instead. That
check is load-bearing, not a nicety: ``_begin_run()`` CREATES a collector when
none is active, so a still-wrapped function would otherwise flush and UPLOAD a
trace after the customer disconnected.

Marvin 3.x runs on pydantic-ai. Where a provider adapter is active UNDER a given
call, that layer reports the request with real token counts; this adapter then
omits its own tokenless ``model.invoke`` for THAT call rather than double-count
it. The dedup is scoped to the call's own root span, never to the trace — N
concurrent calls share one collector under ``trace_context()``, and a trace-wide
test would make each of them mistake a sibling's ``model.invoke`` for its own
deeper layer. Marvin itself surfaces no usage on its primitives, so no
``cost.record`` is ever emitted from here — the pricing hook needs tokens that do
not exist at this layer.

Model resolution deliberately never reads ``kwargs["model"]``: no Marvin 3.x
primitive accepts ``model=``, so on a ``@marvin.fn``-decorated function those
kwargs are the CALLER'S OWN arguments — ``def spec(model: str)`` about car
models would stamp "Civic" as the LLM model name. The real per-call seam is
``agent=``; ``marvin.defaults`` / ``marvin.settings`` back it up. Nothing
resolves -> no ``model.invoke`` at all, because ``model`` is required at ingest
and a placeholder would be a fabricated model name. The ``tool.call`` still
carries the call either way.

Usage::

    adapter = instrument_marvin(client)
    marvin.classify("This is great", labels=["positive", "negative"])
    uninstrument_marvin()
"""

from __future__ import annotations

import time
import uuid
import inspect
import logging
import functools
import importlib.util
from typing import Any, Dict, Tuple, Callable, Optional, FrozenSet
from threading import Lock
from dataclasses import field, dataclass

from ._utils import truncate, safe_serialize
from ..._context import _current_collector
from ..._identity import _API_METHOD_RE, _s, _is_generic
from ._base_framework import FrameworkAdapter
from ..._capture_config import CaptureConfig

log = logging.getLogger(__name__)

# ``import marvin`` calls ensure_db_tables_exist() at module scope
# (marvin/__init__.py), CREATING a SQLite database as an import side effect.
# Probe the spec instead, so importing this adapter — or the registry's
# discover_installed() — never touches the filesystem. The real import happens
# in _on_connect, where the caller has asked for instrumentation.
_HAS_MARVIN = importlib.util.find_spec("marvin") is not None

#: Primitives patched as direct callables. The ``*_async`` twins are SEPARATE
#: module-level coroutine functions in marvin 3.x (not async variants reached
#: through the sync names), so omitting them would leave the entire async
#: surface untraced.
_MARVIN_PRIMITIVES: Tuple[str, ...] = (
    "classify",
    "extract",
    "cast",
    "generate",
    "classify_async",
    "extract_async",
    "cast_async",
    "generate_async",
)

#: Attribute stamped on a traced wrapper (sentinel + escape hatch to the original).
_TRACED_ATTR = "_layerlens_traced"
_ORIGINAL_ATTR = "_layerlens_original"

#: Per-value content caps, mirroring ateam's normalisers.
_ARG_LIMIT = 400
_RESPONSE_LIMIT = 1000

#: Call kwargs never echoed onto a payload, at any capture level.
_EXCLUDED_KWARGS: FrozenSet[str] = frozenset({"api_key"})


def _base_primitive(primitive: str) -> str:
    """The primitive's sync name — ``classify_async`` takes the same arguments as
    ``classify``, so target/label extraction keys off the shared base."""
    suffix = "_async"
    return primitive[: -len(suffix)] if primitive.endswith(suffix) else primitive


def _model_to_str(value: Any) -> Optional[str]:
    """Coerce a Marvin model setting to a model-name string, or None.

    Marvin accepts a plain string (``"openai:gpt-4o"``) or a pydantic-ai
    ``Model`` instance exposing ``model_name``. Anything else resolves to None —
    never a placeholder.
    """
    if isinstance(value, str):
        return value or None
    model_name = getattr(value, "model_name", None)
    if isinstance(model_name, str) and model_name:
        return model_name
    return None


def _agent_model(agent: Any) -> Optional[str]:
    """The model a ``marvin.Agent`` will actually use, or None.

    ``Agent.model`` is the developer's explicit override; ``Agent.get_model()``
    is marvin's own answer (``self.model or marvin.defaults.model``). Both are
    real configured models, so either is honest to report.
    """
    if agent is None:
        return None
    resolved = _model_to_str(getattr(agent, "model", None))
    if resolved:
        return resolved
    getter = getattr(agent, "get_model", None)
    if not callable(getter):
        return None
    try:
        return _model_to_str(getter())
    except Exception:
        log.debug("layerlens: marvin Agent.get_model() failed; no model reported", exc_info=True)
        return None


#: Memo for :func:`_default_agent_names` — marvin's name pool is a module
#: constant, so it is read once per process.
_AGENT_NAME_POOL: Dict[str, FrozenSet[str]] = {}


def _default_agent_names() -> FrozenSet[str]:
    """Marvin's own pool of auto-assigned Agent names, lower-cased.

    An unnamed ``marvin.Agent()`` is given a RANDOM name out of
    ``marvin.agents.names.AGENT_NAMES`` ("HAL 9000", "KITT", …) — a different one
    on every construction. Read from marvin itself rather than hardcoded so the
    guard tracks the installed version.
    """
    cached = _AGENT_NAME_POOL.get("names")
    if cached is not None:
        return cached
    try:
        from marvin.agents.names import AGENT_NAMES  # pyright: ignore[reportMissingImports]

        names = frozenset(str(name).strip().lower() for name in AGENT_NAMES)
    except Exception:
        log.debug("layerlens: marvin's default agent-name pool is unreadable", exc_info=True)
        names = frozenset()
    _AGENT_NAME_POOL["names"] = names
    return names


def _honest_agent_name(agent: Any) -> Optional[str]:
    """A DEVELOPER-declared ``marvin.Agent`` name, or None.

    Marvin auto-names an unnamed agent from its own random pool, so that name is
    a framework default that changes per construction — not a producer-declared
    identity, and fabrication if it reached the Agent column. (A developer who
    genuinely names an agent "HAL 9000" is suppressed too; omitting a real name
    is recoverable, inventing one is not.) The generic/API-method guards are
    _identity.py's own, so this can never disagree with the resolver downstream.
    """
    name = _s(getattr(agent, "name", None))
    if name is None:
        return None
    lowered = name.lower()
    if lowered in _default_agent_names() or _is_generic(name) or _API_METHOD_RE.match(lowered):
        return None
    return name


def _describe_response_model(target: Any) -> Optional[str]:
    """Resolve an ``extract``/``cast``/``generate`` target to a class name."""
    if target is None:
        return None
    if isinstance(target, type):
        return target.__name__
    name = getattr(target, "__name__", None)
    if name:
        return str(name)
    return type(target).__name__


def _describe_labels(labels: Any) -> Optional[str]:
    """Honest description of a ``classify()`` label set (NOT a response model).

    An Enum/type gets its class name; a list/tuple/set gets a short preview.
    Nothing describable (an EMPTY set) resolves to None so the caller omits the
    key outright — an empty ``labels: ""`` would be a placeholder, not a fact.

    The result is CONTENT at every branch and must only ever reach a payload
    through ``_set_if_capturing``: a label set is the caller's own
    classification taxonomy — ``labels=["billing dispute acct 4429", "patient
    consented"]`` is real customer data, and even the Enum branch reports a
    symbol the customer chose to name their categories with.
    """
    if labels is None:
        return None
    if isinstance(labels, type):
        return labels.__name__ or None
    if isinstance(labels, (list, tuple, set, frozenset)):
        return ", ".join(str(v) for v in list(labels)[:10]) or None
    return str(labels) or None


def _detect_marvin_model(module: Any) -> Optional[str]:
    """The model Marvin is configured to use, across 3.x and 2.x, or None.

    Probes ``defaults.model`` then ``settings.agent_model`` (3.x), then the 2.x
    settings tree. Returns None when nothing real is discoverable — callers must
    then OMIT model telemetry rather than fabricate a name.
    """
    if module is None:
        return None
    defaults = getattr(module, "defaults", None)
    if defaults is not None:
        resolved = _model_to_str(getattr(defaults, "model", None))
        if resolved:
            return resolved
    settings = getattr(module, "settings", None)
    if settings is None:
        return None
    resolved = _model_to_str(getattr(settings, "agent_model", None))
    if resolved:
        return resolved
    for path in (
        ("openai", "chat", "completions", "model"),
        ("openai", "completions", "model"),
        ("anthropic", "model"),
        ("azure_openai", "model"),
        ("default_model",),
    ):
        current: Any = settings
        for attr in path:
            current = getattr(current, attr, None)
            if current is None:
                break
        resolved = _model_to_str(current)
        if resolved:
            return resolved
    return None


def _detect_marvin_settings(module: Any) -> Dict[str, Any]:
    """Marvin's active provider/model as an ``environment.config`` config dict.

    Keys are omitted rather than defaulted: an install with nothing discoverable
    yields the honest near-empty ``{"framework": "marvin"}``.
    """
    config: Dict[str, Any] = {"framework": "marvin"}
    model = _detect_marvin_model(module)
    if model:
        config["model"] = model
    settings = getattr(module, "settings", None)
    if settings is not None:
        provider = getattr(settings, "provider", None) or getattr(settings, "default_provider", None)
        if provider is not None:
            config["provider"] = str(provider)
    return config


@dataclass
class _CallContext:
    """Everything resolved BEFORE a primitive runs, carried into the emit path."""

    run_id: str
    span_id: str
    primitive: str
    fn_name: str
    model: Optional[str]
    agent_name: Optional[str]
    response_model: Optional[str]
    labels: Optional[str]
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any] = field(default_factory=dict)
    start_ns: int = 0
    #: This call's own root span — the scope the deeper-layer dedup keys off.
    root_span_id: str = ""


class MarvinAdapter(FrameworkAdapter):
    """Marvin adapter — see module docstring for the instrumentation model."""

    name = "marvin"
    package = "marvin"

    def __init__(self, client: Any, capture_config: Optional[CaptureConfig] = None) -> None:
        super().__init__(client, capture_config)
        # Bounded by construction: at most one entry per patched module attribute
        # (8 primitives + fn), and the patched owner is the marvin module itself.
        self._patched: Dict[str, Tuple[Any, str, Any]] = {}
        self._patch_lock = Lock()
        self._marvin_module: Any = None
        self._config_emitted = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _on_connect(self, target: Any = None, **_kwargs: Any) -> Any:
        module = target
        if module is None:
            # Only import when we must resolve the module ourselves; a caller who
            # supplied one has already provided the framework.
            self._check_dependency(_HAS_MARVIN)
            import marvin  # pyright: ignore[reportMissingImports]

            module = marvin
        self._marvin_module = module

        version = getattr(module, "__version__", None)
        if version:
            self._metadata["framework_version"] = str(version)
        model = _detect_marvin_model(module)
        if model:
            self._metadata["model"] = model

        with self._patch_lock:
            for primitive in _MARVIN_PRIMITIVES:
                self._patch_attr(module, primitive, self._wrap_primitive)
            self._patch_attr(module, "fn", lambda _name, original: self._wrap_fn_decorator(original))
        return module

    def _on_disconnect(self) -> None:
        with self._patch_lock:
            for owner, attr_name, original in self._patched.values():
                current = getattr(owner, attr_name, None)
                if not getattr(current, _TRACED_ATTR, False):
                    # A third party re-patched this attribute after us — leave their
                    # wrapper alone rather than silently clobbering it.
                    continue
                try:
                    setattr(owner, attr_name, original)
                except (AttributeError, TypeError):
                    log.warning("layerlens: could not restore marvin.%s", attr_name, exc_info=True)
            self._patched.clear()
            self._config_emitted = False
            self._marvin_module = None

    def _patch_attr(self, module: Any, attr_name: str, build: Callable[[str, Any], Any]) -> None:
        """Swap ``module.<attr_name>`` for a traced wrapper. Caller holds the lock."""
        original = getattr(module, attr_name, None)
        if original is None or not callable(original):
            return
        key = f"marvin.{attr_name}"
        if key in self._patched or getattr(original, _TRACED_ATTR, False):
            return
        wrapped = build(attr_name, original)
        self._patched[key] = (module, attr_name, original)
        try:
            setattr(module, attr_name, wrapped)
        except (AttributeError, TypeError):
            self._patched.pop(key, None)
            log.warning(
                "layerlens: could not install the marvin wrapper on %s; calls through it will NOT be traced",
                key,
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Wrappers
    # ------------------------------------------------------------------

    def _wrap_primitive(self, primitive: str, original: Any) -> Any:
        if inspect.iscoroutinefunction(original):

            @functools.wraps(original)
            async def traced_async(*args: Any, **kwargs: Any) -> Any:
                return await self._invoke_async(primitive=primitive, original=original, args=args, kwargs=kwargs)

            setattr(traced_async, _ORIGINAL_ATTR, original)
            setattr(traced_async, _TRACED_ATTR, True)
            return traced_async

        @functools.wraps(original)
        def traced_sync(*args: Any, **kwargs: Any) -> Any:
            return self._invoke_sync(primitive=primitive, original=original, args=args, kwargs=kwargs)

        setattr(traced_sync, _ORIGINAL_ATTR, original)
        setattr(traced_sync, _TRACED_ATTR, True)
        return traced_sync

    def _wrap_fn_decorator(self, original_fn: Any) -> Any:
        """Wrap ``marvin.fn``, handling BOTH decoration forms.

        Bare ``@marvin.fn`` calls ``fn(func)`` and gets the decorated function
        back — wrap it. Parametrized ``@marvin.fn(agent=...)`` returns a
        DECORATOR — wrap that, so the function it is eventually applied to is the
        thing traced. Neither form emits anything at decoration time.
        """

        @functools.wraps(original_fn)
        def traced_fn(*decorator_args: Any, **decorator_kwargs: Any) -> Any:
            if len(decorator_args) == 1 and callable(decorator_args[0]) and not decorator_kwargs:
                decorated = original_fn(decorator_args[0])
                if not callable(decorated):
                    return decorated
                return self._wrap_decorated(decorated, model_hint=None, agent_hint=None)

            inner_decorator = original_fn(*decorator_args, **decorator_kwargs)
            if not callable(inner_decorator):
                return inner_decorator
            # ``agent=`` is the decorator's real, developer-declared config — the
            # only honest model/identity source for an @marvin.fn call, since the
            # function's own kwargs belong to the caller.
            declared_agent = decorator_kwargs.get("agent")
            model_hint = _agent_model(declared_agent)
            agent_hint = _honest_agent_name(declared_agent)

            def applied(func: Callable[..., Any]) -> Any:
                decorated = inner_decorator(func)
                if not callable(decorated):
                    return decorated
                return self._wrap_decorated(decorated, model_hint=model_hint, agent_hint=agent_hint)

            return applied

        setattr(traced_fn, _ORIGINAL_ATTR, original_fn)
        setattr(traced_fn, _TRACED_ATTR, True)
        return traced_fn

    def _wrap_decorated(
        self,
        decorated: Any,
        *,
        model_hint: Optional[str],
        agent_hint: Optional[str],
    ) -> Any:
        """Wrap the FINAL ``@marvin.fn``-decorated function with tracing."""
        fn_name = getattr(decorated, "__name__", "fn")

        if inspect.iscoroutinefunction(decorated):

            @functools.wraps(decorated)
            async def traced_decorated_async(*args: Any, **kwargs: Any) -> Any:
                return await self._invoke_async(
                    primitive="fn",
                    original=decorated,
                    args=args,
                    kwargs=kwargs,
                    function_name=fn_name,
                    model_hint=model_hint,
                    agent_hint=agent_hint,
                )

            setattr(traced_decorated_async, _TRACED_ATTR, True)
            return traced_decorated_async

        @functools.wraps(decorated)
        def traced_decorated_sync(*args: Any, **kwargs: Any) -> Any:
            return self._invoke_sync(
                primitive="fn",
                original=decorated,
                args=args,
                kwargs=kwargs,
                function_name=fn_name,
                model_hint=model_hint,
                agent_hint=agent_hint,
            )

        setattr(traced_decorated_sync, _TRACED_ATTR, True)
        return traced_decorated_sync

    # ------------------------------------------------------------------
    # Invocation core
    # ------------------------------------------------------------------

    def _invoke_sync(
        self,
        *,
        primitive: str,
        original: Any,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        function_name: Optional[str] = None,
        model_hint: Optional[str] = None,
        agent_hint: Optional[str] = None,
    ) -> Any:
        if not self._connected:
            return original(*args, **kwargs)
        ctx = self._begin_call(primitive, args, kwargs, function_name, model_hint, agent_hint)
        error: Optional[BaseException] = None
        response: Any = None
        try:
            response = original(*args, **kwargs)
        except BaseException as exc:
            error = exc
            raise
        finally:
            self._finish_call(ctx, response, error)
        return response

    async def _invoke_async(
        self,
        *,
        primitive: str,
        original: Any,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        function_name: Optional[str] = None,
        model_hint: Optional[str] = None,
        agent_hint: Optional[str] = None,
    ) -> Any:
        if not self._connected:
            return await original(*args, **kwargs)
        ctx = self._begin_call(primitive, args, kwargs, function_name, model_hint, agent_hint)
        error: Optional[BaseException] = None
        response: Any = None
        try:
            response = await original(*args, **kwargs)
        except BaseException as exc:
            error = exc
            raise
        finally:
            self._finish_call(ctx, response, error)
        return response

    def _begin_call(
        self,
        primitive: str,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        function_name: Optional[str],
        model_hint: Optional[str],
        agent_hint: Optional[str],
    ) -> _CallContext:
        # Marvin's primitives are AMBIENT module functions with no enclosing run,
        # so each traced call opens its own scope. Inside @trace/trace_context the
        # outer collector is reused (and _end_run does not flush).
        run = self._begin_run()
        # environment.config is deferred to here, not connect(): _emit no-ops
        # without an active collector, so emitting it at patch time would drop it.
        self._emit_config_once()

        call_agent = self._call_agent(primitive, kwargs)
        response_model, labels = self._extract_target_info(primitive, args, kwargs)
        ctx = _CallContext(
            run_id=str(uuid.uuid4()),
            span_id=self._new_span_id(),
            primitive=primitive,
            fn_name=function_name or primitive,
            model=_agent_model(call_agent) or model_hint or _detect_marvin_model(self._marvin_module),
            agent_name=_honest_agent_name(call_agent) or agent_hint,
            response_model=response_model,
            labels=labels,
            args=args,
            kwargs=kwargs,
            root_span_id=run.root_span_id,
        )
        ctx.start_ns = time.time_ns()
        return ctx

    def _finish_call(self, ctx: _CallContext, response: Any, error: Optional[BaseException]) -> None:
        try:
            latency_ms = (time.time_ns() - ctx.start_ns) / 1_000_000
            self._emit_tool_call(ctx, latency_ms, response, error)
            self._emit_model_invoke(ctx, latency_ms, response, error)
        finally:
            # Must run even if an emit raised, or the run's ContextVars leak into
            # the caller's next call.
            self._end_run()

    @staticmethod
    def _call_agent(primitive: str, kwargs: Dict[str, Any]) -> Any:
        """The ``marvin.Agent`` for this call, or None.

        Only the four primitives take ``agent=``. For ``fn`` the kwargs are the
        CALLER'S function arguments, so reading ``agent`` out of them would
        promote customer data to an agent/model identity — the decorator's
        captured hint is the only honest source there.
        """
        if _base_primitive(primitive) == "fn":
            return None
        return kwargs.get("agent")

    @staticmethod
    def _extract_target_info(
        primitive: str, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Tuple[Optional[str], Optional[str]]:
        """Resolve ``(response_model, labels)`` honestly per primitive.

        ``classify()`` takes a LABEL SET — an enum or list of values, which is not
        a response model — so it is reported under ``labels``. ``fn`` resolves to
        neither: its args/kwargs are the CALLER'S, so ``def route(target: str)``
        called as ``route(target="billing")`` would otherwise report a
        response_model derived from customer data.
        """
        base = _base_primitive(primitive)
        if base == "fn":
            return None, None
        if base == "classify":
            labels = kwargs.get("labels")
            if labels is None and len(args) >= 2:
                labels = args[1]
            return None, _describe_labels(labels)
        target = kwargs.get("target")
        if target is None:
            target = kwargs.get("_type")
        if target is None:
            # ``target`` is the 2nd positional for extract/cast but the FIRST for
            # generate(target=None, n=1, ...).
            if base == "generate" and args:
                target = args[0]
            elif base in {"extract", "cast"} and len(args) >= 2:
                target = args[1]
        return _describe_response_model(target), None

    @staticmethod
    def _deeper_layer_reported_model_invoke(ctx: _CallContext) -> bool:
        """True iff a layer BELOW *this* call already reported its request.

        Scoped to THIS CALL'S span, never to the trace. ``_begin_run()`` pushes a
        root span unique to this call, and a provider adapter emitting inside it
        parents its ``model.invoke`` on whatever ``_current_span_id`` holds — i.e.
        exactly this root (``providers/_emit_helpers.py``: ``parent_span_id =
        _current_span_id.get()``). Marvin has not emitted its own ``model.invoke``
        yet when this runs, so a hit can only be a deeper layer's.

        A trace-WIDE count cannot answer this: under ``trace_context()`` +
        ``asyncio.gather`` (batch classification — marvin's canonical async shape)
        N calls share ONE collector, so a concurrent sibling's ``model.invoke``
        reads as "something deeper already reported mine" and every call but the
        first silently drops its own event.

        The boundary, stated honestly: a deeper layer that opens its OWN run scope
        parents its ``model.invoke`` on its own root rather than on ours, so we do
        not recognise it and emit alongside it. That direction fails toward a
        duplicate (marvin's is tokenless and carries no cost, so attribution is
        unaffected) rather than toward silent loss.
        """
        collector = _current_collector.get()
        if collector is None:
            return False
        return any(
            e.get("event_type") == "model.invoke" and e.get("parent_span_id") == ctx.root_span_id
            for e in collector.events
        )

    # ------------------------------------------------------------------
    # Emission
    # ------------------------------------------------------------------

    def _emit_tool_call(
        self,
        ctx: _CallContext,
        latency_ms: float,
        response: Any,
        error: Optional[BaseException],
    ) -> None:
        label = f"marvin.{ctx.fn_name}"
        # ateam also stamps this label as ``agent_id``, a third copy of the same
        # string. atlas's InferAgentGraph reads agent_id as a graph-node identity
        # (graph_inference.go :: nodeIdentityFields), so the primitive's own name
        # renders as the AGENT whenever the caller declared no marvin.Agent —
        # "marvin.classify" is the function that ran, not somebody's agent.
        # Nothing is lost by omitting it: ``tool_name``/``name`` carry the label
        # and ``primitive`` carries the kind.
        payload = self._payload(
            run_id=ctx.run_id,
            tool_name=label,
            name=label,
            primitive=ctx.primitive,
            latency_ms=latency_ms,
            success=error is None,
        )
        if ctx.agent_name:
            payload["agent_name"] = ctx.agent_name
        if ctx.response_model:
            payload["response_model"] = ctx.response_model
        # The label SET is the caller's own taxonomy, not schema — content.
        self._set_if_capturing(payload, "labels", ctx.labels)
        self._set_if_capturing(payload, "input", [truncate(safe_serialize(a), _ARG_LIMIT) for a in ctx.args])
        if response is not None:
            self._set_if_capturing(payload, "output", truncate(safe_serialize(response), _RESPONSE_LIMIT))
        self._stamp_error(payload, error)
        self._emit("tool.call", payload, span_id=ctx.span_id, span_name=label)

    def _emit_model_invoke(
        self,
        ctx: _CallContext,
        latency_ms: float,
        response: Any,
        error: Optional[BaseException],
    ) -> None:
        if not ctx.model:
            # model is required at ingest and a placeholder would be a FABRICATED
            # model name. The call is still traced by the tool.call above.
            log.debug(
                "layerlens: no real model discoverable for marvin.%s — skipping model.invoke "
                "(tool.call still emitted)",
                ctx.fn_name,
            )
            return
        if self._deeper_layer_reported_model_invoke(ctx):
            # Marvin 3.x runs on pydantic-ai: a provider adapter under THIS call
            # already reported this request WITH token counts. Marvin's own view is
            # tokenless, so adding it would duplicate the call and split its cost
            # attribution.
            log.debug(
                "layerlens: a deeper layer already reported model.invoke for marvin.%s — "
                "skipping the tokenless duplicate",
                ctx.fn_name,
            )
            return

        payload = self._payload(
            run_id=ctx.run_id,
            primitive=ctx.primitive,
            model_name=ctx.model,
            model=ctx.model,
            latency_ms=latency_ms,
        )
        if ctx.agent_name:
            payload["agent_name"] = ctx.agent_name
        if ctx.response_model:
            payload["response_model"] = ctx.response_model
        # The label SET is the caller's own taxonomy, not schema — content.
        self._set_if_capturing(payload, "labels", ctx.labels)
        self._set_if_capturing(payload, "args", [truncate(safe_serialize(a), _ARG_LIMIT) for a in ctx.args])
        self._set_if_capturing(
            payload,
            "kwargs",
            {
                k: truncate(safe_serialize(v), _ARG_LIMIT)
                for k, v in ctx.kwargs.items()
                if k not in _EXCLUDED_KWARGS
            },
        )
        if response is not None:
            self._set_if_capturing(payload, "response", truncate(safe_serialize(response), _RESPONSE_LIMIT))
        self._stamp_error(payload, error)
        self._emit("model.invoke", payload, span_id=ctx.span_id, span_name=f"marvin.{ctx.fn_name}")

    def _stamp_error(self, payload: Dict[str, Any], error: Optional[BaseException]) -> None:
        """Record a failure honestly: the CATEGORY always, the free text gated.

        str(exc) echoes the failing arguments (Marvin renders the offending value
        into its own ValueErrors), so it is content. ``error_type`` survives
        capture_content=False, keeping a failure distinguishable from a success —
        and distinguishable even when str(exc) renders empty.
        """
        if error is None:
            return
        payload["error_type"] = type(error).__name__
        error_text = str(error)
        if error_text:
            self._set_if_capturing(payload, "error", truncate(error_text, _ARG_LIMIT))

    def _emit_config_once(self) -> None:
        """One-shot ``environment.config`` per adapter, now that a run is active."""
        with self._patch_lock:
            if self._config_emitted:
                return
            self._config_emitted = True
            module = self._marvin_module
        # ateam stamps agent_id="marvin" here. NO identity is carried instead:
        # "marvin" is the framework's own name, not a producer-declared agent,
        # and atlas nodes every identity key — including agent_id — so the label
        # rendered as a second agent beside the real one and turned a
        # single-agent extraction into Agent = "multi-agent". The framework is
        # already named in ``config["framework"]``, which is not an identity key.
        payload = self._payload(config=_detect_marvin_settings(module))
        self._emit("environment.config", payload, span_id=self._new_span_id(), span_name="marvin:config")


def instrument_marvin(
    client: Any,
    *,
    capture_config: Optional[CaptureConfig] = None,
    marvin_module: Any = None,
) -> MarvinAdapter:
    """Process-wide patch of Marvin's primitive functions.

    Wraps ``classify`` / ``extract`` / ``cast`` / ``generate`` (and their
    ``*_async`` twins) plus ``marvin.fn``, so every call emits a ``tool.call``
    and — when a real model resolves — a ``model.invoke``.

    Args:
        client: The layerlens client events are uploaded through.
        capture_config: Optional capture configuration.
        marvin_module: Optional explicit ``marvin`` module reference. Production
            callers omit this; the adapter imports the real module itself.

    Returns:
        The connected adapter. The patch is reverted by
        :func:`uninstrument_marvin` (or ``adapter.disconnect()``) — always pair
        instrumentation with one of them on shutdown.
    """
    from .._registry import get, register

    existing = get("marvin")
    if existing is not None:
        existing.disconnect()
    adapter = MarvinAdapter(client, capture_config)
    adapter.connect(target=marvin_module)
    register("marvin", adapter)
    return adapter


def uninstrument_marvin() -> None:
    from .._registry import unregister

    unregister("marvin")
