"""Instructor adapter — traced ``create()`` wrappers bound on the patched client.

Instructor builds its patched classes dynamically per call-site, so subclassing
them is brittle. The bound methods on the *instance* are exactly the surface a
customer reaches after ``client = instructor.from_openai(OpenAI())``, so this
adapter wraps those and delegates to the original.

On a real ``Instructor``, ``.chat``, ``.completions`` and ``.messages`` are all
properties returning ``self`` — so ``chat.completions.create``,
``messages.create``, ``completions.create`` and ``create`` resolve to the SAME
(owner, method) slot. The ``_layerlens_traced`` sentinel is what stops the 2nd
through 4th from double-wrapping and emitting a duplicate ``model.invoke``; it is
a correctness guard, not an idempotence nicety.

Retry observation rides Instructor's hooks system (``client.on("parse:error" |
"completion:error", ...)``): one ``tool.call`` / ``instructor.validation_retry``
per REAL observed error, correlated to the in-flight call through a ContextVar.
When the installed Instructor exposes no hooks, no retry telemetry is emitted at
all — the configured maximum is never passed off as an observation, and
``retries_observed`` is omitted rather than defaulted.

Usage::

    adapter = InstructorAdapter(client)
    patched = instructor.from_openai(OpenAI())
    adapter.connect(target=patched)
    profile = patched.chat.completions.create(model="gpt-4o-mini", messages=[...], response_model=UserProfile)
    adapter.disconnect()
"""

from __future__ import annotations

import uuid
import inspect
import logging
import weakref
import functools
import threading
from typing import Any, Dict, List, Tuple, Optional
from contextvars import ContextVar

from ._utils import truncate, safe_serialize
from ..._context import _current_run, _current_collector
from ..._identity import _API_METHOD_RE, _s, _is_generic
from ._base_framework import FrameworkAdapter
from ..._capture_config import CaptureConfig

log = logging.getLogger(__name__)

try:
    import instructor as _instructor  # pyright: ignore[reportMissingImports]

    _HAS_INSTRUCTOR = True
except ImportError:
    _instructor = None  # type: ignore[assignment]
    _HAS_INSTRUCTOR = False

#: Hook names Instructor fires on a retryable failure. ``parse:error`` fires on
#: each Pydantic validation failure, ``completion:error`` on each provider-call
#: failure — both drive a retry inside Instructor's tenacity loop.
_RETRY_HOOKS: Tuple[str, ...] = ("parse:error", "completion:error")

#: Constant tool name for an observed validation/completion retry.
_RETRY_TOOL_NAME = "instructor.validation_retry"

#: Endpoints probed on the patched client, in order. The ``acreate`` variants are
#: dead on modern Instructor (an async client exposes ``create`` as a coroutine
#: function instead) and are retained only for older/legacy patched clients that
#: still expose them; a missing path is skipped, so probing costs nothing.
_CREATE_METHOD_PATHS: Tuple[str, ...] = (
    "chat.completions.create",
    "chat.completions.acreate",
    "messages.create",
    "messages.acreate",
    "completions.create",
    "completions.acreate",
    "create",
    "acreate",
    # Structured-output STREAMING endpoints. These return a (sync/async)
    # generator of partial models / items rather than a finished model, so they
    # are instrumented with deferred emission (see _build_traced_stream). Left
    # unwrapped, a streamed instructor call emitted zero telemetry.
    "chat.completions.create_partial",
    "chat.completions.create_iterable",
    "create_partial",
    "create_iterable",
)

#: The streaming method NAMES (last path segment). ``create_partial`` yields a
#: growing partial model; ``create_iterable`` yields items. Both return a plain
#: generator (sync client) or async generator (async client) — usage + the final
#: object land only when the generator drains, so emission is deferred to drain.
_STREAM_METHODS: frozenset = frozenset({"create_partial", "create_iterable"})

#: The in-flight traced call on this thread / asyncio.Task. Instructor's hooks are
#: registered client-GLOBALLY, so a handler needs this to find the call that
#: actually triggered the error it was handed. A ContextVar isolates concurrent
#: calls on one client per-thread AND per-Task.
_ACTIVE_CALL: ContextVar[Optional[Dict[str, Any]]] = ContextVar("layerlens_instructor_active_call", default=None)

#: Attribute stamped on a traced wrapper (sentinel + escape hatch to the original).
_TRACED_ATTR = "_layerlens_traced"
_ORIGINAL_ATTR = "_layerlens_original"

#: ``(module substring, provider)`` probes, in order. ``azure`` precedes ``openai``
#: because an Azure client's module carries both.
_PROVIDER_MARKERS: Tuple[Tuple[str, str], ...] = (
    ("azure", "azure_openai"),
    ("openai", "openai"),
    ("anthropic", "anthropic"),
    ("google", "google"),
    ("gemini", "google"),
    ("cohere", "cohere"),
    ("mistral", "mistral"),
    ("groq", "groq"),
)


def _detect_provider(client: Any) -> Optional[str]:
    """The underlying provider of a patched Instructor client, or None.

    Probes the module each client class was defined in, then hops to the SDK
    client Instructor wraps (``client.client``) — a patched ``Instructor``'s own
    module is ``instructor.*``, which matches no provider. Returns None rather
    than an "unknown" label when nothing resolves. The hop is bounded by a
    visited-set so a client whose ``.client`` chain forms a cycle terminates
    instead of raising RecursionError.
    """
    seen: set[int] = set()
    current = client
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        module = str(getattr(type(current), "__module__", "") or "")
        for marker, provider in _PROVIDER_MARKERS:
            if marker in module:
                return provider
        current = getattr(current, "client", None)
    return None


def _response_model_name(response_model: Any) -> Optional[str]:
    """Resolve a ``response_model`` argument to its class-name string, or None."""
    if response_model is None:
        return None
    if isinstance(response_model, type):
        return response_model.__name__
    name = getattr(response_model, "__name__", None)
    if name:
        return str(name)
    return type(response_model).__name__


def _configured_max_retries(kwargs: Dict[str, Any]) -> Optional[int]:
    """The caller's ``max_retries`` when it is honestly reportable as a count.

    Instructor accepts an ``int`` or a tenacity ``Retrying``/``AsyncRetrying``.
    Only a plain int is a count: a tenacity object's stop-condition semantics
    cannot be flattened to a number, and ``bool`` is an int subclass but not a
    count. Anything else yields no field — and never a blind ``int()`` that would
    break the customer's call.
    """
    value = kwargs.get("max_retries")
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _honest_agent_name(name: Any) -> Optional[str]:
    """A caller-DECLARED agent name, or None.

    Instructor declares no agent identity of its own, so the only honest source is
    a name the caller passed explicitly. The framework label ("instructor") is
    rejected outright: a framework name is not a producer-declared agent, and
    surfacing one in the Agent column is the generic-identity anti-pattern. An
    unnamed client honestly renders "—".
    """
    resolved = _s(name)
    if resolved is None:
        return None
    lowered = resolved.lower()
    if lowered == InstructorAdapter.name or _is_generic(resolved) or _API_METHOD_RE.match(lowered):
        return None
    return resolved


def _capped_messages(messages: Any) -> Any:
    """Render request messages for embedding, capped like all other content."""
    if not isinstance(messages, (list, tuple)):
        return safe_serialize(messages)
    rendered: List[Any] = []
    for message in messages:
        if isinstance(message, dict):
            rendered.append(
                {
                    "role": str(message.get("role", "")),
                    "content": truncate(safe_serialize(message.get("content")), 1000),
                }
            )
        else:
            rendered.append(truncate(safe_serialize(message), 1000))
    return rendered


def _capped_response(response: Any) -> Any:
    """Render an extracted response for embedding, capped like all other content.

    Keeps the structured ``model_dump()`` while it stays small so downstream can
    read the extracted fields; falls back to a truncated string otherwise, so one
    large extraction cannot blow up the event.
    """
    dumped = safe_serialize(response)
    if isinstance(dumped, dict) and len(repr(dumped)) <= 1000:
        return dumped
    return truncate(dumped if isinstance(dumped, str) else repr(dumped), 1000)


class InstructorAdapter(FrameworkAdapter):
    """Instructor adapter — see module docstring for the instrumentation model."""

    name = "instructor"
    package = "instructor"

    def __init__(self, client: Any, capture_config: Optional[CaptureConfig] = None) -> None:
        super().__init__(client, capture_config)
        # Per-client registries keyed by id(client). The client itself is held only
        # through a weakref whose death callback purges every entry, so a dropped
        # client is collectable and the maps stay bounded; strong refs are limited
        # to the sub-object originals we must restore.
        self._client_refs: Dict[int, weakref.ref[Any]] = {}
        self._wrapped_methods: Dict[int, Dict[str, Tuple[Any, str, Any]]] = {}
        self._hooks: Dict[int, List[Tuple[str, Any]]] = {}
        self._client_meta: Dict[int, Dict[str, Any]] = {}
        self._agent_names: Dict[int, Optional[str]] = {}
        self._config_emitted: set[int] = set()
        self._wrap_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _on_connect(self, target: Any = None, **kwargs: Any) -> None:
        self._check_dependency(_HAS_INSTRUCTOR)
        if target is None:
            raise ValueError(
                "InstructorAdapter requires a patched Instructor client: "
                "adapter.connect(target=instructor.from_openai(OpenAI()))"
            )
        version = getattr(_instructor, "__version__", None)
        if version:
            self._metadata["instructor_version"] = str(version)
        self._instrument_client(target, agent_name=kwargs.get("agent_name"))

    def _on_disconnect(self) -> None:
        with self._wrap_lock:
            refs = list(self._client_refs.items())
        for client_id, ref in refs:
            client = ref()
            if client is not None:
                self._restore(client)
            self._purge(client_id)
        with self._wrap_lock:
            self._client_refs.clear()
            self._wrapped_methods.clear()
            self._hooks.clear()
            self._client_meta.clear()
            self._agent_names.clear()
            self._config_emitted.clear()

    # ------------------------------------------------------------------
    # Instrumentation
    # ------------------------------------------------------------------

    def _instrument_client(self, client: Any, *, agent_name: Optional[str] = None) -> Any:
        """Wrap the patched client's ``create`` endpoints. Idempotent per client."""
        client_id = id(client)
        with self._wrap_lock:
            if client_id in self._wrapped_methods:
                return client
            self._wrapped_methods[client_id] = {}

            def _on_collect(_ref: "weakref.ref[Any]", cid: int = client_id) -> None:
                self._purge(cid)

            try:
                self._client_refs[client_id] = weakref.ref(client, _on_collect)
            except TypeError:
                log.debug(
                    "layerlens: instructor client %s is not weak-referenceable; "
                    "its registry entry is kept until disconnect()",
                    type(client).__name__,
                )

        self._agent_names[client_id] = _honest_agent_name(agent_name)
        meta: Dict[str, Any] = {}
        provider = _detect_provider(client)
        if provider:
            meta["provider"] = provider
        mode = getattr(client, "mode", None)
        if mode is not None:
            meta["mode"] = str(mode)
        self._client_meta[client_id] = meta

        outcomes: Dict[str, int] = {"wrapped": 0, "traced": 0, "absent": 0, "failed": 0}
        for method_path in _CREATE_METHOD_PATHS:
            outcomes[self._wrap_method_path(client, method_path, client_id=client_id)] += 1

        if outcomes["wrapped"] == 0:
            self._purge(client_id)
            # Each cause gets its own truthful message: reporting "no create() found"
            # for a client whose methods exist but are already traced (or whose
            # setattr was swallowed) sends the caller hunting the wrong bug.
            if outcomes["traced"]:
                raise RuntimeError(
                    f"InstructorAdapter: every create() method on {type(client).__name__} is already "
                    "instrumented by another adapter instance, which emits to its own client. "
                    "Disconnect that adapter before instrumenting this client again."
                )
            if outcomes["failed"]:
                raise RuntimeError(
                    f"InstructorAdapter: found a create() method on {type(client).__name__} but could "
                    "not install a wrapper on it (the attribute assignment did not take — a frozen "
                    "or __slots__/__setattr__-guarded client). No calls would be traced."
                )
            raise TypeError(
                "InstructorAdapter could not locate a recognised create() method on "
                f"{type(client).__name__}. Expected one of: chat.completions.create, "
                "messages.create, completions.create, or create."
            )

        self._register_retry_hooks(client, client_id=client_id)
        return client

    def _wrap_method_path(self, client: Any, dotted_path: str, *, client_id: int) -> str:
        """Wrap ``client.<dotted.path>``.

        Returns ``"wrapped"``, ``"traced"`` (already carries the sentinel),
        ``"absent"`` (no such callable), or ``"failed"`` (the assignment did not
        take).
        """
        parts = dotted_path.split(".")
        owner: Any = client
        for attr in parts[:-1]:
            owner = getattr(owner, attr, None)
            if owner is None:
                return "absent"
        method_name = parts[-1]
        original = getattr(owner, method_name, None)
        if original is None or not callable(original):
            return "absent"
        if getattr(original, _TRACED_ATTR, False):
            return "traced"

        traced = self._build_traced(original, client_id, method_name=method_name)
        try:
            setattr(owner, method_name, traced)
        except (AttributeError, TypeError):
            log.warning(
                "layerlens: could not install the instructor wrapper on %s.%s; calls through it will NOT be traced",
                type(owner).__name__,
                method_name,
                exc_info=True,
            )
            return "failed"
        # A client with a __setattr__ guard (frozen dataclass / pydantic model) can
        # swallow the assignment. Recording an un-installed wrapper would report a
        # success that emits nothing for the life of the process, so verify it took.
        if getattr(owner, method_name, None) is not traced:
            log.warning(
                "layerlens: the instructor wrapper on %s.%s did not take (the attribute "
                "is unchanged); calls through it will NOT be traced",
                type(owner).__name__,
                method_name,
            )
            return "failed"
        with self._wrap_lock:
            self._wrapped_methods.setdefault(client_id, {})[dotted_path] = (owner, method_name, original)
        return "wrapped"

    def _build_traced(self, original: Any, client_id: int, *, method_name: Optional[str] = None) -> Any:
        if method_name in _STREAM_METHODS:
            return self._build_traced_stream(original, client_id)

        if inspect.iscoroutinefunction(original):

            @functools.wraps(original)
            async def traced_acreate(*args: Any, **kwargs: Any) -> Any:
                return await self._invoke_async(original, client_id, args, kwargs)

            setattr(traced_acreate, _ORIGINAL_ATTR, original)
            setattr(traced_acreate, _TRACED_ATTR, True)
            return traced_acreate

        @functools.wraps(original)
        def traced_create(*args: Any, **kwargs: Any) -> Any:
            return self._invoke_sync(original, client_id, args, kwargs)

        setattr(traced_create, _ORIGINAL_ATTR, original)
        setattr(traced_create, _TRACED_ATTR, True)
        return traced_create

    def _build_traced_stream(self, original: Any, client_id: int) -> Any:
        """Wrap a streaming ``create_partial`` / ``create_iterable``.

        The call returns a generator, so emission is DEFERRED to when the
        generator drains (the final object + usage are known only then). The run
        is begun eagerly (so a setup error is still surfaced) then detached with
        ``flush=False``; the drain wrapper re-establishes the run to emit + flush.
        An abandoned (never-fully-drained) stream emits nothing — an honest
        non-report beats a fabricated complete result.
        """
        if inspect.isasyncgenfunction(original):

            @functools.wraps(original)
            def traced_astream(*args: Any, **kwargs: Any) -> Any:
                return self._invoke_async_stream(original, client_id, args, kwargs)

            setattr(traced_astream, _ORIGINAL_ATTR, original)
            setattr(traced_astream, _TRACED_ATTR, True)
            return traced_astream

        @functools.wraps(original)
        def traced_stream(*args: Any, **kwargs: Any) -> Any:
            return self._invoke_sync_stream(original, client_id, args, kwargs)

        setattr(traced_stream, _ORIGINAL_ATTR, original)
        setattr(traced_stream, _TRACED_ATTR, True)
        return traced_stream

    def _restore(self, client: Any) -> None:
        client_id = id(client)
        with self._wrap_lock:
            method_map = dict(self._wrapped_methods.get(client_id, {}))
            hooks = list(self._hooks.get(client_id, []))
        for owner, method_name, original in method_map.values():
            current = getattr(owner, method_name, None)
            if not getattr(current, _TRACED_ATTR, False):
                # A third party re-wrapped this slot after us — leave their wrapper
                # alone rather than silently clobbering it.
                continue
            try:
                setattr(owner, method_name, original)
            except (AttributeError, TypeError):
                log.warning(
                    "layerlens: could not restore the original %s.%s",
                    type(owner).__name__,
                    method_name,
                    exc_info=True,
                )
        off = getattr(client, "off", None)
        if callable(off):
            for hook_name, handler in hooks:
                try:
                    off(hook_name, handler)
                except Exception:
                    log.debug("layerlens: removing instructor hook %r failed", hook_name, exc_info=True)

    def _purge(self, client_id: int) -> None:
        """Drop every registry entry for a dead (or restored) client."""
        with self._wrap_lock:
            self._client_refs.pop(client_id, None)
            self._wrapped_methods.pop(client_id, None)
            self._hooks.pop(client_id, None)
            self._client_meta.pop(client_id, None)
            self._agent_names.pop(client_id, None)
            self._config_emitted.discard(client_id)

    # ------------------------------------------------------------------
    # Retry observation (Instructor hooks)
    # ------------------------------------------------------------------

    def _register_retry_hooks(self, client: Any, *, client_id: int) -> None:
        """Subscribe to REAL validation/completion errors via ``client.on``.

        An Instructor build with no hooks system observes no retries, so none are
        reported: absence is honest, and synthesizing them from the configured
        maximum would be fabrication.
        """
        on = getattr(client, "on", None)
        if not callable(on):
            log.debug(
                "layerlens: instructor client %s exposes no hooks system — validation retries will not be observed",
                type(client).__name__,
            )
            return
        registered: List[Tuple[str, Any]] = []
        for hook_name in _RETRY_HOOKS:
            # ``_hook`` binds per-iteration; a closure over the loop variable would
            # report every hook under the last name.
            # instructor also passes attempt_number/max_attempts/is_last_attempt;
            # accept and discard them so a new hook kwarg cannot break the call.
            def handler(*args: Any, _hook: str = hook_name, **_kwargs: Any) -> None:
                self._on_retry_hook(_hook, args)

            try:
                on(hook_name, handler)
            except Exception:
                log.debug("layerlens: registering instructor hook %r failed", hook_name, exc_info=True)
                continue
            registered.append((hook_name, handler))
        if registered:
            with self._wrap_lock:
                self._hooks[client_id] = registered

    def _hooks_active(self, client_id: int) -> bool:
        with self._wrap_lock:
            return client_id in self._hooks

    def _on_retry_hook(self, hook: str, args: Tuple[Any, ...]) -> None:
        """Emit one validation-retry ``tool.call`` for a REAL observed error."""
        ctx = _ACTIVE_CALL.get()
        if ctx is None:
            # Hooks are client-global: this one fired outside a traced call (an
            # uninstrumented code path drove the same client). There is nothing
            # honest to correlate it to, so no orphan event is invented.
            log.debug("layerlens: instructor %s hook fired outside a traced call", hook)
            return
        ctx["attempts"] = int(ctx.get("attempts", 0)) + 1
        error = args[0] if args else None

        payload = self._payload(
            run_id=ctx["run_id"],
            tool_name=_RETRY_TOOL_NAME,
            name=_RETRY_TOOL_NAME,
            attempt=int(ctx["attempts"]),
            success=False,
        )
        self._stamp_agent(payload, ctx["agent_name"])
        if ctx["response_model"]:
            payload["response_model"] = ctx["response_model"]
        if hook:
            payload["hook"] = hook
        if error is not None:
            payload["error_type"] = type(error).__name__
        error_text = str(error) if error is not None else ""
        if error_text:
            # A Pydantic ValidationError renders the offending value verbatim
            # ("input_value='<the caller's raw PII>'"), so this free text is
            # CONTENT. The surviving hook / attempt / success / error_type keep a
            # redacted retry visible instead of blinding it.
            self._set_if_capturing(payload, "error", truncate(error_text, 400))
        self._emit("tool.call", payload, parent_span_id=ctx["span_id"], span_name=_RETRY_TOOL_NAME)

    # ------------------------------------------------------------------
    # Invocation core
    # ------------------------------------------------------------------

    def _invoke_sync(self, original: Any, client_id: int, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        ctx = self._begin_call(client_id, kwargs)
        token = _ACTIVE_CALL.set(ctx)
        error: Optional[BaseException] = None
        response: Any = None
        try:
            response = original(*args, **kwargs)
        except BaseException as exc:
            error = exc
            raise
        finally:
            _ACTIVE_CALL.reset(token)
            self._finish_call(ctx, client_id, response, error)
        return response

    async def _invoke_async(self, original: Any, client_id: int, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        ctx = self._begin_call(client_id, kwargs)
        token = _ACTIVE_CALL.set(ctx)
        error: Optional[BaseException] = None
        response: Any = None
        try:
            response = await original(*args, **kwargs)
        except BaseException as exc:
            error = exc
            raise
        finally:
            _ACTIVE_CALL.reset(token)
            self._finish_call(ctx, client_id, response, error)
        return response

    # ------------------------------------------------------------------
    # Streaming invocation core (deferred-emit; mirrors the mirascope stream seam)
    # ------------------------------------------------------------------

    def _invoke_sync_stream(self, original: Any, client_id: int, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        ctx = self._begin_call(client_id, kwargs)
        token = _ACTIVE_CALL.set(ctx)
        try:
            gen = original(*args, **kwargs)
        except BaseException as exc:
            # create_partial()/create_iterable() setup failed before yielding —
            # surface it like the buffered error path.
            _ACTIVE_CALL.reset(token)
            self._finish_call(ctx, client_id, None, exc)
            raise
        _ACTIVE_CALL.reset(token)
        run = self._get_run()
        # Detach the run's ContextVars; the drain re-establishes them and flushes
        # once the stream completes (usage/final object land only then).
        self._end_run(flush=False)
        return self._drain_sync(gen, ctx, client_id, run)

    def _invoke_async_stream(self, original: Any, client_id: int, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        # ``original`` is an async-generator function: calling it returns the
        # async generator WITHOUT running its body (no await), so the underlying
        # request fires during the drain, not here.
        ctx = self._begin_call(client_id, kwargs)
        token = _ACTIVE_CALL.set(ctx)
        try:
            agen = original(*args, **kwargs)
        except BaseException as exc:
            _ACTIVE_CALL.reset(token)
            self._finish_call(ctx, client_id, None, exc)
            raise
        _ACTIVE_CALL.reset(token)
        run = self._get_run()
        self._end_run(flush=False)
        return self._drain_async(agen, ctx, client_id, run)

    def _drain_sync(self, gen: Any, ctx: Dict[str, Any], client_id: int, run: Any) -> Any:
        state = {"emitted": False, "last": None}

        def emit(error: Optional[BaseException]) -> None:
            if state["emitted"]:
                return
            state["emitted"] = True
            self._emit_stream_call(ctx, client_id, run, state["last"], error)

        try:
            for item in gen:
                state["last"] = item
                yield item
        except GeneratorExit:
            # Caller abandoned mid-drain — incomplete; emit nothing (honest).
            raise
        except BaseException as exc:
            emit(exc)
            raise
        else:
            emit(None)

    async def _drain_async(self, agen: Any, ctx: Dict[str, Any], client_id: int, run: Any) -> Any:
        state = {"emitted": False, "last": None}

        def emit(error: Optional[BaseException]) -> None:
            if state["emitted"]:
                return
            state["emitted"] = True
            self._emit_stream_call(ctx, client_id, run, state["last"], error)

        try:
            async for item in agen:
                state["last"] = item
                yield item
        except GeneratorExit:
            raise
        except BaseException as exc:
            emit(exc)
            raise
        else:
            emit(None)

    def _emit_stream_call(
        self, ctx: Dict[str, Any], client_id: int, run: Any, response: Any, error: Optional[BaseException]
    ) -> None:
        """Re-establish the (detached) run, emit the deferred model.invoke /
        cost.record for the drained stream, then flush if this run owns its
        collector (an ambient collector is flushed by its owner)."""
        if run is None:  # pragma: no cover - always inside _begin_call here
            return
        owns_collector = run._col_token is not None
        run_token = _current_run.set(run)
        col_token = _current_collector.set(run.collector)
        try:
            self._emit_call_events(ctx, client_id, response, error)
        finally:
            _current_collector.reset(col_token)
            _current_run.reset(run_token)
            if owns_collector:
                run.collector.flush()

    def _begin_call(self, client_id: int, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        self._begin_run()
        # environment.config is deferred to here, not connect(): _emit no-ops
        # without an active collector, so emitting it at connect time would drop it
        # silently.
        self._maybe_emit_client_config(client_id)
        self._start_timer("call")
        return {
            "run_id": str(uuid.uuid4()),
            "span_id": self._new_span_id(),
            "response_model": _response_model_name(kwargs.get("response_model")),
            "model": kwargs.get("model"),
            "messages": kwargs.get("messages"),
            "max_retries": _configured_max_retries(kwargs),
            "agent_name": self._agent_names.get(client_id),
            "attempts": 0,
        }

    def _finish_call(
        self,
        ctx: Dict[str, Any],
        client_id: int,
        response: Any,
        error: Optional[BaseException],
    ) -> None:
        try:
            self._emit_call_events(ctx, client_id, response, error)
        finally:
            # Must run even if the emit path raised or the model skip fired, or the
            # run's ContextVars leak into the caller's next call.
            self._end_run()

    def _emit_call_events(
        self,
        ctx: Dict[str, Any],
        client_id: int,
        response: Any,
        error: Optional[BaseException],
    ) -> None:
        latency_ms = self._stop_timer("call")
        model = ctx["model"]
        if not model or not isinstance(model, str):
            # A placeholder would be a FABRICATED model name, so the whole event is
            # dropped rather than invented. NB this adapter deliberately does not emit
            # ``model_name`` at all (``test_instructor.py`` asserts its absence for the
            # framework family) — that requirement is advisory here, and pretending
            # otherwise is what made the contract self-contradictory before LAY-3622
            # F1. See ``layerlens.instrument._ingest_contract``.
            log.debug("layerlens: instructor create() called without a model string — skipping model.invoke")
            return

        payload = self._payload(run_id=ctx["run_id"], model=model)
        self._stamp_agent(payload, ctx["agent_name"])
        meta = self._client_meta.get(client_id, {})
        provider = meta.get("provider")
        if provider:
            payload["provider"] = provider
        if ctx["response_model"]:
            payload["response_model"] = ctx["response_model"]
        if latency_ms is not None:
            payload["latency_ms"] = latency_ms
        if ctx["max_retries"] is not None:
            payload["max_retries_configured"] = ctx["max_retries"]
        if self._hooks_active(client_id):
            # Only a hook-observed count is an observation. Without hooks the key is
            # omitted entirely — never backfilled from max_retries.
            payload["retries_observed"] = int(ctx["attempts"])

        # Instructor stashes the raw provider response on the returned Pydantic
        # model as ``_raw_response``; usage hangs off that, not off the model.
        usage_source = getattr(response, "_raw_response", response)
        tokens = self._normalize_tokens(getattr(usage_source, "usage", None))
        payload.update(tokens)

        if error is not None:
            # error_type/status are metadata and SURVIVE capture_content=False, so a
            # failed extraction stays distinguishable from a success even when the
            # free-text error is stripped — and even when str(exc) renders empty.
            payload["status"] = "error"
            payload["error_type"] = type(error).__name__
            error_text = str(error)
            if error_text:
                self._set_if_capturing(payload, "error", truncate(error_text, 400))
        else:
            payload["status"] = "ok"

        if ctx["messages"]:
            self._set_if_capturing(payload, "messages", _capped_messages(ctx["messages"]))
        if response is not None:
            self._set_if_capturing(payload, "output_message", _capped_response(response))

        self._emit(
            "model.invoke",
            payload,
            span_id=ctx["span_id"],
            span_name=f"instructor:{ctx['response_model'] or 'create'}",
        )

        if tokens:
            cost_payload = self._payload(model=model)
            if provider:
                cost_payload["provider"] = provider
            cost_payload.update(tokens)
            self._emit("cost.record", cost_payload)

    def _maybe_emit_client_config(self, client_id: int) -> None:
        """One-shot ``environment.config`` per client, now that a run is active."""
        with self._wrap_lock:
            if client_id in self._config_emitted:
                return
            self._config_emitted.add(client_id)
        payload = self._payload(**self._client_meta.get(client_id, {}))
        self._stamp_agent(payload, self._agent_names.get(client_id))
        self._emit("environment.config", payload)

    def _stamp_agent(self, payload: Dict[str, Any], agent_name: Optional[str]) -> None:
        """Stamp the caller-declared identity, or nothing.

        ``agent_name`` is the field the Agent column resolves from; ``agent_id`` is
        carried alongside for wire parity. Both stay absent for an unnamed client —
        Instructor declares no agent of its own, and stamping the framework label
        would fabricate one.
        """
        if agent_name:
            payload["agent_name"] = agent_name
            payload["agent_id"] = agent_name


def instrument_instructor(
    client: Any,
    target: Any,
    *,
    capture_config: Optional[CaptureConfig] = None,
    agent_name: Optional[str] = None,
) -> InstructorAdapter:
    """Trace a patched Instructor client's ``create()`` calls.

    Args:
        client: The layerlens client events are uploaded through.
        target: A patched Instructor client (``instructor.from_openai(OpenAI())``).
        capture_config: Optional capture configuration.
        agent_name: The agent identity to stamp. Omit it and the Agent column
            renders "—" — Instructor declares no agent name of its own, so there
            is nothing honest to fill it with.
    """
    from .._registry import get, register

    existing = get("instructor")
    if existing is not None:
        existing.disconnect()
    adapter = InstructorAdapter(client, capture_config)
    adapter.connect(target=target, agent_name=agent_name)
    register("instructor", adapter)
    return adapter


def uninstrument_instructor() -> None:
    from .._registry import unregister

    unregister("instructor")
