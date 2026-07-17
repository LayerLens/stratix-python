"""DSPy adapter — first-party callback bus, with a ``__class__``-swap fallback.

DSPy exposes three distinct instrumentation targets, and this adapter covers
all three from a SINGLE emission source per environment:

- ``connect()`` registers a :class:`dspy.utils.callback.BaseCallback` on
  ``dspy.settings.callbacks`` — DSPy's documented hook. While it is installed
  it is the only module/LM/tool emission path, so ``instrument_program`` /
  ``instrument_lm`` deliberately become no-ops (running both double-counts
  every event).
- For callback-less environments (older DSPy, or DSPy-shaped objects under
  test) ``instrument_program`` / ``instrument_lm`` fall back to swapping
  ``__class__`` for a synthesized traced subclass. Python looks dunders up on
  the TYPE, so ``program.__call__ = wrapper`` is silently ignored; the swap is
  opaque (``isinstance(program, OriginalModule)`` still passes).
- ``instrument_optimizer`` always wraps ``compile`` + the optimizer's
  ``metric``, regardless of the callback: DSPy's callback API does not cover
  optimization, and the score-over-iterations stream is the optimizer's
  headline signal. Each metric evaluation is an ``agent.state.change`` with
  ``state_key="optimization_step"`` carrying the real compile ``run_id``
  (threaded on a contextvar so concurrent compiles never collide).

DSPy modules NEST (``ReAct`` -> ``ChainOfThought`` -> ``Predict``), so the run
lifecycle is owned by the OUTERMOST module only — a ``_begin_run`` per module
start would shred one logical program call into N single-node traces.

Usage::

    adapter = DSPyAdapter(client)
    adapter.connect()  # registers the DSPy callback
    program(question="...")  # traced
    adapter.disconnect()
"""

from __future__ import annotations

import json
import time
import uuid
import logging
import threading
import contextlib
from typing import Any, Dict, List, Tuple, Optional, NamedTuple
from collections import OrderedDict
from contextvars import ContextVar

from ._base_framework import FrameworkAdapter
from ..._capture_config import CaptureConfig

log = logging.getLogger(__name__)

try:
    import dspy  # pyright: ignore[reportMissingImports]

    _HAS_DSPY = True
except ImportError:
    _HAS_DSPY = False

# Probed separately from ``dspy`` itself: a DSPy old enough to lack the callback
# bus is still instrumentable through the ``__class__``-swap fallback, so a
# missing callback module must not read as a missing framework.
try:
    from dspy.utils.callback import BaseCallback  # pyright: ignore[reportMissingImports]

    _HAS_DSPY_CALLBACK = True
except ImportError:
    _HAS_DSPY_CALLBACK = False
    BaseCallback = object  # type: ignore[assignment,misc]

#: Bound on the in-flight callback-state map. DSPy fires paired start/end
#: callbacks so entries are short-lived; the cap only matters when an end
#: callback is lost (framework bug / crash), which would otherwise grow the map
#: for the process lifetime.
_CALL_STARTS_MAX = 4096

#: Provider prefixes DSPy addresses models with, per litellm's ``provider/model``
#: convention.
_KNOWN_MODEL_PREFIXES = frozenset(
    {"openai", "anthropic", "google", "gemini", "mistral", "ollama", "ollama_chat", "azure", "bedrock", "openrouter"}
)


class _ModuleFrame(NamedTuple):
    """One in-flight DSPy module call on the current context's stack."""

    call_id: str
    module_type: str
    agent_name: Optional[str]
    owns_run: bool


class _OptimizerState(NamedTuple):
    """What ``_restore`` needs to put a wrapped optimizer back exactly as found."""

    compile_fn: Any
    compile_was_own: bool
    metric_fn: Any
    metric_was_own: bool


class _BoundedCallStarts:
    """Thread-safe, size-bounded map of in-flight callback state."""

    def __init__(self, cap: int = _CALL_STARTS_MAX) -> None:
        self._data: "OrderedDict[str, Any]" = OrderedDict()
        self._cap = cap
        self._lock = threading.Lock()

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self._cap:
                self._data.popitem(last=False)

    def pop(self, key: str) -> Optional[Any]:
        with self._lock:
            return self._data.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


class DSPyAdapter(FrameworkAdapter):
    """DSPy adapter — see the module docstring for the instrumentation model."""

    name = "dspy"
    package = "dspy"

    def __init__(self, client: Any, capture_config: Optional[CaptureConfig] = None) -> None:
        super().__init__(client, capture_config)
        self._callback: Any = None
        self._wrapped_modules: Dict[int, type] = {}
        self._wrapped_lms: Dict[int, type] = {}
        self._wrapped_optimizers: Dict[int, _OptimizerState] = {}
        # Strong refs so disconnect() can restore every target it swapped.
        self._wrapped_targets: List[Any] = []
        self._call_starts = _BoundedCallStarts()
        # The enclosing module stack for this context: attributes nested LM/tool
        # calls to the program that made them, and marks which frame owns the run.
        self._stack: ContextVar[Tuple[_ModuleFrame, ...]] = ContextVar(f"layerlens_dspy_stack_{id(self)}", default=())
        # Real compile run id, threaded to metric-step events.
        self._compile_run_id: ContextVar[Optional[str]] = ContextVar(
            f"layerlens_dspy_compile_run_{id(self)}", default=None
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _on_connect(self, target: Any = None, **kwargs: Any) -> None:
        """Register the DSPy callback; optionally attach the explicit targets.

        ``target`` is a program; ``lm`` / ``optimizer`` cover DSPy's other two
        instrumentation surfaces (``connect(target=prog, optimizer=opt)``). None
        is required — the callback alone traces every module/LM/tool call, so
        this adapter needs no credentials to connect and therefore QUALIFIES for
        ``auto()`` wiring from a bare client (``_registry.py``'s stated bar).
        """
        self._check_dependency(_HAS_DSPY)
        version = getattr(dspy, "__version__", None)
        if version:
            self._metadata["framework_version"] = str(version)
        self._register_callback()
        if target is not None:
            self.instrument_program(target)
        if kwargs.get("lm") is not None:
            self.instrument_lm(kwargs["lm"])
        if kwargs.get("optimizer") is not None:
            self.instrument_optimizer(kwargs["optimizer"])

    def _on_disconnect(self) -> None:
        self._unregister_callback()
        # Reverse order so a wrapper never observes a half-restored target.
        for target in reversed(self._wrapped_targets):
            self._restore(target)
        self._wrapped_targets.clear()
        self._wrapped_modules.clear()
        self._wrapped_lms.clear()
        self._wrapped_optimizers.clear()
        self._call_starts.clear()

    # ------------------------------------------------------------------
    # DSPy callback bus (the documented first-party hook)
    # ------------------------------------------------------------------

    def _register_callback(self) -> None:
        """Install the callback on ``dspy.settings.callbacks``, or stay off it.

        ``dspy.settings.configure()`` may only be called by the thread that
        first configured settings and raises otherwise, and the bare-attribute
        fallback can be ignored just as quietly. So trust the INSTALLED state,
        never the call: believing a callback is registered when it is not would
        self-disable the ``instrument_*`` fallback (which no-ops whenever a
        callback is present) and emit nothing while reporting connected.
        """
        if not _HAS_DSPY_CALLBACK:
            log.debug("layerlens: dspy.utils.callback unavailable; using the class-swap instrumentation path")
            self._callback = None
            return
        settings = getattr(dspy, "settings", None)
        if settings is None:
            self._callback = None
            return

        cb = _LayerLensDSPyCallback(self)
        callbacks = list(getattr(settings, "callbacks", None) or [])
        callbacks.append(cb)
        try:
            settings.configure(callbacks=callbacks)
        except Exception:
            with contextlib.suppress(Exception):
                settings.callbacks = callbacks

        installed = list(getattr(settings, "callbacks", None) or [])
        if any(c is cb for c in installed):
            self._callback = cb
        else:
            self._callback = None
            log.warning(
                "layerlens: could not install the DSPy callback on dspy.settings; falling back to the "
                "class-swap path — call adapter.instrument_program(program) to trace module calls"
            )

    def _unregister_callback(self) -> None:
        if self._callback is None:
            return
        try:
            settings = getattr(dspy, "settings", None)
            if settings is not None:
                callbacks = [c for c in (getattr(settings, "callbacks", None) or []) if c is not self._callback]
                try:
                    settings.configure(callbacks=callbacks)
                except Exception:
                    with contextlib.suppress(Exception):
                        settings.callbacks = callbacks
        except Exception:
            log.debug("layerlens: could not unregister the DSPy callback", exc_info=True)
        finally:
            self._callback = None

    # ------------------------------------------------------------------
    # Public instrumentation API
    # ------------------------------------------------------------------

    def instrument_program(self, program: Any) -> Any:
        """Trace a DSPy ``Module`` / program via the ``__class__``-swap fallback.

        A no-op when the first-party callback is registered: it already traces
        every module call, and adding a wrapper would emit each call twice.
        """
        program = _require_callable(program, "Module")
        if self._callback is not None:
            log.debug(
                "layerlens: DSPy callback registered — instrument_program(%s) skips the class-swap wrap "
                "(single emission source)",
                type(program).__name__,
            )
            return program

        program_id = id(program)
        with self._lock:
            if program_id in self._wrapped_modules:
                return program
            # Unwrap first: a compiled program is a deepcopy of the swapped
            # student, so instrumenting one would otherwise synthesize
            # ``_LayerLensTraced__LayerLensTraced_MyQA`` and restore to a class
            # this module invented rather than to the developer's own.
            original_class = _unswapped_class(type(program))
            self._wrapped_modules[program_id] = original_class
            self._wrapped_targets.append(program)

        adapter = self

        def traced_call(self_program: Any, *args: Any, **kwargs: Any) -> Any:
            call_id = uuid.uuid4().hex
            inputs = dict(kwargs)
            if args:
                inputs["_args"] = args
            # original_class, NOT type(self_program): after the swap the latter is
            # the synthesized ``_LayerLensTraced_*`` subclass, whose name is this
            # module's own plumbing and would land in the Agent column.
            adapter._enter_module(call_id, original_class, self_program, inputs)
            start_ns = time.time_ns()
            error: Optional[BaseException] = None
            prediction: Any = None
            try:
                prediction = original_class.__call__(self_program, *args, **kwargs)
            except BaseException as exc:
                error = exc
                raise
            finally:
                adapter._exit_module(call_id, prediction, error, start_ns)
            return prediction

        traced_class = type(
            f"_LayerLensTraced_{original_class.__name__}",
            (original_class,),
            {"__call__": traced_call, "_layerlens_traced": True},
        )
        program.__class__ = traced_class
        return program

    def instrument_lm(self, lm: Any) -> Any:
        """Trace a DSPy LM so each call emits a ``model.invoke``.

        A no-op when the first-party callback is registered — ``on_lm_start`` /
        ``on_lm_end`` already trace every LM call.
        """
        lm = _require_callable(lm, "LM")
        if self._callback is not None:
            log.debug(
                "layerlens: DSPy callback registered — instrument_lm(%s) skips the class-swap wrap "
                "(single emission source)",
                type(lm).__name__,
            )
            return lm

        lm_id = id(lm)
        with self._lock:
            if lm_id in self._wrapped_lms:
                return lm
            original_class = _unswapped_class(type(lm))
            self._wrapped_lms[lm_id] = original_class
            self._wrapped_targets.append(lm)

        adapter = self

        def traced_lm_call(self_lm: Any, *args: Any, **kwargs: Any) -> Any:
            call_id = uuid.uuid4().hex
            prompt = args[0] if args else kwargs.get("prompt") or kwargs.get("messages")
            start_ns = time.time_ns()
            error: Optional[BaseException] = None
            response: Any = None
            try:
                response = original_class.__call__(self_lm, *args, **kwargs)
            except BaseException as exc:
                error = exc
                raise
            finally:
                adapter._emit_lm_call(
                    call_id,
                    self_lm,
                    prompt,
                    response,
                    (time.time_ns() - start_ns) / 1_000_000,
                    error,
                )
            return response

        traced_class = type(
            f"_LayerLensTraced_{original_class.__name__}",
            (original_class,),
            {"__call__": traced_lm_call, "_layerlens_traced": True},
        )
        lm.__class__ = traced_class
        return lm

    def instrument_optimizer(self, optimizer: Any) -> Any:
        """Wrap ``optimizer.compile()`` and the optimizer's ``metric``.

        The compile boundary emits an ``agent.input`` / ``agent.output`` pair for
        the whole optimization; every metric evaluation inside it becomes an
        ``agent.state.change`` carrying the compile's ``run_id``, so a dashboard
        can plot score against iteration per compile.

        Always wraps, even when the callback path is active: DSPy's callback API
        does not cover optimizer metrics.
        """
        if not hasattr(optimizer, "compile"):
            raise TypeError(f"DSPyAdapter requires an object with .compile(); got {type(optimizer).__name__}")

        opt_id = id(optimizer)
        with self._lock:
            if opt_id in self._wrapped_optimizers:
                return optimizer
            original_compile = optimizer.compile
            original_metric = getattr(optimizer, "metric", None)
            self._wrapped_optimizers[opt_id] = _OptimizerState(
                compile_fn=original_compile,
                compile_was_own="compile" in vars(optimizer),
                metric_fn=original_metric,
                metric_was_own="metric" in vars(optimizer),
            )
            self._wrapped_targets.append(optimizer)

        optimizer_type = type(optimizer).__name__
        agent_name = _honest_agent_name(type(optimizer))
        # Stable per-optimizer id for metric calls made OUTSIDE a wrapped compile
        # (a user evaluating the metric directly) — never reused across compiles.
        fallback_run_id = f"optimization-{uuid.uuid4()}"
        iter_lock = threading.Lock()
        counter = {"i": 0}

        if callable(original_metric):

            def traced_metric(example: Any, prediction: Any, *m_args: Any, **m_kwargs: Any) -> Any:
                with iter_lock:
                    counter["i"] += 1
                    iteration = counter["i"]
                run_id = self._compile_run_id.get() or fallback_run_id
                payload = self._payload(
                    run_id=run_id,
                    state_key="optimization_step",
                    state_type="optimization",
                    optimizer_type=optimizer_type,
                    iteration=iteration,
                )
                self._set_identity(payload, agent_name)
                try:
                    score = original_metric(example, prediction, *m_args, **m_kwargs)
                except BaseException:
                    # A failed evaluation is still a real iteration — record it
                    # with NO score rather than a fabricated 0.0.
                    self._emit_optimization_step(payload, run_id, iteration)
                    raise
                if isinstance(score, (int, float)):
                    payload["score"] = float(score)
                    payload["new_value"] = float(score)
                self._emit_optimization_step(payload, run_id, iteration)
                return score

            with contextlib.suppress(AttributeError, TypeError):
                optimizer.metric = traced_metric

        def traced_compile(*args: Any, **kwargs: Any) -> Any:
            run_id = str(uuid.uuid4())
            self._begin_run()
            token = self._compile_run_id.set(run_id)
            with iter_lock:
                # Baseline, never the absolute count: the counter is per-optimizer
                # and never resets, so a second compile on the same optimizer would
                # otherwise report the LIFETIME total as this compile's iterations.
                start_i = counter["i"]
            payload_in = self._payload(
                run_id=run_id,
                operation="compile",
                optimizer_type=optimizer_type,
                input_text=f"{optimizer_type}.compile",
            )
            self._set_identity(payload_in, agent_name)
            self._emit("agent.input", payload_in, run_id=run_id, span_name=f"dspy:{optimizer_type}.compile")
            start_ns = time.time_ns()
            error: Optional[BaseException] = None
            result: Any = None
            try:
                result = original_compile(*args, **kwargs)
            except BaseException as exc:
                error = exc
                raise
            finally:
                with iter_lock:
                    iterations = counter["i"] - start_i
                payload = self._payload(
                    run_id=run_id,
                    operation="compile",
                    optimizer_type=optimizer_type,
                    latency_ms=(time.time_ns() - start_ns) / 1_000_000,
                    iterations=iterations,
                )
                self._set_identity(payload, agent_name)
                if error is not None:
                    self._set_if_capturing(payload, "error", _safe_str(str(error), limit=400))
                    payload["error_type"] = type(error).__name__
                self._emit("agent.output", payload, run_id=run_id, span_name=f"dspy:{optimizer_type}.compile")
                # The compiled program is a NEW object (demos attached) — describe
                # it rather than inventing an output_text for it.
                if result is not None:
                    self._emit_program_config(result)
                self._compile_run_id.reset(token)
                self._end_run()
            return result

        optimizer.compile = traced_compile
        return optimizer

    def _restore(self, target: Any) -> None:
        target_id = id(target)
        original_class = self._wrapped_modules.get(target_id) or self._wrapped_lms.get(target_id)
        if original_class is not None:
            with contextlib.suppress(TypeError):
                target.__class__ = original_class
        state = self._wrapped_optimizers.get(target_id)
        if state is not None:
            with contextlib.suppress(AttributeError, TypeError):
                if state.compile_was_own:
                    target.compile = state.compile_fn
                else:
                    # Drop the instance shadow so the class's own bound method is
                    # reachable again, rather than leaving a permanent attribute.
                    target.__dict__.pop("compile", None)
            with contextlib.suppress(AttributeError, TypeError):
                if state.metric_was_own:
                    target.metric = state.metric_fn
                else:
                    target.__dict__.pop("metric", None)

    # ------------------------------------------------------------------
    # Module boundary (shared by the callback and class-swap paths)
    # ------------------------------------------------------------------

    def _enter_module(self, call_id: str, cls: type, instance: Any, inputs: Dict[str, Any]) -> None:
        # The single chokepoint for both emission paths, so neither can leak the
        # synthesized subclass: the callback bus passes a raw ``type(instance)``,
        # which IS the swapped class whenever instrument_program() ran first.
        cls = _unswapped_class(cls)
        frames = self._stack.get()
        # Only the OUTERMOST module owns the run: DSPy programs nest, so a
        # _begin_run per module start would shred one call into N traces. An
        # already-active run (an enclosing compile, an ambient trace_context, or
        # another framework's run) is borrowed, never replaced.
        owns_run = self._get_run() is None
        if owns_run:
            self._begin_run()
        module_type = cls.__name__
        agent_name = _honest_agent_name(cls)
        parent_call_id = frames[-1].call_id if frames else None
        self._stack.set(frames + (_ModuleFrame(call_id, module_type, agent_name, owns_run),))
        # Only the START TIME lives in the bounded map. Identity rides the stack
        # frame, which is a ContextVar and cannot be evicted — so a full map costs
        # a latency number, never a module's identity.
        self._call_starts.put(call_id, time.time_ns())

        self._emit_program_config(instance, cls)

        payload = self._payload(run_id=call_id, module_type=module_type, input_keys=sorted(inputs))
        self._set_identity(payload, agent_name)
        self._set_if_capturing(payload, "input_text", _render_inputs(inputs))
        self._set_if_capturing(payload, "inputs", {k: _safe_str(v, limit=400) for k, v in inputs.items()})
        self._emit(
            "agent.input",
            payload,
            run_id=call_id,
            parent_run_id=parent_call_id,
            span_name=f"dspy:{module_type}",
        )

    def _exit_module(
        self,
        call_id: str,
        prediction: Any,
        error: Optional[BaseException],
        start_ns: Optional[int] = None,
    ) -> None:
        frames = self._stack.get()
        frame: Optional[_ModuleFrame] = None
        parent_call_id: Optional[str] = None
        for index, candidate in enumerate(frames):
            if candidate.call_id == call_id:
                frame = candidate
                parent_call_id = frames[index - 1].call_id if index else None
                break
        # Rebuild by filtering rather than resetting a stashed token: the token
        # would have to live in the evictable call-start map, where losing it
        # leaves the stack permanently set and mis-attributes every later LM/tool
        # event to a stale module.
        self._stack.set(tuple(f for f in frames if f.call_id != call_id))

        entry = self._call_starts.pop(call_id)
        if entry is not None:
            start_ns = entry
        latency_ms = (time.time_ns() - start_ns) / 1_000_000 if start_ns is not None else None

        payload = self._payload(run_id=call_id)
        # No frame means we never saw this module START, so its type and identity
        # are genuinely UNKNOWN. ateam falls back to the literal "Module" here and
        # stamps it into agent_id/agent_name — a fabricated agent in the Agent
        # column. Omit instead: an honest "—" beats an invention.
        if frame is not None:
            payload["module_type"] = frame.module_type
            self._set_identity(payload, frame.agent_name)
        if latency_ms is not None:
            payload["latency_ms"] = latency_ms
        rendered = _render_prediction(prediction)
        if rendered is not None:
            self._set_if_capturing(payload, "prediction", rendered)
            self._set_if_capturing(payload, "output_text", _dumps(rendered))
        if error is not None:
            # error_type is the CATEGORY and always survives; str(exc) is CONTENT
            # (a real dspy AdapterParseError embeds the raw LM response verbatim).
            self._set_if_capturing(payload, "error", _safe_str(str(error), limit=400))
            payload["error_type"] = type(error).__name__
        self._emit(
            "agent.output",
            payload,
            run_id=call_id,
            parent_run_id=parent_call_id,
            span_name=f"dspy:{frame.module_type}" if frame is not None else "dspy:module",
        )
        if frame is not None and frame.owns_run:
            self._end_run()

    def _emit_program_config(self, program: Any, cls: Optional[type] = None) -> None:
        """Describe a program once per run (signature, fields, demos, predictors).

        Scoped to the RUN rather than the adapter: the id() key is only ever
        compared against programs the live call stack still holds, so a collected
        object's id cannot be recycled into a false "already seen" hit — and every
        trace stays self-describing instead of only the first one.

        *cls* is the ORIGINAL class on the class-swap path; ``type(program)``
        there is the synthesized ``_LayerLensTraced_*`` subclass, whose name is
        this module's own plumbing. The compile path has no original class to
        hand over — the compiled program is a fresh ``deepcopy`` the optimizer
        made — so the fallback must unwrap rather than trust ``type()``.
        """
        run = self._get_run()
        if run is None:
            return
        seen = run.data.setdefault("dspy_seen_programs", set())
        if id(program) in seen:
            return
        seen.add(id(program))
        cls = _unswapped_class(cls or type(program))
        payload = self._payload(**_extract_program_metadata(program, cls))
        self._set_identity(payload, _honest_agent_name(cls))
        self._emit("environment.config", payload, span_name=f"dspy:{cls.__name__}")

    # ------------------------------------------------------------------
    # LM boundary
    # ------------------------------------------------------------------

    def _emit_lm_call(
        self,
        call_id: str,
        lm: Any,
        prompt: Any,
        outputs: Any,
        latency_ms: Optional[float],
        error: Optional[BaseException],
    ) -> None:
        raw_model = _resolve_model(lm)
        if not raw_model:
            # THE honesty rule of this adapter: a model.invoke without a
            # resolvable model is invalid at ingest and must never be stamped
            # "unknown" — drop the event instead.
            log.debug(
                "layerlens: skipping DSPy model.invoke for call %s: no resolvable model on %s",
                call_id,
                type(lm).__name__,
            )
            return
        model, provider = _split_model_id(raw_model)
        frames = self._stack.get()
        frame = frames[-1] if frames else None

        payload = self._payload(run_id=call_id, model_name=model, model=model)
        if provider:
            payload["provider"] = provider
        if frame is not None:
            self._set_identity(payload, frame.agent_name)
        self._set_if_capturing(payload, "prompt", _safe_str(prompt, limit=2000))
        self._set_if_capturing(payload, "output", _safe_str(outputs, limit=2000))
        tokens = self._normalize_tokens(_probe_usage(lm, outputs))
        payload.update(tokens)
        if latency_ms is not None:
            payload["latency_ms"] = latency_ms
        if error is not None:
            self._set_if_capturing(payload, "error", _safe_str(str(error), limit=400))
            payload["error_type"] = type(error).__name__
        self._emit(
            "model.invoke",
            payload,
            run_id=call_id,
            parent_run_id=frame.call_id if frame is not None else None,
            span_name=f"dspy:lm:{model}",
        )

        # No usage -> no cost.record. DSPy surfaces usage only when the LM keeps
        # history and the provider returned counts; a zero-token record would
        # price a call we cannot actually measure.
        if tokens:
            cost_payload = self._payload(run_id=call_id, model=model)
            if provider:
                cost_payload["provider"] = provider
            cost_payload.update(tokens)
            self._emit("cost.record", cost_payload)

    # ------------------------------------------------------------------
    # Tool boundary (callback path only — DSPy has no class-swap tool seam)
    # ------------------------------------------------------------------

    def _emit_tool_call(self, call_id: str, outputs: Any, exception: Optional[BaseException]) -> None:
        entry = self._call_starts.pop(f"tool:{call_id}")
        if entry is None:
            # tool_name is only knowable from the start entry, and it is required
            # at ingest — skip rather than fabricate one.
            log.debug("layerlens: skipping DSPy tool.call for call %s: no recorded start (evicted?)", call_id)
            return
        start_ns, tool_name, inputs = entry
        frames = self._stack.get()
        frame = frames[-1] if frames else None

        payload = self._payload(
            run_id=call_id,
            tool_name=tool_name,
            name=tool_name,
            success=exception is None,
            latency_ms=(time.time_ns() - start_ns) / 1_000_000,
        )
        self._set_if_capturing(payload, "input", _safe_str(inputs, limit=1000))
        self._set_if_capturing(payload, "output", _safe_str(outputs, limit=1000))
        if frame is not None:
            self._set_identity(payload, frame.agent_name)
        if exception is not None:
            self._set_if_capturing(payload, "error", _safe_str(str(exception), limit=400))
            payload["error_type"] = type(exception).__name__
        self._emit(
            "tool.call",
            payload,
            run_id=call_id,
            parent_run_id=frame.call_id if frame is not None else None,
            span_name=f"dspy:tool:{tool_name}",
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _emit_optimization_step(self, payload: Dict[str, Any], run_id: str, iteration: int) -> None:
        # payload["run_id"] stays the compile's id so every step correlates to its
        # compile; the SPAN key must still be per-iteration or all N steps collapse
        # onto one span.
        self._emit(
            "agent.state.change",
            payload,
            run_id=f"{run_id}:step:{iteration}",
            parent_run_id=run_id,
            span_name="dspy:optimization_step",
        )

    @staticmethod
    def _set_identity(payload: Dict[str, Any], agent_name: Optional[str]) -> None:
        """Stamp the agent identity ateam carries as ``agent_id`` PLUS the
        ``agent_name`` LayerLens actually resolves from (``_identity.py`` Tier 2;
        no tier reads ``agent_id``, so agent_id alone blanks the Agent column).
        Both are omitted together when there is no honest producer-declared name.
        """
        if agent_name:
            payload["agent_id"] = agent_name
            payload["agent_name"] = agent_name


class _LayerLensDSPyCallback(BaseCallback):  # type: ignore[misc,valid-type]
    """Thin shim from DSPy's callback bus onto the adapter's handlers."""

    def __init__(self, adapter: DSPyAdapter) -> None:
        super().__init__()
        self._adapter = adapter

    def on_module_start(self, call_id: str, instance: Any, inputs: Any) -> None:
        self._adapter._enter_module(call_id, type(instance), instance, _unwrap_inputs(inputs))

    def on_module_end(self, call_id: str, outputs: Any, exception: Optional[BaseException] = None) -> None:
        self._adapter._exit_module(call_id, outputs, exception)

    def on_lm_start(self, call_id: str, instance: Any, inputs: Any) -> None:
        # Keep the LM instance: on_lm_end only receives outputs, and both the
        # model identity and the token counts have to be read off the LM itself.
        prompt = None
        if isinstance(inputs, dict):
            prompt = inputs.get("prompt") or inputs.get("messages")
        self._adapter._call_starts.put(f"lm:{call_id}", (time.time_ns(), instance, prompt))

    def on_lm_end(self, call_id: str, outputs: Any, exception: Optional[BaseException] = None) -> None:
        entry = self._adapter._call_starts.pop(f"lm:{call_id}")
        if entry is None:
            # Without the start entry there is no LM instance, so no model can be
            # resolved — the same honesty skip _emit_lm_call applies.
            log.debug("layerlens: skipping DSPy model.invoke for call %s: no recorded start (evicted?)", call_id)
            return
        start_ns, instance, prompt = entry
        self._adapter._emit_lm_call(
            call_id, instance, prompt, outputs, (time.time_ns() - start_ns) / 1_000_000, exception
        )

    def on_tool_start(self, call_id: str, instance: Any, inputs: Any) -> None:
        tool_name = getattr(instance, "name", None) or type(instance).__name__
        self._adapter._call_starts.put(f"tool:{call_id}", (time.time_ns(), str(tool_name), _unwrap_inputs(inputs)))

    def on_tool_end(self, call_id: str, outputs: Any, exception: Optional[BaseException] = None) -> None:
        self._adapter._emit_tool_call(call_id, outputs, exception)


# ---------------------------------------------------------------------------
# Pure helpers (never import dspy — the class-swap path runs without it)
# ---------------------------------------------------------------------------


def _require_callable(obj: Any, label: str) -> Any:
    """Guard a DSPy target, returning it unnarrowed.

    Kept as a function so the ``callable()`` check does not narrow the caller's
    binding to a plain function type — the ``__class__`` swap below rebinds an
    attribute the narrowed type does not admit.
    """
    if not callable(obj):
        raise TypeError(f"DSPyAdapter requires a callable {label}; got {type(obj).__name__}")
    return obj


def _unswapped_class(cls: type) -> type:
    """The developer's own class behind a synthesized ``_LayerLensTraced_*`` subclass.

    The class-swap fallback rebinds ``obj.__class__`` to a subclass THIS module
    synthesizes, so ``type(obj)`` stops being the developer's class the moment a
    program is instrumented. That name must never escape: ``_honest_agent_name``
    discriminates structurally on the defining module, and the synthesized class
    is defined here — NOT under ``dspy`` — so it sails straight through the
    framework-primitive guard and lands in the Agent column as
    ``_LayerLensTraced_MyQA``, an "agent" no producer ever declared.

    Two live paths hand back a swapped class: DSPy's optimizers ``deepcopy`` the
    student, so a COMPILED program's ``type()`` is the synthesized subclass; and
    ``instrument_program()`` before ``connect()`` swaps first, after which the
    callback bus hands ``type(instance)`` to ``on_module_start``.

    Matched on the marker in the class's OWN ``__dict__`` (never inherited), so a
    user class that happens to subclass a traced one keeps its own honest name.
    """
    while cls.__dict__.get("_layerlens_traced") is True and cls.__bases__:
        cls = cls.__bases__[0]
    return cls


def _honest_agent_name(cls: type) -> Optional[str]:
    """A DSPy class name that the DEVELOPER chose, or None for a framework primitive.

    ``type(module).__name__`` is the only identity DSPy offers, and it is honest
    only for a class the developer declared (``class RAG(dspy.Module)`` -> "RAG").
    DSPy's own primitives (``Predict``, ``ChainOfThought``, ``ReAct``,
    ``BootstrapFewShot``) are plumbing every unnamed program shares; surfacing one
    in the Agent column is the generic-label anti-pattern ``_identity.py`` guards
    against for smolagents/crewai/langchain. Discriminated STRUCTURALLY on the
    defining module — a dspy-owned class is never a producer-declared agent — since
    a name denylist would drift with every DSPy release.
    """
    module = getattr(cls, "__module__", "") or ""
    if module == "dspy" or module.startswith("dspy."):
        return None
    return getattr(cls, "__name__", "") or None


def _unwrap_inputs(inputs: Any) -> Dict[str, Any]:
    """Flatten DSPy's callback ``inputs`` into the real call arguments.

    The bus hands over the wrapper's own frame — ``{"args": (...), "kwargs": {...}}``
    — so a naive ``sorted(inputs)`` reports ``["args", "kwargs"]`` for every module
    on earth instead of the signature fields. Folding positional args under
    ``_args`` matches the class-swap path, so both paths emit the same shape.
    """
    if not isinstance(inputs, dict):
        return {}
    if not set(inputs) <= {"args", "kwargs"}:
        return dict(inputs)
    flat: Dict[str, Any] = {}
    kwargs = inputs.get("kwargs")
    if isinstance(kwargs, dict):
        flat.update(kwargs)
    args = inputs.get("args")
    if args:
        flat["_args"] = args
    return flat


def _safe_str(value: Any, limit: int = 500) -> Optional[str]:
    if value is None:
        return None
    try:
        rendered = str(value)
    except Exception:
        return "<unrenderable>"
    if len(rendered) <= limit:
        return rendered
    return rendered[:limit] + f"...[truncated {len(rendered) - limit} chars]"


def _dumps(value: Any, limit: int = 2000) -> str:
    try:
        return json.dumps(value, sort_keys=True, default=str)[:limit]
    except (TypeError, ValueError):
        return str(_safe_str(value, limit=limit))


def _render_inputs(inputs: Dict[str, Any], limit: int = 2000) -> str:
    return _dumps({k: _safe_str(v, limit=400) for k, v in inputs.items()}, limit=limit)


def _render_prediction(prediction: Any) -> Any:
    """A DSPy ``Prediction`` rendered as a flat {field: value} dict, or None.

    Predictions expose ``.keys()`` listing their output fields; anything else
    falls back to a truncated string rendering.
    """
    if prediction is None:
        return None
    try:
        keys = list(prediction.keys())
    except Exception:
        return _safe_str(prediction, limit=1000)
    return {k: _safe_str(getattr(prediction, k, ""), limit=400) for k in keys}


def _detect_provider_from_model_name(model_name: Optional[str]) -> Optional[str]:
    """Infer a provider for an id that carries no ``provider/`` prefix."""
    if not model_name:
        return None
    lower = model_name.lower()
    if lower.startswith(("gpt", "o1", "o3", "o4")):
        return "openai"
    if "claude" in lower:
        return "anthropic"
    if "gemini" in lower:
        return "google"
    if "mistral" in lower or "mixtral" in lower:
        return "mistral"
    return None


def _split_model_id(model_id: str) -> Tuple[str, Optional[str]]:
    """Split DSPy's litellm-style ``provider/model`` id into (model, provider).

    The pricing table is keyed by the BARE model id — ``is_priced("openai/gpt-4o-mini")``
    is False while ``is_priced("gpt-4o-mini")`` is True — so emitting the prefixed
    id leaves every priced DSPy call unpriced AND indistinguishable from a
    genuinely-unpriced local model (the LAY-3626 fail-closed cost check reads an
    unresolvable model as legitimately free). The split is lossless: the prefix is
    exactly the provider, which rides its own field.
    """
    prefix, sep, rest = model_id.partition("/")
    if sep and rest and prefix.lower() in _KNOWN_MODEL_PREFIXES:
        return rest, prefix.lower()
    return model_id, _detect_provider_from_model_name(model_id)


def _resolve_model(lm: Any) -> Optional[str]:
    """The model id off a live DSPy LM, or None. DSPy moved it between
    attributes across versions (``model``, ``model_name``, ``kwargs['model']``)."""
    if lm is None:
        return None
    kwargs_obj = getattr(lm, "kwargs", None)
    kwargs_model = kwargs_obj.get("model") if isinstance(kwargs_obj, dict) else None
    model = getattr(lm, "model", None) or getattr(lm, "model_name", None) or kwargs_model
    return str(model) if model else None


def _probe_usage(lm: Any, outputs: Any) -> Optional[Dict[str, Any]]:
    """Token usage for ONE LM call, correlated by output identity.

    DSPy's ``on_lm_end`` hands back ``LM.__call__``'s return value — a list of
    strings — which carries no usage at all; the counts live on the LM's history
    entry, appended by that same call before the callback fires. Matching on
    ``entry["outputs"] is outputs`` picks this call's entry without racing a
    concurrent call on the same LM (``history[-1]`` would not). Absent history
    (``dspy.settings.disable_history``) simply yields no tokens.
    """
    history = getattr(lm, "history", None)
    if not isinstance(history, list):
        return None
    for entry in reversed(history):
        if isinstance(entry, dict) and entry.get("outputs") is outputs:
            usage = entry.get("usage")
            return usage if isinstance(usage, dict) else None
    return None


def _extract_program_metadata(program: Any, cls: Optional[type] = None) -> Dict[str, Any]:
    """Signature / field / demo / predictor metadata for a DSPy program.

    Every field is OMITTED when the attribute is absent or raises — nothing is
    defaulted, so a missing signature reads as "not reported", never as empty.
    """
    md: Dict[str, Any] = {"module_type": (cls or type(program)).__name__}

    # Real DSPy signatures are CLASSES (instances of SignatureMeta), so
    # type(sig).__name__ on one yields the metaclass name — distinguish classes
    # from instances and plain strings.
    sig = getattr(program, "signature", None)
    if sig is not None:
        if isinstance(sig, str):
            md["signature"] = sig
        elif isinstance(sig, type):
            md["signature"] = sig.__name__
        else:
            md["signature"] = type(sig).__name__
        try:
            input_fields = getattr(sig, "input_fields", None)
            if input_fields:
                md["input_fields"] = sorted(input_fields.keys())
            output_fields = getattr(sig, "output_fields", None)
            if output_fields:
                md["output_fields"] = sorted(output_fields.keys())
        except Exception:
            log.debug("layerlens: could not read DSPy signature fields", exc_info=True)

    demos = getattr(program, "demos", None)
    if demos is not None:
        with contextlib.suppress(TypeError):
            md["demo_count"] = len(demos)

    predictors = getattr(program, "predictors", None)
    if callable(predictors):
        try:
            md["predictor_count"] = len(list(predictors()))
        except Exception:
            log.debug("layerlens: could not enumerate DSPy predictors", exc_info=True)

    return md


def instrument_dspy(
    client: Any,
    program: Any = None,
    *,
    capture_config: Optional[CaptureConfig] = None,
    lm: Any = None,
    optimizer: Any = None,
) -> DSPyAdapter:
    """Connect a :class:`DSPyAdapter` and attach it to DSPy's three targets.

    A convenience wrapper over the class API (``DSPyAdapter(client).connect()``),
    which stays the primary path and the one ``auto()`` targets. It exists because
    dspy genuinely has three separable surfaces — program, LM, optimizer — and
    only ``optimizer`` requires an explicit hand-off.

    Usage::

        adapter = instrument_dspy(client, my_program, optimizer=BootstrapFewShot(metric=m))
        compiled = optimizer.compile(my_program, trainset=trainset)
        adapter.disconnect()
    """
    adapter = DSPyAdapter(client, capture_config)
    adapter.connect(target=program, lm=lm, optimizer=optimizer)
    return adapter


__all__ = ["DSPyAdapter", "instrument_dspy"]
