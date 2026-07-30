"""BrowserUse adapter — wraps ``Agent.run()`` on the agent instance.

browser-use drives an LLM-controlled browser behind a single public coroutine,
so the adapter wraps the bound ``agent.run`` and walks the REAL
``AgentHistoryList`` it returns:

* steps live on ``history.history`` (each an ``AgentHistory``);
* a step's actions are the LIST at ``step.model_output.action`` (one
  ``ActionModel`` per action, whose ``model_dump()`` is ``{action_name: params}``);
* the page URL is at ``step.state.url``;
* per-action outcomes are the LIST at ``step.result``, index-paired with the
  actions;
* token usage lives on the HISTORY LIST (``history.usage``), not on steps — so
  exactly one ``model.invoke`` is emitted per run, and only when the agent's
  real model (``agent.llm``) is known.

The walk runs in a ``finally``, so a crashed run still reports its real partial
history (browser-use records it on the agent), its real latency and a
``run_failed`` transition; the exception propagates unchanged.

Why wrap rather than subclass: browser-use's Agent class has been renamed and
re-located several times across versions, and the ``on_step_start`` /
``on_step_end`` hooks were only formalised in 0.3. Wrapping the public method
works on every released version and degrades cleanly when the framework
upgrades.

Usage::

    adapter = instrument_browser_use(agent, client=client)
    await agent.run()
    uninstrument_browser_use()
"""

from __future__ import annotations

import inspect
import logging
import weakref
import functools
import threading
import importlib.metadata
from typing import Any, Dict, List, Tuple, Optional

from ._utils import truncate, safe_serialize
from ..._identity import _API_METHOD_RE, _s, _is_generic
from ._base_framework import FrameworkAdapter
from ..._capture_config import CaptureConfig

log = logging.getLogger(__name__)

_HAS_BROWSER_USE = False
try:
    from browser_use import Agent as _AgentCheck  # pyright: ignore[reportMissingImports]

    _HAS_BROWSER_USE = True
    del _AgentCheck
except ImportError:
    pass

#: Attribute names a usage summary may use for each token figure, in probe
#: order — browser-use has renamed these across versions, and 0.13's real
#: ``UsageSummary`` exposes ONLY the ``total_*`` spellings.
_PROMPT_TOKEN_ATTRS = ("total_prompt_tokens", "prompt_tokens", "total_input_tokens", "input_tokens")
_COMPLETION_TOKEN_ATTRS = ("total_completion_tokens", "completion_tokens", "total_output_tokens", "output_tokens")
_TOTAL_TOKEN_ATTRS = ("total_tokens",)
_CACHED_TOKEN_ATTRS = ("total_prompt_cached_tokens", "cached_tokens")
_COST_ATTRS = ("total_cost",)

#: Agent attributes copied into environment.config, when set. ``sensitive_data``
#: is deliberately absent: it holds the user's credentials.
_CONFIG_ATTRS = ("max_steps", "max_failures", "use_vision", "save_conversation_path")

#: Browser fields safe to report. Read off the modern ``browser_profile`` first
#: (0.13 replaced ``agent.browser.config``); no cookie/auth/credential field is
#: ever read. ``viewport``/``viewport_size`` are the new/old spellings.
_BROWSER_ATTRS = ("headless", "viewport", "viewport_size", "user_agent", "executable_path")

#: Browser actions carry HTML/DOM blobs — the caps are deliberately aggressive.
_TASK_LIMIT = 1000
_PARAMS_LIMIT = 400
_EXTRACTED_LIMIT = 500
_ERROR_LIMIT = 400

_SPAN_NAME = "browser_use:agent"


class BrowserUseAdapter(FrameworkAdapter):
    """BrowserUse adapter — see module docstring for the instrumentation model."""

    name = "browser_use"
    package = "browser-use"

    def _check_dependency(self, available: bool) -> None:
        """Point at the standalone install, NOT a ``layerlens[browser-use]`` extra.

        browser-use pins ``openai==2.16.0`` exactly, which is incompatible with
        the ``openai>=2.31.0`` the rest of the SDK needs, so it can't be a
        co-installable extra of this package. It runs in its own environment; the
        adapter itself has no import-time dependency on it.
        """
        if not available:
            raise ImportError(
                "The 'browser-use' package is required for browser_use instrumentation. "
                "It pins an older openai than the rest of the SDK, so install it in a "
                "dedicated environment: pip install browser-use"
            )

    def __init__(self, client: Any, capture_config: Optional[CaptureConfig] = None) -> None:
        super().__init__(client, capture_config)
        # id(agent) -> weakref. The wrapper (and the original bound method, via
        # its closure) lives on the agent instance itself, so the registry holds
        # no strong references — a dropped agent is collected and its entry
        # purged by the weakref callback. Restore-on-disconnect walks every
        # still-live agent (many agents per adapter, keyed by id).
        self._wrapped_agents: Dict[int, "weakref.ref[Any]"] = {}
        self._agent_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _on_connect(self, target: Any = None, **kwargs: Any) -> Any:
        self._check_dependency(_HAS_BROWSER_USE)
        if target is None:
            raise ValueError("BrowserUseAdapter requires a target agent: adapter.connect(target=agent)")
        self._metadata["framework_version"] = _detect_framework_version() or "unknown"
        self.instrument_agent(target, agent_name=kwargs.get("agent_name"))
        return target

    def _on_disconnect(self) -> None:
        with self._agent_lock:
            refs = list(self._wrapped_agents.values())
            self._wrapped_agents.clear()
        for ref in refs:
            agent = ref()
            if agent is not None:
                self._unwrap_agent(agent)

    # ------------------------------------------------------------------
    # Public instrumentation API
    # ------------------------------------------------------------------

    def instrument_agent(self, agent: Any, *, agent_name: Optional[str] = None) -> Any:
        """Wrap ``agent.run()`` so each invocation emits a trace.

        Idempotent, and never double-wraps an agent another adapter instance
        already instrumented (the marker on the wrapper is checked, not just
        this adapter's registry — two adapters would otherwise duplicate every
        event). Returns the agent so callers can chain.
        """
        run_method = getattr(agent, "run", None)
        if run_method is None:
            raise TypeError(f"BrowserUse adapter requires an object with a .run() method; got {type(agent).__name__}")
        if not inspect.iscoroutinefunction(run_method):
            # The wrapper awaits the original; fail here rather than with a
            # TypeError inside the user's first call.
            raise TypeError(
                f"BrowserUse adapter requires an async .run(); {type(agent).__name__}.run is not a coroutine function"
            )
        if getattr(run_method, "_layerlens_traced", False):
            return agent

        agent_id = id(agent)
        with self._agent_lock:
            if agent_id in self._wrapped_agents:
                return agent

        agent.run = self._build_traced_run(agent, run_method, agent_name)

        def _on_collect(_ref: "weakref.ref[Any]", aid: int = agent_id) -> None:
            self._purge(aid)

        with self._agent_lock:
            self._wrapped_agents[agent_id] = weakref.ref(agent, _on_collect)
        return agent

    def _purge(self, agent_id: int) -> None:
        """Weakref callback: drop the registry entry for a collected agent."""
        with self._agent_lock:
            self._wrapped_agents.pop(agent_id, None)

    def _unwrap_agent(self, agent: Any) -> None:
        instance_attr = vars(agent).get("run") if hasattr(agent, "__dict__") else None
        if instance_attr is None or not getattr(instance_attr, "_layerlens_traced", False):
            return  # not our wrapper (or already restored) — leave it alone
        original = getattr(instance_attr, "_layerlens_original", None)
        try:
            del agent.run
        except (AttributeError, TypeError):
            log.debug("layerlens: could not restore Agent.run() on %s", id(agent), exc_info=True)
            return
        # The original is normally the bound class method — deleting the
        # instance attribute re-exposes it. Only a pre-existing instance
        # attribute has to be reinstated explicitly.
        if original is not None and getattr(original, "__self__", None) is not agent:
            agent.run = original

    # ------------------------------------------------------------------
    # Run wrapping
    # ------------------------------------------------------------------

    def _build_traced_run(self, agent: Any, original_run: Any, agent_name: Optional[str]) -> Any:
        @functools.wraps(original_run)
        async def traced_run(*args: Any, **kwargs: Any) -> Any:
            self._begin_run()
            error: Optional[BaseException] = None
            result: Any = None
            resolved_name: Optional[str] = None
            try:
                # Telemetry is best-effort; the customer's automation is not.
                # A payload that cannot be built (an unrenderable task, a hostile
                # attribute) must never stop the run from happening.
                try:
                    resolved_name = _resolve_agent_name(agent, agent_name)
                    self._emit_run_input(agent, resolved_name, args, kwargs)
                except Exception:
                    log.debug("layerlens: could not emit the browser_use run input", exc_info=True)
                self._start_timer("run")
                try:
                    result = await original_run(*args, **kwargs)
                except BaseException as exc:
                    error = exc
                    raise
                finally:
                    latency_ms = self._stop_timer("run")
                    # A malformed history must never break the user's run, and
                    # _end_run must run regardless so the trace is flushed.
                    try:
                        self._emit_run_events(agent, resolved_name, result, error, latency_ms)
                    except Exception:
                        log.debug("layerlens: could not emit the browser_use run history", exc_info=True)
            finally:
                self._end_run()
            return result

        traced_run._layerlens_original = original_run  # type: ignore[attr-defined]
        traced_run._layerlens_traced = True  # type: ignore[attr-defined]
        return traced_run

    # ------------------------------------------------------------------
    # Run boundary events
    # ------------------------------------------------------------------

    def _emit_run_input(self, agent: Any, agent_name: Optional[str], args: Any, kwargs: Dict[str, Any]) -> None:
        root = self._get_root_span()
        task = _safe_getattr(agent, "task", "") or (kwargs.get("task") if kwargs else "")
        payload = self._identity_payload(agent_name)
        self._set_if_capturing(payload, "input_text", truncate(safe_serialize(task), _TASK_LIMIT))
        # max_steps is a run() parameter in browser-use, not an agent attribute
        # (it is absent from Agent.__init__), and it is the FIRST positional —
        # ``await agent.run(20)`` is idiomatic. Report it only when the caller
        # actually declared it; the framework's own default is not a declaration.
        max_steps = args[0] if args else kwargs.get("max_steps")
        if isinstance(max_steps, int) and not isinstance(max_steps, bool):
            payload["max_steps"] = max_steps
        self._emit("agent.input", payload, span_id=root, parent_span_id=None, span_name=_SPAN_NAME)
        self._emit_agent_config(agent, agent_name)

    def _emit_run_events(
        self,
        agent: Any,
        agent_name: Optional[str],
        result: Any,
        error: Optional[BaseException],
        latency_ms: Optional[float],
    ) -> None:
        history = _resolve_history(agent, result)
        steps = _resolve_steps(history)
        self._emit_history_events(history=history, steps=steps, agent=agent, agent_name=agent_name)
        self._emit_run_output(history, result, agent_name, steps, error, latency_ms)
        self._emit_state_change(agent_name, error)

    def _emit_run_output(
        self,
        history: Any,
        result: Any,
        agent_name: Optional[str],
        steps: List[Any],
        error: Optional[BaseException],
        latency_ms: Optional[float],
    ) -> None:
        payload = self._identity_payload(agent_name)
        self._set_if_capturing(payload, "output_text", truncate(_final_output_text(history, result), _TASK_LIMIT))
        payload["total_steps"] = len(steps)
        if latency_ms is not None:
            payload["latency_ms"] = latency_ms
        if error is not None:
            # The free-text error echoes the failing task/URL, so it is content
            # and is gated; error_type is a CATEGORY and always survives, so a
            # failure stays visible under capture_content=False.
            payload["error_type"] = type(error).__name__
            self._set_if_capturing(payload, "error", truncate(str(error), _ERROR_LIMIT))
        else:
            payload["status"] = "ok"
        self._emit(
            "agent.output",
            payload,
            span_id=self._get_root_span(),
            parent_span_id=None,
            span_name=_SPAN_NAME,
        )

    def _emit_state_change(self, agent_name: Optional[str], error: Optional[BaseException]) -> None:
        # The state_key + old/new value shape is ingest-required; the previous
        # shape (a bare event_subtype) was rejected at ingest.
        payload = self._identity_payload(agent_name)
        payload["state_key"] = "run_status"
        payload["state_type"] = "run_failed" if error else "run_complete"
        payload["old_value"] = "running"
        payload["new_value"] = "failed" if error else "complete"
        if error is not None:
            payload["error_type"] = type(error).__name__
            self._set_if_capturing(payload, "error", truncate(str(error), _ERROR_LIMIT))
        self._emit("agent.state.change", payload)

    # ------------------------------------------------------------------
    # History walking (real browser-use shapes)
    # ------------------------------------------------------------------

    def _emit_history_events(self, *, history: Any, steps: List[Any], agent: Any, agent_name: Optional[str]) -> None:
        for step_index, step in enumerate(steps):
            try:
                self._emit_step_actions(step=step, step_index=step_index, agent_name=agent_name)
            except Exception:
                log.debug("layerlens: failed to emit browser_use history step %d", step_index, exc_info=True)
        self._emit_run_usage(history=history, agent=agent, agent_name=agent_name)

    def _emit_step_actions(self, *, step: Any, step_index: int, agent_name: Optional[str]) -> None:
        """Emit one tool.call per REAL action in ``step.model_output.action``."""
        model_output = getattr(step, "model_output", None)
        actions = getattr(model_output, "action", None) if model_output is not None else None
        if actions is None:
            return
        if not isinstance(actions, list):
            actions = [actions]

        state = getattr(step, "state", None)
        url = getattr(state, "url", None) if state is not None else None
        results = getattr(step, "result", None)
        if not isinstance(results, list):
            results = [results] if results is not None else []

        for action_index, action in enumerate(actions):
            named = _action_name_and_params(action)
            if named is None:
                log.debug(
                    "layerlens: no real action name for browser_use step %d action %d (%s) — skipping",
                    step_index,
                    action_index,
                    type(action).__name__,
                )
                continue
            action_name, params = named
            # An action past the end of the results list has NO outcome — the
            # neighbouring result is not its own.
            result_obj = results[action_index] if action_index < len(results) else None
            success = getattr(result_obj, "success", None) if result_obj is not None else None
            error = getattr(result_obj, "error", None) if result_obj is not None else None
            extracted = getattr(result_obj, "extracted_content", None) if result_obj is not None else None

            payload = self._identity_payload(agent_name)
            payload["step_index"] = step_index
            payload["action_index"] = action_index
            payload["tool_name"] = action_name
            # A browsed URL routinely carries query-string PII / session tokens.
            self._set_if_capturing(payload, "url", str(url) if url else None)
            if isinstance(success, bool):
                payload["success"] = success
            self._set_if_capturing(payload, "input", truncate(safe_serialize(params), _PARAMS_LIMIT))
            self._set_if_capturing(payload, "output", truncate(str(extracted), _EXTRACTED_LIMIT) if extracted else None)
            self._set_if_capturing(payload, "error", truncate(str(error), _ERROR_LIMIT) if error else None)
            # NO latency_ms: the history exposes no per-ACTION timing (N actions
            # share one step) and dividing a step's duration across them is
            # fabrication.
            self._emit(
                "tool.call",
                payload,
                span_id=self._new_span_id(),
                parent_span_id=self._get_root_span(),
                span_name=f"browser_use:{action_name}",
            )

    def _emit_run_usage(self, *, history: Any, agent: Any, agent_name: Optional[str]) -> None:
        """Emit ONE model.invoke (+ cost.record) from the history-level usage.

        Usage lives on the ``AgentHistoryList``, not on steps. When the agent's
        real model name is unknowable nothing is emitted at all — an
        unattributable invocation is fabricated data.
        """
        llm = _safe_getattr(agent, "llm", None)
        model = _extract_model_name_from_llm(llm)
        if not model:
            log.debug("layerlens: agent.llm reports no real model name — skipping browser_use model.invoke")
            return
        usage = getattr(history, "usage", None) if history is not None else None
        tokens = _usage_tokens(usage)
        provider = _detect_provider_from_llm(llm)

        payload = self._identity_payload(agent_name)
        payload["model"] = model
        if provider:
            payload["provider"] = provider
        payload.update(tokens)
        # NO latency_ms: the LLM-only share of the run duration is browser time
        # plus LLM time and is not reported by the framework.
        self._emit("model.invoke", payload)
        self._emit_cost_record(usage, tokens, model, provider, agent_name)

    def _emit_cost_record(
        self,
        usage: Any,
        tokens: Dict[str, int],
        model: str,
        provider: Optional[str],
        agent_name: Optional[str],
    ) -> None:
        """Emit the run's cost when it is HONESTLY derivable.

        browser-use prices the run itself (``UsageSummary.total_cost``), and its
        own figure is preferred — setting ``cost_usd`` makes the shared
        price-on-emit chokepoints skip (they fill, never clobber). Otherwise the
        shared pricer derives it, but ONLY from prompt/completion rates: a usage
        with no REAL prompt/completion count (a total-only usage, or one whose
        prompt/completion are zero) would be stamped with a fabricated $0.00, so
        no cost.record is emitted at all in those cases.
        """
        if not tokens:
            return
        payload = self._identity_payload(agent_name)
        payload["model"] = model
        if provider:
            payload["provider"] = provider
        payload.update(tokens)
        cached = _probe_int(usage, _CACHED_TOKEN_ATTRS)
        if cached:
            payload["cached_tokens"] = cached
        cost = _probe_float(usage, _COST_ATTRS)
        if cost is not None and cost > 0:
            payload["cost_usd"] = cost
        elif not tokens.get("tokens_prompt") and not tokens.get("tokens_completion"):
            # Tests the VALUES, not merely key presence: a prompt/completion of
            # zero prices to exactly the same fabricated $0.00 an absent one
            # would, so both must take the honest omission.
            return
        self._emit("cost.record", payload)

    # ------------------------------------------------------------------
    # environment.config
    # ------------------------------------------------------------------

    def _emit_agent_config(self, agent: Any, agent_name: Optional[str]) -> None:
        """Emitted per run so every trace carries its own config node."""
        payload = self._identity_payload(agent_name)
        payload["config"] = self._extract_agent_config(agent)
        self._emit(
            "environment.config",
            payload,
            span_id=self._new_span_id(),
            parent_span_id=self._get_root_span(),
            span_name="browser_use:config",
        )

    def _extract_agent_config(self, agent: Any) -> Dict[str, Any]:
        """A config snapshot: what was this agent set up to do.

        Every field is OMITTED when unset — never defaulted. Only an explicit
        safe-field allowlist is read, so no credential (``sensitive_data``,
        cookies, auth) is ever captured.
        """
        config: Dict[str, Any] = {"framework": self.name}
        # The task is the user's raw natural-language instruction — content.
        self._set_if_capturing(config, "task", _safe_getattr(agent, "task", None))
        for attr in _CONFIG_ATTRS:
            value = _safe_getattr(agent, attr, None)
            if value is not None:
                config[attr] = value

        llm = _safe_getattr(agent, "llm", None)
        if llm is not None:
            model = _extract_model_name_from_llm(llm)
            if model:
                config["model"] = model
            provider = _detect_provider_from_llm(llm)
            if provider:
                config["provider"] = provider

        # 0.13 replaced ``agent.browser.config`` with ``browser_profile``;
        # probe the modern object first and fall back to the legacy ``.config``
        # rather than silently capturing nothing.
        profile = (
            _safe_getattr(agent, "browser_profile", None)
            or _safe_getattr(agent, "browser_session", None)
            or _safe_getattr(agent, "browser", None)
            or _safe_getattr(agent, "browser_context", None)
        )
        source = profile
        legacy = _safe_getattr(source, "config", None) if source is not None else None
        if legacy is not None:
            source = legacy
        if source is not None:
            for field in _BROWSER_ATTRS:
                value = _safe_getattr(source, field, None)
                if value is not None:
                    config[f"browser.{field}"] = str(value) if field == "executable_path" else value
        return config

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _identity_payload(self, agent_name: Optional[str]) -> Dict[str, Any]:
        """A payload seeded with the framework marker and, when the producer
        DECLARED one, the honest agent identity. A real ``browser_use.Agent``
        has no ``name`` attribute, so an unnamed agent stays blank rather than
        surfacing a placeholder as if the developer had chosen it."""
        payload = self._payload()
        if agent_name:
            payload["agent_name"] = agent_name
            payload["agent_id"] = agent_name
        return payload


# -- Module-level helpers ---------------------------------------------------


def _safe_getattr(obj: Any, attr: str, default: Any = None) -> Any:
    """``getattr`` that never lets a custom ``__getattr__`` escape into the
    user's run — a hostile/lazy attribute must not break instrumentation."""
    try:
        return getattr(obj, attr, default)
    except Exception:
        log.debug("layerlens: failed to read %s.%s", type(obj).__name__, attr, exc_info=True)
        return default


def _detect_framework_version() -> Optional[str]:
    """The installed browser-use version, or None.

    Read from the distribution metadata: browser-use exposes no ``__version__``
    attribute (verified absent on 0.13), so probing the module silently reports
    nothing and the drift between adapter and framework goes unseen.
    """
    try:
        return importlib.metadata.version("browser-use")
    except importlib.metadata.PackageNotFoundError:
        pass
    try:
        import browser_use  # pyright: ignore[reportMissingImports]
    except ImportError:
        return None
    version = getattr(browser_use, "__version__", None)
    return version if isinstance(version, str) and version else None


def _resolve_agent_name(agent: Any, explicit: Optional[str]) -> Optional[str]:
    """The producer-DECLARED agent name, honest-guarded — or None.

    browser-use's ``Agent`` has no ``name`` attribute and no name/id constructor
    parameter, so the only honest source is a caller-supplied ``agent_name``.
    Reuses the shared identity guard so a generic label or a dotted API-method
    string is never surfaced as an agent.
    """
    raw = explicit if isinstance(explicit, str) and explicit else _safe_getattr(agent, "name", None)
    name = _s(raw)
    if name is None or _is_generic(name) or _API_METHOD_RE.match(name.lower()):
        return None
    return name


def _resolve_history(agent: Any, result: Any) -> Any:
    """Locate the AgentHistoryList: the run() result, else the agent's own.

    On the error path there is no result, so the agent's partial history is the
    honest fallback (browser-use records it as the run proceeds).
    """
    for candidate in (
        result,
        _safe_getattr(_safe_getattr(agent, "state", None), "history", None),
        _safe_getattr(agent, "history", None),
    ):
        if candidate is None:
            continue
        if hasattr(candidate, "history") or isinstance(candidate, list):
            return candidate
    return None


def _resolve_steps(history: Any) -> List[Any]:
    """Extract the step list from an AgentHistoryList (or a plain list)."""
    if history is None:
        return []
    steps = getattr(history, "history", None)
    if isinstance(steps, list):
        return steps
    if isinstance(history, list):
        return history
    try:
        return list(history)
    except TypeError:
        return []


def _action_name_and_params(action: Any) -> Optional[Tuple[str, Any]]:
    """Resolve the REAL action name (+ params) from an ActionModel, or None.

    browser-use's ``ActionModel.model_dump()`` yields ``{action_name: params}``.
    Dict and attribute shapes are also supported. Returns None when no real name
    is determinable — callers MUST skip rather than emit a placeholder.
    """
    dumped: Any = None
    if isinstance(action, dict):
        dumped = action
    elif hasattr(action, "model_dump"):
        try:
            dumped = action.model_dump(exclude_unset=True, exclude_none=True)
        except TypeError:
            try:
                dumped = action.model_dump()
            except Exception:
                dumped = None
        except Exception:
            dumped = None
    if isinstance(dumped, dict):
        # Legacy explicit shape: {"name": ..., "args": ...}
        name_field = dumped.get("name")
        if isinstance(name_field, str) and name_field:
            return name_field, dumped.get("args")
        for key, value in dumped.items():
            if value is not None:
                return str(key), value
    for attr in ("action_name", "name"):
        name_attr = getattr(action, attr, None)
        if isinstance(name_attr, str) and name_attr:
            return name_attr, getattr(action, "args", None)
    return None


def _usage_tokens(usage: Any) -> Dict[str, int]:
    """Real token figures from a browser-use UsageSummary.

    The shared ``_normalize_tokens`` probes only ``prompt_tokens``/
    ``input_tokens``; the real ``UsageSummary`` exposes neither, so it drops the
    genuine prompt/completion counts and leaves a total that cannot be priced.
    Only the ATTRIBUTE NAMES deviate from the shared helper — its all-zero strip
    is preserved verbatim below, because callers gate on ``if tokens:``.
    """
    tokens: Dict[str, int] = {}
    prompt = _probe_int(usage, _PROMPT_TOKEN_ATTRS)
    completion = _probe_int(usage, _COMPLETION_TOKEN_ATTRS)
    total = _probe_int(usage, _TOTAL_TOKEN_ATTRS)
    if prompt is not None:
        tokens["tokens_prompt"] = prompt
    if completion is not None:
        tokens["tokens_completion"] = completion
    if total is not None:
        tokens["tokens_total"] = total
    elif prompt is not None and completion is not None:
        # Never a partial sum — only when BOTH are real.
        tokens["tokens_total"] = prompt + completion
    # An ALL-zero usage is the ABSENCE of a measurement, not a measured zero:
    # browser-use assigns this exact summary whenever its token service recorded
    # no LLM entry (a crash or Ctrl-C before the first call, a 0-step run).
    # Reporting it as counts would price it to a fabricated $0.00. A real zero
    # ALONGSIDE a real count is a genuine figure and survives.
    if tokens and not any(tokens.values()):
        return {}
    return tokens


def _probe_int(obj: Any, attrs: Tuple[str, ...]) -> Optional[int]:
    value = _probe_number(obj, attrs)
    return int(value) if value is not None else None


def _probe_float(obj: Any, attrs: Tuple[str, ...]) -> Optional[float]:
    value = _probe_number(obj, attrs)
    return float(value) if value is not None else None


def _probe_number(obj: Any, attrs: Tuple[str, ...]) -> Optional[float]:
    if obj is None:
        return None
    for attr in attrs:
        value = obj.get(attr) if isinstance(obj, dict) else _safe_getattr(obj, attr, None)
        # bools are ints in Python — True must never become a count of 1.
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _final_output_text(history: Any, result: Any) -> Optional[str]:
    """The run's final output, resolved honestly from the history.

    Never stringifies the history object as if it were the agent's answer.
    """
    if history is not None:
        final_result = getattr(history, "final_result", None)
        if callable(final_result):
            try:
                final = final_result()
                if final:
                    return str(final)
            except Exception:
                log.debug("layerlens: history.final_result() failed", exc_info=True)
    if result is not None and not hasattr(result, "history"):
        return str(result)
    return None


def _extract_model_name_from_llm(llm: Any) -> Optional[str]:
    """The REAL model name from an llm wrapper, or None.

    Probes the ``model_name``/``model`` string attributes only. The wrapper's
    class name is NOT a model name and is never substituted — callers must omit
    model telemetry when this returns None.
    """
    if llm is None:
        return None
    for attr in ("model_name", "model"):
        value = _safe_getattr(llm, attr, None)
        if isinstance(value, str) and value:
            return value
    return None


def _detect_provider_from_llm(llm: Any) -> Optional[str]:
    """The llm's provider — its own declaration first, else inferred."""
    if llm is None:
        return None
    # browser-use's own llm wrappers declare this; sniffing is the fallback for
    # a custom/third-party wrapper.
    declared = _safe_getattr(llm, "provider", None)
    if isinstance(declared, str) and declared:
        return declared

    cls_name = type(llm).__name__.lower()
    if "openai" in cls_name:
        return "openai"
    if "anthropic" in cls_name or "claude" in cls_name:
        return "anthropic"
    if "google" in cls_name or "gemini" in cls_name or "vertex" in cls_name:
        return "google"
    if "azure" in cls_name:
        return "azure_openai"
    if "bedrock" in cls_name:
        return "aws_bedrock"

    model = str(_extract_model_name_from_llm(llm) or "").lower()
    if model.startswith(("gpt", "o1", "o3")):
        return "openai"
    if "claude" in model:
        return "anthropic"
    if "gemini" in model:
        return "google"
    return None


def instrument_browser_use(
    agent: Any,
    *,
    client: Any = None,
    capture_config: Optional[CaptureConfig] = None,
    agent_name: Optional[str] = None,
) -> BrowserUseAdapter:
    """Attach a layerlens adapter to a BrowserUse ``Agent``.

    The agent's ``run()`` is wrapped in place; ``uninstrument_browser_use()``
    restores it. When an adapter is already registered the SAME adapter is
    reused so a second agent is added rather than the first being unwrapped.

    Args:
        agent: A BrowserUse ``Agent`` with an async ``.run()``.
        client: The ``layerlens.Stratix`` instance events are uploaded through.
        capture_config: Optional ``CaptureConfig``; defaults to ``standard()``.
        agent_name: Agent identity stamped on every event. A real browser-use
            Agent declares no name, so without this the Agent column stays
            honestly blank.
    """
    from .._registry import get, register

    existing = get("browser_use")
    if isinstance(existing, BrowserUseAdapter):
        existing.instrument_agent(agent, agent_name=agent_name)
        return existing
    adapter = BrowserUseAdapter(client, capture_config=capture_config)
    adapter.connect(target=agent, agent_name=agent_name)
    register("browser_use", adapter)
    return adapter


def uninstrument_browser_use() -> None:
    from .._registry import unregister

    unregister("browser_use")


__all__ = ["BrowserUseAdapter", "instrument_browser_use", "uninstrument_browser_use"]
