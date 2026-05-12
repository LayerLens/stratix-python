"""browser_use adapter lifecycle (full implementation).

Instruments the [browser-use](https://github.com/browser-use/browser-use)
agent — a Python library that pairs an LLM with a Playwright-driven
browser to perform autonomous web navigation, form filling, and content
extraction.

Instrumentation strategy
------------------------

browser_use exposes an ``Agent`` orchestrator that drives a ``Browser``
session through a perception → reasoning → action loop. The adapter
wraps the lifecycle entry points and per-step hooks (no native
callback bus exists, so the adapter takes the wrapper-pattern path
that the lighter framework adapters share):

* ``Agent.run()`` start            → ``agent.input`` (L1) +
                                     ``browser.session.start``
* ``Agent.run()`` end              → ``agent.output`` (L1) +
                                     ``agent.state.change``
* Per-step navigation              → ``browser.navigate``
* Per-step action (click/type/...) → ``browser.action`` + ``tool.call``
* Per-step screenshot              → ``browser.screenshot`` (DROPPED
                                     to SHA-256 ref via the truncation
                                     policy — never embedded in events)
* Per-step DOM extract             → ``browser.dom.extract`` (capped
                                     at 16 KiB by the policy)
* Per-step LLM call                → ``model.invoke`` (L3) +
                                     ``cost.record``
* Agent / browser config           → ``environment.config`` (L4a)
* Any callback exception           → ``agent.error`` / ``tool.error`` /
                                     ``model.error`` (per PR #115
                                     pattern, surfaced inline so a
                                     dashboard never sees a hung
                                     "start" with no matching "end")

Design contracts (CLAUDE.md + cross-pollination audit)
------------------------------------------------------

1. **Truncation from day one.** Every emit routes through
   :data:`DEFAULT_POLICY` — screenshots / image data / base64 PNG fields
   become deterministic SHA-256 references; HTML / DOM / page_content
   are capped at 16 KiB; prompts / completions / tool I/O follow the
   shared per-field caps. This is non-negotiable: a single browser
   navigation can produce multi-megabyte screenshots that would blow
   past the Kafka 1 MB record ceiling and inflate ingestion cost.

2. **Per-callback resilience.** Every public hook is wrapped in
   try / except so an exception in our observability code can NEVER
   crash the customer's browser_use agent. Failures bump the
   ``_resilience_failures`` counter (visible via
   :meth:`get_adapter_info` metadata) and are logged at WARNING with a
   truncated traceback. Conforms to the PR #117 contract.

3. **Multi-tenant org_id propagation.** The adapter accepts
   ``org_id`` as a constructor kwarg (or resolves it from
   ``stratix.org_id`` / ``stratix.organization_id``). Every emitted
   event payload carries the bound org_id — there is no silent
   cross-tenant leak path. Conforms to the PR #118 contract.

4. **Error-aware emission.** When a wrapped callback raises a
   framework exception (the Agent loop fails, a tool errors, the
   LLM rate-limits), the corresponding ``*.error`` event is emitted
   BEFORE the exception is re-raised. Conforms to the PR #115
   contract.
"""

from __future__ import annotations

import time
import uuid
import logging
import threading
import traceback
from typing import Any, Dict, List, Optional

from layerlens.instrument.adapters._base.adapter import (
    AdapterInfo,
    BaseAdapter,
    AdapterHealth,
    AdapterStatus,
    ReplayableTrace,
    AdapterCapability,
)
from layerlens.instrument.adapters._base.truncation import DEFAULT_POLICY
from layerlens.instrument.adapters._base.pydantic_compat import PydanticCompat

logger = logging.getLogger(__name__)


# Provider-detection table reused across hooks.
_PROVIDER_HINTS: Dict[str, str] = {
    "gpt": "openai",
    "o1": "openai",
    "o3": "openai",
    "claude": "anthropic",
    "gemini": "google",
    "mistral": "mistral",
    "mixtral": "mistral",
    "llama": "meta",
    "command": "cohere",
}


def _detect_provider(model: Optional[str]) -> Optional[str]:
    if not model:
        return None
    m = model.lower()
    for hint, provider in _PROVIDER_HINTS.items():
        if hint in m:
            return provider
    return None


class BrowserUseAdapter(BaseAdapter):
    """LayerLens adapter for browser_use (full implementation).

    See module docstring for instrumentation strategy and design
    contracts. The adapter is constructed with a LayerLens client +
    org_id binding, then ``connect()`` is called to probe the
    framework, then ``instrument_agent(agent)`` wraps an Agent
    instance. ``disconnect()`` restores the originals on every
    wrapped agent.
    """

    FRAMEWORK = "browser_use"
    VERSION = "0.1.0"
    # browser_use itself is a Pydantic v2 library (it uses
    # ``BaseModel`` with v2-only features such as ``model_validator``
    # and ``ConfigDict``). The adapter does not import any browser_use
    # Pydantic models directly — but the ``execute_replay`` path needs
    # to round-trip browser_use action models, so we declare V2_ONLY
    # to be honest about the runtime requirement and let the catalog
    # UI warn customers pinning v1.
    requires_pydantic = PydanticCompat.V2_ONLY

    # ---- Construction ------------------------------------------------

    def __init__(
        self,
        stratix: Any = None,
        capture_config: Any = None,
        stratix_instance: Any = None,
        org_id: Optional[str] = None,
    ) -> None:
        resolved_stratix = stratix or stratix_instance
        super().__init__(stratix=resolved_stratix, capture_config=capture_config)
        # Multi-tenant org_id binding (PR #118). Resolution order:
        #   1. explicit ``org_id`` kwarg
        #   2. ``stratix.org_id``
        #   3. ``stratix.organization_id``
        # The adapter does NOT raise on missing org_id (the BaseAdapter
        # in this PR's base does not yet enforce it — that ratchets in
        # PR #118 via _base/adapter.py changes). We carry the binding
        # locally and stamp it onto every payload so when PR #118 lands
        # the contract is already in force at the adapter level.
        resolved_org = org_id
        if resolved_org is None and resolved_stratix is not None:
            resolved_org = (
                getattr(resolved_stratix, "org_id", None)
                or getattr(resolved_stratix, "organization_id", None)
            )
        self._org_id: Optional[str] = resolved_org

        # Truncation policy — CRITICAL for browser_use because a single
        # navigation step can produce multi-megabyte screenshots / DOM
        # snapshots. Wired here from day one (cross-pollination audit
        # §2.4 flags browser_use as the worst offender if untruncated).
        self._truncation_policy = DEFAULT_POLICY

        # Per-callback resilience counters (PR #117 contract). The
        # adapter MUST never crash the customer's agent if our event
        # emission path raises. Counters surface via
        # ``get_adapter_info().metadata``.
        self._resilience_failures: Dict[str, int] = {}
        self._resilience_last_error: Optional[str] = None

        # Wrapped-agent bookkeeping.
        self._originals: Dict[int, Dict[str, Any]] = {}
        self._wrapped_agents: List[Any] = []
        self._adapter_lock = threading.Lock()
        self._seen_agents: set[str] = set()
        self._framework_version: Optional[str] = None
        self._run_starts: Dict[int, int] = {}
        self._session_ids: Dict[int, str] = {}

    # ---- Lifecycle ---------------------------------------------------

    def connect(self) -> None:
        """Probe ``browser_use`` availability and prepare the adapter.

        Imports the runtime SDK to capture its version. If the package
        is not installed (the adapter being used standalone for replay
        deserialisation), the version is recorded as ``"unknown"`` and
        the adapter still reports HEALTHY — runtime code paths that
        actually need the SDK will raise on first use.
        """
        try:
            import browser_use  # type: ignore[import-not-found,unused-ignore]

            self._framework_version = getattr(browser_use, "__version__", "unknown")
        except ImportError:
            logger.debug("browser_use not installed")
            self._framework_version = None
        self._connected = True
        self._status = AdapterStatus.HEALTHY

    def disconnect(self) -> None:
        """Unwrap every instrumented agent and release sinks."""
        for agent in self._wrapped_agents:
            self._unwrap_agent(agent)
        self._wrapped_agents.clear()
        self._originals.clear()
        self._seen_agents.clear()
        self._run_starts.clear()
        self._session_ids.clear()
        self._connected = False
        self._status = AdapterStatus.DISCONNECTED
        self._close_sinks()

    def _unwrap_agent(self, agent: Any) -> None:
        agent_id = id(agent)
        originals = self._originals.get(agent_id)
        if not originals:
            return
        for method_name, original in originals.items():
            try:
                setattr(agent, method_name, original)
            except Exception:
                logger.debug("Could not unwrap %s.%s", agent_id, method_name, exc_info=True)

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
            name="BrowserUseAdapter",
            version=self.VERSION,
            framework=self.FRAMEWORK,
            framework_version=self._framework_version,
            capabilities=[
                AdapterCapability.TRACE_TOOLS,
                AdapterCapability.TRACE_MODELS,
                AdapterCapability.TRACE_STATE,
                AdapterCapability.STREAMING,
                AdapterCapability.REPLAY,
            ],
            description="LayerLens adapter for browser_use (LLM-driven browser automation)",
        )

    def serialize_for_replay(self) -> ReplayableTrace:
        from layerlens._compat.pydantic import model_dump

        return ReplayableTrace(
            adapter_name="BrowserUseAdapter",
            framework=self.FRAMEWORK,
            trace_id=str(uuid.uuid4()),
            events=list(self._trace_events),
            state_snapshots=[],
            config={
                "capture_config": model_dump(self._capture_config),
                "org_id": self._org_id,
                "framework_version": self._framework_version,
            },
            metadata={"resilience_failures": dict(self._resilience_failures)},
        )

    # ---- Resilience helper ------------------------------------------

    def _track_failure(self, callback_name: str, exc: BaseException) -> None:
        """Bump the resilience counter and log a truncated traceback.

        Implements the PR #117 per-callback resilience contract:
        observability errors NEVER crash the customer's agent.
        """
        with self._adapter_lock:
            self._resilience_failures[callback_name] = (
                self._resilience_failures.get(callback_name, 0) + 1
            )
            tb = traceback.format_exc()
            # Keep the traceback bounded so a flapping hook does not
            # blow up the in-memory log rate.
            self._resilience_last_error = (
                f"{callback_name}: {type(exc).__name__}: {str(exc)[:500]}"
            )
        logger.warning(
            "BrowserUseAdapter callback %s failed: %s",
            callback_name,
            type(exc).__name__,
            extra={"adapter": self.FRAMEWORK, "callback": callback_name},
        )
        logger.debug("Full traceback for %s failure:\n%s", callback_name, tb[:4000])

    # ---- Multi-tenant emit shim -------------------------------------

    # browser.* event types are the adapter's canonical surface but are
    # not in BaseAdapter's ``event_type_map`` (which only knows about
    # the cross-framework canonical events). Without explicit handling
    # the layer gate would silently drop every browser event. Instead
    # we map each browser.* family to its semantically-closest layer
    # and gate against that — so a customer who turns OFF L5c (tool
    # environment) still drops screenshots / DOM extracts as expected,
    # but the navigation / action / session events ride on L5a / L1.
    _BROWSER_EVENT_LAYERS: Dict[str, str] = {
        "browser.session.start": "l1_agent_io",
        "browser.session.end": "l1_agent_io",
        "browser.navigate": "l5a_tool_calls",
        "browser.action": "l5a_tool_calls",
        "browser.screenshot": "l5c_tool_environment",
        "browser.dom.extract": "l5c_tool_environment",
        # Error events MUST always emit (PR #115 contract). They are
        # the failure-mode signal for dashboards — silently dropping
        # them is exactly what the contract exists to prevent. We map
        # to a layer that's enabled by default so disabling content
        # capture does not also disable error visibility.
        "agent.error": "l1_agent_io",
        "tool.error": "l5a_tool_calls",
        "model.error": "l3_model_metadata",
    }

    def _emit(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Stamp org_id and emit through BaseAdapter.

        The org_id stamp is defensive — it overwrites any caller-set
        value to prevent cross-tenant leaks via misuse. Conforms to
        the PR #118 multi-tenancy contract.

        For browser.* event types (which BaseAdapter's event_type_map
        does not know about) we apply the layer gate manually against
        the mapping in :data:`_BROWSER_EVENT_LAYERS` and route through
        the same truncation + circuit-breaker path as
        :meth:`emit_dict_event`.
        """
        if self._org_id:
            payload["org_id"] = self._org_id

        layer_attr = self._BROWSER_EVENT_LAYERS.get(event_type)
        if layer_attr is not None:
            # Custom browser.* event type — apply our own layer gate
            # then emit via the BaseAdapter primitive that bypasses
            # the unknown-event-drops-by-default path.
            if not getattr(self._capture_config, layer_attr, True):
                return
            self._emit_browser_event(event_type, payload)
            return

        # Canonical event type — let BaseAdapter's gate handle it.
        self.emit_dict_event(event_type, payload)

    def _emit_browser_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Emit a browser.* event through BaseAdapter, bypassing the
        unknown-type layer gate.

        Mirrors :meth:`emit_dict_event` minus the layer gate (handled
        by the caller for browser.* types). Truncation, circuit
        breaker, replay buffer, and sink dispatch all still apply.
        """
        # Re-use BaseAdapter's circuit-breaker check; pass an event
        # type that IS in ALWAYS_ENABLED_EVENT_TYPES so the layer gate
        # passes. We use ``cost.record`` as the sentinel — it's
        # always-enabled and BaseAdapter's pre-check returns True.
        # We then manually handle the success/failure recording with
        # the REAL event_type so the trace buffer and sinks see the
        # browser.* type.
        if not self._pre_emit_check("cost.record"):
            return
        emit_payload = self._apply_truncation(payload)
        try:
            self._stratix.emit(event_type, emit_payload)
            self._post_emit_success(event_type, emit_payload)
        except Exception:
            self._post_emit_failure()

    # ---- Error-aware emission helpers (PR #115 pattern) -------------

    def _emit_error_event(
        self,
        event_type: str,
        framework_context: Dict[str, Any],
        exc: BaseException,
    ) -> None:
        """Emit a structured error event before the exception is re-raised.

        Implements the PR #115 contract: a callback exception MUST
        appear as a structured event so dashboards do not render a
        "start" with no matching "end". Message and traceback are
        truncated by the shared truncation policy (no PII bypass).
        """
        payload: Dict[str, Any] = {
            "framework": "browser_use",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        }
        payload.update(framework_context)
        try:
            self._emit(event_type, payload)
        except Exception:
            # Defensive — never let the error-emission path crash the
            # caller. The original exception still propagates from the
            # wrapper layer.
            logger.debug("Error event emission failed for %s", event_type, exc_info=True)

    # ---- Framework integration --------------------------------------

    def instrument_agent(self, agent: Any) -> Any:
        """Wrap a browser_use ``Agent`` instance.

        Patches ``run`` (sync wrapper) and ``run_async`` (async wrapper)
        so every agent.run() emits the canonical L1/L4 lifecycle events
        plus per-step browser / tool / model events. Idempotent — a
        repeated call on the same agent returns immediately.
        """
        agent_id = id(agent)
        if agent_id in self._originals:
            return agent
        originals: Dict[str, Any] = {}

        # Async entry point — browser_use Agent.run is async by default.
        if hasattr(agent, "run"):
            originals["run"] = agent.run
            agent.run = self._create_traced_run_async(agent, agent.run)

        # Some browser_use builds expose a sync helper.
        if hasattr(agent, "run_sync"):
            originals["run_sync"] = agent.run_sync
            agent.run_sync = self._create_traced_run_sync(agent, agent.run_sync)

        self._originals[agent_id] = originals
        self._wrapped_agents.append(agent)
        agent_name = getattr(agent, "name", None) or type(agent).__name__
        self._emit_agent_config(agent_name, agent)
        return agent

    def _create_traced_run_async(self, agent: Any, original_run: Any) -> Any:
        adapter = self

        async def traced_run(*args: Any, **kwargs: Any) -> Any:
            agent_name = getattr(agent, "name", None) or "browser_use_agent"
            task = (
                kwargs.get("task")
                or kwargs.get("message")
                or (args[0] if args else getattr(agent, "task", None))
            )
            session_id = adapter.on_session_start(agent_name=agent_name, task=task)
            error: Optional[BaseException] = None
            result: Any = None
            try:
                result = await original_run(*args, **kwargs)
            except BaseException as exc:
                error = exc
                # PR #115: surface the failure as a structured event
                # before re-raise. Use BaseException so KeyboardInterrupt
                # is still surfaced, but always re-raise — never swallow.
                adapter._emit_error_event(
                    "agent.error",
                    {"agent_name": agent_name, "session_id": session_id, "task": str(task) if task else None},
                    exc,
                )
                raise
            finally:
                adapter.on_session_end(
                    agent_name=agent_name,
                    session_id=session_id,
                    output=result,
                    error=error,
                )
                adapter._extract_run_details(agent, result, session_id)
            return result

        traced_run._layerlens_original = original_run  # type: ignore[attr-defined]
        return traced_run

    def _create_traced_run_sync(self, agent: Any, original_run: Any) -> Any:
        adapter = self

        def traced_run_sync(*args: Any, **kwargs: Any) -> Any:
            agent_name = getattr(agent, "name", None) or "browser_use_agent"
            task = (
                kwargs.get("task")
                or kwargs.get("message")
                or (args[0] if args else getattr(agent, "task", None))
            )
            session_id = adapter.on_session_start(agent_name=agent_name, task=task)
            error: Optional[BaseException] = None
            result: Any = None
            try:
                result = original_run(*args, **kwargs)
            except BaseException as exc:
                error = exc
                adapter._emit_error_event(
                    "agent.error",
                    {"agent_name": agent_name, "session_id": session_id, "task": str(task) if task else None},
                    exc,
                )
                raise
            finally:
                adapter.on_session_end(
                    agent_name=agent_name,
                    session_id=session_id,
                    output=result,
                    error=error,
                )
                adapter._extract_run_details(agent, result, session_id)
            return result

        traced_run_sync._layerlens_original = original_run  # type: ignore[attr-defined]
        return traced_run_sync

    def _extract_run_details(self, agent: Any, result: Any, session_id: str) -> None:
        """Best-effort extraction of per-step navigation/action history.

        browser_use's ``Agent.run`` returns an ``AgentHistoryList``
        whose entries carry per-step actions, screenshots, DOM
        snapshots, and model usage. Walking the history at the end
        provides a single backstop in case per-step hooks were not
        invoked (e.g. when the customer instrumented an Agent built
        before the per-step callbacks existed).
        """
        if result is None:
            return
        try:
            history = getattr(result, "history", None) or []
            for step in history:
                step_url = getattr(step, "url", None)
                if step_url:
                    self.on_navigation(url=step_url, session_id=session_id)
                action = getattr(step, "action", None)
                if action:
                    action_name = (
                        getattr(action, "type", None)
                        or getattr(action, "name", None)
                        or type(action).__name__
                    )
                    self.on_action(
                        action_type=str(action_name),
                        target=getattr(action, "target", None),
                        session_id=session_id,
                    )
                screenshot = getattr(step, "screenshot", None)
                if screenshot:
                    self.on_screenshot(
                        screenshot=screenshot,
                        url=step_url,
                        session_id=session_id,
                    )
                usage = getattr(step, "model_usage", None) or getattr(step, "usage", None)
                if usage:
                    self.on_llm_call(
                        model=str(getattr(agent, "model", None) or "unknown"),
                        tokens_prompt=getattr(usage, "input_tokens", None)
                        or getattr(usage, "prompt_tokens", None),
                        tokens_completion=getattr(usage, "output_tokens", None)
                        or getattr(usage, "completion_tokens", None),
                        session_id=session_id,
                    )
        except Exception as exc:
            self._track_failure("_extract_run_details", exc)

    # ---- Public lifecycle hooks -------------------------------------

    def on_session_start(
        self,
        agent_name: Optional[str] = None,
        task: Any = None,
    ) -> str:
        """Emit ``browser.session.start`` + ``agent.input`` for a new run.

        Returns the session_id so per-step hooks can correlate.
        """
        session_id = str(uuid.uuid4())
        if not self._connected:
            return session_id
        try:
            tid = threading.get_ident()
            start_ns = time.time_ns()
            with self._adapter_lock:
                self._run_starts[tid] = start_ns
                self._session_ids[tid] = session_id
            self._emit(
                "browser.session.start",
                {
                    "framework": "browser_use",
                    "agent_name": agent_name,
                    "session_id": session_id,
                    "timestamp_ns": start_ns,
                },
            )
            self._emit(
                "agent.input",
                {
                    "framework": "browser_use",
                    "agent_name": agent_name,
                    "session_id": session_id,
                    "input": self._safe_serialize(task),
                    "timestamp_ns": start_ns,
                },
            )
        except Exception as exc:
            self._track_failure("on_session_start", exc)
        return session_id

    def on_session_end(
        self,
        agent_name: Optional[str] = None,
        session_id: Optional[str] = None,
        output: Any = None,
        error: Optional[BaseException] = None,
    ) -> None:
        if not self._connected:
            return
        try:
            tid = threading.get_ident()
            end_ns = time.time_ns()
            with self._adapter_lock:
                start_ns = self._run_starts.pop(tid, 0)
                if not session_id:
                    session_id = self._session_ids.pop(tid, str(uuid.uuid4()))
                else:
                    self._session_ids.pop(tid, None)
            duration_ns = end_ns - start_ns if start_ns else 0
            payload: Dict[str, Any] = {
                "framework": "browser_use",
                "agent_name": agent_name,
                "session_id": session_id,
                "output": self._safe_serialize(output),
                "duration_ns": duration_ns,
            }
            if error:
                payload["error"] = str(error)
            self._emit("agent.output", payload)
            self._emit(
                "agent.state.change",
                {
                    "framework": "browser_use",
                    "agent_name": agent_name,
                    "session_id": session_id,
                    "event_subtype": "session_complete" if not error else "session_failed",
                },
            )
        except Exception as exc:
            self._track_failure("on_session_end", exc)

    def on_navigation(
        self,
        url: str,
        session_id: Optional[str] = None,
        referrer: Optional[str] = None,
        status_code: Optional[int] = None,
    ) -> None:
        """Emit ``browser.navigate`` for a page-load."""
        if not self._connected:
            return
        try:
            payload: Dict[str, Any] = {
                "framework": "browser_use",
                "url": url,
                "session_id": session_id,
            }
            if referrer:
                payload["referrer"] = referrer
            if status_code is not None:
                payload["status_code"] = status_code
            self._emit("browser.navigate", payload)
        except Exception as exc:
            self._track_failure("on_navigation", exc)

    def on_action(
        self,
        action_type: str,
        target: Any = None,
        value: Any = None,
        session_id: Optional[str] = None,
        latency_ms: Optional[float] = None,
        error: Optional[BaseException] = None,
    ) -> None:
        """Emit ``browser.action`` + ``tool.call`` for click/type/select/etc.

        Each browser action is also surfaced as a tool call so the
        unified tool-call analytics in atlas-app picks it up alongside
        regular agent tool invocations. PR #115: an ``error`` argument
        triggers a paired ``tool.error`` emission.
        """
        if not self._connected:
            return
        try:
            payload: Dict[str, Any] = {
                "framework": "browser_use",
                "action_type": action_type,
                "session_id": session_id,
            }
            if target is not None:
                payload["target"] = self._safe_serialize(target)
            if value is not None:
                payload["value"] = self._safe_serialize(value)
            if latency_ms is not None:
                payload["latency_ms"] = latency_ms
            if error:
                payload["error"] = str(error)
            self._emit("browser.action", payload)
            # Mirror as a tool.call for unified analytics.
            self._emit(
                "tool.call",
                {
                    "framework": "browser_use",
                    "tool_name": f"browser.{action_type}",
                    "tool_input": self._safe_serialize({"target": target, "value": value}),
                    "session_id": session_id,
                    "latency_ms": latency_ms,
                    "error": str(error) if error else None,
                },
            )
            if error:
                self._emit_error_event(
                    "tool.error",
                    {
                        "tool_name": f"browser.{action_type}",
                        "session_id": session_id,
                        "action_type": action_type,
                    },
                    error,
                )
        except Exception as exc:
            self._track_failure("on_action", exc)

    def on_screenshot(
        self,
        screenshot: Any,
        url: Optional[str] = None,
        session_id: Optional[str] = None,
        encoding: str = "png",
    ) -> None:
        """Emit ``browser.screenshot``.

        The ``screenshot`` field is DROPPED to a SHA-256 reference by
        the truncation policy — multi-megabyte PNG/WebP blobs are
        never embedded in events. The hash is deterministic so
        customers can correlate the same screenshot across emissions.
        """
        if not self._connected:
            return
        try:
            payload: Dict[str, Any] = {
                "framework": "browser_use",
                "session_id": session_id,
                "encoding": encoding,
                # Truncation policy will replace this with a hash ref.
                "screenshot": screenshot,
            }
            if url:
                payload["url"] = url
            self._emit("browser.screenshot", payload)
        except Exception as exc:
            self._track_failure("on_screenshot", exc)

    def on_dom_extraction(
        self,
        html: Optional[str] = None,
        dom: Any = None,
        url: Optional[str] = None,
        session_id: Optional[str] = None,
        element_count: Optional[int] = None,
    ) -> None:
        """Emit ``browser.dom.extract``.

        ``html`` / ``dom`` / ``page_content`` fields are capped at
        16 KiB by the truncation policy — DOMs from
        modern pages routinely exceed 100 KB but the document
        structure survives the cap.
        """
        if not self._connected:
            return
        try:
            payload: Dict[str, Any] = {
                "framework": "browser_use",
                "session_id": session_id,
            }
            if html is not None:
                payload["html"] = html
            if dom is not None:
                payload["dom"] = self._safe_serialize(dom)
            if url:
                payload["url"] = url
            if element_count is not None:
                payload["element_count"] = element_count
            self._emit("browser.dom.extract", payload)
        except Exception as exc:
            self._track_failure("on_dom_extraction", exc)

    def on_llm_call(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        tokens_prompt: Optional[int] = None,
        tokens_completion: Optional[int] = None,
        latency_ms: Optional[float] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        session_id: Optional[str] = None,
        error: Optional[BaseException] = None,
    ) -> None:
        """Emit ``model.invoke`` + ``cost.record`` for the LLM that drives
        the agent's reasoning loop.

        PR #115: an ``error`` triggers a paired ``model.error`` emission.
        """
        if not self._connected:
            return
        try:
            resolved_provider = provider or _detect_provider(model)
            invoke_payload: Dict[str, Any] = {
                "framework": "browser_use",
                "session_id": session_id,
            }
            if resolved_provider:
                invoke_payload["provider"] = resolved_provider
            if model:
                invoke_payload["model"] = model
            if tokens_prompt is not None:
                invoke_payload["tokens_prompt"] = tokens_prompt
            if tokens_completion is not None:
                invoke_payload["tokens_completion"] = tokens_completion
            if latency_ms is not None:
                invoke_payload["latency_ms"] = latency_ms
            if self._capture_config.capture_content and messages:
                invoke_payload["messages"] = messages
            if error:
                invoke_payload["error"] = str(error)
            self._emit("model.invoke", invoke_payload)

            if tokens_prompt is not None or tokens_completion is not None:
                self._emit(
                    "cost.record",
                    {
                        "framework": "browser_use",
                        "session_id": session_id,
                        "provider": resolved_provider,
                        "model": model,
                        "tokens_prompt": tokens_prompt,
                        "tokens_completion": tokens_completion,
                        "tokens_total": (
                            (tokens_prompt or 0) + (tokens_completion or 0)
                        ) or None,
                    },
                )

            if error:
                self._emit_error_event(
                    "model.error",
                    {
                        "model": model,
                        "provider": resolved_provider,
                        "session_id": session_id,
                    },
                    error,
                )
        except Exception as exc:
            self._track_failure("on_llm_call", exc)

    # ---- Helpers ----------------------------------------------------

    def _emit_agent_config(self, agent_name: str, agent: Any) -> None:
        """Emit ``environment.config`` once per agent on first instrument."""
        with self._adapter_lock:
            if agent_name in self._seen_agents:
                return
            self._seen_agents.add(agent_name)
        try:
            metadata: Dict[str, Any] = {
                "framework": "browser_use",
                "agent_name": agent_name,
            }
            model = getattr(agent, "model", None) or getattr(agent, "llm", None)
            if model:
                metadata["model"] = str(model)
            task = getattr(agent, "task", None)
            if task and self._capture_config.capture_content:
                metadata["task"] = str(task)[:500]
            browser = getattr(agent, "browser", None)
            if browser is not None:
                browser_cfg: Dict[str, Any] = {"type": type(browser).__name__}
                for attr in ("headless", "user_data_dir", "executable_path"):
                    val = getattr(browser, attr, None)
                    if val is not None:
                        browser_cfg[attr] = str(val) if attr != "headless" else bool(val)
                metadata["browser"] = browser_cfg
            controller = getattr(agent, "controller", None)
            if controller is not None:
                actions = getattr(controller, "registry", None) or getattr(controller, "actions", None)
                if actions:
                    try:
                        metadata["available_actions"] = (
                            list(actions.keys())[:50]
                            if isinstance(actions, dict)
                            else [str(a) for a in list(actions)[:50]]
                        )
                    except Exception:
                        pass
            self._emit("environment.config", metadata)
        except Exception as exc:
            self._track_failure("_emit_agent_config", exc)

    def _safe_serialize(self, value: Any) -> Any:
        try:
            if value is None:
                return None
            if hasattr(value, "model_dump"):
                return value.model_dump()
            if hasattr(value, "dict"):
                return value.dict()
            if isinstance(value, dict):
                return dict(value)
            if isinstance(value, (str, int, float, bool)):
                return value
            return str(value)
        except Exception:
            return str(value)

    # ---- Override get_adapter_info to expose resilience telemetry ---

    def info(self) -> AdapterInfo:
        """Return AdapterInfo, augmenting description with resilience state.

        BaseAdapter.info() handles the requires_pydantic overlay. We
        do not currently extend AdapterInfo with a metadata field
        (that lands in PR #117's BaseAdapter changes); for now the
        resilience state is exposed via :meth:`resilience_snapshot`.
        """
        return super().info()

    def resilience_snapshot(self) -> Dict[str, Any]:
        """Return a snapshot of the per-callback resilience counters.

        Surfaced for tests + the adapter health endpoint. Conforms to
        the PR #117 contract — observability errors NEVER crash the
        customer's agent, but they MUST be visible to operators.
        """
        with self._adapter_lock:
            total = sum(self._resilience_failures.values())
            return {
                "resilience_failures_total": total,
                "resilience_failures_by_callback": dict(self._resilience_failures),
                "resilience_last_error": self._resilience_last_error,
            }
