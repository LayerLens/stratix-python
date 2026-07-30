"""AWS Bedrock Agents adapter (``InvokeAgent``) using boto3 event hooks.

The real ``bedrock-agent-runtime.invoke_agent(...)`` response is
``{'completion': <botocore.eventstream.EventStream>, 'contentType', 'sessionId',
'ResponseMetadata'}`` — the agent's answer and **all** orchestration/model/tool
traces stream as events *inside* ``completion``, which the customer consumes
lazily. There is no top-level ``outputText`` / ``trace`` key (LAY-3600).

So this adapter wraps ``completion`` in a transparent proxy: as the customer
iterates ``response["completion"]`` it (a) yields every original event unchanged
and (b) observes it — accumulating ``chunk`` bytes into the answer and emitting
``model.invoke`` / ``cost.record`` / ``tool.call`` / ``agent.handoff`` /
``agent.error`` from the real ``orchestrationTrace`` shapes, with a final
``agent.output`` when the stream is drained.

The run is opened synchronously in the ``provide-client-params`` hook
(``agent.input`` + ``environment.config``), detached without flushing in the
``after-call`` hook (the proxy is installed there), and the captured collector
is flushed by the proxy when the customer finishes draining the stream. The
proxy re-establishes the run's ContextVars only for the duration of each
emission (the ``openai_agents`` pattern), so it is correct regardless of which
thread / task drains the stream.

Usage::

    client = boto3.client("bedrock-agent-runtime")
    adapter = BedrockAgentsAdapter(ll_client)
    adapter.connect(target=client)
    response = client.invoke_agent(agentId=..., enableTrace=True, ...)
    for event in response["completion"]:   # adapter observes as you iterate
        ...
    adapter.disconnect()
"""

from __future__ import annotations

import base64
import logging
import threading
from typing import Any, Set, Dict, List, Optional
from contextlib import contextmanager

from ._utils import safe_serialize
from ..._context import _current_run, _current_collector
from ._base_framework import FrameworkAdapter
from ..._capture_config import CaptureConfig
from ..providers.pricing import BEDROCK_PRICING, calculate_cost
from ..providers.token_usage import NormalizedTokenUsage

log = logging.getLogger(__name__)

try:
    import boto3  # pyright: ignore[reportMissingImports]  # noqa: F401

    _HAS_BOTO3 = True
except ImportError:
    _HAS_BOTO3 = False


_BEFORE_HOOK = "provide-client-params.bedrock-agent-runtime.InvokeAgent"
_AFTER_HOOK = "after-call.bedrock-agent-runtime.InvokeAgent"
# Fires when the call raises before a response is parsed (transport errors).
# Modeled service errors (HTTP >= 300) still go through ``after-call`` with an
# error ``parsed`` dict, so they are handled by ``_after_invoke``.
_ERROR_HOOK = "after-call-error.bedrock-agent-runtime.InvokeAgent"

#: FilePart caps (AWS: 0..5 OutputFiles, each ``bytes`` 0..1,000,000). Raw bytes
#: are only embedded (base64) under content capture and within these caps; the
#: metadata (name/type/size) is always recorded.
_MAX_FILES = 5
_MAX_FILE_BYTES = 1_000_000

#: orchestrationTrace siblings that carry a model invocation (each a real,
#: separately-priced model call with its own ``modelInvocationInput/Output``).
_MODEL_PHASES = (
    "orchestrationTrace",
    "preProcessingTrace",
    "postProcessingTrace",
    "routingClassifierTrace",
)


class BedrockAgentsAdapter(FrameworkAdapter):
    """AWS Bedrock Agents adapter using boto3 event hooks + a completion proxy."""

    name = "bedrock_agents"
    package = "bedrock"

    def __init__(self, client: Any, capture_config: Optional[CaptureConfig] = None) -> None:
        super().__init__(client, capture_config)
        self._boto_client: Optional[Any] = None
        self._warned_no_trace = False
        # In-flight completion proxies: each invoke_agent opens one; it is
        # discarded once finished. disconnect() finishes any straggler so an
        # un-drained stream still flushes a (partial) trace.
        self._pending: Set["_CompletionProxy"] = set()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _on_connect(self, target: Any = None, **kwargs: Any) -> None:
        self._check_dependency(_HAS_BOTO3)
        if target is None:
            raise ValueError("connect() requires a bedrock-agent-runtime boto3 client as target")
        self._boto_client = target
        event_system = target.meta.events
        event_system.register(_BEFORE_HOOK, self._before_invoke)
        event_system.register(_AFTER_HOOK, self._after_invoke)
        event_system.register(_ERROR_HOOK, self._on_invoke_error)

    def _on_disconnect(self) -> None:
        if self._boto_client is not None:
            try:
                ev = self._boto_client.meta.events
                ev.unregister(_BEFORE_HOOK, self._before_invoke)
                ev.unregister(_AFTER_HOOK, self._after_invoke)
                ev.unregister(_ERROR_HOOK, self._on_invoke_error)
            except Exception:
                log.debug("layerlens: could not unregister boto3 event hooks", exc_info=True)
            self._boto_client = None
        # Finish any stream the customer never fully drained so its trace is
        # not lost (partial output + flush).
        with self._lock:
            stragglers = list(self._pending)
            self._pending.clear()
        for proxy in stragglers:
            proxy.finish()

    def _discard_pending(self, proxy: "_CompletionProxy") -> None:
        with self._lock:
            self._pending.discard(proxy)

    # ------------------------------------------------------------------
    # boto3 event hooks
    # ------------------------------------------------------------------

    def _before_invoke(self, **kwargs: Any) -> None:
        if not self._connected:
            return
        try:
            params = kwargs.get("params", {})
            agent_id = params.get("agentId", "unknown")

            if not params.get("enableTrace") and not self._warned_no_trace:
                self._warned_no_trace = True
                log.warning(
                    "layerlens: bedrock_agents InvokeAgent called without enableTrace=True; "
                    "model/cost/tool/handoff events will not be captured (only agent.input + text output)."
                )

            run = self._begin_run()
            run.data["agent_id"] = agent_id
            self._start_timer("invoke")

            self._emit_agent_config(agent_id, params)

            root = self._get_root_span()
            payload = self._payload(
                agent_id=agent_id,
                session_id=params.get("sessionId"),
                enable_trace=params.get("enableTrace", False),
            )
            self._set_if_capturing(payload, "input", params.get("inputText"))
            self._emit(
                "agent.input",
                payload,
                span_id=root,
                parent_span_id=None,
                span_name="bedrock.invoke_agent",
            )
        except Exception:
            log.warning("layerlens: error in _before_invoke", exc_info=True)

    def _after_invoke(self, **kwargs: Any) -> None:
        if not self._connected:
            return
        run = self._get_run()
        if run is None:
            return
        try:
            parsed = kwargs.get("parsed", {})
            latency_ms = self._stop_timer("invoke")
            session_id = parsed.get("sessionId") if isinstance(parsed, dict) else None
            # AWS-native correlation anchor: the InvokeAgent request id (CloudTrail;
            # lets a LayerLens trace point deterministically at the AWS call).
            meta = parsed.get("ResponseMetadata") if isinstance(parsed, dict) else None
            request_id = meta.get("RequestId") if isinstance(meta, dict) else None
            agent_id = run.data.get("agent_id", "unknown")
            owns_collector = run._col_token is not None

            completion = parsed.get("completion") if isinstance(parsed, dict) else None
            if completion is None or not hasattr(completion, "__iter__"):
                # No stream to observe (defensive — the real wire always has one).
                self._emit_empty_output(session_id, latency_ms, request_id)
                self._end_run()
                return

            proxy = _CompletionProxy(
                adapter=self,
                run=run,
                source=completion,
                agent_id=agent_id,
                session_id=session_id,
                latency_ms=latency_ms,
                owns_collector=owns_collector,
                request_id=request_id,
            )
            parsed["completion"] = proxy
            with self._lock:
                self._pending.add(proxy)
            # Detach the run from the ContextVars (so they do not leak into the
            # customer's code) WITHOUT flushing — the proxy flushes the captured
            # collector once the customer drains the stream.
            self._end_run(flush=False)
        except Exception:
            log.warning("layerlens: error in _after_invoke", exc_info=True)
            try:
                self._end_run()
            except Exception:
                log.debug("layerlens: _end_run failed after _after_invoke error", exc_info=True)

    def _on_invoke_error(self, **kwargs: Any) -> None:
        """End the run if InvokeAgent fails before ``after-call`` fires.

        Only transport-level failures reach here; modeled service errors flow
        through ``_after_invoke`` (botocore emits ``after-call`` with an error
        ``parsed`` before raising). Without this, ``_before_invoke``'s run would
        leak into the customer's ContextVars after the failed call.
        """
        if not self._connected:
            return
        run = self._get_run()
        if run is None:
            return
        try:
            exc = kwargs.get("exception")
            if exc is not None:
                payload = self._payload(error=str(exc), error_type=type(exc).__name__)
                self._emit("agent.error", payload, span_name="bedrock.error")
        except Exception:
            log.warning("layerlens: error in _on_invoke_error", exc_info=True)
        finally:
            self._end_run()

    def _emit_empty_output(
        self, session_id: Optional[str], latency_ms: Optional[float], request_id: Optional[str] = None
    ) -> None:
        root = self._get_root_span()
        payload = self._payload(session_id=session_id)
        if latency_ms is not None:
            payload["latency_ms"] = latency_ms
        if request_id:
            payload["aws_request_id"] = str(request_id)
        self._emit("agent.output", payload, span_id=root, span_name="bedrock.invoke_agent")

    # ------------------------------------------------------------------
    # Environment config (per run — every trace gets its own)
    # ------------------------------------------------------------------

    def _emit_agent_config(self, agent_id: str, params: Dict[str, Any]) -> None:
        self._emit(
            "environment.config",
            self._payload(
                agent_id=agent_id,
                agent_alias_id=params.get("agentAliasId"),
                enable_trace=params.get("enableTrace", False),
            ),
            span_name="bedrock.config",
        )


# ======================================================================
# Completion EventStream proxy
# ======================================================================


class _CompletionProxy:
    """Transparent, single-read proxy over the ``completion`` EventStream.

    Iterating yields every source event unchanged while the adapter observes it.
    Emission runs against the captured run (its collector + root span) with the
    run's ContextVars re-established only for the duration of each emit, so it
    works no matter which thread / task drains the stream. The trace is flushed
    once, when the stream is drained (or on adapter disconnect).
    """

    def __init__(
        self,
        *,
        adapter: BedrockAgentsAdapter,
        run: Any,
        source: Any,
        agent_id: str,
        session_id: Optional[str],
        latency_ms: Optional[float],
        owns_collector: bool,
        request_id: Optional[str] = None,
    ) -> None:
        self._adapter = adapter
        self._run = run
        self._source = source
        self._agent_id = agent_id
        self._session_id = session_id
        self._latency_ms = latency_ms
        self._owns_collector = owns_collector
        self._request_id = request_id

        self._chunks: List[bytes] = []
        self._final_text: Optional[str] = None
        self._pending_inputs: Dict[str, Dict[str, Any]] = {}
        self._model_ids: Dict[str, str] = {}
        self._last_model_id: Optional[str] = None

        self._lock = threading.Lock()
        self._done = False
        self._errored = False
        self._returned_control = False
        self._warned_sealed = False

    # -- iteration (transparency) --

    def __iter__(self):
        try:
            for event in self._source:
                self._observe(event)
                yield event
        except Exception as exc:  # genuine stream error (e.g. EventStreamError)
            self._record_error(exc)
            raise  # transparency: the customer sees the same exception
        finally:
            self.finish()

    def close(self) -> Any:
        """Flush the trace, then close the underlying stream (EventStream API)."""
        self.finish()
        src_close = getattr(self._source, "close", None)
        if callable(src_close):
            return src_close()
        return None

    def __getattr__(self, name: str) -> Any:
        # Delegate any other EventStream API (e.g. get_initial_response) to the
        # source. Guarded so it never recurses on our own attributes.
        if name.startswith("_"):
            raise AttributeError(name)
        source = self.__dict__.get("_source")
        if source is None:
            raise AttributeError(name)
        return getattr(source, name)

    # -- context re-establishment for emission --

    @contextmanager
    def _active(self):
        run_token = _current_run.set(self._run)
        col_token = _current_collector.set(self._run.collector)
        try:
            if not self._owns_collector and not self._warned_sealed and self._run.collector.sealed:
                self._warned_sealed = True
                log.warning(
                    "layerlens: bedrock_agents completion stream drained after its "
                    "trace context closed; trace events are dropped. Drain "
                    "response['completion'] before the trace_context() block exits."
                )
            yield
        finally:
            _current_collector.reset(col_token)
            _current_run.reset(run_token)

    # -- emission (injects the run-level AWS correlation anchor) --

    def _emit(self, event_type: str, payload: Dict[str, Any], **kwargs: Any) -> None:
        """Emit via the adapter, stamping the InvokeAgent request id on every
        proxy-produced event so any event points back to the AWS call."""
        if self._request_id:
            payload.setdefault("aws_request_id", str(self._request_id))
        adapter_emit = self._adapter._emit
        adapter_emit(event_type, payload, **kwargs)

    # -- observation (never breaks the customer stream) --

    def _observe(self, event: Dict[str, Any]) -> None:
        try:
            with self._active():
                self._dispatch(event)
        except Exception:
            log.warning("layerlens: error observing bedrock_agents event", exc_info=True)

    def _dispatch(self, event: Dict[str, Any]) -> None:
        if not isinstance(event, dict):
            return
        if "chunk" in event:
            chunk = event.get("chunk") or {}
            data = chunk.get("bytes") if isinstance(chunk, dict) else None
            if data:
                self._chunks.append(data)
            return
        if "returnControl" in event:
            rc = event.get("returnControl")
            if isinstance(rc, dict):
                self._on_return_control(rc)
            return
        if "files" in event:
            files = event.get("files")
            if isinstance(files, dict):
                self._on_files(files)
            return
        trace = event.get("trace")
        if not isinstance(trace, dict):
            return
        inner = trace.get("trace")
        if not isinstance(inner, dict):
            return
        if isinstance(inner.get("failureTrace"), dict):
            self._on_failure(inner["failureTrace"])
            return
        if isinstance(inner.get("guardrailTrace"), dict):
            self._on_guardrail(inner["guardrailTrace"])
            return
        phase = next((inner[p] for p in _MODEL_PHASES if isinstance(inner.get(p), dict)), None)
        if phase is None:
            return
        mi = phase.get("modelInvocationInput")
        if isinstance(mi, dict):
            self._stash_model_input(mi)
        mo = phase.get("modelInvocationOutput")
        if isinstance(mo, dict):
            self._on_model_output(mo)
        ii = phase.get("invocationInput")
        if isinstance(ii, dict):
            tid = ii.get("traceId")
            if tid:
                self._pending_inputs[tid] = ii
        obs = phase.get("observation")
        if isinstance(obs, dict):
            self._on_observation(obs)

    # -- per-trace emitters --

    def _stash_model_input(self, mi: Dict[str, Any]) -> None:
        model_id = mi.get("foundationModel")
        if not model_id:
            return
        self._last_model_id = model_id
        tid = mi.get("traceId")
        if tid:
            self._model_ids[tid] = model_id

    def _on_model_output(self, mo: Dict[str, Any]) -> None:
        metadata = mo.get("metadata") or {}
        usage = metadata.get("usage") or {} if isinstance(metadata, dict) else {}
        # Prefer the model stashed for this exact traceId. Only fall back to the
        # last-seen model when the output has no traceId at all — never borrow
        # another phase's model for a present-but-unmatched traceId.
        trace_id = mo.get("traceId")
        model_id = self._model_ids.get(trace_id) if trace_id else self._last_model_id
        tokens_prompt = int(usage.get("inputTokens") or 0) if isinstance(usage, dict) else 0
        tokens_completion = int(usage.get("outputTokens") or 0) if isinstance(usage, dict) else 0

        span_id = self._adapter._new_span_id()
        payload = self._adapter._payload(provider="aws_bedrock")
        if model_id:
            payload["model"] = model_id
        if tokens_prompt:
            payload["tokens_prompt"] = tokens_prompt
        if tokens_completion:
            payload["tokens_completion"] = tokens_completion
        if tokens_prompt or tokens_completion:
            payload["tokens_total"] = tokens_prompt + tokens_completion
        if trace_id:
            payload["bedrock_trace_id"] = str(trace_id)
        # the underlying model call's id -> the CloudWatch model-invocation-logging record
        client_request_id = metadata.get("clientRequestId") if isinstance(metadata, dict) else None
        if client_request_id:
            payload["client_request_id"] = str(client_request_id)
        # Fill the response_id column from the honest per-response identifier: the
        # underlying model call's id, else the InvokeAgent AWS RequestId (G3).
        resp_id = client_request_id or getattr(self, "_request_id", None)
        if resp_id:
            payload["response_id"] = str(resp_id)
        self._emit("model.invoke", payload, span_id=span_id, span_name="bedrock.model")

        if tokens_prompt or tokens_completion:
            cost_payload = self._adapter._payload(
                provider="aws_bedrock",
                tokens_prompt=tokens_prompt,
                tokens_completion=tokens_completion,
                tokens_total=tokens_prompt + tokens_completion,
            )
            if trace_id:
                cost_payload["bedrock_trace_id"] = str(trace_id)
            if model_id:
                cost_payload["model"] = model_id
                cost_usd = calculate_cost(
                    model_id,
                    NormalizedTokenUsage(prompt_tokens=tokens_prompt, completion_tokens=tokens_completion),
                    BEDROCK_PRICING,
                )
                if cost_usd is not None:
                    cost_payload["cost_usd"] = cost_usd
            self._emit("cost.record", cost_payload, span_id=span_id)

    def _on_observation(self, obs: Dict[str, Any]) -> None:
        tid = obs.get("traceId")
        inp = self._pending_inputs.get(tid or "", {})
        if isinstance(obs.get("actionGroupInvocationOutput"), dict):
            self._on_action_group(obs["actionGroupInvocationOutput"], inp, tid)
        elif isinstance(obs.get("knowledgeBaseLookupOutput"), dict):
            self._on_knowledge_base(obs["knowledgeBaseLookupOutput"], inp, tid)
        elif isinstance(obs.get("agentCollaboratorInvocationOutput"), dict):
            self._on_collaborator(obs["agentCollaboratorInvocationOutput"], inp, tid)
        elif isinstance(obs.get("codeInterpreterInvocationOutput"), dict):
            self._on_code_interpreter(obs["codeInterpreterInvocationOutput"], inp, tid)
        elif isinstance(obs.get("repromptResponse"), dict):
            self._on_reprompt(obs["repromptResponse"], tid)
        elif obs.get("type") == "ASK_USER":
            self._on_ask_user(obs, tid)
        elif isinstance(obs.get("finalResponse"), dict):
            text = obs["finalResponse"].get("text")
            if text:
                self._final_text = str(text)

    def _on_action_group(self, output: Dict[str, Any], inp: Dict[str, Any], trace_id: Optional[str] = None) -> None:
        ag_in = inp.get("actionGroupInvocationInput", {}) if isinstance(inp, dict) else {}
        ag_in = ag_in if isinstance(ag_in, dict) else {}
        payload = self._adapter._payload(
            tool_name=ag_in.get("actionGroupName", "unknown"),
            tool_type="action_group",
        )
        if trace_id:
            payload["bedrock_trace_id"] = str(trace_id)
        for src, dst in (
            ("function", "function"),
            ("verb", "verb"),
            ("apiPath", "api_path"),
            ("executionType", "execution_type"),
        ):
            val = ag_in.get(src)
            if val:
                payload[dst] = str(val)
        invocation_id = ag_in.get("invocationId")
        if invocation_id:
            payload["invocation_id"] = str(invocation_id)
        self._adapter._set_if_capturing(
            payload, "input", safe_serialize(ag_in.get("parameters") or ag_in.get("requestBody"))
        )
        self._adapter._set_if_capturing(payload, "output", output.get("text"))
        self._emit("tool.call", payload, span_name="bedrock.action_group")

    def _on_knowledge_base(self, output: Dict[str, Any], inp: Dict[str, Any], trace_id: Optional[str] = None) -> None:
        kb_in = inp.get("knowledgeBaseLookupInput", {}) if isinstance(inp, dict) else {}
        kb_in = kb_in if isinstance(kb_in, dict) else {}
        payload = self._adapter._payload(
            tool_name=kb_in.get("knowledgeBaseId", "knowledge_base"),
            tool_type="knowledge_base_retrieval",
        )
        if trace_id:
            payload["bedrock_trace_id"] = str(trace_id)
        self._adapter._set_if_capturing(payload, "input", kb_in.get("text"))
        refs = output.get("retrievedReferences")
        if isinstance(refs, list):
            payload["num_results"] = len(refs)
            sources: List[str] = []
            for ref in refs:
                if not isinstance(ref, dict):
                    continue
                location = ref.get("location") or {}
                s3 = location.get("s3Location") or {} if isinstance(location, dict) else {}
                uri = s3.get("uri") if isinstance(s3, dict) else None
                if uri:
                    sources.append(str(uri))
            if sources:
                payload["retrieval_sources"] = sources[:20]
        self._adapter._set_if_capturing(payload, "output", safe_serialize(refs))
        self._emit("tool.call", payload, span_name="bedrock.knowledge_base")

    def _on_code_interpreter(self, output: Dict[str, Any], inp: Dict[str, Any], trace_id: Optional[str] = None) -> None:
        """Map a code-interpreter observation to ``agent.code`` (LAY-3609).

        Correlates the executed source (stashed ``invocationInput`` by traceId)
        with its execution result. A code run that *errors* is still
        ``agent.code`` (the diagnostic) — never ``agent.error``, which is
        reserved for ``failureTrace``. ``errored`` is a structural flag kept
        visible so failures stay observable; the error string itself is a
        traceback (can embed runtime/user values) and is content, so it — like
        the code + output — is gated.
        """
        ci_in = inp.get("codeInterpreterInvocationInput", {}) if isinstance(inp, dict) else {}
        ci_in = ci_in if isinstance(ci_in, dict) else {}
        payload = self._adapter._payload(language="python")
        if trace_id:
            payload["bedrock_trace_id"] = str(trace_id)
        self._adapter._set_if_capturing(payload, "code", ci_in.get("code"))
        self._adapter._set_if_capturing(payload, "output", output.get("executionOutput"))
        error = output.get("executionError")
        if error:
            payload["errored"] = True
            self._adapter._set_if_capturing(payload, "execution_error", str(error))
        if output.get("executionTimeout"):
            payload["execution_timeout"] = True
        files = output.get("files")
        if isinstance(files, list) and files:
            payload["num_files"] = len(files)
            names = [str(f.get("name")) if isinstance(f, dict) else str(f) for f in files]
            payload["files"] = names[:20]
        self._emit("agent.code", payload, span_name="bedrock.code_interpreter")

    def _on_reprompt(self, reprompt: Dict[str, Any], trace_id: Optional[str] = None) -> None:
        """Map a REPROMPT observation to ``agent.step`` (LAY-3610).

        A reprompt is the agent correcting a malformed tool/parser/KB output by
        re-prompting the model — a *successful self-correction*, NOT an error
        (``agent.error`` stays reserved for ``failureTrace`` to keep error counts
        honest). The corrective text is content and is gated.
        """
        payload = self._adapter._payload(step_type="reprompt")
        if trace_id:
            payload["bedrock_trace_id"] = str(trace_id)
        source = reprompt.get("source")
        if source:
            payload["reprompt_source"] = str(source)
        self._adapter._set_if_capturing(payload, "text", reprompt.get("text"))
        self._emit("agent.step", payload, span_name="bedrock.reprompt")

    def _on_ask_user(self, obs: Dict[str, Any], trace_id: Optional[str] = None) -> None:
        """Map an ASK_USER observation to ``agent.step`` (LAY-3610).

        The agent pauses to ask the human for more information — distinct from a
        reprompt (which asks the *model*). The question text is also this turn's
        answer to the user, so it still flows to the terminal ``agent.output``.
        """
        final = obs.get("finalResponse")
        text = final.get("text") if isinstance(final, dict) else None
        payload = self._adapter._payload(step_type="ask_user")
        if trace_id:
            payload["bedrock_trace_id"] = str(trace_id)
        self._adapter._set_if_capturing(payload, "text", text)
        self._emit("agent.step", payload, span_name="bedrock.ask_user")
        if text:
            self._final_text = str(text)

    def _on_collaborator(self, output: Dict[str, Any], inp: Dict[str, Any], trace_id: Optional[str] = None) -> None:
        collab_in = inp.get("agentCollaboratorInvocationInput", {}) if isinstance(inp, dict) else {}
        collab_in = collab_in if isinstance(collab_in, dict) else {}
        # from_agent (the supervisor) is always the honest constructor-injected
        # agent id; to_agent is honest only when AWS names the collaborator —
        # omit rather than fabricate the literal "collaborator" (F9).
        to_agent = collab_in.get("agentCollaboratorName") or output.get("agentCollaboratorName")
        payload = self._adapter._payload(
            from_agent=self._agent_id,
            reason="supervisor_delegation",
        )
        if to_agent:
            payload["to_agent"] = str(to_agent)
        if trace_id:
            payload["bedrock_trace_id"] = str(trace_id)
        self._adapter._set_if_capturing(payload, "input", safe_serialize(collab_in.get("input")))
        self._adapter._set_if_capturing(payload, "output", safe_serialize(output.get("output")))
        self._emit("agent.handoff", payload, span_name="bedrock.handoff")

    def _on_failure(self, failure: Dict[str, Any]) -> None:
        payload = self._adapter._payload(
            error=str(failure.get("failureReason") or "agent failure"),
            error_type="agent_failure",
            status="error",
        )
        code = failure.get("failureCode")
        if isinstance(code, int):
            payload["error_code"] = code
        self._emit("agent.error", payload, span_name="bedrock.error")

    def _on_guardrail(self, guardrail: Dict[str, Any]) -> None:
        """Map a ``guardrailTrace`` to ``policy.violation`` (LAY-3607).

        A guardrail intervention is a *policy outcome*, not an agent failure —
        it is NOT ``agent.error`` (keeps error counts honest), and the terminal
        ``agent.output`` still stands. ``action == "NONE"`` means the guardrail
        let the turn through, so nothing is emitted. Input vs output stage is
        derived from which assessments list is populated.
        """
        if guardrail.get("action") != "INTERVENED":
            return
        input_assessments = guardrail.get("inputAssessments")
        output_assessments = guardrail.get("outputAssessments")
        stage = "input" if isinstance(input_assessments, list) and input_assessments else "output"
        policies: List[Dict[str, Any]] = []
        for assessment in (input_assessments or []) + (output_assessments or []):
            if isinstance(assessment, dict):
                policies.extend(self._flatten_guardrail_assessment(assessment))
        payload = self._adapter._payload(action="INTERVENED", stage=stage, policies=policies)
        tid = guardrail.get("traceId")
        if tid:
            payload["bedrock_trace_id"] = str(tid)
        self._emit("policy.violation", payload, span_name="bedrock.guardrail")

    @staticmethod
    def _flatten_guardrail_assessment(assessment: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Flatten one GuardrailAssessment into canonical policy entries.

        Structural metadata only (policy type / name / filter or entity type /
        action) — never the raw matched text, which is sensitive and is left to
        the model output under content gating.
        """
        policies: List[Dict[str, Any]] = []
        topic_policy = assessment.get("topicPolicy")
        if isinstance(topic_policy, dict):
            for topic in topic_policy.get("topics") or []:
                if isinstance(topic, dict):
                    policies.append({"type": "topic", "name": topic.get("name"), "action": topic.get("action")})
        content_policy = assessment.get("contentPolicy")
        if isinstance(content_policy, dict):
            for filt in content_policy.get("filters") or []:
                if isinstance(filt, dict):
                    entry = {"type": "content", "filter_type": filt.get("type"), "action": filt.get("action")}
                    confidence = filt.get("confidence")
                    if confidence is not None:
                        entry["confidence"] = confidence
                    policies.append(entry)
        pii_policy = assessment.get("sensitiveInformationPolicy")
        if isinstance(pii_policy, dict):
            for pii in pii_policy.get("piiEntities") or []:
                if isinstance(pii, dict):
                    policies.append({"type": "pii", "entity_type": pii.get("type"), "action": pii.get("action")})
            for regex in pii_policy.get("regexes") or []:
                if isinstance(regex, dict):
                    policies.append({"type": "pii", "name": regex.get("name"), "action": regex.get("action")})
        word_policy = assessment.get("wordPolicy")
        if isinstance(word_policy, dict):
            for _word in word_policy.get("customWords") or []:
                if isinstance(_word, dict):
                    policies.append({"type": "word", "action": _word.get("action")})
            for managed in word_policy.get("managedWordLists") or []:
                if isinstance(managed, dict):
                    policies.append({"type": "word", "list_type": managed.get("type"), "action": managed.get("action")})
        return policies

    def _on_return_control(self, rc: Dict[str, Any]) -> None:
        """Map a top-level ``returnControl`` to ``tool.call`` (LAY-3608).

        RETURN_CONTROL hands the tool invocation back to the customer's own app;
        the result is supplied on the NEXT ``InvokeAgent`` call, so there is no
        ``output`` here. It also means this turn produced no answer, so set
        ``_returned_control`` to suppress the otherwise-misleading empty
        ``agent.output`` in ``finish()`` (mirrors the ``_errored`` suppression).
        """
        self._returned_control = True
        invocation_id = rc.get("invocationId")
        inputs = rc.get("invocationInputs")
        if not isinstance(inputs, list):
            return
        for item in inputs:
            if not isinstance(item, dict):
                continue
            fn = item.get("functionInvocationInput")
            api = item.get("apiInvocationInput")
            fn = fn if isinstance(fn, dict) else None
            api = api if isinstance(api, dict) else None
            spec = fn or api
            if spec is None:
                continue
            payload = self._adapter._payload(
                tool_name=spec.get("actionGroup", "unknown"),
                tool_type="return_control",
            )
            if invocation_id:
                payload["invocation_id"] = str(invocation_id)
            if fn is not None and fn.get("function"):
                payload["function"] = str(fn["function"])
            if api is not None:
                if api.get("apiPath"):
                    payload["api_path"] = str(api["apiPath"])
                if api.get("httpMethod"):
                    payload["verb"] = str(api["httpMethod"])
            # Multi-agent attribution (the input carries these only in collaborator mode).
            agent_id = item.get("agentId")
            if agent_id:
                payload["from_agent"] = str(agent_id)
            collaborator = item.get("collaboratorName")
            if collaborator:
                payload["collaborator"] = str(collaborator)
            self._adapter._set_if_capturing(
                payload, "input", safe_serialize(spec.get("parameters") or spec.get("requestBody"))
            )
            self._emit("tool.call", payload, span_name="bedrock.return_control")

    def _on_files(self, files_part: Dict[str, Any]) -> None:
        """Map a top-level ``files`` FilePart to ``agent.code`` (LAY-3611).

        These are the binary artifacts the code interpreter delivers (charts,
        CSVs, exports). Metadata (name / MIME type / size) is always recorded;
        the raw bytes are embedded as base64 only under content capture, capped
        at ``_MAX_FILE_BYTES`` per file and ``_MAX_FILES`` files, so a trace is
        never bloated by large blobs.
        """
        output_files = files_part.get("files")
        if not isinstance(output_files, list) or not output_files:
            return
        capturing = self._adapter._config.capture_content
        artifacts: List[Dict[str, Any]] = []
        for output_file in output_files[:_MAX_FILES]:
            if not isinstance(output_file, dict):
                continue
            entry: Dict[str, Any] = {}
            if output_file.get("name"):
                entry["name"] = str(output_file["name"])
            if output_file.get("type"):
                entry["type"] = str(output_file["type"])
            data = output_file.get("bytes")
            if isinstance(data, (bytes, bytearray)):
                entry["size_bytes"] = len(data)
                if capturing and len(data) <= _MAX_FILE_BYTES:
                    entry["data"] = base64.b64encode(bytes(data)).decode("ascii")
            elif isinstance(data, str):
                entry["size_bytes"] = len(data)
                if capturing and len(data) <= _MAX_FILE_BYTES:
                    entry["data"] = data
            artifacts.append(entry)
        if not artifacts:
            return
        payload = self._adapter._payload(num_files=len(artifacts), files=artifacts)
        self._emit("agent.code", payload, span_name="bedrock.files")

    def _record_error(self, exc: BaseException) -> None:
        self._errored = True  # agent.error is authoritative — suppress the terminal agent.output
        try:
            with self._active():
                payload = self._adapter._payload(error=str(exc), error_type=type(exc).__name__)
                self._emit("agent.error", payload, span_name="bedrock.error")
        except Exception:
            log.warning("layerlens: error recording bedrock_agents stream error", exc_info=True)

    # -- finish (idempotent: emit agent.output + flush exactly once) --

    def finish(self) -> None:
        with self._lock:
            if self._done:
                return
            self._done = True
        self._adapter._discard_pending(self)
        try:
            # On a failed stream the recorded agent.error is the terminal event,
            # and on a return-of-control turn the answer comes next turn — in
            # both cases don't emit a (misleading) terminal agent.output.
            if not self._errored and not self._returned_control:
                with self._active():
                    output = b"".join(self._chunks).decode("utf-8", "replace") if self._chunks else self._final_text
                    payload = self._adapter._payload(session_id=self._session_id)
                    if self._latency_ms is not None:
                        payload["latency_ms"] = self._latency_ms
                    self._adapter._set_if_capturing(payload, "output", output or None)
                    self._emit(
                        "agent.output",
                        payload,
                        span_id=self._run.root_span_id,
                        span_name="bedrock.invoke_agent",
                    )
        except Exception:
            log.warning("layerlens: error finishing bedrock_agents trace", exc_info=True)
        finally:
            if self._owns_collector:
                try:
                    self._run.collector.flush()
                except Exception:
                    log.warning("layerlens: error flushing bedrock_agents trace", exc_info=True)
