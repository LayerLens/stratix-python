from __future__ import annotations

import uuid
import logging
from typing import Any, Dict, List, Optional

from .._base import resilient_callback
from ._utils import truncate, new_span_id
from ..._collector import TraceCollector
from ._base_framework import FrameworkAdapter
from ..._capture_config import CaptureConfig

log = logging.getLogger(__name__)

try:
    import httpx  # pyright: ignore[reportMissingImports]

    _HAS_HTTPX = True
except ImportError:
    _HAS_HTTPX = False


# ---------------------------------------------------------------------------
# Langfuse observation type -> LayerLens event type mapping
# ---------------------------------------------------------------------------


class LangfuseAdapter(FrameworkAdapter):
    """Bidirectional trace sync adapter for Langfuse.

    This adapter is a batch sync pipeline, **not** a real-time instrumentation
    wrapper.  It connects to a Langfuse instance via API keys and supports:

    * **Import** -- pull traces from Langfuse, normalise observations into flat
      LayerLens events, and emit them through :class:`TraceCollector`.
    * **Export** -- convert flat LayerLens events back to Langfuse's
      trace / generation / span format and POST them via the ingestion API.

    Usage::

        adapter = LangfuseAdapter(client)
        adapter.connect(public_key="pk-lf-...", secret_key="sk-lf-...", host="https://cloud.langfuse.com")

        # Pull traces from Langfuse into LayerLens
        adapter.import_traces(limit=50)

        # Push LayerLens events to Langfuse
        adapter.export_traces(events_by_trace={"trace-id": [event1, event2]})

        adapter.disconnect()
    """

    name = "langfuse"
    package = "httpx"

    def __init__(self, client: Any, capture_config: Optional[CaptureConfig] = None) -> None:
        super().__init__(client, capture_config)

        # Langfuse connection state
        self._public_key: Optional[str] = None
        self._secret_key: Optional[str] = None
        self._host: Optional[str] = None
        self._http: Optional[Any] = None  # httpx.Client

        # Incremental sync cursor (ISO-8601 timestamp of last imported trace)
        self._last_cursor: Optional[str] = None

    # ------------------------------------------------------------------
    # BaseAdapter interface
    # ------------------------------------------------------------------

    def _on_connect(self, target: Any = None, **kwargs: Any) -> None:
        """Connect to a Langfuse instance.

        Keyword arguments
        -----------------
        public_key:
            Langfuse public API key.
        secret_key:
            Langfuse secret API key.
        host:
            Langfuse API base URL (default ``https://cloud.langfuse.com``).
        """
        self._check_dependency(_HAS_HTTPX)
        public_key = kwargs.get("public_key")
        secret_key = kwargs.get("secret_key")
        host = kwargs.get("host")

        if not public_key or not secret_key:
            raise ValueError("Both 'public_key' and 'secret_key' are required to connect to Langfuse.")

        self._public_key = public_key
        self._secret_key = secret_key
        self._host = (host or "https://cloud.langfuse.com").rstrip("/")
        if self._host:
            self._metadata["host"] = self._host

        self._http = httpx.Client(
            base_url=self._host,
            auth=(self._public_key, self._secret_key),
            timeout=30.0,
            headers={"Content-Type": "application/json"},
        )

        # Validate connectivity with a lightweight request
        try:
            resp = self._http.get("/api/public/traces", params={"limit": 1})
            resp.raise_for_status()
        except Exception as exc:
            self._http.close()
            self._http = None
            raise ConnectionError(f"Failed to connect to Langfuse at {self._host}: {exc}") from exc

        log.info("layerlens: Langfuse adapter connected to %s", self._host)

    def _on_disconnect(self) -> None:
        with self._lock:
            if self._http is not None:
                try:
                    self._http.close()
                except Exception:
                    log.warning("layerlens: error closing Langfuse HTTP client", exc_info=True)
                self._http = None
            self._public_key = None
            self._secret_key = None
            self._host = None
            self._last_cursor = None
        # base class handles self._connected = False and self._metadata.clear()

    # ------------------------------------------------------------------
    # Import: Langfuse -> LayerLens
    # ------------------------------------------------------------------

    def import_traces(
        self,
        *,
        since: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> int:
        """Fetch traces from Langfuse and emit them as flat LayerLens events.

        Parameters
        ----------
        since:
            ISO-8601 timestamp.  Only traces updated after this time are
            fetched.  Falls back to the internal cursor from the last import.
        limit:
            Maximum number of traces to fetch (Langfuse page size, default 50).

        Returns
        -------
        int
            Number of traces imported.
        """
        self._require_connected()

        params: Dict[str, Any] = {}
        cursor = since or self._last_cursor
        if cursor is not None:
            params["fromTimestamp"] = cursor
        if limit is not None:
            params["limit"] = limit

        try:
            resp = self._http.get("/api/public/traces", params=params)  # type: ignore[union-attr]
            resp.raise_for_status()
            body = resp.json()
        except Exception:
            log.warning("layerlens: failed to list Langfuse traces", exc_info=True)
            return 0

        traces = body.get("data", [])
        if not traces:
            return 0

        imported = 0
        for trace_summary in traces:
            # ``_import_single_trace`` is wrapped with @resilient_callback
            # — a malformed trace becomes a logged warning + failure
            # counter increment, NOT a halt of the batch import. Track
            # success via the success counter we maintain manually.
            before = self._resilience.total_failures
            self._import_single_trace(trace_summary)
            if self._resilience.total_failures == before:
                imported += 1

        # Advance cursor to the most recent trace timestamp
        latest = traces[0].get("updatedAt") or traces[0].get("timestamp")
        if latest:
            self._last_cursor = latest

        log.info("layerlens: imported %d Langfuse traces", imported)
        return imported

    @resilient_callback(callback_name="_import_single_trace")
    def _import_single_trace(self, trace_summary: Dict[str, Any]) -> None:
        """Fetch a full trace and emit events via TraceCollector."""
        trace_id = trace_summary["id"]

        resp = self._http.get(f"/api/public/traces/{trace_id}")  # type: ignore[union-attr]
        resp.raise_for_status()
        trace = resp.json()

        collector = TraceCollector(self._client, self._config)
        root_span_id = new_span_id()

        # Emit agent.input from trace input
        trace_input = trace.get("input")
        if trace_input is not None:
            collector.emit(
                "agent.input",
                {
                    "framework": "langfuse",
                    "langfuse_trace_id": trace_id,
                    "content": truncate(str(trace_input), max_len=4000),
                    "name": trace.get("name", ""),
                    "metadata": _safe_dict(trace.get("metadata")),
                },
                span_id=root_span_id,
                span_name=trace.get("name"),
            )

        # Process observations (generations, spans, events). Inner call
        # is wrapped with @resilient_callback — a malformed observation
        # is logged + counted, not propagated.
        observations = trace.get("observations", [])
        for obs in observations:
            self._import_observation(collector, obs, root_span_id)

        # Scores (Langfuse "annotations") — wrapped in a resilient
        # helper so one bad score doesn't abort the whole trace import.
        for score in trace.get("scores", []) or []:
            self._import_score(collector, trace, trace_id, root_span_id, score)

        # Emit agent.output from trace output
        trace_output = trace.get("output")
        if trace_output is not None:
            out_payload: Dict[str, Any] = {
                "framework": "langfuse",
                "langfuse_trace_id": trace_id,
                "content": truncate(str(trace_output), max_len=4000),
            }
            session_id = trace.get("sessionId")
            if session_id:
                out_payload["session_id"] = session_id
            collector.emit(
                "agent.output",
                out_payload,
                span_id=root_span_id,
                parent_span_id=None,
                span_name=trace.get("name"),
            )

        collector.flush()

    @resilient_callback(callback_name="_import_score")
    def _import_score(
        self,
        collector: TraceCollector,
        trace: Dict[str, Any],
        trace_id: str,
        root_span_id: str,
        score: Dict[str, Any],
    ) -> None:
        """Emit one Langfuse score as a LayerLens evaluation.result event."""
        score_payload: Dict[str, Any] = {
            "framework": "langfuse",
            "langfuse_trace_id": trace_id,
            "name": score.get("name"),
            "value": score.get("value"),
            "source": score.get("source"),
            "data_type": score.get("dataType"),
            "observation_id": score.get("observationId"),
        }
        comment = score.get("comment")
        if comment:
            score_payload["comment"] = truncate(str(comment), max_len=2000)
        # Session clustering: Langfuse groups related traces via sessionId.
        # Carry it through so downstream session-level analytics work.
        session_id = score.get("sessionId") or trace.get("sessionId")
        if session_id:
            score_payload["session_id"] = session_id
        collector.emit(
            "evaluation.result",
            score_payload,
            span_id=new_span_id(),
            parent_span_id=root_span_id,
        )

    @resilient_callback(callback_name="_import_observation")
    def _import_observation(
        self,
        collector: TraceCollector,
        obs: Dict[str, Any],
        root_span_id: str,
    ) -> None:
        """Convert a single Langfuse observation to LayerLens event(s)."""
        obs_type = obs.get("type", "").upper()
        obs_id = obs.get("id", new_span_id())
        span_id = new_span_id()
        parent_id = obs.get("parentObservationId")
        parent_span = new_span_id() if parent_id else root_span_id

        base_payload: Dict[str, Any] = {
            "framework": "langfuse",
            "langfuse_observation_id": obs_id,
            "name": obs.get("name", ""),
        }

        if obs_type == "GENERATION":
            self._import_generation(collector, obs, span_id, parent_span, base_payload)
        elif obs_type == "SPAN":
            self._import_span(collector, obs, span_id, parent_span, base_payload)
        elif obs_type == "EVENT":
            payload = {**base_payload}
            status_msg = obs.get("statusMessage")
            if status_msg:
                payload["status_message"] = truncate(str(status_msg), max_len=4000)
            obs_input = obs.get("input")
            if obs_input is not None:
                payload["input"] = truncate(str(obs_input), max_len=4000)
            collector.emit(
                "agent.state.change",
                payload,
                span_id=span_id,
                parent_span_id=parent_span,
                span_name=obs.get("name"),
            )

    def _import_generation(
        self,
        collector: TraceCollector,
        obs: Dict[str, Any],
        span_id: str,
        parent_span: str,
        base_payload: Dict[str, Any],
    ) -> None:
        """Import a Langfuse generation as model.invoke + cost.record."""
        model = obs.get("model", "")
        usage = obs.get("usage") or {}
        prompt_tokens = usage.get("promptTokens", 0) or usage.get("input", 0) or 0
        completion_tokens = usage.get("completionTokens", 0) or usage.get("output", 0) or 0
        total_tokens = usage.get("totalTokens", 0) or (prompt_tokens + completion_tokens)

        payload: Dict[str, Any] = {**base_payload, "model": model}
        if prompt_tokens:
            payload["tokens_prompt"] = prompt_tokens
        if completion_tokens:
            payload["tokens_completion"] = completion_tokens
        if total_tokens:
            payload["tokens_total"] = total_tokens

        obs_input = obs.get("input")
        if obs_input is not None:
            payload["messages"] = truncate(str(obs_input), max_len=4000)
        obs_output = obs.get("output")
        if obs_output is not None:
            payload["output_message"] = truncate(str(obs_output), max_len=4000)

        collector.emit(
            "model.invoke",
            payload,
            span_id=span_id,
            parent_span_id=parent_span,
            span_name=obs.get("name"),
        )

        # Emit cost.record alongside generation with fuller breakdown
        if prompt_tokens or completion_tokens:
            cost_payload: Dict[str, Any] = {
                "framework": "langfuse",
                "model": model,
                "tokens_prompt": prompt_tokens,
                "tokens_completion": completion_tokens,
                "tokens_total": total_tokens,
            }
            # Include cost breakdown (input/output/cache/audio) and detailed
            # usage pieces so dashboards can attribute spend beyond a single
            # lump-sum number.
            cost_details = obs.get("costDetails") or {}
            total_cost = obs.get("calculatedTotalCost") or obs.get("totalCost")
            if total_cost is not None:
                cost_payload["cost_usd"] = total_cost
            if cost_details:
                # Normalize well-known cost breakdown keys so downstream UIs
                # don't each need to know about Langfuse's JSON shape.
                for src, dst in (
                    ("input", "cost_input_usd"),
                    ("output", "cost_output_usd"),
                    ("total", "cost_total_usd"),
                    ("inputCache", "cost_input_cache_usd"),
                    ("outputReasoning", "cost_output_reasoning_usd"),
                    ("audio", "cost_audio_usd"),
                ):
                    val = cost_details.get(src)
                    if val is not None:
                        cost_payload[dst] = val
                cost_payload["cost_details"] = cost_details
            usage_details = obs.get("usageDetails") or {}
            for src, dst in (
                ("cacheRead", "cached_tokens"),
                ("cacheCreation", "cache_creation_tokens"),
                ("reasoning", "reasoning_tokens"),
                ("audio", "audio_tokens"),
            ):
                val = usage_details.get(src)
                if val is not None:
                    try:
                        cost_payload[dst] = int(val)
                    except (TypeError, ValueError):
                        pass

            collector.emit(
                "cost.record",
                cost_payload,
                span_id=span_id,
                parent_span_id=parent_span,
            )

    def _import_span(
        self,
        collector: TraceCollector,
        obs: Dict[str, Any],
        span_id: str,
        parent_span: str,
        base_payload: Dict[str, Any],
    ) -> None:
        """Import a Langfuse span as tool.call or agent.code."""
        payload: Dict[str, Any] = {**base_payload}
        obs_input = obs.get("input")
        obs_output = obs.get("output")

        if obs_input is not None:
            payload["input"] = truncate(str(obs_input), max_len=4000)
        if obs_output is not None:
            payload["output"] = truncate(str(obs_output), max_len=4000)

        # Heuristic: spans whose name contains code-related keywords
        # map to agent.code, others to tool.call
        name = (obs.get("name") or "").lower()
        event_type = "agent.code" if any(kw in name for kw in ("code", "exec", "sandbox")) else "tool.call"

        collector.emit(
            event_type,
            payload,
            span_id=span_id,
            parent_span_id=parent_span,
            span_name=obs.get("name"),
        )

    # ------------------------------------------------------------------
    # Export: LayerLens -> Langfuse
    # ------------------------------------------------------------------

    def export_traces(
        self,
        *,
        events_by_trace: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ) -> int:
        """Convert flat LayerLens events to Langfuse format and ingest them.

        Parameters
        ----------
        events_by_trace:
            Mapping of ``{trace_id: [event_dict, ...]}`` to export. Each
            event dict should have at minimum ``event_type`` and ``payload``.

        Returns
        -------
        int
            Number of traces successfully exported.
        """
        self._require_connected()

        if not events_by_trace:
            return 0

        exported = 0
        for trace_id, events in events_by_trace.items():
            before = self._resilience.total_failures
            self._export_single_trace(trace_id, events)
            if self._resilience.total_failures == before:
                exported += 1

        log.info("layerlens: exported %d traces to Langfuse", exported)
        return exported

    @resilient_callback(callback_name="_export_single_trace")
    def _export_single_trace(self, trace_id: str, events: List[Dict[str, Any]]) -> None:
        """Build + POST a single trace's ingestion batch.

        Wrapped with @resilient_callback so a single bad trace doesn't
        abort the rest of the batch export. The success/failure of each
        trace is tracked via the resilience tracker.
        """
        batch = self._build_ingestion_batch(trace_id, events)
        if batch:
            self._post_ingestion(batch)

    def _build_ingestion_batch(
        self,
        trace_id: str,
        events: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Convert a list of flat events into Langfuse ingestion batch items."""
        batch: List[Dict[str, Any]] = []
        langfuse_trace_id = uuid.uuid4().hex

        # Collect agent.input / agent.output to form the trace envelope
        trace_input: Optional[str] = None
        trace_output: Optional[str] = None
        trace_name: Optional[str] = None

        for evt in events:
            etype = evt.get("event_type", "")
            payload = evt.get("payload", {})

            if etype == "agent.input":
                trace_input = payload.get("content") or payload.get("messages")
                trace_name = trace_name or payload.get("name")
            elif etype == "agent.output":
                trace_output = payload.get("content") or payload.get("output_message")

        # Trace envelope
        trace_body: Dict[str, Any] = {
            "id": langfuse_trace_id,
            "name": trace_name or f"layerlens-{trace_id[:8]}",
            "metadata": {"layerlens_trace_id": trace_id},
        }
        if trace_input is not None:
            trace_body["input"] = trace_input
        if trace_output is not None:
            trace_body["output"] = trace_output

        batch.append(
            {
                "id": uuid.uuid4().hex,
                "type": "trace-create",
                "timestamp": _iso_now(),
                "body": trace_body,
            }
        )

        # Convert individual events to observations
        for evt in events:
            etype = evt.get("event_type", "")
            payload = evt.get("payload", {})
            span_id = evt.get("span_id", new_span_id())
            span_name = evt.get("span_name")

            if etype == "model.invoke":
                batch.append(
                    self._event_to_generation(
                        langfuse_trace_id,
                        span_id,
                        span_name,
                        payload,
                    )
                )
            elif etype == "tool.call":
                batch.append(
                    self._event_to_span(
                        langfuse_trace_id,
                        span_id,
                        span_name,
                        payload,
                    )
                )
            elif etype in ("agent.input", "agent.output"):
                # Already handled in trace envelope
                continue
            elif etype == "cost.record":
                # Cost is embedded in generation; skip standalone
                continue
            else:
                # Emit as generic Langfuse event
                batch.append(
                    self._event_to_langfuse_event(
                        langfuse_trace_id,
                        span_id,
                        span_name,
                        etype,
                        payload,
                    )
                )

        return batch

    @staticmethod
    def _event_to_generation(
        trace_id: str,
        span_id: str,
        span_name: Optional[str],
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "traceId": trace_id,
            "name": span_name or payload.get("name", "generation"),
            "model": payload.get("model", ""),
            "metadata": {"layerlens_span_id": span_id},
        }
        messages = payload.get("messages")
        if messages is not None:
            body["input"] = messages
        output_msg = payload.get("output_message")
        if output_msg is not None:
            body["output"] = output_msg

        # Token usage
        usage: Dict[str, int] = {}
        prompt_tokens = payload.get("tokens_prompt")
        if prompt_tokens:
            usage["promptTokens"] = prompt_tokens
        completion_tokens = payload.get("tokens_completion")
        if completion_tokens:
            usage["completionTokens"] = completion_tokens
        total = payload.get("tokens_total")
        if total:
            usage["totalTokens"] = total
        if usage:
            body["usage"] = usage

        return {
            "id": uuid.uuid4().hex,
            "type": "generation-create",
            "timestamp": _iso_now(),
            "body": body,
        }

    @staticmethod
    def _event_to_span(
        trace_id: str,
        span_id: str,
        span_name: Optional[str],
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "traceId": trace_id,
            "name": span_name or payload.get("tool_name", "span"),
            "metadata": {"layerlens_span_id": span_id},
        }
        inp = payload.get("input")
        if inp is not None:
            body["input"] = inp
        out = payload.get("output")
        if out is not None:
            body["output"] = out

        return {
            "id": uuid.uuid4().hex,
            "type": "span-create",
            "timestamp": _iso_now(),
            "body": body,
        }

    @staticmethod
    def _event_to_langfuse_event(
        trace_id: str,
        span_id: str,
        span_name: Optional[str],
        event_type: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "traceId": trace_id,
            "name": span_name or event_type,
            "metadata": {"layerlens_span_id": span_id, "event_type": event_type},
            "input": payload,
        }
        return {
            "id": uuid.uuid4().hex,
            "type": "event-create",
            "timestamp": _iso_now(),
            "body": body,
        }

    def _post_ingestion(self, batch: List[Dict[str, Any]]) -> None:
        """POST a batch to the Langfuse ingestion endpoint."""
        resp = self._http.post(  # type: ignore[union-attr]
            "/api/public/ingestion",
            json={"batch": batch},
        )
        resp.raise_for_status()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _require_connected(self) -> None:
        if not self._connected or self._http is None:
            raise RuntimeError("LangfuseAdapter is not connected. Call connect() first.")


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _safe_dict(value: Any) -> Dict[str, Any]:
    """Coerce *value* to a dict, returning ``{}`` on failure."""
    if isinstance(value, dict):
        return value
    return {}


def _iso_now() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
