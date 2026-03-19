"""
Langfuse <-> STRATIX Bidirectional Field Mapper

Maps Langfuse trace/observation structures to STRATIX canonical events
and vice versa.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


class LangfuseToSTRATIXMapper:
    """
    Maps Langfuse traces and observations to STRATIX canonical event dicts.

    Each Langfuse trace produces multiple STRATIX events:
    - trace.input  -> agent.input (L1)
    - trace.output -> agent.output (L1)
    - span         -> agent.code (L2) or tool.call (L5a)
    - generation   -> model.invoke (L3) + cost.record (Cross)
    - metadata     -> environment.config (L4a)
    - errors       -> policy.violation (Cross)
    """

    def map_trace(self, trace: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Map a complete Langfuse trace (with observations) to STRATIX events.

        Args:
            trace: Langfuse trace dict from the API, including nested observations.

        Returns:
            List of STRATIX event dicts ready for ingestion.
        """
        trace_id = trace.get("id", str(uuid.uuid4()))
        events: list[dict[str, Any]] = []
        timestamp = trace.get("timestamp", datetime.now(timezone.utc).isoformat())
        seq = 0

        # Trace-level metadata (L4a)
        metadata = trace.get("metadata")
        if metadata:
            events.append(self._make_event(
                event_type="environment.config",
                trace_id=trace_id,
                timestamp=timestamp,
                sequence_id=seq,
                payload={
                    "config_type": "langfuse_trace_metadata",
                    "config": metadata,
                    "framework": "langfuse",
                },
                langfuse_metadata=self._extract_trace_metadata(trace),
            ))
            seq += 1

        # Trace input -> agent.input (L1)
        trace_input = trace.get("input")
        if trace_input is not None:
            events.append(self._make_event(
                event_type="agent.input",
                trace_id=trace_id,
                timestamp=timestamp,
                sequence_id=seq,
                payload={
                    "agent_id": trace.get("name", "langfuse_agent"),
                    "input_text": self._to_str(trace_input),
                    "input": trace_input,
                    "framework": "langfuse",
                },
                langfuse_metadata=self._extract_trace_metadata(trace),
            ))
            seq += 1

        # Sort observations by start_time for temporal ordering
        observations = trace.get("observations", [])
        observations = sorted(
            observations,
            key=lambda o: o.get("startTime", o.get("start_time", "")),
        )

        for obs in observations:
            obs_events = self._map_observation(obs, trace_id, seq)
            events.extend(obs_events)
            seq += len(obs_events)

        # Trace output -> agent.output (L1)
        trace_output = trace.get("output")
        if trace_output is not None:
            end_time = trace.get("endTime", trace.get("end_time", timestamp))
            events.append(self._make_event(
                event_type="agent.output",
                trace_id=trace_id,
                timestamp=end_time or timestamp,
                sequence_id=seq,
                payload={
                    "agent_id": trace.get("name", "langfuse_agent"),
                    "output_text": self._to_str(trace_output),
                    "output": trace_output,
                    "framework": "langfuse",
                },
                langfuse_metadata=self._extract_trace_metadata(trace),
            ))

        return events

    def _map_observation(
        self,
        obs: dict[str, Any],
        trace_id: str,
        start_seq: int,
    ) -> list[dict[str, Any]]:
        """Map a single Langfuse observation to STRATIX event(s)."""
        obs_type = obs.get("type", "SPAN").upper()
        timestamp = obs.get("startTime", obs.get("start_time", ""))

        if obs_type == "GENERATION":
            return self._map_generation(obs, trace_id, timestamp, start_seq)
        elif obs_type == "SPAN":
            return self._map_span(obs, trace_id, timestamp, start_seq)
        else:
            # EVENT or unknown type — map as agent.code
            return self._map_span(obs, trace_id, timestamp, start_seq)

    def _map_generation(
        self,
        obs: dict[str, Any],
        trace_id: str,
        timestamp: str,
        seq: int,
    ) -> list[dict[str, Any]]:
        """Map a Langfuse generation to model.invoke + cost.record."""
        events: list[dict[str, Any]] = []

        model = obs.get("model", obs.get("modelId"))
        usage = obs.get("usage", obs.get("promptTokens"))

        # Compute latency
        latency_ms = self._compute_latency_ms(obs)

        # Normalize token usage
        if isinstance(usage, dict):
            prompt_tokens = usage.get("promptTokens", usage.get("input", 0))
            completion_tokens = usage.get("completionTokens", usage.get("output", 0))
            total_tokens = usage.get("totalTokens", usage.get("total", 0))
        else:
            prompt_tokens = obs.get("promptTokens", 0)
            completion_tokens = obs.get("completionTokens", 0)
            total_tokens = obs.get("totalTokens", 0)

        # model.invoke (L3)
        payload: dict[str, Any] = {
            "provider": "langfuse",
            "model": model,
            "tokens_prompt": prompt_tokens or 0,
            "tokens_completion": completion_tokens or 0,
            "tokens_total": total_tokens or (prompt_tokens or 0) + (completion_tokens or 0),
            "framework": "langfuse",
        }
        if latency_ms is not None:
            payload["latency_ms"] = latency_ms

        # Include model parameters if present
        model_params = obs.get("modelParameters")
        if model_params:
            payload["parameters"] = model_params

        # Check for errors
        level = obs.get("level", "").upper()
        status_message = obs.get("statusMessage", "")
        if level == "ERROR":
            payload["error"] = status_message or "Generation error"

        events.append(self._make_event(
            event_type="model.invoke",
            trace_id=trace_id,
            timestamp=timestamp,
            sequence_id=seq,
            payload=payload,
        ))

        # cost.record (Cross-cutting)
        total_cost = obs.get("totalCost", obs.get("calculatedTotalCost"))
        if total_cost is not None and total_cost > 0:
            events.append(self._make_event(
                event_type="cost.record",
                trace_id=trace_id,
                timestamp=timestamp,
                sequence_id=seq + 1,
                payload={
                    "model": model,
                    "cost_usd": total_cost,
                    "tokens_prompt": prompt_tokens or 0,
                    "tokens_completion": completion_tokens or 0,
                    "framework": "langfuse",
                },
            ))

        # Error/warning observations -> policy.violation
        if level in ("ERROR", "WARNING"):
            events.append(self._make_event(
                event_type="policy.violation",
                trace_id=trace_id,
                timestamp=timestamp,
                sequence_id=seq + len(events),
                payload={
                    "violation_type": "error" if level == "ERROR" else "warning",
                    "description": status_message or f"Generation {level.lower()}",
                    "source": "langfuse_observation",
                    "observation_id": obs.get("id"),
                    "framework": "langfuse",
                },
            ))

        return events

    def _map_span(
        self,
        obs: dict[str, Any],
        trace_id: str,
        timestamp: str,
        seq: int,
    ) -> list[dict[str, Any]]:
        """Map a Langfuse span to tool.call or agent.code."""
        name = obs.get("name", "")
        obs_input = obs.get("input")
        obs_output = obs.get("output")
        latency_ms = self._compute_latency_ms(obs)
        level = obs.get("level", "").upper()
        status_message = obs.get("statusMessage", "")

        # Determine if this is a tool call (metadata hint or naming convention)
        metadata = obs.get("metadata", {}) or {}
        is_tool = (
            metadata.get("type") == "TOOL"
            or name.lower().startswith("tool_")
            or name.lower().startswith("tool:")
            or metadata.get("tool_name")
        )

        events: list[dict[str, Any]] = []

        if is_tool:
            # tool.call (L5a)
            payload: dict[str, Any] = {
                "tool_name": metadata.get("tool_name", name),
                "framework": "langfuse",
            }
            if obs_input is not None:
                payload["input"] = obs_input
            if obs_output is not None:
                payload["output"] = obs_output
            if latency_ms is not None:
                payload["latency_ms"] = latency_ms
            if level == "ERROR":
                payload["error"] = status_message or "Tool error"

            events.append(self._make_event(
                event_type="tool.call",
                trace_id=trace_id,
                timestamp=timestamp,
                sequence_id=seq,
                payload=payload,
            ))
        else:
            # agent.code (L2)
            payload = {
                "step_name": name,
                "framework": "langfuse",
            }
            if obs_input is not None:
                payload["input"] = obs_input
            if obs_output is not None:
                payload["output"] = obs_output
            if latency_ms is not None:
                payload["latency_ms"] = latency_ms

            events.append(self._make_event(
                event_type="agent.code",
                trace_id=trace_id,
                timestamp=timestamp,
                sequence_id=seq,
                payload=payload,
            ))

        # Error/warning -> policy.violation
        if level in ("ERROR", "WARNING"):
            events.append(self._make_event(
                event_type="policy.violation",
                trace_id=trace_id,
                timestamp=timestamp,
                sequence_id=seq + 1,
                payload={
                    "violation_type": "error" if level == "ERROR" else "warning",
                    "description": status_message or f"Span {level.lower()}",
                    "source": "langfuse_observation",
                    "observation_id": obs.get("id"),
                    "framework": "langfuse",
                },
            ))

        return events

    # --- Helpers ---

    @staticmethod
    def _make_event(
        event_type: str,
        trace_id: str,
        timestamp: str,
        sequence_id: int,
        payload: dict[str, Any],
        langfuse_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Construct a normalized STRATIX event dict."""
        event: dict[str, Any] = {
            "event_type": event_type,
            "trace_id": trace_id,
            "timestamp": timestamp,
            "sequence_id": sequence_id,
            "payload": payload,
        }
        if langfuse_metadata:
            event["metadata"] = langfuse_metadata
        return event

    @staticmethod
    def _extract_trace_metadata(trace: dict[str, Any]) -> dict[str, Any]:
        """Extract Langfuse-specific metadata from a trace."""
        meta: dict[str, Any] = {
            "langfuse_trace_id": trace.get("id"),
        }
        if trace.get("sessionId"):
            meta["langfuse_session_id"] = trace["sessionId"]
        if trace.get("userId"):
            meta["langfuse_user_id"] = trace["userId"]
        if trace.get("tags"):
            meta["langfuse_tags"] = trace["tags"]
        if trace.get("scores"):
            meta["langfuse_scores"] = trace["scores"]
        return meta

    @staticmethod
    def _compute_latency_ms(obs: dict[str, Any]) -> float | None:
        """Compute latency from observation start/end times."""
        start = obs.get("startTime", obs.get("start_time"))
        end = obs.get("endTime", obs.get("end_time"))
        if not start or not end:
            return None
        try:
            if isinstance(start, str):
                start_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
            else:
                start_dt = start
            if isinstance(end, str):
                end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
            else:
                end_dt = end
            delta = end_dt - start_dt
            return delta.total_seconds() * 1000
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _to_str(value: Any) -> str:
        """Convert a value to string representation."""
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            import json
            return json.dumps(value)
        return str(value)


class STRATIXToLangfuseMapper:
    """
    Maps STRATIX canonical events back to Langfuse trace/observation structures.

    Used for exporting STRATIX traces to Langfuse.
    """

    def map_events_to_trace(
        self,
        events: list[dict[str, Any]],
        trace_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Map a list of STRATIX events to a Langfuse trace with observations.

        Returns a dict with 'trace' (trace body) and 'observations' (list of observations).
        """
        trace_id = trace_id or str(uuid.uuid4())

        trace_body: dict[str, Any] = {
            "id": trace_id,
            "name": "stratix-export",
            "tags": ["stratix-exported"],
            "metadata": {"stratix_trace_id": trace_id},
        }
        observations: list[dict[str, Any]] = []

        for event in events:
            event_type = event.get("event_type", "")
            payload = event.get("payload", {})
            timestamp = event.get("timestamp", datetime.now(timezone.utc).isoformat())
            event_metadata = event.get("metadata", {})

            if event_type == "agent.input":
                trace_body["input"] = payload.get("input", payload.get("input_text"))
                if not trace_body.get("name") or trace_body["name"] == "stratix-export":
                    agent_id = payload.get("agent_id")
                    if agent_id:
                        trace_body["name"] = agent_id

            elif event_type == "agent.output":
                trace_body["output"] = payload.get("output", payload.get("output_text"))

            elif event_type == "model.invoke":
                obs = self._make_generation(payload, timestamp, trace_id)
                observations.append(obs)

            elif event_type == "tool.call":
                obs = self._make_tool_span(payload, timestamp, trace_id)
                observations.append(obs)

            elif event_type == "agent.code":
                obs = self._make_default_span(payload, timestamp, trace_id)
                observations.append(obs)

            elif event_type == "cost.record":
                # Cost is attached to corresponding generation — find matching
                self._attach_cost(observations, payload)

            elif event_type == "environment.config":
                config = payload.get("config", {})
                existing_meta = trace_body.get("metadata", {})
                existing_meta["environment_config"] = config
                trace_body["metadata"] = existing_meta

            elif event_type == "agent.handoff":
                obs = self._make_handoff_span(payload, timestamp, trace_id)
                observations.append(obs)

            elif event_type == "agent.state.change":
                obs = self._make_state_span(payload, timestamp, trace_id)
                observations.append(obs)

        return {"trace": trace_body, "observations": observations}

    @staticmethod
    def _make_generation(
        payload: dict[str, Any],
        timestamp: str,
        trace_id: str,
    ) -> dict[str, Any]:
        """Create a Langfuse generation observation from model.invoke event."""
        gen: dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "traceId": trace_id,
            "type": "GENERATION",
            "name": payload.get("model", "unknown-model"),
            "startTime": timestamp,
            "model": payload.get("model"),
        }
        # Token usage
        usage: dict[str, Any] = {}
        if payload.get("tokens_prompt"):
            usage["promptTokens"] = payload["tokens_prompt"]
        if payload.get("tokens_completion"):
            usage["completionTokens"] = payload["tokens_completion"]
        if payload.get("tokens_total"):
            usage["totalTokens"] = payload["tokens_total"]
        if usage:
            gen["usage"] = usage

        # Parameters
        if payload.get("parameters"):
            gen["modelParameters"] = payload["parameters"]

        # Latency -> end time
        if payload.get("latency_ms"):
            gen["endTime"] = timestamp  # Approximate

        # Error
        if payload.get("error"):
            gen["level"] = "ERROR"
            gen["statusMessage"] = payload["error"]

        return gen

    @staticmethod
    def _make_tool_span(
        payload: dict[str, Any],
        timestamp: str,
        trace_id: str,
    ) -> dict[str, Any]:
        """Create a Langfuse TOOL span from tool.call event."""
        span: dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "traceId": trace_id,
            "type": "SPAN",
            "name": payload.get("tool_name", "unknown-tool"),
            "startTime": timestamp,
            "metadata": {"type": "TOOL"},
        }
        if payload.get("input") is not None:
            span["input"] = payload["input"]
        if payload.get("output") is not None:
            span["output"] = payload["output"]
        if payload.get("error"):
            span["level"] = "ERROR"
            span["statusMessage"] = payload["error"]
        return span

    @staticmethod
    def _make_default_span(
        payload: dict[str, Any],
        timestamp: str,
        trace_id: str,
    ) -> dict[str, Any]:
        """Create a Langfuse DEFAULT span from agent.code event."""
        span: dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "traceId": trace_id,
            "type": "SPAN",
            "name": payload.get("step_name", "execution-step"),
            "startTime": timestamp,
        }
        if payload.get("input") is not None:
            span["input"] = payload["input"]
        if payload.get("output") is not None:
            span["output"] = payload["output"]
        return span

    @staticmethod
    def _make_handoff_span(
        payload: dict[str, Any],
        timestamp: str,
        trace_id: str,
    ) -> dict[str, Any]:
        """Create a Langfuse span for agent.handoff event."""
        return {
            "id": str(uuid.uuid4()),
            "traceId": trace_id,
            "type": "SPAN",
            "name": f"handoff:{payload.get('from_agent', '?')}->{payload.get('to_agent', '?')}",
            "startTime": timestamp,
            "metadata": {
                "type": "HANDOFF",
                "from_agent": payload.get("from_agent"),
                "to_agent": payload.get("to_agent"),
                "context": payload.get("context"),
            },
        }

    @staticmethod
    def _make_state_span(
        payload: dict[str, Any],
        timestamp: str,
        trace_id: str,
    ) -> dict[str, Any]:
        """Create a Langfuse span for agent.state.change event."""
        return {
            "id": str(uuid.uuid4()),
            "traceId": trace_id,
            "type": "SPAN",
            "name": f"state-change:{payload.get('state_type', 'unknown')}",
            "startTime": timestamp,
            "metadata": {
                "type": "STATE_CHANGE",
                "before": payload.get("before"),
                "after": payload.get("after"),
            },
        }

    @staticmethod
    def _attach_cost(
        observations: list[dict[str, Any]],
        cost_payload: dict[str, Any],
    ) -> None:
        """Attach cost to the matching generation observation."""
        model = cost_payload.get("model")
        cost_usd = cost_payload.get("cost_usd")
        if cost_usd is None:
            return

        # Find a matching generation by model name
        for obs in reversed(observations):
            if obs.get("type") == "GENERATION":
                if model is None or obs.get("model") == model:
                    obs["totalCost"] = cost_usd
                    return
        # No match — attach to last generation if any
        for obs in reversed(observations):
            if obs.get("type") == "GENERATION":
                obs["totalCost"] = cost_usd
                return
