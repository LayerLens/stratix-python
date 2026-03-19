"""STRATIX Event Parser - Generic event extraction from STRATIX traces."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelInvocation:
    """Represents a model/LLM invocation event."""
    model: str
    prompt: str
    response: str
    tokens_in: int
    tokens_out: int
    latency_ms: float
    timestamp: str
    event_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCall:
    """Represents a tool call event."""
    tool_name: str
    inputs: dict[str, Any]
    output: Any
    latency_ms: float
    timestamp: str
    event_id: str = ""
    success: bool = True
    error: str | None = None


@dataclass
class StateChange:
    """Represents a state mutation event."""
    node: str
    field: str
    old_hash: str
    new_hash: str
    timestamp: str
    event_id: str = ""


class EventParser:
    """Generic event extraction from STRATIX traces."""

    LAYERS = {"L1", "L2", "L3", "L4", "L5a", "L5b"}

    def extract_by_layer(self, events: list[dict], layer: str) -> list[dict]:
        """Extract events by layer (L1, L2, L3, L4, L5a, L5b)."""
        if layer not in self.LAYERS:
            raise ValueError(f"Invalid layer: {layer}")
        return [e for e in events if self._get_layer(e) == layer]

    def extract_model_invocations(self, events: list[dict]) -> list[ModelInvocation]:
        """Extract all LLM/model invocation events."""
        result = []
        for event in events:
            if self._get_event_type(event) in ("model_invoke", "llm_call", "model_call"):
                p = self._get_payload(event)
                result.append(ModelInvocation(
                    model=p.get("model", "unknown"),
                    prompt=p.get("prompt", p.get("input", "")),
                    response=p.get("response", p.get("output", "")),
                    tokens_in=p.get("tokens_in", p.get("input_tokens", 0)),
                    tokens_out=p.get("tokens_out", p.get("output_tokens", 0)),
                    latency_ms=p.get("latency_ms", p.get("duration_ms", 0.0)),
                    timestamp=self._get_timestamp(event),
                    event_id=self._get_event_id(event),
                ))
        return result

    def extract_tool_calls(self, events: list[dict]) -> list[ToolCall]:
        """Extract all tool call events."""
        result = []
        for event in events:
            if self._get_event_type(event) in ("tool_call", "tool_invoke", "function_call"):
                p = self._get_payload(event)
                result.append(ToolCall(
                    tool_name=p.get("tool_name", p.get("name", "unknown")),
                    inputs=p.get("inputs", p.get("arguments", {})),
                    output=p.get("output", p.get("result")),
                    latency_ms=p.get("latency_ms", p.get("duration_ms", 0.0)),
                    timestamp=self._get_timestamp(event),
                    event_id=self._get_event_id(event),
                    success=p.get("success", True),
                    error=p.get("error"),
                ))
        return result

    def extract_state_changes(self, events: list[dict]) -> list[StateChange]:
        """Extract state mutation events."""
        result = []
        for event in events:
            if self._get_event_type(event) in ("state_change", "state_mutation", "agent_state_change"):
                p = self._get_payload(event)
                result.append(StateChange(
                    node=p.get("node", p.get("agent", "unknown")),
                    field=p.get("field", p.get("key", "")),
                    old_hash=p.get("old_hash", self._hash(p.get("old_value"))),
                    new_hash=p.get("new_hash", self._hash(p.get("new_value"))),
                    timestamp=self._get_timestamp(event),
                    event_id=self._get_event_id(event),
                ))
        return result

    def extract_by_type(self, events: list[dict], event_type: str) -> list[dict]:
        """Extract events by event_type field."""
        return [e for e in events if self._get_event_type(e) == event_type]

    def extract_by_agent(self, events: list[dict], agent_id: str) -> list[dict]:
        """Extract events by agent ID."""
        return [e for e in events if self._get_agent_id(e) == agent_id]

    def _get_layer(self, event: dict) -> str | None:
        p = self._get_payload(event)
        return p.get("layer") or event.get("identity", {}).get("layer")

    def _get_event_type(self, event: dict) -> str:
        identity = event.get("identity", {})
        return identity.get("event_type") or event.get("event_type") or self._get_payload(event).get("event_type", "unknown")

    def _get_payload(self, event: dict) -> dict:
        return event.get("payload", event)

    def _get_timestamp(self, event: dict) -> str:
        from datetime import datetime
        ts = event.get("identity", {}).get("timestamps", {}).get("created_at")
        return ts or event.get("timestamp") or datetime.utcnow().isoformat()

    def _get_event_id(self, event: dict) -> str:
        return event.get("identity", {}).get("span_id") or event.get("event_id", "")

    def _get_agent_id(self, event: dict) -> str:
        return event.get("identity", {}).get("agent_id") or event.get("agent_id", "")

    def _hash(self, value: Any) -> str:
        if value is None:
            return ""
        try:
            return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:16]
        except Exception:
            return hashlib.sha256(str(value).encode()).hexdigest()[:16]
