"""Self-tests for the event-schema contract lock (LAY-3583 / T9).

The lock itself is enforced through the capture fixtures (every adapter unit
suite validates its uploaded events); these tests prove the validator can
actually fail — a guard that cannot fail guards nothing.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from tests.instrument._event_schema import (
    EventSchemaViolation,
    validate_event,
    validate_events,
)


def _event(event_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return {"event_type": event_type, "payload": payload}


class TestValidatorCatchesDrift:
    def test_unknown_event_type_fails(self) -> None:
        problems = validate_event(_event("model.invocation", {"framework": "newfw"}))
        assert problems and "unknown event type" in problems[0]

    def test_duration_ns_outside_drift_table_fails(self) -> None:
        problems = validate_event(_event("model.invoke", {"framework": "langchain", "duration_ns": 12}))
        assert problems and "duration_ns" in problems[0]

    def test_duration_ns_inside_drift_table_passes(self) -> None:
        assert validate_event(_event("agent.output", {"framework": "smolagents", "duration_ns": 12})) == []

    def test_unknown_usage_key_fails(self) -> None:
        problems = validate_event(_event("model.invoke", {"provider": "openai", "usage": {"promptTokens": 5}}))
        assert problems and "usage keys" in problems[0]

    def test_non_int_usage_value_fails(self) -> None:
        problems = validate_event(_event("model.invoke", {"usage": {"prompt_tokens": "5"}}))
        assert problems and "must be int" in problems[0]

    def test_mixed_token_vocabularies_fail(self) -> None:
        problems = validate_event(_event("cost.record", {"framework": "x", "tokens_prompt": 1, "prompt_tokens": 1}))
        assert problems and "mixes framework token vocabulary" in problems[0]

    def test_cost_record_without_tokens_fails(self) -> None:
        problems = validate_event(_event("cost.record", {"provider": "openai"}))
        assert problems and "without any token counts" in problems[0]

    def test_bad_latency_type_fails(self) -> None:
        problems = validate_event(_event("tool.call", {"latency_ms": "fast"}))
        assert problems and "latency_ms" in problems[0]

    def test_compliant_provider_event_passes(self) -> None:
        assert (
            validate_event(
                _event(
                    "model.invoke",
                    {
                        "provider": "openai",
                        "model": "gpt-4",
                        "latency_ms": 12.5,
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                    },
                )
            )
            == []
        )

    def test_validate_events_raises_with_context(self) -> None:
        with pytest.raises(EventSchemaViolation, match="LAY-3583"):
            validate_events([_event("not.a.thing", {})])
