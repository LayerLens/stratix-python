"""Self-tests for the event-schema contract lock (LAY-3583 / T9).

The lock is enforced *after* each test by the ``_enforce_schema_lock`` autouse
fixture (root ``conftest.py``): the capture helpers record every uploaded event
via ``record_for_schema_lock`` and the fixture validates them once the test body
returns — i.e. OUTSIDE the production upload path, which swallows exceptions and
previously made the lock a silent no-op (LAY-3613). These tests prove both that
the validator can actually fail (a guard that cannot fail guards nothing) and
that the record→validate wiring is in place.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from tests.instrument._event_schema import (
    KNOWN_EVENT_TYPES,
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


class TestEnforcementWiring:
    """Guard the LAY-3613 fix END TO END: a unit suite that uploads an
    unregistered event must FAIL, because the autouse ``_enforce_schema_lock``
    fixture validates *after* the test body — outside the swallowing upload path.

    This runs an inner pytest (``pytester``) so the autouse teardown's effect is
    observable as a real outcome. If someone deleted ``_enforce_schema_lock``
    (re-introducing the bug where validation is swallowed by ``_upload_sync``),
    the inner suite would report 0 errors and THIS test would fail — which a
    direct ``validate_events(...)`` call could never detect.
    """

    def test_autouse_lock_fails_a_suite_that_uploads_an_unregistered_type(self, pytester):
        pytester.makeconftest(
            # Re-export the real autouse fixture + recorder so the inner session
            # enforces exactly as the repo does (same module-global buffer).
            "from tests.instrument.conftest import _enforce_schema_lock, record_for_schema_lock  # noqa: F401"
        )
        pytester.makepyfile(
            test_inner="""
            from tests.instrument.conftest import record_for_schema_lock

            def test_uploads_unregistered_event():
                # Body passes; the autouse teardown must reject this unregistered type.
                record_for_schema_lock([{"event_type": "totally.unregistered.type", "payload": {}}])
            """
        )
        result = pytester.runpytest()
        # Body passes, teardown raises EventSchemaViolation -> reported as an error.
        result.assert_outcomes(passed=1, errors=1)

    def test_all_emitted_event_constants_are_registered(self) -> None:
        """Every event-type constant the SDK emits (``layerlens.instrument._events``)
        must be in ``KNOWN_EVENT_TYPES``. This is a structural guard: a new emit
        constant cannot ship unregistered, and registered strings whose only
        emitters bypass the capture fixtures — ``a2a.agent.discovered`` /
        ``a2a.delegation`` (the a2a suites assert adapter internals directly, not
        through an upload) — are still pinned to the source of truth, so a typo
        between the emit constant and the registry is caught even when a2a isn't
        installed.
        """
        from layerlens.instrument import _events

        emitted = {
            value
            for name, value in vars(_events).items()
            if name.isupper() and isinstance(value, str) and "." in value and value.islower()
        }
        unregistered = sorted(emitted - set(KNOWN_EVENT_TYPES))
        assert not unregistered, f"event-type constants emitted but not registered in the schema lock: {unregistered}"

    def test_record_for_schema_lock_feeds_the_validator(self) -> None:
        # Unit-level check of the recorder + validator the fixture composes (kept
        # alongside the end-to-end guard above).
        from tests.instrument import conftest as root_conftest

        root_conftest._pending_schema_events.clear()
        root_conftest.record_for_schema_lock([_event("totally.unregistered.type", {})])
        try:
            with pytest.raises(EventSchemaViolation):
                validate_events(root_conftest._pending_schema_events)
        finally:
            root_conftest._pending_schema_events.clear()
