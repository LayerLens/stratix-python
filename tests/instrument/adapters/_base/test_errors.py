"""Unit tests for the shared error-event emission helper.

Cross-pollinates the AutoGen ``wrappers.py:94-108`` and LangChain
``on_*_error`` callback patterns to all ten "lighter" runtime adapters.
This test module exercises the helper itself; per-adapter integration
tests in ``tests/instrument/adapters/frameworks/`` confirm that each
target adapter actually invokes the helper from its callbacks.
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest import mock

import pytest

from layerlens.instrument.adapters._base import (
    MAX_MESSAGE_CHARS,
    SAFE_CONTEXT_KEYS,
    DEFAULT_EVENT_TYPE,
    MAX_TRACEBACK_CHARS,
    MAX_TRACEBACK_FRAMES,
    AdapterInfo,
    BaseAdapter,
    AdapterHealth,
    AdapterStatus,
    ReplayableTrace,
    AdapterCapability,
    emit_error_event,
    build_error_payload,
)
from layerlens.instrument.adapters._base.errors import (
    _scrub_secrets,
    _resolve_org_id,
    _format_traceback,
)

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


class _RecordingStratix:
    """Captures emitted events without contacting any real client."""

    def __init__(self, org_id: str | None = None) -> None:
        self.events: List[Dict[str, Any]] = []
        self.org_id = org_id

    def emit(self, *args: Any, **kwargs: Any) -> None:
        if len(args) == 2 and isinstance(args[0], str):
            self.events.append({"event_type": args[0], "payload": args[1]})


class _StubAdapter(BaseAdapter):
    """Minimal concrete BaseAdapter for testing the helper.

    Implements the abstract surface only — no framework integration.
    """

    FRAMEWORK = "stub"
    VERSION = "0.0.0"

    def connect(self) -> None:
        self._connected = True
        self._status = AdapterStatus.HEALTHY

    def disconnect(self) -> None:
        self._connected = False
        self._status = AdapterStatus.DISCONNECTED

    def health_check(self) -> AdapterHealth:
        return AdapterHealth(
            status=self._status,
            framework_name=self.FRAMEWORK,
            adapter_version=self.VERSION,
        )

    def get_adapter_info(self) -> AdapterInfo:
        return AdapterInfo(
            name="StubAdapter",
            version=self.VERSION,
            framework=self.FRAMEWORK,
            capabilities=[AdapterCapability.TRACE_TOOLS],
        )

    def serialize_for_replay(self) -> ReplayableTrace:
        return ReplayableTrace(
            adapter_name="StubAdapter",
            framework=self.FRAMEWORK,
            trace_id="test",
        )


def _raise_at_depth(depth: int, exc: Exception) -> None:
    """Raise ``exc`` after ``depth`` frames of recursion.

    Used to produce predictable traceback frame counts.
    """
    if depth <= 0:
        raise exc
    _raise_at_depth(depth - 1, exc)


@pytest.fixture
def adapter() -> _StubAdapter:
    a = _StubAdapter(stratix=_RecordingStratix())
    a.connect()
    return a


@pytest.fixture
def stratix(adapter: _StubAdapter) -> _RecordingStratix:
    return adapter._stratix  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Public API: emit_error_event
# ---------------------------------------------------------------------------


def test_emit_error_event_creates_policy_violation_by_default(
    adapter: _StubAdapter, stratix: _RecordingStratix
) -> None:
    """Default event type matches the AutoGen / LangChain convention."""
    try:
        raise ValueError("rate limit hit")
    except ValueError as exc:
        emit_error_event(adapter, exc)

    assert len(stratix.events) == 1
    evt = stratix.events[0]
    assert evt["event_type"] == DEFAULT_EVENT_TYPE == "policy.violation"


def test_emit_error_event_includes_exception_type_and_message(
    adapter: _StubAdapter, stratix: _RecordingStratix
) -> None:
    try:
        raise RuntimeError("upstream API down")
    except RuntimeError as exc:
        emit_error_event(adapter, exc)

    payload = stratix.events[0]["payload"]
    assert payload["exception_type"] == "RuntimeError"
    assert payload["message"] == "upstream API down"
    assert payload["framework"] == "stub"
    assert payload["severity"] == "error"


def test_emit_error_event_propagates_org_id_from_stratix(adapter: _StubAdapter) -> None:
    """Multi-tenant: org_id flows from the adapter's stratix client."""
    adapter._stratix = _RecordingStratix(org_id="org_acme_42")
    try:
        raise ValueError("boom")
    except ValueError as exc:
        emit_error_event(adapter, exc)

    payload = adapter._stratix.events[0]["payload"]  # type: ignore[attr-defined]
    assert payload["org_id"] == "org_acme_42"


def test_emit_error_event_explicit_org_id_overrides_stratix(adapter: _StubAdapter) -> None:
    adapter._stratix = _RecordingStratix(org_id="org_default")
    try:
        raise ValueError("x")
    except ValueError as exc:
        emit_error_event(adapter, exc, org_id="org_override")

    payload = adapter._stratix.events[0]["payload"]  # type: ignore[attr-defined]
    assert payload["org_id"] == "org_override"


def test_emit_error_event_omits_org_id_when_unavailable(
    adapter: _StubAdapter, stratix: _RecordingStratix
) -> None:
    """Missing org_id means no field on payload (sinks treat absent as platform tenant)."""
    try:
        raise ValueError("x")
    except ValueError as exc:
        emit_error_event(adapter, exc)
    assert "org_id" not in stratix.events[0]["payload"]


def test_emit_error_event_truncates_long_message(
    adapter: _StubAdapter, stratix: _RecordingStratix
) -> None:
    """Defends event sinks from megabyte-scale exception messages."""
    long_msg = "x" * (MAX_MESSAGE_CHARS * 4)
    try:
        raise RuntimeError(long_msg)
    except RuntimeError as exc:
        emit_error_event(adapter, exc)

    payload = stratix.events[0]["payload"]
    assert len(payload["message"]) <= MAX_MESSAGE_CHARS
    assert payload["message"].endswith("...")


def test_emit_error_event_truncates_traceback_frames(
    adapter: _StubAdapter, stratix: _RecordingStratix
) -> None:
    """Older (shallower) frames are dropped first."""
    try:
        _raise_at_depth(MAX_TRACEBACK_FRAMES * 2, RuntimeError("deep"))
    except RuntimeError as exc:
        emit_error_event(adapter, exc)

    payload = stratix.events[0]["payload"]
    tb = payload["traceback"]
    # Frame count proxy: count occurrences of "File " markers from the
    # standard traceback.format_tb output. Should be <= MAX_TRACEBACK_FRAMES.
    frame_count = tb.count("File ")
    assert frame_count <= MAX_TRACEBACK_FRAMES


def test_emit_error_event_truncates_traceback_to_char_limit(adapter: _StubAdapter) -> None:
    """Even with few frames, total tb length is bounded."""
    huge_locals = "x" * (MAX_TRACEBACK_CHARS * 2)
    try:
        _ = huge_locals
        raise RuntimeError("trigger")
    except RuntimeError as exc:
        # Patch format_tb to return a single huge frame string
        with mock.patch(
            "layerlens.instrument.adapters._base.errors.traceback.format_tb",
            return_value=["File <fake>:1 in <module>\n    " + ("y" * (MAX_TRACEBACK_CHARS * 4)) + "\n"],
        ):
            emit_error_event(adapter, exc)

    payload = adapter._stratix.events[0]["payload"]  # type: ignore[attr-defined]
    assert len(payload["traceback"]) <= MAX_TRACEBACK_CHARS


def test_emit_error_event_filters_unsafe_context_keys(
    adapter: _StubAdapter, stratix: _RecordingStratix
) -> None:
    """Only allow-listed keys reach the payload — protects against PII leakage."""
    try:
        raise ValueError("err")
    except ValueError as exc:
        emit_error_event(
            adapter,
            exc,
            context={
                "tool_name": "web_search",  # SAFE — allow-listed
                "user_email": "alice@example.com",  # UNSAFE — dropped
                "raw_input": "user said: my password is hunter2",  # UNSAFE — dropped
                "model": "gpt-5",  # SAFE
            },
        )

    payload = stratix.events[0]["payload"]
    assert payload["tool_name"] == "web_search"
    assert payload["model"] == "gpt-5"
    assert "user_email" not in payload
    assert "raw_input" not in payload


def test_emit_error_event_redacts_secrets_in_message(
    adapter: _StubAdapter, stratix: _RecordingStratix
) -> None:
    """API keys / bearer tokens / sk-* patterns are redacted from messages."""
    try:
        raise RuntimeError("auth failed: api_key=sk-abc123def456ghi789jklmnop and Bearer xyz789")
    except RuntimeError as exc:
        emit_error_event(adapter, exc)

    payload = stratix.events[0]["payload"]
    assert "sk-abc123def456" not in payload["message"]
    assert "xyz789" not in payload["message"]
    assert "REDACTED" in payload["message"]


def test_emit_error_event_re_raise_pattern_preserves_exception() -> None:
    """The helper does NOT swallow the exception; callers must re-raise."""
    adapter = _StubAdapter(stratix=_RecordingStratix())
    adapter.connect()

    def framework_call() -> None:
        try:
            raise RuntimeError("framework failed")
        except RuntimeError as exc:
            emit_error_event(adapter, exc)
            raise

    with pytest.raises(RuntimeError, match="framework failed"):
        framework_call()

    # Event was still emitted before re-raise
    assert len(adapter._stratix.events) == 1  # type: ignore[attr-defined]
    assert adapter._stratix.events[0]["event_type"] == "policy.violation"  # type: ignore[attr-defined]


def test_emit_error_event_supports_custom_event_type(
    adapter: _StubAdapter, stratix: _RecordingStratix
) -> None:
    """Adapters may emit ``agent.error`` or ``tool.error`` instead of policy.violation."""
    try:
        raise ValueError("tool blew up")
    except ValueError as exc:
        emit_error_event(adapter, exc, event_type="agent.error", severity="warning")

    evt = stratix.events[0]
    assert evt["event_type"] == "agent.error"
    assert evt["payload"]["severity"] == "warning"


def test_emit_error_event_handles_exception_with_no_traceback(
    adapter: _StubAdapter, stratix: _RecordingStratix
) -> None:
    """Constructed exceptions (never raised) have no __traceback__."""
    exc = ValueError("never raised")
    emit_error_event(adapter, exc)

    payload = stratix.events[0]["payload"]
    assert payload["traceback"] == ""
    assert payload["exception_type"] == "ValueError"
    assert payload["message"] == "never raised"


def test_emit_error_event_handles_exception_with_broken_str(
    adapter: _StubAdapter, stratix: _RecordingStratix
) -> None:
    """Exceptions whose __str__ raises must not crash the helper."""

    class BadException(Exception):
        def __str__(self) -> str:
            raise RuntimeError("can't stringify")

    try:
        raise BadException()
    except BadException as exc:
        emit_error_event(adapter, exc)

    # repr() fallback should produce SOMETHING non-empty, but at minimum
    # the event must still be emitted with a known exception_type.
    payload = stratix.events[0]["payload"]
    assert payload["exception_type"] == "BadException"


def test_emit_error_event_skips_when_circuit_open(adapter: _StubAdapter) -> None:
    """Respects the circuit breaker — does not bypass BaseAdapter protections."""
    # Force-open the circuit.
    adapter._circuit_open = True
    adapter._circuit_opened_at = 1e18  # cooldown effectively never expires

    try:
        raise ValueError("x")
    except ValueError as exc:
        emit_error_event(adapter, exc)

    # When the circuit is open the event is dropped at emit_dict_event time.
    assert adapter._stratix.events == []  # type: ignore[attr-defined]


def test_emit_error_event_records_event_type_in_trace_buffer(
    adapter: _StubAdapter,
) -> None:
    """Event participates in BaseAdapter's _trace_events list (used for replay)."""
    try:
        raise RuntimeError("x")
    except RuntimeError as exc:
        emit_error_event(adapter, exc)

    assert any(e["event_type"] == "policy.violation" for e in adapter._trace_events)


# ---------------------------------------------------------------------------
# build_error_payload (exposed for inspection / extension by adapters)
# ---------------------------------------------------------------------------


def test_build_error_payload_returns_fresh_dict_each_call(adapter: _StubAdapter) -> None:
    """Mutations by one caller must not leak into other invocations."""
    try:
        raise ValueError("x")
    except ValueError as exc:
        p1 = build_error_payload(adapter, exc)
        p1["mutated"] = True
        p2 = build_error_payload(adapter, exc)

    assert "mutated" not in p2


def test_build_error_payload_includes_exception_module(adapter: _StubAdapter) -> None:
    """Exception module helps disambiguate ``RateLimitError`` from different libs."""
    try:
        raise ValueError("x")
    except ValueError as exc:
        payload = build_error_payload(adapter, exc)

    assert payload["exception_module"] == "builtins"


def test_build_error_payload_drops_empty_framework_override(adapter: _StubAdapter) -> None:
    """Empty-string framework in context must NOT clobber adapter.FRAMEWORK."""
    try:
        raise ValueError("x")
    except ValueError as exc:
        payload = build_error_payload(adapter, exc, context={"framework": ""})

    assert payload["framework"] == "stub"


def test_safe_context_keys_does_not_include_pii_keys() -> None:
    """Lint test: the allow-list must not accidentally accept PII keys."""
    pii_keys = {"user_email", "raw_input", "password", "api_key", "secret", "token"}
    leaked = set(SAFE_CONTEXT_KEYS) & pii_keys
    assert not leaked, f"SAFE_CONTEXT_KEYS leaks PII keys: {leaked}"


# ---------------------------------------------------------------------------
# Internal helpers (white-box tests for security-critical code)
# ---------------------------------------------------------------------------


def test_scrub_secrets_redacts_api_key_assignment() -> None:
    text = 'Failed: api_key="sk-real-secret-1234567890"'
    scrubbed = _scrub_secrets(text)
    assert "sk-real-secret" not in scrubbed
    assert "REDACTED" in scrubbed


def test_scrub_secrets_redacts_bearer_token() -> None:
    text = "401 Unauthorized: Authorization: Bearer eyJabc123def456"
    scrubbed = _scrub_secrets(text)
    assert "eyJabc123def456" not in scrubbed
    assert "REDACTED" in scrubbed


def test_scrub_secrets_redacts_openai_style_sk_token() -> None:
    text = "Token sk-abcdefghijklmnopqrstuv was rejected"
    scrubbed = _scrub_secrets(text)
    assert "sk-abcdefghijklmnopqrstuv" not in scrubbed


def test_scrub_secrets_is_idempotent() -> None:
    text = 'api_key="sk-real-secret-1234567890"'
    once = _scrub_secrets(text)
    twice = _scrub_secrets(once)
    assert once == twice


def test_scrub_secrets_handles_empty_string() -> None:
    assert _scrub_secrets("") == ""


def test_format_traceback_returns_empty_for_unraised_exception() -> None:
    """Constructed-but-not-raised exceptions have no traceback."""
    exc = ValueError("never raised")
    assert _format_traceback(exc) == ""


def test_resolve_org_id_falls_back_to_tenant_id_attr(adapter: _StubAdapter) -> None:
    """Some clients expose tenant_id rather than org_id."""

    class _TenantStratix:
        tenant_id = "tenant_99"

        def emit(self, *args: Any, **kwargs: Any) -> None: ...

    adapter._stratix = _TenantStratix()  # type: ignore[assignment]
    assert _resolve_org_id(adapter, None) == "tenant_99"


def test_resolve_org_id_returns_none_when_attribute_access_raises(adapter: _StubAdapter) -> None:
    """A misbehaving stratix client must not crash the error path."""

    class _AngryStratix:
        @property
        def org_id(self) -> str:
            raise RuntimeError("nope")

        @property
        def tenant_id(self) -> str:
            raise RuntimeError("nope")

        def emit(self, *args: Any, **kwargs: Any) -> None: ...

    adapter._stratix = _AngryStratix()  # type: ignore[assignment]
    assert _resolve_org_id(adapter, None) is None
