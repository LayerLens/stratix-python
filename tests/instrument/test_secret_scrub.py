"""Secret scrubbing + runtime guard (credential-sprawl defense).

Secrets leak independently of capture_content: a provider exception that echoes
an API key is uploaded under the DEFAULT config. These tests prove (a) the
production scrubber strips the real shapes, (b) the real provider error path is
scrubbed end-to-end, (c) benign text is not over-scrubbed, and (d) the runtime
guard actually bites.
"""

from __future__ import annotations

import json

import pytest

from layerlens.instrument._context import _current_collector
from layerlens.instrument._collector import TraceCollector
from layerlens.instrument._secret_scrub import REDACTION, safe_error, find_secrets, scrub_secrets
from layerlens.instrument._capture_config import CaptureConfig

from ._secret_scan import scan_for_secrets

# Run as a required CI gate via `-m invariant` (see .github/workflows/invariants.yaml).
pytestmark = pytest.mark.invariant

SECRETS = [
    "sk-proj-ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    "sk-ant-api03-ZZZZZZZZZZZZZZZZZZZZ1234",
    "AKIAIOSFODNN7EXAMPLE",
    "Bearer abcdef0123456789ABCDEF",
    "postgres://admin:hunter2@db.internal:5432/app",
    "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NSJ9.abcdef123456",
    # added after adversarial review (vertex/azure/stripe/github in scope)
    "AIzaSyA1234567890abcdefghijklmnopqrstuv",
    "AccountKey=abcd1234EFGH5678ijkl9012MNOP3456==",
    "sk_live_ABCDEFGHIJKLMNOP1234567890",
    "password=SuperSecret123",
    "ghp_abcdefghijklmnopqrstuvwxyz0123456789",
]


@pytest.mark.parametrize("secret", SECRETS)
def test_scrub_removes_known_secret_shapes(secret: str) -> None:
    text = f"boom: {secret} happened"
    scrubbed = scrub_secrets(text)
    assert secret not in scrubbed, f"secret survived scrub: {secret}"
    assert REDACTION in scrubbed
    assert find_secrets(text), "find_secrets failed to flag a known secret"


def test_scrub_does_not_touch_benign_text() -> None:
    benign = "ValueError: list index out of range (model gpt-4o, 3 tokens)"
    assert scrub_secrets(benign) == benign, "benign error text was over-scrubbed"
    assert not find_secrets(benign)


def test_safe_error_scrubs_exception_string() -> None:
    exc = ValueError("Incorrect API key provided: sk-proj-ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    out = safe_error(exc)
    assert "sk-proj-" not in out and REDACTION in out


def test_provider_emit_llm_error_scrubs_api_key_real_path() -> None:
    """Real provider error path: the uploaded agent.error must not carry the key,
    but error_type must survive (observability preserved)."""
    from layerlens.instrument.adapters.providers._emit_helpers import emit_llm_error

    collector = TraceCollector(object(), CaptureConfig.standard())  # default: capture_content=True
    token = _current_collector.set(collector)
    try:
        emit_llm_error(
            "openai.chat.completions",
            ValueError("Incorrect API key provided: sk-proj-LEAKLEAKLEAKLEAKLEAK0123456789"),
            12.0,
        )
    finally:
        _current_collector.reset(token)
    events = collector.events
    blob = json.dumps([e["payload"] for e in events], default=str)
    assert "sk-proj-" not in blob, "provider error leaked the API key into the uploaded payload"
    assert events[0]["payload"].get("error_type") == "ValueError", "error_type over-scrubbed"


def test_guard_bites_on_seeded_secret() -> None:
    """The runtime guard must FAIL on a seeded secret — proves the CI net is live,
    not vacuous (hardcoded literal so it is independent of the shared regex)."""
    events = [{"event_type": "agent.error", "payload": {"error": "key sk-ant-api03-AAAAAAAAAAAAAAAAAAAA1234"}}]
    with pytest.raises(AssertionError, match="secret"):
        scan_for_secrets(events)


def test_guard_passes_clean_events() -> None:
    events = [{"event_type": "agent.error", "payload": {"error_type": "ValueError", "status": "error"}}]
    scan_for_secrets(events)  # must not raise


def test_collector_chokepoint_scrubs_error_under_default_config() -> None:
    """The chokepoint runs even under the DEFAULT capture_content=True (where
    redact_payload is a no-op) — so any adapter's str(exc) error is scrubbed,
    not just the 2 provider sites that call safe_error."""
    collector = TraceCollector(object(), CaptureConfig.standard())  # capture_content=True
    collector.emit(
        "agent.error",
        {"error": "auth failed: sk-proj-LEAKLEAKLEAKLEAKLEAK0123456789", "error_type": "AuthError"},
        span_id="s1",
    )
    payload = collector.events[0]["payload"]
    assert "sk-proj-" not in json.dumps(payload), "chokepoint did not scrub error under default config"
    assert payload.get("error_type") == "AuthError", "category over-scrubbed"


def test_real_mcp_error_path_scrubbed_without_safe_error_call() -> None:
    """A framework/protocol error site that does NOT call safe_error (mcp tool
    wrapper) is still scrubbed by the collector chokepoint, under default config."""
    from layerlens.instrument.adapters.protocols.mcp.adapter import MCPProtocolAdapter
    from layerlens.instrument.adapters.protocols.mcp.tool_wrapper import wrap_mcp_tool_call

    adapter = MCPProtocolAdapter(capture_config=CaptureConfig.standard())
    collector = TraceCollector(object(), CaptureConfig.standard())
    token = _current_collector.set(collector)
    try:

        def broken(**_kw):
            raise RuntimeError("upstream rejected key AKIAIOSFODNN7EXAMPLE")

        with pytest.raises(RuntimeError):
            wrap_mcp_tool_call(broken, adapter)(name="charge", arguments={"q": "x"})
    finally:
        _current_collector.reset(token)
    blob = json.dumps([e["payload"] for e in collector.events], default=str)
    assert "AKIAIOSFODNN7EXAMPLE" not in blob, "mcp error path leaked an AWS key (chokepoint missed it)"
