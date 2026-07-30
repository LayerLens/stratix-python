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
from layerlens.instrument._secret_scrub import REDACTION, safe_error, find_secrets, scrub_payload, scrub_secrets
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


class TestSecretNetEnforcementWiring:
    """Prove the credential-sprawl net is actually WIRED into the autouse
    ``_enforce_schema_lock`` teardown (LAY-3572 / B25), not merely callable
    directly. A unit suite that uploads an event carrying a secret must FAIL —
    if someone dropped the ``scan_for_secrets(events)`` line from the fixture,
    the inner suite would report 0 errors and this test would fail (which a
    direct ``scan_for_secrets(...)`` call could never detect)."""

    def test_autouse_net_fails_a_suite_that_uploads_a_secret(self, pytester) -> None:
        pytester.makeconftest(
            "from tests.instrument.conftest import _enforce_schema_lock, record_for_schema_lock  # noqa: F401"
        )
        pytester.makepyfile(
            test_inner="""
            from tests.instrument.conftest import record_for_schema_lock

            def test_uploads_a_secret():
                # schema-valid agent.error (has a category) so the ERROR is from
                # the secret net specifically, not the schema validator.
                record_for_schema_lock([
                    {"event_type": "agent.error",
                     "payload": {"error": "auth failed: sk-ant-api03-AAAAAAAAAAAAAAAAAAAA1234",
                                 "error_type": "AuthError"}}
                ])
            """
        )
        result = pytester.runpytest()
        result.assert_outcomes(passed=1, errors=1)


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


# ---------------------------------------------------------------------------
# Nested secrets in error structures (LAY-3572 / R2 / B24) — provider SDKs and
# structured-error bodies put str(exc)/details inside a dict or list under the
# error key, which the top-level-only chokepoint missed.
# ---------------------------------------------------------------------------


def test_chokepoint_scrubs_secret_nested_in_error_dict() -> None:
    collector = TraceCollector(object(), CaptureConfig.standard())  # default capture_content=True
    collector.emit(
        "agent.error",
        {
            "error": {"detail": "auth failed: sk-proj-LEAKLEAKLEAKLEAKLEAK0123456789", "code": "E_AUTH"},
            "error_type": "AuthError",
        },
        span_id="s1",
    )
    payload = collector.events[0]["payload"]
    assert "sk-proj-" not in json.dumps(payload), "secret nested inside an error dict not scrubbed by the chokepoint"
    assert payload.get("error_type") == "AuthError", "category over-scrubbed"


def test_chokepoint_scrubs_secret_in_error_list() -> None:
    collector = TraceCollector(object(), CaptureConfig.standard())
    collector.emit(
        "agent.error",
        {"error_message": ["validation failed", "token AKIAIOSFODNN7EXAMPLE rejected"], "error_code": "E1"},
        span_id="s1",
    )
    assert "AKIAIOSFODNN7EXAMPLE" not in json.dumps(collector.events[0]["payload"]), (
        "secret in an error list not scrubbed"
    )


# PEM private keys + Azure SAS signatures (LAY-3572 / B26) — high-value shapes
# the pattern set did not cover.
_NEW_SECRET_SHAPES = [
    "-----BEGIN RSA PRIVATE KEY-----",
    "-----BEGIN PRIVATE KEY-----",
    "-----BEGIN OPENSSH PRIVATE KEY-----",
    "https://acct.blob.core.windows.net/c/b?sv=2021-08-06&sig=aB3dEfGh%2FIjKlMnOpQrStUvWx0123456789%3D",
]


@pytest.mark.parametrize("secret", _NEW_SECRET_SHAPES)
def test_scrub_removes_pem_and_azure_sas(secret: str) -> None:
    assert find_secrets(secret), f"new secret shape not detected: {secret!r}"
    scrubbed = scrub_secrets(f"config blob: {secret} trailing")
    assert REDACTION in scrubbed
    assert "BEGIN" not in scrubbed.replace("config blob", ""), "PEM header survived scrub"
    if "sig=" in secret:
        assert "aB3dEfGh" not in scrubbed, "Azure SAS signature survived scrub"


# ---------------------------------------------------------------------------
# BROAD secret scan over ALL string fields (LAY-3625 / A10 — user-approved
# 2026-06-25). Secrets ride in tool args, model output, agui patch values, and
# elicitation prompts — NOT just the 4 ERROR_KEYS. Under the shipping default
# (capture_content=True) redact_payload is a no-op, so a secret in tool.call
# arguments / model.invoke output uploaded CLEARTEXT to atlas-app. The collector
# chokepoint now scrubs every string value, orthogonal to capture_content.
# ---------------------------------------------------------------------------

_NON_ERROR_LEAK_CASES = [
    pytest.param(
        "tool.call",
        {"tool": "charge_card", "arguments": {"api_key": "sk-proj-LEAKLEAKLEAKLEAKLEAK0123456789"}},
        "sk-proj-",
        id="tool_call_arguments",
    ),
    pytest.param(
        "model.invoke",
        {"output_message": {"role": "assistant", "content": "your key is sk-ant-api03-AAAAAAAAAAAAAAAAAAAA1234"}},
        "sk-ant-",
        id="model_invoke_output",
    ),
    pytest.param(
        "agui.state",
        {"operations": [{"op": "add", "path": "/creds", "value": "AKIAIOSFODNN7EXAMPLE"}]},
        "AKIAIOSFODNN7EXAMPLE",
        id="agui_state_delta_value",
    ),
    pytest.param(
        "mcp.elicitation",
        {"title": "enter token ghp_abcdefghijklmnopqrstuvwxyz0123456789", "phase": "request"},
        "ghp_",
        id="mcp_elicitation_title",
    ),
]


@pytest.mark.parametrize("event_type,payload,secret", _NON_ERROR_LEAK_CASES)
def test_chokepoint_scrubs_secret_in_non_error_field(event_type: str, payload: dict, secret: str) -> None:
    """A10: a secret in tool args / model output / agui patch value / elicitation
    title must be scrubbed at the collector chokepoint even when content capture
    is ON (capture_content=True, the full()/opt-in path where redact_payload is a
    no-op — secrets are orthogonal to capture_content). The pre-A10 scrub only
    touched ERROR_KEYS, so these all leaked. Bite: narrow scrub_payload back to
    ERROR_KEYS and this goes RED."""
    collector = TraceCollector(object(), CaptureConfig(capture_content=True))
    collector.emit(event_type, dict(payload), span_id="s1")
    blob = json.dumps(collector.events[0]["payload"], default=str)
    assert secret not in blob, f"{event_type}: secret in a non-error field leaked past the broad chokepoint"
    assert REDACTION in blob, f"{event_type}: expected the secret to be replaced with the redaction marker"


def test_broad_scrub_preserves_benign_nested_content() -> None:
    """The broad scan must not over-scrub benign nested content (the regexes are
    specific). A normal tool result with no secret-shaped values is untouched."""
    collector = TraceCollector(object(), CaptureConfig(capture_content=True))
    payload = {"tool": "search", "arguments": {"query": "weather in Paris"}, "result": {"temp": 20, "ok": True}}
    collector.emit("tool.call", dict(payload), span_id="s1")
    out = collector.events[0]["payload"]
    assert out["arguments"]["query"] == "weather in Paris", "benign nested content over-scrubbed"
    assert REDACTION not in json.dumps(out, default=str), "benign payload incorrectly redacted"


# ---------------------------------------------------------------------------
# Credit-card PAN + CVC (A15 / UCP-Q2 — PCI). ACP rfc.delegate_payment §277:
# "logs MUST NOT contain full PAN or CVC"; UCP: "Never log raw credentials". A
# card number reaching ANY uploaded string must be scrubbed at the collector
# chokepoint regardless of capture_content — the commerce content-key redaction
# only fires under capture_content=False and only on known keys, so a card that
# rides an unmapped field / the default content-on config needs the scrub net.
# The PAN pattern is Luhn-gated + card-grouped so a benign 16-digit order id is
# NOT over-scrubbed.
# ---------------------------------------------------------------------------

# Luhn-VALID test PANs (the official scheme test numbers) in every grouping form.
_PANS = [
    "4111111111111111",  # Visa, contiguous
    "4111 1111 1111 1111",  # Visa, space-grouped
    "4111-1111-1111-1111",  # Visa, dash-grouped
    "5555555555554444",  # Mastercard
    "378282246310005",  # Amex (15 digits)
    "4012888888881881",  # Visa
    "6011111111111117",  # Discover
]


@pytest.mark.parametrize("pan", _PANS)
def test_scrub_removes_card_pan(pan: str) -> None:
    text = f"charge failed for card {pan} at merchant"
    scrubbed = scrub_secrets(text)
    assert pan not in scrubbed, f"PAN survived scrub: {pan!r}"
    assert REDACTION in scrubbed
    assert "card_pan" in find_secrets(text), f"find_secrets failed to flag a real PAN: {pan!r}"


# Benign 16-ish-digit values that are NOT Luhn-valid cards: order ids, tracking
# numbers, timestamps, plain large numbers. These must NOT be over-scrubbed
# (keeping the PAN pattern specific — Luhn + card grouping).
_NOT_PANS = [
    "1234567890123456",  # 16 digits, Luhn-INVALID
    "ORDER-1234567890123456",  # order id with a letter+dash prefix
    "9999999999999999",  # Luhn-invalid run
    "2026-06-26T12:00:00",  # timestamp
    "tracking 1Z999AA10123456784",  # alphanumeric tracking number
    "amount 1234.56 usd",  # money, not a card
    # A Luhn-VALID card-shaped run FUSED inside a larger alphanumeric token (a
    # hex hash, a base64 id) is NOT a standalone PAN: a real card number in a log
    # is delimited by non-word chars, never welded to hex letters. The keyed-HMAC
    # action_context_hash (a2ui/ap2/a2a) is a random sha256 hexdigest that
    # intermittently contains such a run — it was corrupted by the PAN scrubber.
    "sha256:aa4111111111111111ff",  # Luhn-valid Visa run between hex letters
    "deadbeef4012888888881881cafe",  # Luhn-valid Visa run inside a hex digest
]


@pytest.mark.parametrize("value", _NOT_PANS)
def test_benign_card_like_value_not_over_scrubbed(value: str) -> None:
    assert "card_pan" not in find_secrets(value), f"benign value mis-flagged as a PAN: {value!r}"
    assert scrub_secrets(value) == value, f"benign card-like value over-scrubbed: {value!r}"


# CVC / CVV only in a LABELED context (a bare 3-4 digit number is everywhere).
_CVCS = ["cvc: 737", '"cvv":"4321"', "security_code=123", "card_verification_value: 1234", "cvc2=321"]


@pytest.mark.parametrize("text", _CVCS)
def test_scrub_removes_labeled_cvc(text: str) -> None:
    assert "card_cvc" in find_secrets(text), f"labeled CVC not flagged: {text!r}"
    assert REDACTION in scrub_secrets(text)


def test_bare_three_digit_number_not_scrubbed_as_cvc() -> None:
    benign = "there were 737 items in the cart"
    assert "card_cvc" not in find_secrets(benign), "bare 3-digit number mis-flagged as a CVC"
    assert scrub_secrets(benign) == benign, "benign 3-digit number over-scrubbed"


def test_chokepoint_scrubs_pan_in_commerce_field_under_default_config() -> None:
    """A real PAN riding a commerce.* event field is scrubbed at the collector
    chokepoint even under the content-on config (capture_content=True, where
    redact_payload is a no-op for content keys). Bite: remove the ``card_pan``
    pattern from SECRET_PATTERNS -> the PAN survives -> RED."""
    collector = TraceCollector(object(), CaptureConfig(capture_content=True))
    collector.emit(
        "commerce.checkout_completed",
        {"session_id": "s1", "card": {"number": "4111 1111 1111 1111", "cvc": "737"}, "amount": 49.99},
        span_id="s1",
    )
    blob = json.dumps(collector.events[0]["payload"], default=str)
    assert "4111 1111 1111 1111" not in blob, "card PAN leaked past the collector chokepoint under content-on"
    assert REDACTION in blob, "expected the PAN to be replaced with the redaction marker"


def test_hash_digest_with_embedded_pan_shaped_run_not_over_scrubbed() -> None:
    """Regression + the exact CI flake this reproduces: a keyed-HMAC digest
    (a2ui/ap2/a2a ``action_context_hash`` = ``"sha256:"`` + a random hexdigest)
    can, by chance, contain a 13-19 digit run that passes Luhn. The PAN scrubber
    must NOT corrupt the digest — a real card number is a standalone token, never
    fused to a digest's hex letters. ``scrub_payload`` runs on EVERY uploaded
    event, so an over-match here silently mangles a linkage-critical hash.

    Bite: revert the ``card_pan`` boundary to ``(?<![\\d.-])...(?![\\d.-])``
    (letters not excluded) and the embedded run is redacted -> RED. This is the
    intermittent ``test_a2ui_adapter::test_no_content_config_keeps_the_hash``
    failure, made deterministic here."""
    # 4111111111111111 is a Luhn-valid Visa test PAN, here welded between hex
    # letters exactly as it appears inside a random sha256 hexdigest.
    digest = "sha256:aabbcc4111111111111111ddeeff"
    out = scrub_payload({"action_context_hash": digest})
    assert out["action_context_hash"] == digest, "hash digest corrupted by PAN over-scrub"
    assert "card_pan" not in find_secrets(digest), "PAN-shaped run fused in a hash mis-flagged as a card"


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
