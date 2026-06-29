"""Secret scrubbing for telemetry strings (credential-sprawl defense).

Secrets are ORTHOGONAL to ``capture_content``: a provider SDK exception string
routinely embeds the customer's API key / bearer token / connection string
(e.g. ``AuthenticationError: Incorrect API key provided: sk-proj-...``), and
that error rides ``agent.error`` which is uploaded even under the DEFAULT
``capture_content=True``. ``redact_payload`` does not help — it only acts on
content fields under ``capture_content=False``. So error strings (and any other
free text that could carry a credential) must be scrubbed before they enter a
payload, regardless of capture config.

:func:`safe_error` wraps ``str(exc)`` at the provider error sites; the test
harness imports :data:`SECRET_PATTERNS` for a runtime guard that fails CI if any
secret-shaped value reaches an uploaded event (so the two can never drift).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Pattern

REDACTION = "[REDACTED-SECRET]"

# Specific enough to avoid flagging benign text. Each entry: (name, compiled).
SECRET_PATTERNS: List[tuple[str, Pattern[str]]] = [
    # OpenAI / Anthropic family keys (sk-, sk-ant-, sk-proj-, sk-svcacct-)
    ("openai_key", re.compile(r"sk-(?:ant-|proj-|svcacct-)?[A-Za-z0-9_-]{16,}")),
    # Stripe / underscore-form keys (sk_live_, rk_live_, sk_test_)
    ("stripe_key", re.compile(r"\b[sr]k_(?:live|test)_[A-Za-z0-9]{16,}")),
    # Google / Gemini / GCP API keys
    ("google_key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    # AWS access key ids (long-term AKIA + temporary ASIA)
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    # Slack (xoxb-/xoxp-...) / GitHub (ghp_/gho_/...) tokens
    ("slack_github_token", re.compile(r"\b(?:xox[baprs]-[A-Za-z0-9-]{10,}|gh[pousr]_[A-Za-z0-9]{20,})\b")),
    # Bearer tokens
    ("bearer", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}")),
    # api-key / secret / password / token assignment (api_key=..., password: ...)
    (
        "secret_assign",
        re.compile(
            r"""(?i)\b(?:api[-_]?key|secret|password|passwd|access[-_]?token|auth[-_]?token)["']?\s*[:=]\s*["']?[A-Za-z0-9._/+-]{8,}"""
        ),
    ),
    # x-api-key header echo
    ("x_api_key", re.compile(r"""(?i)\bx-api-key["']?\s*[:=]\s*\S+""")),
    # AWS session token header marker
    ("aws_session_token", re.compile(r"(?i)\bx-amz-security-token\b")),
    # Azure storage / service-bus shared key
    ("azure_account_key", re.compile(r"(?i)\bAccountKey=[A-Za-z0-9+/]{16,}={0,2}")),
    # DB / broker connection strings with embedded credentials user:pass@host
    (
        "conn_string",
        re.compile(r"(?i)\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|amqp)://[^:@/\s]+:[^@/\s]+@"),
    ),
    # JWTs (three base64url segments)
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{6,}")),
]

# Free-text fields that routinely carry ``str(exc)`` — scrubbed unconditionally
# at the collector chokepoint (secrets leak under the DEFAULT
# capture_content=True, which redact_payload does not touch).
ERROR_KEYS = ("error", "error_message", "execution_error", "status_message")


def scrub_secrets(text: str) -> str:
    """Replace every secret-shaped substring in *text* with a redaction marker."""
    if not text:
        return text
    for _name, pattern in SECRET_PATTERNS:
        text = pattern.sub(REDACTION, text)
    return text


def safe_error(exc: object) -> str:
    """``str(exc)`` with any embedded secret scrubbed. Use at error-emit sites."""
    return scrub_secrets(str(exc))


def find_secrets(text: str) -> List[str]:
    """Return the names of every secret pattern that matches *text* (for guards)."""
    if not text:
        return []
    return [name for name, pattern in SECRET_PATTERNS if pattern.search(text)]


def scrub_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Scrub secrets from the free-text error fields of an event payload.

    The COLLECTOR-SIDE chokepoint: runs unconditionally on every event (secrets
    must be scrubbed regardless of ``capture_content``), covering every adapter's
    ``str(exc)`` error site at once instead of relying on each emit site calling
    ``safe_error``. Copy-on-write — only allocates when something is scrubbed.
    """
    scrubbed = payload
    for key in ERROR_KEYS:
        value = payload.get(key)
        if isinstance(value, str):
            cleaned = scrub_secrets(value)
            if cleaned != value:
                if scrubbed is payload:
                    scrubbed = dict(payload)
                scrubbed[key] = cleaned
    return scrubbed
