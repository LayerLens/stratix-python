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
from typing import Any, Dict, List, Pattern, FrozenSet

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
    # PEM private keys (RSA/EC/OPENSSH/DSA/plain) — header alone is a strong
    # signal; the optional body+footer are swept too when present.
    (
        "private_key_pem",
        re.compile(
            r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----(?:[\s\S]*?-----END (?:[A-Z0-9 ]+ )?PRIVATE KEY-----)?"
        ),
    ),
    # Azure Storage / Service-Bus SAS signature query parameter (?sv=...&sig=...).
    ("azure_sas", re.compile(r"(?i)[?&]sig=[A-Za-z0-9%/+]{16,}")),
    # Credit-card PAN (PCI / ACP rfc.delegate_payment §277 "logs MUST NOT
    # contain full PAN or CVC"; UCP "Never log raw credentials"). 13-19 digits
    # in card-grouping form: contiguous (4111111111111111) or grouped by single
    # spaces/dashes (4111 1111 1111 1111 / 4111-1111-1111-1111). The regex alone
    # is a CANDIDATE — every match is Luhn-validated (see ``_LUHN_VALIDATED``)
    # before it is flagged/scrubbed, so a random 16-digit value or an order id
    # with letters is NOT over-scrubbed (only mod-10-valid card numbers are).
    # The boundaries exclude ``\w`` (letters/digits/underscore) as well as ``.``
    # and ``-`` so a card-shaped digit run FUSED inside a larger alphanumeric
    # token — e.g. a Luhn-passing run that lands, by chance, inside a random
    # sha256 hexdigest (the keyed-HMAC ``action_context_hash``) — is NOT matched.
    # A real PAN in a log is delimited by whitespace/punctuation, never welded to
    # hex letters, so widening the boundary loses no genuine card.
    ("card_pan", re.compile(r"(?<![\w.-])(?:\d[ -]?){12,18}\d(?![\w.-])")),
    # CVC / CVV — a bare 3-4 digit number is everywhere, so this matches ONLY in
    # a LABELED context (cvc: 123 / "cvv":"4321" / security_code=123). The label
    # makes it specific without a checksum.
    (
        "card_cvc",
        re.compile(
            r"""(?i)\b(?:cvc2?|cvv2?|cid|security[ _-]?code|card[ _-]?verification[ _-]?(?:value|code))\b["']?\s*[:=]\s*["']?\d{3,4}\b"""
        ),
    ),
]

# Pattern names whose regex is only a CANDIDATE matcher; a hit is kept only when
# the matched text passes :func:`_luhn_ok`. Keeps the PAN pattern from nuking a
# 16-digit order id / tracking number that happens to look card-shaped.
_LUHN_VALIDATED: FrozenSet[str] = frozenset({"card_pan"})


def _luhn_ok(text: str) -> bool:
    """True iff the digits in *text* (13-19 of them) satisfy the Luhn mod-10
    checksum used by every real card scheme. A random/sequential number fails
    this ~90% of the time, so requiring it keeps the PAN scrubber specific."""
    digits = [int(c) for c in text if c.isdigit()]
    if not (13 <= len(digits) <= 19):
        return False
    total = 0
    # Double every second digit from the right.
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


# Free-text fields that routinely carry ``str(exc)`` — scrubbed unconditionally
# at the collector chokepoint (secrets leak under the DEFAULT
# capture_content=True, which redact_payload does not touch).
ERROR_KEYS = ("error", "error_message", "execution_error", "status_message")


def scrub_secrets(text: str) -> str:
    """Replace every secret-shaped substring in *text* with a redaction marker."""
    if not text:
        return text
    for name, pattern in SECRET_PATTERNS:
        if name in _LUHN_VALIDATED:
            # Only redact candidate matches that pass the checksum, so a benign
            # 16-digit value is left intact (CPython re.sub returns the input
            # unchanged when the callback redacts nothing — identity preserved).
            text = pattern.sub(lambda m: REDACTION if _luhn_ok(m.group(0)) else m.group(0), text)
        else:
            text = pattern.sub(REDACTION, text)
    return text


def safe_error(exc: object) -> str:
    """``str(exc)`` with any embedded secret scrubbed. Use at error-emit sites."""
    return scrub_secrets(str(exc))


def find_secrets(text: str) -> List[str]:
    """Return the names of every secret pattern that matches *text* (for guards)."""
    if not text:
        return []
    hits: List[str] = []
    for name, pattern in SECRET_PATTERNS:
        if name in _LUHN_VALIDATED:
            # A checksum-validated pattern flags only when a candidate match
            # actually passes Luhn (consistent with scrub_secrets).
            if any(_luhn_ok(m.group(0)) for m in pattern.finditer(text)):
                hits.append(name)
        elif pattern.search(text):
            hits.append(name)
    return hits


def _scrub_value(value: object) -> object:
    """Recursively scrub secret-shaped substrings from every string nested in
    *value*. COPY-ON-WRITE: returns the SAME object when nothing matched (the
    common case — the hot path allocates nothing), otherwise a scrubbed copy.

    Identity preservation relies on :func:`scrub_secrets` returning the original
    ``str`` object when no pattern matches (CPython ``re.sub`` returns the input
    unchanged on no-op), so a deeply-nested clean payload is detected as clean
    without a value-compare per node.
    """
    if isinstance(value, str):
        return scrub_secrets(value)
    if isinstance(value, dict):
        out = value
        for k, v in value.items():
            cleaned = _scrub_value(v)
            if cleaned is not v:
                if out is value:
                    out = dict(value)
                out[k] = cleaned
        return out
    if isinstance(value, list):
        cleaned_items = [_scrub_value(v) for v in value]
        if any(ci is not orig for ci, orig in zip(cleaned_items, value)):
            return cleaned_items
        return value
    if isinstance(value, tuple):
        cleaned_items = [_scrub_value(v) for v in value]
        if any(ci is not orig for ci, orig in zip(cleaned_items, value)):
            return tuple(cleaned_items)
        return value
    return value


def scrub_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Scrub secret-shaped values from EVERY string field of an event payload.

    The COLLECTOR-SIDE chokepoint: runs unconditionally on every event (secrets
    are ORTHOGONAL to ``capture_content`` — a credential in tool-call arguments,
    model output, an AG-UI state-delta value, or an elicitation prompt ships
    under the DEFAULT ``capture_content=True`` where ``redact_payload`` is a
    no-op). Broadened from the 4 ``ERROR_KEYS`` to the whole payload
    (LAY-3625 / A10, user-approved 2026-06-25) so a leaked key anywhere is
    caught, not just in an error string. Copy-on-write via :func:`_scrub_value`
    — a clean payload (the common case) is returned unchanged with no allocation.
    """
    cleaned = _scrub_value(payload)
    return cleaned if isinstance(cleaned, dict) else payload
