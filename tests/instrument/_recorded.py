"""Recorded-real-response corpus — loader, replay seams, scrub guard (LAY-3614).

The corpus closes the systemic gap that let ``agentforce`` and ``bedrock_agents``
ship green on *fictional* schemas: no automated CI layer exercised a **real**
provider/service response shape (unit doubles *are* the assumed shape, the
matrix uses fake models, the live suite runs in no CI workflow). We capture each
adapter's real upstream response **once, offline** (real creds, like the live
suite), scrub it, commit it under ``tests/fixtures/recorded/<adapter>/<scenario>.json``,
and **replay it in CI** through the SDK's own transport seam so the adapter's
real parser runs against the real shape — no creds, no network, no spend.

The one design rule that makes this work (non-negotiable):
**record UPSTREAM of the parser, assert DOWNSTREAM of it.** The fixture is the
provider's raw transport response (= our adapter's *input*); the assertion is
the adapter's emitted *events*. Recording the adapter's *output* and replaying
that would just rebuild the self-shape mirror with extra steps — so a fixture is
always the thing we do not control.

Three replay seams, one per transport family:

* ``http`` — ``httpx.MockTransport`` returning the recorded body. The real SDK
  client still does its real deserialization (the proven ``test_azure_openai``
  pattern). Covers openai, anthropic, azure_openai, litellm, langfuse,
  agentforce, and any framework whose model client accepts an ``http_client=``.
* ``boto3`` — ``botocore.stub.Stubber`` + ``StreamingBody`` for the bedrock
  provider's ``invoke_model`` / ``converse``.
* ``eventstream`` — a single-read ``_FakeEventStream`` injected into
  ``parsed['completion']`` for the bedrock_agents ``InvokeAgent`` stream that
  ``Stubber`` cannot model.

**Honest limit (document it, don't over-trust it):** a fixture is a *snapshot*.
It catches "our parser vs a real shape we captured", **not** "the provider
changed shape since we captured". The ``captured_at`` provenance makes staleness
visible; the periodic gated **live smoke** (a future session) is the freshness
check.
"""

from __future__ import annotations

import io
import os
import re
import json
import base64
from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path

import httpx

# tests/instrument/_recorded.py  ->  tests/fixtures/recorded
RECORDED_ROOT = Path(__file__).resolve().parent.parent / "fixtures" / "recorded"

#: Every committed fixture must stamp these so staleness is visible later.
PROVENANCE_KEYS = ("provider", "sdk_version", "model", "scenario", "captured_at")

#: Marks a fixture seeded from the SDK type / a documented example because no
#: credentials exist to capture it live (azure_openai, google_vertex). These are
#: honestly weaker than a live capture and are flagged so the gap stays visible.
PENDING_CREDS = "pending-creds"


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def recorded_path(adapter: str, scenario: str = "default") -> Path:
    return RECORDED_ROOT / adapter / f"{scenario}.json"


def load_recorded(adapter: str, scenario: str = "default") -> Dict[str, Any]:
    """Load a committed fixture. Raises ``FileNotFoundError`` (the RED state for
    a replay test written before its fixture is captured)."""
    path = recorded_path(adapter, scenario)
    if not path.is_file():
        raise FileNotFoundError(
            f"no recorded fixture for {adapter}/{scenario} at {path} — "
            f"capture it offline with tests/fixtures/record_corpus.py (LAY-3614)."
        )
    with open(path) as f:
        fixture = json.load(f)
    prov = fixture.get("provenance", {})
    missing = [k for k in PROVENANCE_KEYS if k not in prov]
    if missing:
        raise ValueError(f"{path} is missing provenance keys: {missing}")
    return fixture


def iter_recorded_files() -> List[Path]:
    if not RECORDED_ROOT.is_dir():
        return []
    return sorted(RECORDED_ROOT.rglob("*.json"))


# ---------------------------------------------------------------------------
# Scrub guard — capture-time redaction + a committed-fixture leak scan
# ---------------------------------------------------------------------------

#: Patterns that must never appear in a committed fixture. Used both at capture
#: time (to redact) and by ``tests/test_recorded_corpus.py`` (to fail the build
#: if a secret slipped through). Keep these specific enough to avoid flagging
#: legitimate recorded content (model ids, token counts, prose).
_SECRET_PATTERNS: Tuple[Tuple[str, "re.Pattern[str]"], ...] = (
    ("openai_key", re.compile(r"sk-(?:proj-)?[A-Za-z0-9_-]{20,}")),
    ("anthropic_key", re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}")),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("bearer_token", re.compile(r"[Bb]earer\s+[A-Za-z0-9._\-]{20,}")),
    ("jwt", re.compile(r"eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}")),
    ("aws_account_id_in_arn", re.compile(r"arn:aws[^:]*:[^:]*:[^:]*:(\d{12}):")),
    ("google_api_key", re.compile(r"AIza[0-9A-Za-z_\-]{30,}")),
    ("slack_token", re.compile(r"xox[abprs]-[0-9A-Za-z-]{10,}")),
)

#: Header/body names whose values are always credentials — redacted wholesale at
#: capture time regardless of the value pattern (covers OAuth token-response
#: bodies, e.g. Salesforce ``access_token`` / ``signature``, that match no
#: generic secret regex).
_SECRET_HEADER_KEYS = frozenset(
    {
        "authorization",
        "api-key",
        "x-api-key",
        "x-amz-security-token",
        "x-goog-api-key",
        "openai-organization",
        "cookie",
        "set-cookie",
        "access_token",
        "refresh_token",
        "id_token",
        "signature",
        "client_secret",
        "private_key",
    }
)

_REDACTED = "[REDACTED]"
_ACCOUNT_PLACEHOLDER = "000000000000"


def find_secret_leaks(text: str) -> List[str]:
    """Return ``"<name>: <match>"`` for every secret pattern found in ``text``.

    The scrubbed account-id placeholder is not a leak — only a *real* 12-digit
    account id inside an ARN counts.
    """
    hits: List[str] = []
    for name, pat in _SECRET_PATTERNS:
        for m in pat.finditer(text):
            if name == "aws_account_id_in_arn" and m.group(1) == _ACCOUNT_PLACEHOLDER:
                continue
            hits.append(f"{name}: {m.group(0)[:24]}...")
    return hits


def scrub(obj: Any) -> Any:
    """Recursively redact secrets from a capture before it is written to disk.

    * Auth/credential headers are replaced wholesale.
    * AWS account ids inside ARNs are replaced with a stable placeholder.
    * Any string value matching a secret pattern is replaced with ``[REDACTED]``.
    """
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            if isinstance(k, str) and k.lower() in _SECRET_HEADER_KEYS:
                out[k] = _REDACTED
            else:
                out[k] = scrub(v)
        return out
    if isinstance(obj, list):
        return [scrub(v) for v in obj]
    if isinstance(obj, str):
        return _scrub_str(obj)
    return obj


def _scrub_str(s: str) -> str:
    # Account ids first (keep the surrounding ARN intact for shape realism).
    s = re.sub(r"(arn:aws[^:]*:[^:]*:[^:]*:)\d{12}(:)", rf"\g<1>{_ACCOUNT_PLACEHOLDER}\g<2>", s)
    for name, pat in _SECRET_PATTERNS:
        if name == "aws_account_id_in_arn":
            continue
        s = pat.sub(_REDACTED, s)
    return s


# ---------------------------------------------------------------------------
# Replay seam 1 — http (httpx.MockTransport)
# ---------------------------------------------------------------------------


def mock_transport(fixture: Dict[str, Any]) -> Tuple[httpx.MockTransport, List[httpx.Request]]:
    """Build an ``httpx.MockTransport`` that serves a fixture's recorded HTTP
    interactions in order, recording each outgoing request for assertions.

    The returned transport is injected through the SDK's ``http_client=`` seam
    so the *real* client does its real routing + deserialization against the
    recorded body. Identical principle to ``test_azure_openai._make_client``.
    """
    interactions = fixture["interactions"]
    if not interactions:
        raise ValueError("http fixture has no interactions")
    requests: List[httpx.Request] = []
    state = {"i": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        idx = min(state["i"], len(interactions) - 1)
        state["i"] += 1
        resp = interactions[idx]["response"]
        status = resp.get("status_code", 200)
        headers = resp.get("headers") or {}
        if "json" in resp and resp["json"] is not None:
            return httpx.Response(status, json=resp["json"], headers=headers)
        return httpx.Response(status, text=resp.get("text", ""), headers=headers)

    return httpx.MockTransport(handler), requests


def recorded_http_client(fixture: Dict[str, Any]) -> Tuple[httpx.Client, List[httpx.Request]]:
    """Convenience: a real ``httpx.Client`` bound to the fixture's MockTransport."""
    transport, requests = mock_transport(fixture)
    return httpx.Client(transport=transport), requests


# ---------------------------------------------------------------------------
# Replay seam 2 — boto3 (botocore Stubber + StreamingBody)
# ---------------------------------------------------------------------------


def _streaming_body(body: bytes) -> Any:
    from botocore.response import StreamingBody

    return StreamingBody(io.BytesIO(body), len(body))


def bedrock_stub_response(fixture: Dict[str, Any]) -> Dict[str, Any]:
    """Build the ``stubber.add_response`` payload for a recorded bedrock op.

    ``invoke_model`` carries a single-read ``StreamingBody`` rebuilt from the
    recorded body bytes; ``converse`` is a plain modeled dict.
    """
    op = fixture["operation"]
    resp = fixture["response"]
    meta = {"ResponseMetadata": {"RequestId": fixture.get("request_id", "00000000-0000-0000-0000-000000000000")}}
    if op == "invoke_model":
        body = base64.b64decode(resp["body_b64"])
        return {
            "body": _streaming_body(body),
            "contentType": resp.get("content_type", "application/json"),
            **meta,
        }
    if op == "converse":
        out = {k: v for k, v in resp.items()}
        out.update(meta)
        return out
    raise ValueError(f"unsupported bedrock operation: {op}")


# ---------------------------------------------------------------------------
# Replay seam 3 — eventstream (_FakeEventStream for InvokeAgent completion)
# ---------------------------------------------------------------------------


class FakeEventStream:
    """Single-read stand-in for ``botocore.eventstream.EventStream``.

    Iterating consumes it exactly like the real single-use wire stream (a second
    pass yields nothing) — this catches an adapter that drains the stream inside
    its ``after-call`` hook. An optional ``error`` is raised once events are
    exhausted, to simulate a mid/end ``EventStreamError``. Mirrors the proven
    helper in ``test_bedrock_agents_doubles.py``.
    """

    def __init__(self, events: List[Dict[str, Any]], *, error: Optional[BaseException] = None) -> None:
        self._events = list(events)
        self._idx = 0
        self._error = error
        self.closed = False

    def __iter__(self) -> "FakeEventStream":
        return self

    def __next__(self) -> Dict[str, Any]:
        if self._idx >= len(self._events):
            if self._error is not None:
                err, self._error = self._error, None
                raise err
            raise StopIteration
        event = self._events[self._idx]
        self._idx += 1
        return event

    def close(self) -> None:
        self.closed = True

    def get_initial_response(self) -> Dict[str, Any]:
        return {"status_code": 200}


def _decode_bytes(obj: Any) -> Any:
    """Rehydrate the capture's ``{"__bytes_b64__": "..."}`` markers back to the
    real ``bytes`` the adapter expects (e.g. an InvokeAgent ``chunk.bytes``)."""
    if isinstance(obj, dict):
        if set(obj) == {"__bytes_b64__"}:
            return base64.b64decode(obj["__bytes_b64__"])
        return {k: _decode_bytes(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode_bytes(v) for v in obj]
    return obj


def fake_completion_stream(fixture: Dict[str, Any], *, error: Optional[BaseException] = None) -> FakeEventStream:
    events = [_decode_bytes(e) for e in fixture["events"]]
    return FakeEventStream(events, error=error)


# ---------------------------------------------------------------------------
# Capture-mode gate (used only by tests/fixtures/record_corpus.py — never CI)
# ---------------------------------------------------------------------------


def recording_enabled() -> bool:
    """True only in the gated offline capture pass (never set in CI)."""
    return os.environ.get("LAYERLENS_RECORD") == "1"
