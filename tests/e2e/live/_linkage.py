"""Platform-side inbound-linkage verification for the live suite.

After a trace upload, the Stratix platform's ``InboundLinkageService`` stamps
``traces.integration_id`` (synchronously, at trace-create) when the uploading
app API key matches a registered ``sdk_adapter`` inbound integration, records a
bucket in ``integration_metric_buckets``, and a periodic sweeper (~30s) flips
that integration's status ``Inactive -> Healthy``. These are ASYNC /
eventually-consistent, so status is polled with a bounded timeout — never
asserted immediately after upload.

Machine-agnostic by design. The ``integration_id`` is read back from the trace
**API** (``client.traces.get(id).integration_id``), so no DB access is needed.
Linkage assertions are opt-in via env so the suite stays green on a machine
that has not registered an audit integration:

* ``LAYERLENS_LIVE_INTEGRATION_ID`` — when set, assert the uploaded trace links
  to *exactly* this integration id and (unless disabled) poll its status to
  ``Healthy``. When unset, the helper only *records* what the API returns and
  makes no hard assertion.
* ``LAYERLENS_LIVE_LINKAGE_TIMEOUT`` — status poll budget in seconds (default 90,
  ~3 sweep intervals).
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

import httpx

from ._timing import LINKAGE_STATUS_POLL_INTERVAL_S, LINKAGE_STATUS_TIMEOUT_DEFAULT_S

INTEGRATION_ID_ENV = "LAYERLENS_LIVE_INTEGRATION_ID"
_TIMEOUT = float(os.environ.get("LAYERLENS_LIVE_LINKAGE_TIMEOUT", str(LINKAGE_STATUS_TIMEOUT_DEFAULT_S)))


def expected_integration_id() -> Optional[str]:
    val = os.environ.get(INTEGRATION_ID_ENV)
    return val.strip() if val and val.strip() else None


def linkage_configured() -> bool:
    """True when an expected integration id is configured for hard assertions."""
    return expected_integration_id() is not None


def linkage_skip_reason() -> Optional[str]:
    """Reason string for an explicit ``pytest.skip`` when linkage cannot be
    asserted, else ``None``. Callers should skip *visibly* with this rather than
    let an unconfigured check report a silent green that proves no attribution.
    """
    if linkage_configured():
        return None
    return f"linkage assertion requires {INTEGRATION_ID_ENV} (an audit sdk_adapter integration id)"


def trace_integration_id(client: Any, trace_id: str) -> Optional[str]:
    """The ``integration_id`` stamped on the trace, via the API read-back."""
    trace = client.traces.get(trace_id)
    return getattr(trace, "integration_id", None) if trace is not None else None


def _get_integration(client: Any, integration_id: str) -> Optional[Dict[str, Any]]:
    url = f"{client.base_url}organizations/{client.organization_id}/integrations/{integration_id}"
    try:
        resp = httpx.get(url, headers=client.auth_headers, timeout=30)
    except Exception:
        return None
    if resp.status_code != 200:
        return None
    body = resp.json()
    if isinstance(body, dict) and "data" in body and "status" in body:
        body = body["data"]
    return body if isinstance(body, dict) else None


def poll_status_healthy(
    client: Any,
    integration_id: str,
    *,
    timeout: float = _TIMEOUT,
    interval: float = LINKAGE_STATUS_POLL_INTERVAL_S,
) -> Optional[str]:
    """Poll the integration until ``status == 'Healthy'`` or timeout; return last status seen.

    The transition is driven by the platform's ~30s sweeper — budgets in ``_timing``.
    """
    deadline = time.time() + timeout
    last: Optional[str] = None
    while time.time() < deadline:
        doc = _get_integration(client, integration_id)
        if doc is not None:
            last = doc.get("status")
            if last == "Healthy":
                return last
        time.sleep(interval)
    return last


def verify_linkage(
    client: Any,
    trace_id: str,
    *,
    poll_status: bool = True,
    require: bool = False,
) -> Dict[str, Any]:
    """Verify inbound linkage for an uploaded trace.

    The **proof of linkage is the stamped ``integration_id``** read back from the
    trace API. (Integration ``status == "Healthy"`` is NOT proof a given trace was
    received — it only reflects the integration's active flag flipped by the ~30s
    sweeper, so it is a soft, best-effort signal, never the attribution check.)

    Behavior:

    * ``LAYERLENS_LIVE_INTEGRATION_ID`` set → assert the stamp equals it and (if
      ``poll_status``) poll the integration to ``Healthy`` within the budget.
    * ``require=True`` (and the env unset) → assert the trace was stamped with
      *some* integration_id (an API-key upload that should link must not silently
      go unlinked). Use this instead of the default record-only behavior when the
      test knows linkage must occur; callers lacking the env should ``pytest.skip``
      with :func:`linkage_skip_reason` rather than rely on a silent pass.
    * neither → record-only (no assertion); the caller is responsible for skipping
      visibly if it intended to assert.

    Set ``LAYERLENS_LIVE_LINKAGE_POLL_STATUS=0`` to assert only the id match and
    skip the status poll (useful when the caller toggles connectors active/inactive
    around each upload and can't wait for the sweeper). Returns a result row.
    """
    stamped = trace_integration_id(client, trace_id)
    result: Dict[str, Any] = {"integration_id": stamped, "linked": stamped is not None}
    expected = expected_integration_id()
    if expected:
        assert stamped == expected, f"linkage: trace {trace_id} integration_id={stamped!r} != expected {expected!r}"
        if poll_status and os.environ.get("LAYERLENS_LIVE_LINKAGE_POLL_STATUS", "1") != "0":
            status = poll_status_healthy(client, expected)
            result["status"] = status
            # Soft signal only — Healthy reflects the integration's active flag, not
            # this trace's receipt; the integration_id stamp asserted above is the proof.
            assert status == "Healthy", (
                f"linkage: integration {expected} status={status!r} (expected 'Healthy' within {_TIMEOUT:.0f}s)"
            )
    elif require:
        assert stamped is not None, (
            f"linkage required: trace {trace_id} was not stamped with any integration_id "
            f"(an API-key upload that should link to a registered sdk_adapter integration went unlinked)"
        )
    return result
