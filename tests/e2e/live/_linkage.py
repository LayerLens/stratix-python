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


def verify_linkage(client: Any, trace_id: str, *, poll_status: bool = True) -> Dict[str, Any]:
    """Verify inbound linkage for an uploaded trace.

    Always records the stamped ``integration_id`` (from the API). When
    ``LAYERLENS_LIVE_INTEGRATION_ID`` is set, asserts an exact match and — if
    ``poll_status`` — that the integration reaches ``Healthy`` within the budget.
    Set ``LAYERLENS_LIVE_LINKAGE_POLL_STATUS=0`` to assert only the id match and
    skip the status poll (useful when the caller toggles connectors active/inactive
    around each upload and can't wait for the sweeper).
    Returns a result row for the report.
    """
    stamped = trace_integration_id(client, trace_id)
    result: Dict[str, Any] = {"integration_id": stamped, "linked": stamped is not None}
    expected = expected_integration_id()
    if expected:
        assert stamped == expected, f"linkage: trace {trace_id} integration_id={stamped!r} != expected {expected!r}"
        if poll_status and os.environ.get("LAYERLENS_LIVE_LINKAGE_POLL_STATUS", "1") != "0":
            status = poll_status_healthy(client, expected)
            result["status"] = status
            assert status == "Healthy", (
                f"linkage: integration {expected} status={status!r} (expected 'Healthy' within {_TIMEOUT:.0f}s)"
            )
    return result
