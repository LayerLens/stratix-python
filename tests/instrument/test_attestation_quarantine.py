"""Attestation fail-CLOSED: unattested traces are quarantined, not uploaded
(A9 / R8 / LAY-3628, product-approved 2026-06-25).

Before: ``_build_trace_payload`` caught any chain-build exception, set
``attestation={"attestation_error": ...}``, and ``flush()`` uploaded the trace
anyway — an unattested trace passed every runtime net (the integrity guarantee
was void). Now ``flush()`` routes a trace whose attestation can't be built to a
quarantine sink instead of the normal upload, and a chain explicitly
``terminate()``-d (a safety-stop / policy halt) also fails closed.

Bite: revert flush() to always ``enqueue_upload`` and the quarantine tests go RED.
"""

from __future__ import annotations

import pytest

from layerlens.instrument import _collector as cm
from layerlens.instrument._collector import TraceCollector
from layerlens.instrument._capture_config import CaptureConfig

pytestmark = pytest.mark.invariant


@pytest.fixture
def routed(monkeypatch):
    """Capture what flush() routes where: uploaded vs quarantined."""
    uploaded: list = []
    quarantined: list = []
    monkeypatch.setattr(cm, "enqueue_upload", lambda client, payload: uploaded.append(payload))
    monkeypatch.setattr(cm, "_quarantine_sink", quarantined.append)
    return uploaded, quarantined


def _collector_with_event() -> TraceCollector:
    c = TraceCollector(object(), CaptureConfig())
    c.emit("agent.input", {"agent_name": "a"}, span_id="s1")
    return c


def test_attested_trace_is_uploaded(routed) -> None:
    uploaded, quarantined = routed
    c = _collector_with_event()
    c.flush()
    assert len(uploaded) == 1, "a well-attested trace must upload normally"
    assert not quarantined
    assert uploaded[0]["attestation"].get("root_hash"), "uploaded trace missing root_hash"


def test_chain_build_failure_quarantines_not_uploads(routed, monkeypatch) -> None:
    uploaded, quarantined = routed
    c = _collector_with_event()

    def boom() -> None:
        raise RuntimeError("hash chain corrupt")

    monkeypatch.setattr(c._chain, "finalize", boom)
    c.flush()
    assert not uploaded, "unattested trace was UPLOADED (attestation fail-open / R8)"
    assert len(quarantined) == 1, "unattested trace was not quarantined"
    assert "attestation_error" in quarantined[0]["attestation"]
    # events are preserved in the quarantined payload (quarantine != silent drop)
    assert quarantined[0]["events"], "quarantined trace dropped its events"


def test_terminated_chain_fails_closed(routed) -> None:
    """A safety-stop terminates the chain; the trace then fails closed at flush."""
    uploaded, quarantined = routed
    c = _collector_with_event()
    c.terminate("policy.violation: prompt-injection halt")
    c.flush()
    assert not uploaded, "a terminated (non-attestable) trace was uploaded"
    assert len(quarantined) == 1, "terminated trace not quarantined"


def test_quarantine_default_sink_is_safe_when_unset(monkeypatch) -> None:
    """With no sink configured, an unattested trace must still NOT upload (it is
    dropped from the normal path + logged), never silently presented as attested."""
    uploaded: list = []
    monkeypatch.setattr(cm, "enqueue_upload", lambda client, payload: uploaded.append(payload))
    monkeypatch.setattr(cm, "_quarantine_sink", None)
    c = _collector_with_event()
    monkeypatch.setattr(c._chain, "finalize", lambda: (_ for _ in ()).throw(RuntimeError("x")))
    c.flush()  # must not raise even with no sink
    assert not uploaded, "unattested trace uploaded when no quarantine sink set"
