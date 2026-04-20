"""Sample: UCP universal commerce flow."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from adapters._shared import capture_events  # type: ignore[import-not-found]

from layerlens.instrument.adapters.protocols.ucp import instrument_ucp, uninstrument_ucp


class _FakeUCPClient:
    def discover_suppliers(self, *, query: str):
        return [{"id": "acme", "name": "Acme"}, {"id": "widgets", "name": "Widgets Inc"}]

    def browse_catalog(self, *, supplier_id: str, query: str):
        return [{"id": f"item-{i}"} for i in range(5)]

    def start_checkout(self, *, supplier_id: str, session_id: str):
        return {"session_id": session_id, "status": "started"}

    def complete_checkout(self, session_id: str, *, supplier_id: str, amount: float):
        return {"session_id": session_id, "status": "completed"}

    def issue_refund(self, *, session_id: str, amount: float, reason: str):
        return {"ok": True}


def main() -> None:
    client = _FakeUCPClient()
    instrument_ucp(client)
    try:
        with capture_events("ucp"):
            client.discover_suppliers(query="books")
            client.browse_catalog(supplier_id="acme", query="novel")
            client.start_checkout(supplier_id="acme", session_id="sess-1")
            client.complete_checkout("sess-1", supplier_id="acme", amount=29.99)
    finally:
        uninstrument_ucp()


if __name__ == "__main__":
    main()
