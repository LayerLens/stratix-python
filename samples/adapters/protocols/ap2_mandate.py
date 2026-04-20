"""Sample: AP2 payments mandate chain with guardrails."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from adapters._shared import capture_events  # type: ignore[import-not-found]

from layerlens.instrument.adapters.protocols.ap2 import (
    AP2Guardrails,
    instrument_ap2,
    uninstrument_ap2,
)


class _FakeAP2Client:
    def create_intent_mandate(
        self, *, mandate_id: str, amount: float, merchant: str, expires_at: float | None = None
    ) -> dict:
        return {"mandate_id": mandate_id}

    def sign_payment_mandate(self, *, mandate_id: str, amount: float, merchant: str) -> dict:
        return {"mandate_id": mandate_id, "signature": "sig-xyz"}

    def issue_receipt(self, *, receipt_id: str, mandate_id: str, amount: float, merchant: str) -> dict:
        return {"receipt_id": receipt_id}


def main() -> None:
    client = _FakeAP2Client()
    guardrails = AP2Guardrails(max_transaction=100.0, merchant_whitelist=["Bookstore"])
    instrument_ap2(client, guardrails=guardrails)
    try:
        with capture_events("ap2"):
            client.create_intent_mandate(mandate_id="m-1", amount=50, merchant="Bookstore")
            client.sign_payment_mandate(mandate_id="m-1", amount=50, merchant="Bookstore")
            client.issue_receipt(receipt_id="r-1", mandate_id="m-1", amount=50, merchant="Bookstore")
    finally:
        uninstrument_ap2()


if __name__ == "__main__":
    main()
