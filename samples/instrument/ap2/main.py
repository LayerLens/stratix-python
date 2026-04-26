"""Sample: simulate an AP2 commerce flow with the LayerLens adapter.

Walks the three-stage AP2 authorization chain in-memory: intent mandate
(with org-policy guardrail evaluation) → payment mandate signing →
payment receipt issuance. The adapter emits ``commerce.intent.*`` +
``commerce.mandate.*`` + ``commerce.payment.*`` events that ship to
atlas-app via ``HttpEventSink``.

The sample uses a fake signature, fake merchant, and fake payment ID —
no real payment processor is contacted.

Required environment:

* ``LAYERLENS_STRATIX_API_KEY`` — your LayerLens API key (optional).
* ``LAYERLENS_STRATIX_BASE_URL`` — atlas-app base URL (optional).

Run::

    pip install 'layerlens[protocols-ap2]'
    python -m samples.instrument.ap2.main
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone, timedelta

from layerlens.instrument.adapters._base import CaptureConfig
from layerlens.instrument.transport.sink_http import HttpEventSink
from layerlens.instrument.adapters.protocols.ap2 import AP2Adapter


def main() -> int:
    sink = HttpEventSink(
        adapter_name="ap2",
        path="/telemetry/spans",
        max_batch=10,
        flush_interval_s=1.0,
    )

    adapter = AP2Adapter(capture_config=CaptureConfig.standard())
    adapter.add_sink(sink)
    adapter.connect()

    org_id = "org-sample"
    agent_id = "agent-shopper"

    try:
        adapter.configure_policy(
            org_id=org_id,
            max_single_tx=500.0,
            daily_limit=2000.0,
            allowed_merchants=["merchant-shopify"],
        )

        intent_expiry = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        violations = adapter.on_intent_mandate_created(
            mandate_id="im-1",
            description="Buy 1 t-shirt under $50.",
            org_id=org_id,
            merchants=["merchant-shopify"],
            max_amount=50.0,
            currency="USD",
            intent_expiry=intent_expiry,
            agent_id=agent_id,
        )

        if violations:
            print(f"Guardrail violations: {violations}", file=sys.stderr)
            return 1

        adapter.on_payment_mandate_signed(
            mandate_id="im-1",
            payment_details_id="pd-1",
            total_amount=49.99,
            merchant_agent="merchant-shopify",
            org_id=org_id,
            currency="USD",
            payment_method="CARD",
            signature="fake-jwt-for-sample",
            agent_id=agent_id,
        )

        adapter.on_payment_receipt_issued(
            mandate_id="im-1",
            payment_id="pay-1",
            amount=49.99,
            org_id=org_id,
            currency="USD",
            status="success",
            merchant_confirmation_id="conf-1",
        )

        if hasattr(sink, "stats"):
            stats = sink.stats()
            print(f"Batches sent: {stats.get('batches_sent', 0)}")
        print("Emitted AP2 events: intent + validated + mandate.signed + receipt")
    except Exception as exc:
        print(f"AP2 scenario failed: {exc}", file=sys.stderr)
        return 1
    finally:
        sink.close()
        adapter.disconnect()

    print("Telemetry shipped. Check the LayerLens dashboard adapter health page.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
