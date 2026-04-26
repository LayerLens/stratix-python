"""Sample: simulate a UCP commerce flow with the LayerLens adapter.

Walks the UCP commerce lifecycle in-memory: supplier discovery → catalog
browse → checkout creation → checkout completion. The adapter emits
``commerce.supplier.discovered`` + ``commerce.catalog.browsed`` +
``commerce.checkout.*`` events that ship to atlas-app via
``HttpEventSink``.

The sample uses fake supplier IDs and order IDs — no real UCP server
is contacted.

Required environment:

* ``LAYERLENS_STRATIX_API_KEY`` — your LayerLens API key (optional).
* ``LAYERLENS_STRATIX_BASE_URL`` — atlas-app base URL (optional).

Run::

    pip install 'layerlens[protocols-ucp]'
    python -m samples.instrument.ucp.main
"""

from __future__ import annotations

import sys

from layerlens.instrument.adapters._base import CaptureConfig
from layerlens.instrument.transport.sink_http import HttpEventSink
from layerlens.instrument.adapters.protocols.ucp import UCPAdapter


def main() -> int:
    sink = HttpEventSink(
        adapter_name="ucp",
        path="/telemetry/spans",
        max_batch=10,
        flush_interval_s=1.0,
    )

    adapter = UCPAdapter(capture_config=CaptureConfig.standard())
    adapter.add_sink(sink)
    adapter.connect()

    org_id = "org-sample"

    try:
        adapter.on_supplier_discovered(
            supplier_id="sup-acme",
            name="Acme Supplies",
            profile_url="https://acme.example.com/.well-known/ucp.json",
            org_id=org_id,
            capabilities=["catalog", "checkout", "refunds"],
            discovery_method="well_known",
        )

        adapter.on_catalog_browsed(
            supplier_id="sup-acme",
            org_id=org_id,
            items_viewed=18,
            items_selected=2,
        )

        adapter.on_checkout_created(
            checkout_session_id="cs-1",
            supplier_id="sup-acme",
            line_items=[
                {"item_id": "sku-1", "quantity": 2, "unit_price": 24.99},
            ],
            total_amount=49.98,
            org_id=org_id,
            currency="USD",
            idempotency_key="idem-1",
        )

        adapter.on_checkout_completed(
            checkout_session_id="cs-1",
            org_id=org_id,
            order_id="ord-1",
        )

        if hasattr(sink, "stats"):
            stats = sink.stats()
            print(f"Batches sent: {stats.get('batches_sent', 0)}")
        print("Emitted UCP events: supplier + catalog + checkout.created + checkout.completed")
    except Exception as exc:
        print(f"UCP scenario failed: {exc}", file=sys.stderr)
        return 1
    finally:
        sink.close()
        adapter.disconnect()

    print("Telemetry shipped. Check the LayerLens dashboard adapter health page.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
