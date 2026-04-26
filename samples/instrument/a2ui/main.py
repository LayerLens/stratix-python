"""Sample: simulate an A2UI surface lifecycle with the LayerLens adapter.

Constructs an in-memory A2UI session — a surface is created, a user
clicks a confirm button, and the adapter emits two ``commerce.ui.*``
events that ship to atlas-app via ``HttpEventSink``.

The PII-sensitive ``context`` dict is hashed before emission (hard
guarantee in the adapter), so the cleartext cart total never leaves
this process — only the sha256 of the context dict appears on the event.

Required environment:

* ``LAYERLENS_STRATIX_API_KEY`` — your LayerLens API key (optional).
* ``LAYERLENS_STRATIX_BASE_URL`` — atlas-app base URL (optional).

Run::

    pip install 'layerlens[protocols-a2ui]'
    python -m samples.instrument.a2ui.main
"""

from __future__ import annotations

import sys

from layerlens.instrument.adapters._base import CaptureConfig
from layerlens.instrument.transport.sink_http import HttpEventSink
from layerlens.instrument.adapters.protocols.a2ui import A2UIAdapter


def main() -> int:
    sink = HttpEventSink(
        adapter_name="a2ui",
        path="/telemetry/spans",
        max_batch=10,
        flush_interval_s=1.0,
    )

    adapter = A2UIAdapter(capture_config=CaptureConfig.standard())
    adapter.add_sink(sink)
    adapter.connect()

    org_id = "org-sample"

    try:
        adapter.on_surface_created(
            surface_id="surf-checkout-1",
            org_id=org_id,
            root_component_id="cmp-checkout-root",
            component_count=12,
        )

        adapter.on_user_action(
            surface_id="surf-checkout-1",
            action_name="confirm_purchase",
            org_id=org_id,
            component_id="cmp-confirm-btn",
            context={"cart_total": 49.99, "currency": "USD"},
        )

        adapter.on_user_action(
            surface_id="surf-checkout-1",
            action_name="select_payment_method",
            org_id=org_id,
            component_id="cmp-payment-select",
            context={"method": "card"},
        )

        if hasattr(sink, "stats"):
            stats = sink.stats()
            print(f"Batches sent: {stats.get('batches_sent', 0)}")
        print("Emitted A2UI events: surface_created + 2x user_action")
    except Exception as exc:
        print(f"A2UI scenario failed: {exc}", file=sys.stderr)
        return 1
    finally:
        sink.close()
        adapter.disconnect()

    print("Telemetry shipped. Check the LayerLens dashboard adapter health page.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
