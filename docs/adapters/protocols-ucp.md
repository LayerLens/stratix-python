# UCP (Universal Commerce Protocol) adapter

`layerlens.instrument.adapters.protocols.ucp.UCPAdapter` instruments the
Universal Commerce Protocol — supplier discovery, catalog browsing,
checkout sessions, and order refunds.

## Install

```bash
pip install 'layerlens[protocols-ucp]'
```

The `protocols-ucp` extra has no required dependencies; the adapter
operates on protocol payloads, not on a specific SDK.

## Quick start

```python
from layerlens.instrument.adapters.protocols.ucp import UCPAdapter
from layerlens.instrument.transport.sink_http import HttpEventSink

sink = HttpEventSink(adapter_name="ucp")
adapter = UCPAdapter()
adapter.add_sink(sink)
adapter.connect()

org_id = "org-123"

adapter.on_supplier_discovered(
    supplier_id="sup-acme",
    name="Acme Supplies",
    profile_url="https://acme.example.com/.well-known/ucp.json",
    org_id=org_id,
    capabilities=["catalog", "checkout"],
    discovery_method="well_known",
)

adapter.on_checkout_created(
    checkout_session_id="cs-1",
    supplier_id="sup-acme",
    line_items=[{"item_id": "sku-1", "quantity": 2, "unit_price": 24.99}],
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

adapter.disconnect()
sink.close()
```

## What's wrapped

`UCPAdapter` exposes hooks the host calls at each lifecycle stage:

- `on_supplier_discovered(supplier_id, name, profile_url, org_id,
  capabilities, discovery_method)`
- `on_catalog_browsed(supplier_id, org_id, items_viewed, items_selected)`
- `on_checkout_created(checkout_session_id, supplier_id, line_items,
  total_amount, org_id, currency, idempotency_key)`
- `on_checkout_completed(checkout_session_id, org_id, order_id)`
- `on_order_refunded(order_id, refund_amount, currency, org_id, reason)`

Checkout sessions are tracked in-process from `created` to `completed`,
and the duration is computed and logged.

## Events emitted

| Event | Layer | When |
|---|---|---|
| `commerce.supplier.discovered` | L7b | Per `on_supplier_discovered`. |
| `commerce.catalog.browsed` | L7b | Per `on_catalog_browsed`. |
| `commerce.checkout.created` | L7b | Per `on_checkout_created`. |
| `commerce.checkout.completed` | L7b | Per `on_checkout_completed`. |
| `commerce.order.refunded` | L7b | Per `on_order_refunded`. |
| `commerce.supplier.event` | L7b | Per `on_supplier_event` callback (catch-all). |

All `commerce.*` events bypass `CaptureConfig` gating via
`ALWAYS_ENABLED_EVENT_TYPES`.

## UCP specifics

- **Discovery methods**: `well_known` (RFC 8615 `/.well-known/ucp.json`),
  `registry` (a registry returned the supplier), `referral` (another
  agent referred us).
- **Catalog browse rollups**: `on_catalog_browsed` is summary-only — only
  `items_viewed` + `items_selected` counts are captured to keep the
  payload size bounded for high-frequency browsing.
- **Idempotency**: `on_checkout_created` accepts an `idempotency_key` so
  retries can be correlated. The platform uses this key to dedupe
  duplicate checkout creation events.
- **Semantic memory**: when `memory_service=` is set, supplier metadata
  is stored as a semantic memory entry on first discovery.
- **Session-level duration**: duration is computed from
  `_session_start_times[checkout_session_id]` to the completion call.

## Capture config

`commerce.*` events are always captured regardless of `CaptureConfig`
flags.

## BYOK

Not applicable — UCP authentication is per-supplier and managed by the
host application's UCP client.
