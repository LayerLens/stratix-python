# UCP (Universal Commerce Protocol) instrumentation sample

End-to-end demo of `UCPAdapter` — walks the UCP commerce lifecycle
in-memory: **supplier discovery** → **catalog browse** → **checkout
creation** → **checkout completion**. Uses a fake supplier
(`sup-acme`), a fake order (`ord-1`), and idempotent line items. No
real UCP server is contacted.

## Prerequisites

```bash
pip install 'layerlens[protocols-ucp]'
```

(The `protocols-ucp` extra is empty — no peer library is required.)

Optional environment, only needed if you want the events to actually
ship to a LayerLens deployment:

| Variable                       | Purpose                            |
|--------------------------------|------------------------------------|
| `LAYERLENS_STRATIX_API_KEY`    | Bearer token for atlas-app ingest. |
| `LAYERLENS_STRATIX_BASE_URL`   | atlas-app base URL.                |

## Run

```bash
uv run python -m samples.instrument.ucp.main
```

## Expected output

The sample drives four adapter hook calls. Event-type strings come
straight from `_commerce.py`:

| Event                              | Source hook                              |
|------------------------------------|------------------------------------------|
| `commerce.supplier.discovered`     | `adapter.on_supplier_discovered(...)`    |
| `commerce.catalog.browsed`         | `adapter.on_catalog_browsed(...)`        |
| `commerce.checkout.created`        | `adapter.on_checkout_created(...)`       |
| `commerce.checkout.completed`      | `adapter.on_checkout_completed(...)`     |

The sample prints a one-line summary and the sink's `batches_sent`
counter on the way out.

## What this demonstrates

- `UCPAdapter.connect()` brings the adapter to `HEALTHY` with no peer
  library installed.
- `on_supplier_discovered` records a supplier with
  `discovery_method="well_known"` and capability list
  `["catalog", "checkout", "refunds"]`.
- `on_catalog_browsed` records 18 items viewed / 2 selected for the
  same supplier.
- `on_checkout_created` opens a checkout session
  (`cs-1`, `total_amount=49.98`, `idempotency_key="idem-1"`).
- `on_checkout_completed` closes the session with the resulting
  `order_id="ord-1"`. The adapter computes the cart-to-completion
  duration internally.
- `HttpEventSink` batching with `max_batch=10`, `flush_interval_s=1.0`.
- Clean shutdown via `adapter.disconnect()` + `sink.close()` in a
  `finally` block.

`commerce.*` events bypass `CaptureConfig` gating via
`ALWAYS_ENABLED_EVENT_TYPES`. The full hook-to-event mapping is in
[`docs/adapters/protocols-ucp.md`](../../../docs/adapters/protocols-ucp.md).
